# Copyright (c) 2016 rllab contributors
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
from scipy import linalg as la
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.oracle import rlOracle
from rl.core.function_approximators.policies import Policy
from rl.core.utils.math_utils import compute_explained_variance


class ValueBasedPolicyGradientWithTrajCV(rlOracle):
    def __init__(self, policy, ae, avgtype='sum',
                 cvtype='state', n_cv_steps=1, cv_decay=1.0, n_ac_samples=100, sim=None,
                 cv_onestep_weighting=False,
                 switch_from_cvtype_state_at_itr=None):
        # Consider delta and gamma, but no importance sampling capability yet.
        assert cvtype in ['nocv', 'state', 'traj']
        assert isinstance(ae, ValueBasedAE)
        assert isinstance(policy, Policy)

        self._policy = copy.deepcopy(policy)
        self._ae = ae
        self._avgtype = avgtype
        self._cvtype = cvtype
        self._n_cv_steps = n_cv_steps  # number of difference estimators to use, starting from t
        self._cv_decay = cv_decay  # additional decay for difference estimators for future steps
        self._n_ac_samples = n_ac_samples  # number of MC samples for E_A
        self._cv_onestep_weighting = cv_onestep_weighting
        self._sim = sim
        self._ac_dim, self._ob_dim = policy.y_shape[0], policy.x_shape[0]
        self._ro = None

        self._update_vfn = (cvtype != 'nocv')
        self._update_dyn = ((cvtype == 'traj') and (sim is not None)
                            and hasattr(sim, 'learned_predict') and sim.learned_predict)

        # Warm up for 'traj' cvtype, first do 'state' for some iterations.
        # TEMPORARILY change cvtype.
        self._switched_back = False
        if cvtype == 'traj' and switch_from_cvtype_state_at_itr is not None:
            self._switch_from_cvtype_state_at_itr = switch_from_cvtype_state_at_itr
        else:
            self._switch_from_cvtype_state_at_itr = None
        if self._switch_from_cvtype_state_at_itr is not None:
            self._saved_cvtype = self._cvtype
            self._cvtype = 'state'

    def update(self, ro, policy, itr=None, **kwargs):
        if (itr is not None and self._switch_from_cvtype_state_at_itr is not None and
                itr >= self._switch_from_cvtype_state_at_itr and not self._switched_back):
            print('Switch to fancy cv: {} from {}'.format(self._saved_cvtype, self._cvtype))
            self._cvtype = self._saved_cvtype
            self._switched_back = True
        self._policy.assign(policy)  # NOTE sync BOTH variables and parameters
        self._ro = ro

    def update_vfn(self, ro, **kwargs):
        if self._update_vfn:
            return self._ae.update(ro, **kwargs)
        return .0, .0, .0

    def update_dyn(self, ro, **kwargs):
        if self._update_dyn:
            obs_curr = np.concatenate([r.obs[:-1] for r in ro])
            obs_next = np.concatenate([r.obs[1:] for r in ro])
            acs = np.concatenate([r.acs for r in ro])
            return self._sim.predict.__self__.update(np.hstack([obs_curr, acs]),
                                                      obs_next, **kwargs)
        return .0, .0, .0

    def evaluate_vfn(self, ro):
        # per-step importance due to off policy
        w = np.concatenate(self._ae.weights(ro)) if self._ae.use_is else 1.0
        vhat = (w*np.concatenate(self._ae.qfns(ro, self._ae.pe_lambd))).reshape([-1, 1])  # target
        ev = compute_explained_variance(self._ae.vfn(ro['obs_short']), vhat)
        return ev

    def predict_vfns(self, obs, dones=None):
        # Return a flat np array.
        vs = np.ravel(self._ae.vfn.predict(obs))
        if dones is not None:
            vs[dones] = .0
        return vs

    def approximate_qfns(self, obs, acs, before_last=None):
        # Return a flat np array.
        assert len(obs) == len(acs)
        # rws should be based on acs without processing.
        rws = self._sim.batch_reward(obs[:, :-1], sts=None, acs=acs)
        next_obs = self._sim.predict(np.hstack([obs, acs]))
        next_dones = self._sim.batch_is_done(next_obs[:, :-1])
        if before_last is not None:
            assert len(before_last) == len(next_dones)
            next_dones[before_last] = True
        vs = self.predict_vfns(next_obs, next_dones)
        qs = rws + self._ae.delta * vs
        return qs

    def _verify_batch_reward(self, ro):
        # Verify batch_rewards!!!!
        batch_rws = self._sim.batch_reward(ro['obs_short'][:, :-1], sts=None, acs=ro['acs'])
        isclose_kwargs = {'atol': 1e-5, 'rtol': 1e-5}
        assert np.allclose(batch_rws, ro['rws_short'], **isclose_kwargs)

    def grad(self, x):

        # Assign policy variables.
        self._policy.variable = x
        # Note that x is not used. _policy is set in update().
        # g WithOut cv, Difference Estimators for the CURrent step and for the FUTure steps:
        gwocv, decur, defut = .0, .0, .0
        # Go through rollout one by one.
        if self._cv_onestep_weighting:
            cv_onestep_ws = self._ae.weights(self._ro, policy=self._policy)
        else:
            cv_onestep_ws = [np.ones(len(r)) for r in self._ro]
        for i, rollout in enumerate(self._ro):
            # Gradient without CV: gwocv.
            # +1 to consider the appending default vf.
            delta_decay = self._ae.delta ** np.arange(len(rollout)+1)
            delta_decay = np.triu(la.circulant(delta_decay).T, k=0)
            # Q function from MC samples.
            # Ignore the last item in rws, which is default vfn,
            # so that its len is the same as rollout.
            qmc = np.ravel(np.matmul(delta_decay, rollout.rws[:, None]))  # T
            qmc = qmc[:-1]
            # Neglect the last rw, which is the default v,
            # Mixing of G_t, gamma is the discount in problem definition.
            gamma_decay = self._ae.gamma ** np.arange(len(rollout))  # T
            nqmc = self._policy.logp_grad(rollout.obs_short, rollout.acs, gamma_decay * cv_onestep_ws[i] * qmc)
            gwocv += nqmc
            # Difference estimators: decur and defut.
            if self._cvtype == 'nocv':
                pass
            elif self._cvtype == 'state':
                vhat = self.predict_vfns(rollout.obs_short)
                decur += self._policy.logp_grad(rollout.obs_short, rollout.acs, gamma_decay * cv_onestep_ws[i] * vhat)
            elif self._cvtype == 'traj':
                # Use np array operations to avoid another for loop over steps.
                # CV for step t.
                before_last = np.zeros(len(rollout), dtype=bool)  # T
                before_last[-1] = True  # the one before last
                qhat = self.approximate_qfns(rollout.obs_short, rollout.acs, before_last)  # T
                # The same randomness for all the steps to reduce variance.
                # I: number of ac samples, rit: ac randomness in I T size.
                rit = np.random.normal(size=[self._n_ac_samples, self._ac_dim])  # I x d_a
                rit = np.exp(self._policy.lstd) * rit  # NOTE need to scale
                rit = np.tile(rit, [len(rollout), 1])  # I T x d_a
                oit = np.repeat(rollout.obs_short, self._n_ac_samples, axis=0)  # T x d_o -> I T x d_o
                ait = self._policy.derandomize(oit, rit)  # I T x d_a
                before_last_it = np.repeat(before_last, self._n_ac_samples)
                qhatit = self.approximate_qfns(oit, ait, before_last_it)  # I T
                vhat = self.predict_vfns(rollout.obs_short)  # for reducing the variance of enqhat
                vhatit = np.repeat(vhat, self._n_ac_samples, axis=0)  # I T
                gamma_decay_it = np.repeat(gamma_decay, self._n_ac_samples, axis=0)  # I T
                advhatit = qhatit - vhatit
                # Compute gamma^t E_A [ N_t (Qhat_t - Vhat_t)], for each t
                # Approximate of E_A [N Qhat]
                enqhat = self._policy.logp_grad(oit, ait, gamma_decay_it*advhatit)
                enqhat /= self._n_ac_samples
                nqhat = self._policy.logp_grad(rollout.obs_short, rollout.acs,
                                               cv_onestep_ws[i]*gamma_decay*qhat)  # weighting
                decur += nqhat - enqhat
                # Compute gamma^t E_A [(delta*cv_decay)^{k-t} Qhat_k], for each k, for each t
                eqhat = np.reshape(qhatit, [len(rollout), self._n_ac_samples])  # T x I
                eqhat = np.mean(eqhat, axis=1)  # T, take average
                # decaytt with zero diagonal terms. with shape T x T. each row is for t.
                # Something like:
                # 0.0  0.9  0.81
                #      0.0  0.9
                #           0.0
                decaytt = (self._ae.delta * self._cv_decay) ** np.arange(len(rollout))  # T
                if self._n_cv_steps is not None:
                    decaytt[min(self._n_cv_steps, len(rollout)):] = .0
                decaytt = np.triu(la.circulant(decaytt).T, k=1)  # T x T
                # Broadcasting gamma_decay to each column such that each row is multiplied by
                # gamma^t.
                decaytt = decaytt * gamma_decay[:, None]
                advhat = cv_onestep_ws[i] * qhat - eqhat
                advhat = np.ravel(np.matmul(decaytt, advhat[:, None]))
                defut += self._policy.logp_grad(rollout.obs_short, rollout.acs, advhat)

        if self._avgtype == 'avg':
            scale = 1.0 / np.prod([len(rollout) for rollout in self._ro])
        elif self._avgtype == 'sum':
            scale = 1.0 / len(self._ro)
        else:
            raise ValueError
        gwocv *= scale
        decur *= scale
        defut *= scale
        g = -(gwocv - decur - defut)
        return {'g': g, 'gwocv': gwocv, 'decur': decur, 'defut': defut}

    @property
    def ro(self):
        return self._ro