# Copyright (c) 2016 rllab contributors
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pdb
import copy
import numpy as np
from scipy import linalg as la
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.oracle import rlOracle
from rl.core.oracles import LikelihoodRatioOracle
from rl.core.function_approximators.policies import Policy
from rl.core.datasets import Dataset


class ValueBasedPolicyGradient(rlOracle):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """

    def __init__(self, policy, ae,
                 use_is='one', avg_type='sum',
                 biased=True, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # define the internal oracle
        assert isinstance(policy, Policy)
        self._policy = copy.deepcopy(policy)  # just a template
        self._or = LikelihoodRatioOracle(
            self._logp_fun, self._logp_grad,
            biased=biased,  # basic mvavg
            use_log_loss=use_log_loss, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one', 'multi', None]
        self._use_is = use_is  # use importance sampling for polcy gradient
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._scale = None
        self._ro = None

    def _logp_fun(self, x):
        self._policy.variable = x
        return self._policy.logp(self.ro['obs_short'], self.ro['acs'])

    def _logp_grad(self, x, fs):
        self._policy.variable = x
        return self._policy.logp_grad(self.ro['obs_short'], self.ro['acs'], fs)

    @property
    def ro(self):
        return self._ro

    def fun(self, x):
        return self._or.fun(x) * self._scale

    def grad(self, x):
        return self._or.grad(x) * self._scale

    def update(self, ro, policy, update_nor=True, update_vfn=True, **kwargs):
        # Sync policies' parameters.
        self._policy.assign(policy)  # NOTE sync BOTH variables and parameters
        # Compute adv.
        self._ro = ro
        advs, _ = self._ae.advs(self.ro, use_is=self._use_is)
        adv = np.concatenate(advs)
        self._scale = 1.0 if self._avg_type == 'avg' else len(adv)/len(advs)
        # Update the loss function.
        if self._or._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self._use_is:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self._policy))
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self._use_is in ['one', 'multi']
            w_or_logq = ro['lps']
        # Update the LikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=update_nor)  # loss is negative reward
        # Update the value function at the end, so it's unbiased.
        if update_vfn:
            return self.update_vfn(ro, **kwargs)

    def update_vfn(self, ro, **kwargs):
        return self._ae.update(ro, **kwargs)


class ValuedBasedParameterExploringPolicyGradient(ValueBasedPolicyGradient):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

           \nabla p(\theta)   E_{d_\pi} E_{\pi} [ \sum r ]

        where p(\theta) is the distribution of policy parameters.
    """

    def __init__(self, distribution, *args, use_is=None, **kwargs):
        use_is = None  # importance sampling is not considered
        super().__init__(distribution, *args, use_is=use_is, **kwargs)
        self._distribution = self._policy  # so it's not confusing
        del self._policy
        assert self._use_is is None
        assert self._avg_type in ['sum']
        del self._avg_type
        self._scale = 1.0
        self.sampled_vars = None

    def _logp_fun(self, x):
        self._distribution.variable = x
        z = np.empty((len(self.sampled_vars), 0))
        return self._distribution.logp(z, self.sampled_vars)

    def _logp_grad(self, x, fs):
        self._distribution.variable = x
        z = np.empty((len(self.sampled_vars), 0))
        return self._distribution.logp_grad(z, self.sampled_vars, fs)

    def update(self, ro, distribution, update_nor=True, update_vfn=True, **kwargs):
        # NOTE we assume the sampled policies (given in `ro` as an attribute
        # `pol_var`) are i.i.d.  according to `distribution`, while the
        # behvaior policy which is used to collect the data could be different.
        # When `use_is` is "multi", the effects of behavior policy will be
        # corrected.

        # Make sure the sampled policy for a rollout is saved.
        self.sampled_vars = np.array([r.pol_var for r in ro])

        # Sync parameters.
        self._distribution.assign(distribution)
        # Compute adv.
        self._ro = ro

        def compute_adv(r):
            advs, _ = self._ae.advs([r], use_is=self._use_is)
            return advs[0][0]  # we only concern the adv at the first time step
        adv = np.array([compute_adv(r) for r in self.ro])

        # Update the loss function.
        # NOTE the weight here is for `distribution`
        if self._or._use_log_loss:
            # - \E_{\theta} log p(\theta) E_{ob ~ p0} E_{ac ~ \pi | ob} [ adv(ob, ac) ]
            w_or_logq = np.ones_like(adv)  # w
        else:  # False or None
            # - \E_{\theta} E_{ob ~ p0} E_{ac ~ \pi | ob} [ adv(ob, ac) ]
            w_or_logq = self._logp_fun(self._distribution.variable)  # log_q

        # Update the LikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=update_nor)  # loss is negative reward
        # Update the value function at the end, so it's unbiased.
        if update_vfn:
            return self.update_vfn(ro, **kwargs)


class ValueBasedExpertGradientByRandomTimeSampling(rlOracle):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """

    def __init__(self, policy, ae,
                 use_is='one',
                 biased=True, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # Define the internal oracles
        assert isinstance(policy, Policy)
        self._policy = copy.deepcopy(policy)  # just a template

        def logp_fun(x, ro):
            self._policy.variable = x
            return self._policy.logp(ro['obs_short'], ro['acs'])

        def logp_grad(x, fs, ro):
            self._policy.variable = x
            return self._policy.logp_grad(ro['obs_short'], ro['acs'], fs)

        self._scale_or, self._scale_cv = 0., 0.
        self._ro_or, self._ro_cv = None, None
        # noisy gradient
        self._or = LikelihoodRatioOracle(
            lambda var: logp_fun(var, self._ro_or),
            lambda var, fs: logp_grad(var, fs, self._ro_or),
            biased=biased,  # basic mvavg
            use_log_loss=use_log_loss, normalized_is=normalized_is)
        # another oracle for control variate's bias
        self._cv = LikelihoodRatioOracle(
            lambda var: logp_fun(var, self._ro_cv),
            lambda var, fs: logp_grad(var, fs, self._ro_cv),
            biased=biased,  # basic mvavg
            use_log_loss=use_log_loss, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one']  # , 'multi']
        self._use_is = use_is  # use importance sampling for polcy gradient

    @property
    def ro(self):
        return self._ro_or + self._ro_cv

    def fun(self, x):
        f1 = 0. if self._ro_exp is None else self._or.fun(x)*self._scale_or
        f2 = 0. if self._ro_pol is None else self._cv.fun(x)*self._scale_cv
        return f1+f2

    def grad(self, x):
        g1 = np.zeros_like(x) if self._ro_or is None \
            else self._or.grad(x)*self._scale_or
        g2 = np.zeros_like(x) if self._ro_cv is None \
            else self._cv.grad(x)*self._scale_cv
        print('noisy grad', np.linalg.norm(g1), 'cv grad', np.linalg.norm(g2))
        return g1+g2

    def update(self, ro_exp=None, ro_pol=None, policy=None, update_nor=True, **kwargs):
        """ Need to provide either `ro_exp` or `ro_pol`, and `policy`.

            `ro_exp` is used to compute an unbiased but noisy estimate of

                E_{pi}[\nabla \pi(s,a) \hat{A}_{\pi^*}(s,a)]

            when \hat{A}_{\pi^*} given by `self._or` is unbiased.

            `ro_pol` provides a biased gradient which can be used as a control
            variate (when `ro_exp` is provided) or just to define a biased
            oracle.
        """
        assert (ro_exp is not None) or (ro_pol is not None)
        assert policy is not None

        # Sync policies' parameters.
        self._policy.assign(policy)  # NOTE sync BOTH variables and parameters
        # Update the oracles
        n_rollouts = len(ro_exp) if ro_pol is None else len(ro_pol)
        self._ro_or = None
        if ro_exp is not None:
            # compute adv
            if len(ro_exp) > 0:
                advs, _ = self._ae.advs(ro_exp, use_is=self._use_is)
                advs = [a[0:1]*r.scale for a, r in zip(advs, ro_exp)]
                adv = np.concatenate(advs)
                if ro_pol is not None:  # compute the control variate
                    advs_cv, _ = self._ae.advs(ro_exp, use_is=self._use_is, lambd=0.)
                    advs_cv = [a[0:1]*r.scale for a, r in zip(advs_cv, ro_exp)]
                    adv -= np.concatenate(advs_cv)
                logq = np.concatenate([r.lps[0:1] for r in ro_exp])
                # update noisy oracle
                self._scale_or = len(adv)/n_rollouts
                self._or.update(-adv, logq, update_nor=update_nor)  # loss is negative reward
                self._ro_or = Dataset([r[0:1] for r in ro_exp])  # for defining logp

        self._ro_cv = None
        if ro_pol is not None:
            # update biased oracle
            advs, _ = self._ae.advs(ro_pol, use_is=self._use_is, lambd=0.)
            adv = np.concatenate(advs)
            self._scale_cv = len(adv)/n_rollouts
            logq = ro_pol['lps']
            self._cv.update(-adv, logq, update_nor=update_nor)  # loss is negative reward
            self._ro_cv = ro_pol  # for defining logp

        # Update the value function at the end, so it's unbiased.
        if ro_exp is not None:
            return self._ae.update(ro_exp, **kwargs)
        else:  # when biased gradient is used
            return self._ae.update(ro_pol, **kwargs)


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
        # Warm up for 'traj' cvtype, first do 'state' for some iterations.
        if cvtype == 'traj' and switch_from_cvtype_state_at_itr is not None:
            self._switch_from_cvtype_state_at_itr = switch_from_cvtype_state_at_itr
        else:
            self._switch_from_cvtype_state_at_itr = None

        self._ac_dim, self._ob_dim = policy.y_shape[0], policy.x_shape[0]
        self._ro = None

        self._switched_back = False
        if self._switch_from_cvtype_state_at_itr is not None:
            self._saved_cvtype = self._cvtype
            self._cvtype = 'state'

        self._update_vfn = (cvtype != 'nocv')
        self._update_dyn = (cvtype == 'traj')

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
            acs = self.preprocess_acs(acs)
            return self._sim._predict.__self__.update(np.hstack([obs_curr, acs]),
                                                      obs_next, **kwargs)
        return .0, .0, .0

    def preprocess_acs(self, acs):
        # XXX only suit for DartEnv.
        # Use the outpus as inputs to dynamics model to ease learning.
        acs = np.clip(acs, *self._sim._action_clip)
        # acs = acs * self._sim._action_scale[:, None]  # broadcasting to rows
        return acs

    def predict_vfns(self, obs, dones=None):
        # Return a flat np array.
        vs = np.ravel(self._ae.vfn.predict(obs))
        if dones is not None:
            vs[dones] = .0
        return vs

    def approximate_qfns(self, obs, acs):
        # Return a flat np array.
        assert len(obs) == len(acs)
        # rws should be based on acs without processing.
        rws = self._sim._batch_reward(obs, sts=None, acs=acs)
        acs1 = self.preprocess_acs(acs)
        next_obs = self._sim._predict(np.hstack([obs, acs1]))
        next_dones = self._sim._batch_is_done(next_obs)
        vs = self.predict_vfns(next_obs, next_dones)
        qs = rws + self._ae.delta * vs
        return qs

    def grad(self, x):
        # Assign policy variables.
        self._policy.variable = x
        # Note that x is not used. _policy is set in update().
        # g WithOut cv, Difference Estimators for the CURrent step and for the FUTure steps:
        gwocv, decur, defut = .0, .0, .0
        # Go through rollout one by one.
        if self._cv_onestep_weighting:
            cv_onestep_ws = self._ae.weights(self._ro, policy=self.policy)
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
            nqmc = self._policy.logp_grad(rollout.obs_short, rollout.acs, gamma_decay * qmc)
            gwocv += nqmc
            # Difference estimators: decur and defut.
            if self._cvtype == 'nocv':
                pass
            elif self._cvtype == 'state':
                vhat = self.predict_vfns(rollout.obs_short)
                decur += self._policy.logp_grad(rollout.obs_short, rollout.acs, gamma_decay * vhat)
            elif self._cvtype == 'traj':
                # Use np array operations to avoid another for loop over steps.
                # CV for step t.
                qhat = self.approximate_qfns(rollout.obs_short, rollout.acs)  # T
                # The same randomness for all the steps to reduce variance.
                # I: number of ac samples, rit: ac randomness in I T size.
                rit = np.random.normal(size=[self._n_ac_samples, self._ac_dim])  # I x d_a
                rit = np.tile(rit, [len(rollout), 1])  # I T x d_a
                oit = np.repeat(rollout.obs_short, self._n_ac_samples, axis=0)  # T x d_o -> I T x d_o
                ait = self._policy.derandomize(oit, rit)  # I T x d_a
                qhatit = self.approximate_qfns(oit, ait)  # I T
                vhatit = self.predict_vfns(rollout.obs_short)  # for reducing the variance of enqhat
                vhatit = np.repeat(vhatit, self._n_ac_samples)  # I T
                gamma_decay_it = np.repeat(gamma_decay, self._n_ac_samples)  # I T
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
                # Broadcasting gamma_decay to each row (t).
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
