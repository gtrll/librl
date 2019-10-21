# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy as cp
import functools
import numpy as np
import psutil
from rl.algorithms.algorithm import Algorithm, PolicyAgent
from rl.algorithms.utils import get_learner
from rl.oracles.rl_oracles_cv import ValueBasedPolicyGradientWithTrajCV
from rl import online_learners as ol
from rl.policies import Policy
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz


class PolicyGradientWithTrajCV(Algorithm):
    def __init__(self, policy, ae,
                 scheduler_kwargs=None, learner_kwargs=None,
                 n_warm_up_itrs=None, n_pretrain_itrs=1,
                 train_vfn_using_sim=False, vfn_sim_ro_kwargs=None, vfn_sim=None,
                 or_kwargs=None, ss_sim=None):
        # ss_sim: Single-Step simulator to construct Q function.
        assert isinstance(policy, Policy)
        self.policy = policy

        # Create online learner.
        scheduler = ol.scheduler.PowerScheduler(**scheduler_kwargs)
        self.learner = get_learner(policy=policy, scheduler=scheduler, **learner_kwargs)
        self._optimizer = learner_kwargs['optimizer']

        # Create oracle.
        # ae is only for value function estimation, not used for adv computation,
        # therefore use_is can be set to 'one'.
        self.ae = ae
        self.oracle = ValueBasedPolicyGradientWithTrajCV(policy, self.ae, sim=ss_sim, **or_kwargs)

        # Misc.
        self._train_vfn_using_sim = train_vfn_using_sim
        self._vfn_sim_ro_kwargs = cp.deepcopy(vfn_sim_ro_kwargs)
        self._vfn_sim = vfn_sim  # is an MDP
        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs = n_warm_up_itrs
        self._itr = 0

    def get_policy(self):
        return self.policy

    def agent(self, mode):
        return PolicyAgent(self.policy)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                ros, _ = gen_ro(self.agent('behavior'))
                ro = self.merge(ros)
                self.oracle.update_vfn(ro)
                self.oracle.update_dyn(ro)
                self.policy.update(xs=ro['obs_short'])

    def update(self, ros, agents):
        # Aggregate data
        ro = self.merge(ros)
        evs = {}  # explained variances

        with timed('Update oracle'):
            self.oracle.update(ro, self.policy, itr=self._itr)

        if self._train_vfn_using_sim:
            with timed('Collect sim samples for updating vfn'):
                sim_ros, _ = self._vfn_sim.run(self.agent('behavior'), **self._vfn_sim_ro_kwargs)
                sim_ro = self.merge(sim_ros)
            with timed('Update vfn using sim'):
                _, evs['vfn_extra_ev0'], evs['vfn_extra_ev1'] = self.oracle.update_vfn(sim_ro)

        with timed('Compute policy gradient'):
            grads = self.oracle.grad(self.policy.variable)
            g = grads['g']

        with timed('Update vfn and dyn'):
            if not self._train_vfn_using_sim:
                _, evs['vfn_ev0'], evs['vfn_ev1'] = self.oracle.update_vfn(ro)
            else:
                evs['vfn_ev0'] = evs['vfn_ev1'] = self.oracle.evaluate_vfn(ro)
            _, evs['dyn_ev0'], evs['dyn_ev1'] = self.oracle.update_dyn(ro)

        with timed('Policy update'):
            if isinstance(self.learner, ol.FisherOnlineOptimizer):
                if self._optimizer == 'trpo_wl':  # use also the loss function
                    self.learner.update(g, ro=ro, policy=self.policy, loss_fun=self.oracle.fun)
                else:
                    self.learner.update(g, ro=ro, policy=self.policy)
            else:
                self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        if hasattr(self.policy, 'lstd'):
            logz.log_tabular('std', np.mean(np.exp(self.policy.lstd)))
        for name, grad in grads.items():
            logz.log_tabular('{}_norm'.format(name), np.linalg.norm(grad))
        for name, ev in evs.items():
            logz.log_tabular(name, ev)
        logz.log_tabular('memory_mb', psutil.Process().memory_info().rss / 1024.0 / 1024.0)
        self._itr += 1

        # Update input normalizer for whitening.
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

    @staticmethod
    def merge(ros):
        """ Merge a list of Dataset instances. """
        return functools.reduce(lambda x, y: x+y, ros)
