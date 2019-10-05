# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy as cp
import functools
import numpy as np
import psutil
from rl.algorithms.algorithm import Algorithm, PolicyAgent
from rl.algorithms.utils import get_learner
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradientWithTrajCV
from rl import online_learners as ol
from rl.policies import Policy
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz


class PolicyGradientWithTrajCV(Algorithm):
    def __init__(self, policy, vfn,
                 optimizer='adam',
                 lr=1e-3, c=1e-3, max_kl=0.1,
                 horizon=None, gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2, n_warm_up_itrs=None, n_pretrain_itrs=1,
                 sim=None, or_kwargs=None,
                 extra_vfn_training=False, vfn_ro_kwargs=None
                 ):

        assert isinstance(policy, Policy)
        self.vfn = vfn
        self.policy = policy

        # Create online learner.
        scheduler = ol.scheduler.PowerScheduler(lr, c=c)
        self.learner = get_learner(optimizer, policy, scheduler, max_kl)
        self._optimizer = optimizer

        # Create oracle.
        # ae is only for value function estimation, not used for adv computation,
        # therefore use_is can be set to 'one'.
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd,
                               horizon=horizon, use_is='one', max_n_batches=max_n_batches)
        self.oracle = ValueBasedPolicyGradientWithTrajCV(policy, self.ae, sim=sim, **or_kwargs)

        # Misc.
        self._extra_vfn_training = extra_vfn_training
        self._vfn_ro_kwargs = cp.deepcopy(vfn_ro_kwargs)
        self._n_pretrain_itrs = n_pretrain_itrs
        self._experimenter = None
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
                self.policy.update(xs=ro['obs_short'])
                self.oracle.update_vfn(ro)
                self.oracle.update_dyn(ro)

    def update(self, ros, agents):
        # Aggregate data
        ro = self.merge(ros)
        evs = {}

        with timed('Update oracle'):
            self.oracle.update(ro, self.policy)

        if self._extra_vfn_training:
            with timed('Update vfn (Extra)'):
                vfn_ros, _ = self._experimenter.gen_ro(self.agent('behavior'), to_log=False,
                                                       ro_kwargs=self._vfn_ro_kwargs)
                vfn_ro = self.merge(vfn_ros)
                _, evs['vfn_ev0'], evs['vfn_ev1'] = self.oracle.update_vfn(vfn_ro)

        with timed('Compute policy gradient'):
            grads = self.oracle.grad(self.policy.variable)
            g = grads['g']

        with timed('Policy update'):
            if isinstance(self.learner, ol.FisherOnlineOptimizer):
                if self._optimizer == 'trpo_wl':  # use also the loss function
                    self.learner.update(g, ro=ro, policy=self.policy, loss_fun=self.oracle.fun)
                else:
                    self.learner.update(g, ro=ro, policy=self.policy)
            else:
                self.learner.update(g)
            self.policy.variable = self.learner.x
            
        with timed('Update vfn and dyn if needed'):
            if not self._extra_vfn_training:
                _, evs['vfn_ev0'], evs['vfn_ev1'] = self.oracle.update_vfn(ro)
            _, evs['dyn_ev0'], evs['dyn_ev1'] = self.oracle.update_dyn(ro)

        # Update input normalizer for whitening
        # Put it here so that the policy for data collection is the same one with respective to
        # which policy gradient is computed.
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        if hasattr(self.policy, 'lstd'):
            logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        for name, grad in grads.items():
            logz.log_tabular('{}_norm'.format(name), np.linalg.norm(grad))
        for name, ev in evs.items():
            logz.log_tabular(name, ev)
        logz.log_tabular('memory_mb', psutil.Process().memory_info().rss / 1024.0 / 1024.0)
        self._itr += 1

    @staticmethod
    def merge(ros):
        """ Merge a list of Dataset instances. """
        return functools.reduce(lambda x, y: x+y, ros)
