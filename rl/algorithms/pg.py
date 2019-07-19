import numpy as np
from rl.algorithms.algorithm import Algorithm
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.tf2_rl_oracles import tfValueBasedPolicyGradient
from rl.core.online_learners import base_algorithms as balg
from rl.core.online_learners import BasicOnlineOptimizer
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz

class PolicyGradient(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, vfn, eta=1e-3,
                 gamma=1.0, delta=0.999, lambd=0.):
        self.vfn = vfn
        self.policy = policy
        # create online learner
        x0 = self.policy.variable
        scheduler = PowerScheduler(eta)
        self.learner = BasicOnlineOptimizer(balg.Adam(x0, scheduler))
        # create oracle
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd, v_target='monte-carlo')
        self.oracle =tfValueBasedPolicyGradient(policy, self.ae)

    def pi(self, ob, t, done):
        return self.policy(ob)

    def pi_ro(self, ob, t, done):
        return self.policy(ob)

    def logp(self, obs, acs):
        return self.policy.logp(obs, acs)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            ro = gen_ro(self.pi, logp=self.logp)
            self.oracle.update(ro, self.policy)

    def update(self, ro):
        # Update input normalizer for whitening
        self.policy.update(ro['obs_short'])

        # Correction Step (Model-free)
        with timed('Update oracle'):
            self.oracle.update(ro, self.policy)

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy)

        with timed('Policy update'):
            self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
