# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.sims import create_sim_env
from rl.algorithms import PolicyGradientWithTrajCV
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp = ps.setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=c['init_lstd'],
                                   units=c['policy_units'])

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                              units=c['value_units'])
    sim = create_sim_env(mdp.env, np.random.randint(np.iinfo(np.int32).max),
                         dyn_units=c['dyn_units'], predict_residue=c['predict_residue'])

    # Create algorithm
    alg = PolicyGradientWithTrajCV(policy, vfn,
                                   gamma=mdp.gamma, horizon=mdp.horizon,
                                   sim=sim,
                                   **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIG = {
    'top_log_dir': 'log',
    'exp_name': 'cp',
    'seed': 9,
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 4,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'save_freq': None,
        },
        'rollout_kwargs': {
            'min_n_samples': 20,
            'max_n_rollouts': None,
        },
    },
    'algorithm': {
        'optimizer': 'adam',
        'lr': 0.001,
        'c': 0.01,
        'max_kl': 0.1,
        'delta': 0.99,
        'lambd': 0.0,
        'max_n_batches': 2,  # for ae
        'n_warm_up_itrs': None,  # policy nor update
        'n_pretrain_itrs': 1,
        'or_kwargs': {
            'cvtype': 'traj',
            'n_cv_steps': None,
            'cv_decay': 1.0,
            'n_ac_samples': 500,
            'switch_from_cvtype_state_at_itr': None,
        },
    },
    'policy_units': (64,),
    'value_units': (128, 128),
    'dyn_units': (128, 128),
    'predict_residue': True,
    'init_lstd': -1.0,
}


if __name__ == '__main__':
    main(CONFIG)
