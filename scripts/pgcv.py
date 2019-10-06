# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.experimenter import MDP
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

    # Simulator for one-step simulation in TrajCV.
    sim = create_sim_env(mdp.env, np.random.randint(np.iinfo(np.int32).max),
                         use_time_info=mdp.use_time_info,
                         **c['vfn_dyn_kwargs'])

    # Create mdp for collecting extra samples for training vf.
    inacc_env = ps.create_env(mdp.env.env.spec.id,
                              seed=np.random.randint(np.iinfo(np.int32).max),
                              inacc=c['vfn_mdp_inacc'])
    c_mdp = dict(c['mdp'])
    del c_mdp['envid']
    vfn_mdp = MDP(inacc_env, **c_mdp)

    # Create algorithm
    alg = PolicyGradientWithTrajCV(policy, vfn,
                                   gamma=mdp.gamma, horizon=mdp.horizon,
                                   sim=sim, vfn_mdp=vfn_mdp,
                                   **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'],
                           ro_kwargs_pretrain=c['experimenter']['rollout_kwargs_pretrain'])
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
        'use_time_info': True,
    },
    'vfn_mdp_inacc': 0.1,  # bias / inaccuracy in the model for training vfn
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'save_freq': None,
        },
        'rollout_kwargs_pretrain': {
            'min_n_samples': 20,
            'max_n_rollouts': None,
        },
        'rollout_kwargs': {
            'min_n_samples': 20,
            'max_n_rollouts': None,
        },
    },
    'algorithm': {
        'optimizer': 'natgrad',
        'lr': 0.05,
        'c': 0.01,
        'max_kl': 0.1,
        'delta': 0.999,
        'lambd': 1.0,
        'max_n_batches': 1,  # for ae
        'n_warm_up_itrs': 0,  # policy nor update
        'n_pretrain_itrs': 1,
        'or_kwargs': {
            'cvtype': 'traj',
            'n_cv_steps': None,
            'cv_decay': 1.0,
            'n_ac_samples': 500,
            'cv_onestep_weighting': True,  # to reduce bias
            'switch_from_cvtype_state_at_itr': None,
        },
        'vfn_ro_kwargs': {
            'min_n_samples': 100,
            'max_n_rollouts': None,
        },
        'extra_vfn_training': False,
    },
    'policy_units': (64,),
    'value_units': (128, 128),
    'vfn_dyn_kwargs': {
        'units': (128, 128),
        'predict_residue': True,
        'max_n_samples': 200000,
        'max_n_batches': None,
    },
    'init_lstd': -1.0,
}


if __name__ == '__main__':
    main(CONFIG)