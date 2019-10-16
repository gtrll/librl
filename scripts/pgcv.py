# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.experimenter import MDP
from rl.sims import create_sim_dartenv
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
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy', **c['pol_kwargs'])
    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function', **c['vfn_kwargs'])

    # Simulator for Single-Step simulation in TrajCV.
    ss_sim = create_sim_dartenv(mdp.env,
                            seed=np.random.randint(np.iinfo(np.int32).max),
                            use_time_info=mdp.use_time_info, **c['ss_sim_kwargs'])

    # Create mdp for collecting extra samples for training vf.
    vfn_sim = ps.create_dartenv(mdp.env.env.spec.id,
                                seed=np.random.randint(np.iinfo(np.int32).max),
                                inacc=c['vfn_sim_inacc'])
    conf = dict(c['mdp'])
    del conf['envid']
    vfn_sim = MDP(vfn_sim, **conf)

    # Create learner.
    c['algorithm']['ae_kwargs']['gamma'] = mdp.gamma
    c['algorithm']['ae_kwargs']['horizon'] = mdp.horizon
    alg = PolicyGradientWithTrajCV(policy, vfn, vfn_sim=vfn_sim, ss_sim=ss_sim, **c['algorithm'])

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
        # 'envid': 'DartReacher3d-v1',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 1,
        'use_time_info': True,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'save_freq': None,
        },
        'rollout_kwargs_pretrain': {
            'min_n_samples': 1000,
            'max_n_rollouts': None,
        },
        'rollout_kwargs': {
            'min_n_samples': 1000,
            'max_n_rollouts': None,
        },
    },
    'algorithm': {
        'scheduler_kwargs': {
            'eta': 0.1,
            'c': 0.01,
        },
        'learner_kwargs': {
            'optimizer': 'rnatgrad',
            'max_kl': 0.1,
        },
        'ae_kwargs': {
            'delta': 0.99,
            'lambd': 1.0,
            'max_n_batches': 0,
            'use_is': None,
        },
        'or_kwargs': {
            'cvtype': 'traj',
            'n_cv_steps': None,
            'cv_decay': 1.0,
            'n_ac_samples': 200,
            'cv_onestep_weighting': False,  # to reduce bias
            'switch_from_cvtype_state_at_itr': 5,
        },
        'vfn_sim_ro_kwargs': {
            'min_n_samples': 1000,
            'max_n_rollouts': None,
        },  # the sim for training vfn
        'train_vfn_using_sim': False,
        'n_warm_up_itrs': 0,  # policy nor update
        'n_pretrain_itrs': 1,

    },
    'pol_kwargs': {
        'units': (64,),
        'init_lstd': -1.0,
    },
    'vfn_kwargs': {
        'units': (128, 128),
    },
    'ss_sim_kwargs': {
        'inacc': None,  # set to None to use learned dyn
        'dyn_kwargs': {
            'units': (128, 128),
            'predict_residue': True,
            'max_n_samples': 50000,
            'max_n_batches': None,
            'clip': True,
            'scale': True,
        },
    },
    'vfn_sim_inacc': 0.1,  # bias / inaccuracy in the simulator for training vfn
}


if __name__ == '__main__':
    main(CONFIG)
