# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from functools import partial
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.experimenter import MDP, OneStepper
from rl.sims import Dynamics
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.algorithms import PolicyGradientWithTrajCV
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


def main(c):

    # Setup logz and save c.
    ps.configure_log(c)

    # Fix the randomness in tf and np.
    ps.fix_random(c['seed'])
    
    # Create mdp.
    create_dartenv = partial(ps.create_dartenv,
                             envid=c['mdp']['envid'],
                             seed=None,
                             use_time_info=c['mdp']['use_time_info'])
    env = create_dartenv()
    mdp = MDP(env, **c['mdp']['mdp_kwargs'])
    # mdp = setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy', **c['pol_kwargs'])
    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function', **c['vfn_kwargs'])
    ae = ValueBasedAE(policy, vfn,
                      gamma=mdp.gamma, horizon=env.spec.max_episode_steps,
                      **c['ae_kwargs'])

    # Simulator for Single-Step simulation in TrajCV.
    if c['algorithm']['or_kwargs']['cvtype'] == 'traj':
        if c['ss_sim']['type'] == 'biased':
            ss_sim = create_dartenv(bias=c['ss_sim']['bias'])
        elif c['ss_sim']['type'] == 'learn_dyn':
            dyn_sup = Dynamics(env.state.shape, ac_shape, env=env, **c['ss_sim']['dyn_kwargs'])
            ss_sim = create_dartenv(dyn_sup=dyn_sup)
        else:
            raise ValueError
        ss_sim = OneStepper(ss_sim, n_processes=c['ss_sim']['n_processes'])
    else:
        ss_sim = None

    # Create mdp for collecting extra samples for training vf.
    if c['algorithm']['train_vfn_using_sim']:
        vfn_sim = create_dartenv(bias=c['vfn_sim_bias'])
        vfn_sim = MDP(vfn_sim, **c['mdp']['mdp_kwargs'])
    else:
        vfn_sim = None

    # Create learner.
    alg = PolicyGradientWithTrajCV(policy, ae, vfn_sim=vfn_sim, ss_sim=ss_sim, **c['algorithm'])

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
        'use_time_info': False,
        'mdp_kwargs': {
            'gamma': 1.0,
            'n_processes': 4,
        },
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'final_save': False,
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
            'eta': 0.05,
            'c': 0.01,
        },
        'learner_kwargs': {
            'optimizer': 'rnatgrad',
            'max_kl': 0.1,
        },
        'or_kwargs': {
            'cvtype': 'traj',
            'n_cv_steps': None,
            'cv_decay': 1.0,
            'n_ac_samples': 10,
            'cv_onestep_weighting': False,  # to reduce bias
            'switch_from_cvtype_state_at_itr': 3,
            'enqhat_with_vfn': True,
            'same_ac_rand': False,
        },
        'vfn_sim_ro_kwargs': {
            'min_n_samples': 1000,
            'max_n_rollouts': None,
        },  # the sim for training vfn
        'train_vfn_using_sim': False,
        'n_warm_up_itrs': 3,  # policy nor update
        'n_pretrain_itrs': 1,

    },
    'pol_kwargs': {
        'units': (64,),
        'init_lstd': -1.0,
    },
    'vfn_kwargs': {
        'units': (128, 128),
    },
    'ae_kwargs': {
        'delta': 0.999,
        'lambd': 1.0,
        'max_n_batches': 1,
        'use_is': None,
        'pe_lambd': 1.0,
    },
    'ss_sim': {
        'type': 'biased',  # learn_dyn or biased
        'bias': 0.0,  # set to None to use learned dyn
        'dyn_kwargs': {
            'units': (128, 128),
            'predict_residue': True,
            'max_n_samples': 50000,
            'max_n_batches': None,
            'clip': True,
            'scale': True,
        },
        'n_processes': 4,  # used when biased sim is used, i.e. bias is not None
    },
    'vfn_sim_bias': 0.1,  # bias / bias in the simulator for training vfn
}


if __name__ == '__main__':
    main(CONFIG)
