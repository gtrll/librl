# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

range_common = [
    # [['seed'], [x for x in range(4)]],
    [['seed'], [0]],
    # [['exp_name'], ['unbiasedwithev']],
    [['mdp', 'n_processes'], [16]],
    [['experimenter', 'rollout_kwargs_pretrain', 'min_n_samples'], [10000]],
    [['ss_sim_kwargs', 'n_processes'], [16]],
]

range_comparison = [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [None]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [5]],
]

range_cv = [
    [['algorithm', 'or_kwargs', 'n_ac_samples'], [500]],
    [['algorithm', 'or_kwargs', 'cv_decay'], [1.0]],
    [['algorithm', 'train_vfn_using_sim'], [True]],
    [['algorithm', 'vfn_sim_ro_kwargs', 'min_n_samples'], [50000]],
    [['vfn_sim_inacc'], [0.0]],   # biased
]

range_upper_bound = range_common + [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [50000]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [None]],
    [['algorithm', 'or_kwargs', 'cvtype'], ['state']],
]


range_nocv = range_common + range_comparison + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['nocv']],
]

range_state = range_common + range_comparison + range_cv + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['state']],
]

range_state_action = range_common + range_comparison + range_cv + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [1]],
    [['ss_sim_kwargs', 'inacc'], [0.1]],
]

range_traj = range_common + range_comparison + range_cv + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [None]],
    [['ss_sim_kwargs', 'inacc'], [0.1]],
]
