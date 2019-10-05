# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_common = [
    [['seed'], [x for x in range(1)]],
    [['mdp', 'n_processes'], [8]],
    [['algorithm', 'n_pretrain_itrs'], [1]],
    [['algorithm', 'n_warm_up_itrs'], [0]],    
]

range_comparison = [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [None]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [5]],
]

range_cv = [
    [['algorithm', 'or_kwargs', 'n_ac_samples'], [1000]],
    [['algorithm', 'extra_vfn_training'], [True]],
    [['algorithm', 'vfn_ro_kwargs', 'min_n_samples'], [50000]],
    [['vfn_mdp_inacc'], [0.0]], 
]

range_upper_bound = range_common + [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [10000]],
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
]

range_traj = range_common + range_comparison + range_cv + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [None]],
]
