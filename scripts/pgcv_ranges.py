# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_common = [
    [['seed'], [x for x in range(8)]],
    [['mdp', 'n_processes'], [1]],
    [['algorithm', 'n_pretrain_itrs'], [5]],
]

range_upper_bound = range_common + [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [10000]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [None]],
    [['algorithm', 'or_kwargs', 'cvtype'], ['nocv']],    
]

range_extra = [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [2000]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [None]],
    [['algorithm', 'or_kwargs', 'n_ac_samples'], [1000]],
]

range_nocv = range_common + range_extra + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['nocv']],
]

range_state = range_common + range_extra + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['state']],
]

range_state_action = range_common + range_extra + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [1]],
    [['algorithm', 'extra_vfn_training'], [True]],
    [['algorithm', 'vfn_ro_kwargs', 'min_n_samples'], [10000]],
]

range_traj = range_common + range_extra + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [None]],
    [['algorithm', 'extra_vfn_training'], [True]],
    [['algorithm', 'vfn_ro_kwargs', 'min_n_samples'], [10000]],
]
