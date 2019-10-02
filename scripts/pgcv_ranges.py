# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_common = [
    [['seed'], [x * 100 for x in range(1)]],
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [2000]],
    [['mdp', 'n_processes'], [2]],
    [['algorithm', 'n_pretrain_itrs'], [1]],
]

range_nocv = [
    [['algorithm', 'or_kwargs', 'cvtype'], ['nocv']],
]
range_nocv = R.merge_ranges(range_common, range_nocv)

range_state = [
    [['algorithm', 'or_kwargs', 'cvtype'], ['state']],
]
range_state = R.merge_ranges(range_common, range_state)

range_state_action = [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [1]],
]
range_state_action = R.merge_ranges(range_common, range_state_action)

range_traj = [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj']],
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [None]],
]
range_traj = R.merge_ranges(range_common, range_traj)

