# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_common = [
    [['exp_name'], ['lasttime']],
    # [['seed'], [x for x in range(1)]],
    [['seed'], [0]],    
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [10000]],
    [['algorithm', 'n_warm_up_itrs'], [0]],
]

range_lambd = [
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
]
range_lambd = R.merge_ranges(range_common,  range_lambd)


range_delta = range_common + [
    [['algorithm', 'delta'], [0.99]],
    [['algorithm', 'lambd'], [1.]],
]





