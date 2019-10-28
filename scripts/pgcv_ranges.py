# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

range_common = [
    [['seed'], [x for x in range(8)]],
    # [['exp_name'], ['new-20k']],
    [['ae_kwargs', 'delta'], [0.999]],
    [['mdp', 'mdp_kwargs', 'n_processes'], [4]],
    [['mdp', 'use_time_info'], [False]],
    [['ss_sim', 'n_processes'], [4]],
    [['experimenter', 'rollout_kwargs_pretrain', 'min_n_samples'], [2000]],
    [['algorithm', 'scheduler_kwargs', 'eta'], [0.05]],
    [['algorithm', 'learner_kwargs', 'optimizer'], ['rnatgrad']],
]

range_ro = [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [None]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [5]],
]

range_vfn_sim = [
    [['algorithm', 'train_vfn_using_sim'], [True]],
    [['algorithm', 'vfn_sim_ro_kwargs', 'min_n_samples'], [50000]],
    [['vfn_sim_bias'], [0.0]],   # biased
]

range_no_vfn_sim = [
    [['algorithm', 'train_vfn_using_sim'], [False]],
]

range_ac_sample = [
    [['algorithm', 'or_kwargs', 'n_ac_samples'], [100]],
    [['algorithm', 'or_kwargs', 'enqhat_with_vfn'], [True]],
    [['algorithm', 'or_kwargs', 'same_ac_rand'], [False]],
    [['ss_sim', 'type'], ['biased']],
    [['ss_sim', 'bias'], [0.0]],
]

range_no_ac_sample = [
    [['ss_sim', 'type'], [None]],
]

range_traj = [
    [['algorithm', 'or_kwargs', 'n_cv_steps'], [None]],
    [['algorithm', 'or_kwargs', 'cv_decay'], [1.0]],
]


# Different CVs
range_upper_bound = range_common + range_no_vfn_sim + range_no_ac_sample + [
    [['experimenter', 'rollout_kwargs', 'min_n_samples'], [20000]],
    [['experimenter', 'rollout_kwargs', 'max_n_rollouts'], [None]],
    [['algorithm', 'or_kwargs', 'cvtype'], ['s']],
]

range_nocv = range_common + range_ro + range_no_vfn_sim + range_no_ac_sample + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['nocv']],
]

range_s = range_common + range_ro + range_vfn_sim + range_no_ac_sample + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['s']],
]

range_sa = range_common + range_ro + range_vfn_sim + range_ac_sample + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['sa']],
]

range_traj_s = range_common + range_ro + range_vfn_sim + range_ac_sample + range_traj + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj-s']],
]

range_traj_sa = range_common + range_ro + range_vfn_sim + range_ac_sample + range_traj + [
    [['algorithm', 'or_kwargs', 'cvtype'], ['traj-sa']],
]
