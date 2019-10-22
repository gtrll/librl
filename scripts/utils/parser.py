# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import time
import git
import gym
import types
from functools import partial
import tensorflow as tf
import numpy as np
from rl.experimenter import MDP
from rl.core.utils import logz
from rl.sims import wrappers as wp


def configure_log(config, unique_log_dir=False):
    """ Configure output directory for logging. """

    # parse config to get log_dir
    top_log_dir = config['top_log_dir']
    log_dir = config['exp_name']
    seed = config['seed']

    # create dirs
    os.makedirs(top_log_dir, exist_ok=True)
    if unique_log_dir:
        log_dir += '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(top_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, '{}'.format(seed))
    os.makedirs(log_dir, exist_ok=True)

    # Log commit number.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config['git_commit_sha'] = sha

    # save config
    logz.configure_output_dir(log_dir)
    logz.save_params(config)


def _t_state(t, horizon):
    return t / horizon


def create_dartenv(envid, seed=None, use_time_info=True, bias=None, dyn_sup=None):
    env = gym.make(envid)
    t_state = partial(_t_state, horizon=env.spec.max_episode_steps)
    if bias:
        env = wp.AugDartEnv(env, bias=bias)
    else:
        env = wp.AugDartEnv(env)
    if dyn_sup:
        env = wp.LearnDyn(env, dyn_sup=dyn_sup)
    if use_time_info:
        env = wp.ObWithTime(env, t_state)
    env = wp.ReturnState(env)
    env.seed(seed)
    return env

def fix_random(seed):
    tf.keras.backend.clear_session()
    # fix randomness
    if tf.__version__[0] == '2':
        tf.random.set_seed(seed)
    else:
        tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)    

def setup_mdp(c, seed):
    """ Set seed and then create an MDP. """
    c = dict(c)
    envid = c['envid']
    env = create_dartenv(envid, seed)
    fix_random(seed)
    del c['envid']
    mdp = MDP(env, **c)
    return mdp
