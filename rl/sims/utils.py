# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import pdb
from rl import sims

# Env.
ENVID2MODELENV = {
    'DartCartPole-v1': sims.Cartpole,
    'DartHopper-v1': sims.Hopper,
    'DartSnake7Link-v1': sims.Snake,
    'DartWalker3d-v1': sims.Walker3d,
    'DartDog-v1': sims.Dog,
    # 'DartReacher-v1': sims.Reacher2D,
    'DartReacher3d-v1': sims.Reacher,
}


def create_sim_env(env, seed, use_time_info, **dyn_kwargs):
    if use_time_info:
        ob_shape = (len(env.reset()) + 1, )  # assume no time info
    else:
        ob_shape = (len(env.reset()), )  # assume no time info
    ac_shape = env.action_space.shape
    dyn = sims.Dynamics(ob_shape, ac_shape, name='dynamics', **dyn_kwargs)
    envid = env.env.spec.id
    sim = ENVID2MODELENV[envid](env, predict=dyn.predict, seed=seed)
    return sim
