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


def create_sim_env(env, seed, dyn_units, predict_residue):
    ob_shape = (len(env.reset()), )  # assume no time info
    ac_shape = env.action_space.shape
    dyn = sims.Dynamics(ob_shape, ac_shape, units=dyn_units, predict_residue=predict_residue)
    envid = env.env.spec.id
    sim = ENVID2MODELENV[envid](env, predict=dyn.predict, seed=seed)
    return sim
