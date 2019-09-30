# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from rl import sims
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP

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


def create_sim_env(env, seed, dyn_units):
    ob_dim = len(env.reset())
    ac_dim = env.action_space.shape[0]
    dyn = SuperRobustKerasMLP((ob_dim + ac_dim,), (ob_dim,), name='dynamics', units=dyn_units)
    envid = env.env.spec.id
    sim = ENVID2MODELENV[envid](env, predict=dyn.predict, seed=seed)
    return sim
