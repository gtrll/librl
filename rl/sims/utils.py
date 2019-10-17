# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from rl import sims
from rl.experimenter import SimOne
from scripts.utils import parser as ps

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


def create_sim_dartenv(env, inacc=None, horizon=None, seed=None, use_time_info=True, n_processes=1,
                       **dyn_kwargs):

    if use_time_info:
        ob_shape = (len(env.observation_space.low) + 1, )
    else:
        ob_shape = (len(env.observation_space.low), )
    ac_shape = env.action_space.shape
    envid = env.env.spec.id
    if inacc is None:
        dyn = sims.Dynamics(ob_shape, ac_shape, name='dynamics', env=env, **dyn_kwargs)
        sim = ENVID2MODELENV[envid](env, predict=dyn.predict, seed=seed)
    else:
        sim = ps.create_dartenv(envid, seed, inacc=inacc)
        sim = SimOne(sim, horizon=horizon, use_time_info=use_time_info, n_processes=n_processes)
    return sim
