from functools import partial
import numpy as np
from gym.envs.dart import DartEnv
from rl.experimenter import MDP, wrap_gym_env
from rl.core.utils.mp_utils import Worker, JobRunner


class SimOne:

    def __init__(self, env, horizon=None, use_time_info=True, n_processes=1):
        self.env = env
        self.horizon = horizon
        self.use_time_info = use_time_info
        self._n_processes = n_processes
        self._job_runner = None

        t_state = partial(MDP.t_state, horizon=horizon) if use_time_info else None
        self._sim_one = partial(self.sim_one, env=self.env, t_state=t_state)
        assert isinstance(env.env, DartEnv)

    @staticmethod
    def sim_one(env, sts, acs, tms, t_state=None):
        # tms is a list. use a list of None to turn it off.
        def set_and_step(st, ac, tm):
            env.set_state(st[:len(st)//2], st[len(st)//2:])  # qpos, qvel
            _, next_ob, rw, next_dn, info = env.step(ac)
            if tm is not None and t_state is not None:
                next_ob = np.concatenate([next_ob.flatten(), (t_state(tm+1),)])  # time step for next
            return next_ob, rw, next_dn, info

        # XX wrap env if necessary.
        try:
            _, _ = env.reset()
        except ValueError:
            env = wrap_gym_env(env)
        next_obs, rws, next_dns = [], [], []
        for st, ac, tm in zip(sts, acs, tms):
            next_ob, rw, next_dn, _ = set_and_step(st, ac, tm)
            next_obs.append(next_ob)
            rws.append(rw)
            next_dns.append(next_dn)

        return next_obs, rws, next_dns

    def run(self, sts, acs, tms=None):

        def get_idx(n, m):
            # Evenly put n items into m bins.
            # if n = 8, m = 4, returns (0, 4, 8)
            # if n = 10, m = 4, returns (0, 4, 8, 10)
            idx = np.arange((n + (m-1)) // m) * m  # python 3, truncated division
            idx[-1] = n
            return idx

        tms = [None] * len(sts) if tms is None else tms
        if self._n_processes > 1:
            if self._job_runner is None:
                workers = [Worker(method=self._sim_one) for _ in range(self._n_processes)]
                self._job_runner = JobRunner(workers)
            idx = get_idx(len(sts), self._n_processes)
            intervals = [(idx[i], idx[i+1]) for i in range(len(idx)-1)]
            jobs = []
            for a, b in intervals:
                kwargs = {'sts': sts[a:b], 'acs': acs[a:b], 'tms': tms[a:b]}
                jobs.append(([], kwargs))
            res = self._job_runner.run(jobs)
            next_obs = np.vstack([r[0] for r in res])
            rws = np.hstack([r[1] for r in res])
            next_dns = np.hstack([r[2] for r in res])
        else:
            kwargs = {'sts': sts, 'acs': acs, 'tms': tms}
            next_obs, rws, next_dns = self._sim_one(**kwargs)
        return next_obs, rws, next_dns
