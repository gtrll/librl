import pdb
from functools import partial
import numpy as np
from gym.envs.dart import DartEnv
from rl.experimenter import MDP
from rl.sims import ReturnState
from rl.core.utils.mp_utils import Worker, JobRunner


class SimOne(MDP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job_runner = None
        self._sim_one = partial(self.sim_one, env=self.env)
        assert self.env.is_class(ReturnState)

    @staticmethod
    def sim_one(env, sts, acs, tms):
        def set_and_step(st, ac, tm):
            env.reset(state=st, tm=tm)
            next_ob, rw, next_dn, _, _ = env.step(ac)
            return next_ob, rw, next_dn

        next_obs, rws, next_dns = [], [], []
        for st, ac, tm in zip(sts, acs, tms):
            next_ob, rw, next_dn  = set_and_step(st, ac, tm)
            next_obs.append(next_ob)
            rws.append(rw)
            next_dns.append(next_dn)

        return next_obs, rws, next_dns

    def run(self, sts, acs, tms):

        def get_idx(n, m):
            # Evenly put n items into m bins.
            # if n = 8, m = 4, returns (0, 4, 8)
            # if n = 10, m = 4, returns (0, 4, 8, 10)
            idx = ((n + (m-1)) // m) * np.arange(m)  # python 3, truncated division
            idx[-1] = n
            return idx

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
