import tensorflow as tf
import numpy as np
from rl.core.utils.tf2_utils import array_to_ts
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


class Dynamics(SuperRobustKerasMLP):
    def __init__(self, ob_shape, ac_shape, units, name='dynamics', predict_residue=True):
        assert len(ob_shape) == 1
        assert len(ac_shape) == 1
        self._predict_residue = predict_residue
        self._ob_dim = ob_shape[0]
        super().__init__((ob_shape[0] + ac_shape[0], ), ob_shape, name=name, units=units)

    def ts_predict(self, ts_xs, **kwargs):
        # next_obs / residue = fun(obs, acs)
        ts_ys = super().ts_predict(ts_xs, **kwargs)
        if not self._predict_residue:
            return ts_ys
        ts_obs = ts_xs[:, :self._ob_dim]
        return ts_obs + ts_ys
