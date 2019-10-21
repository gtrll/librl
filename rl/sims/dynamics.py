import tensorflow as tf
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


class Dynamics(SuperRobustKerasMLP):
    def __init__(self, st_shape, ac_shape, predict_residue=True, env=None, clip=True, scale=True,
                 **sup_kwargs):
        self._ac_clip, self._ac_scale = None, None
        if clip:
            self._ac_clip = [env.action_space.low, env.action_space.high]
        if scale:
            self._ac_scale = env.action_scale
        assert len(st_shape) == 1
        assert len(ac_shape) == 1
        self._predict_residue = predict_residue
        self._st_dim = st_shape[0]
        self._ac_dim = ac_shape[0]
        super().__init__((self._st_dim + self._ac_dim, ), st_shape, **sup_kwargs)

    def ts_predict(self, ts_xs, **kwargs):
        ts_sts, ts_acs = ts_xs[:, :self._st_dim], ts_xs[:, self._st_dim:]
        if self._ac_clip is not None:
            ts_acs = tf.clip_by_value(ts_acs, *self._ac_clip)
        if self._ac_scale is not None:
            ts_acs = ts_acs * self._ac_scale  # scale
        ts_xs = tf.concat([ts_sts, ts_acs], 1)
        ts_ys = super().ts_predict(ts_xs, **kwargs)

        if not self._predict_residue:
            return ts_ys
        return ts_sts + ts_ys
