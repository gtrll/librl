import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
from gym.envs.dart import DartEnv
from rl.core.function_approximators.supervised_learners import SupervisedLearner
import pydart2 as pydart


class Wrapper(gym.Wrapper):
    # Patch for gym.
    # Currently, only public class method can be accessed, defined in __getattr__ method
    # of gym Wrapper class.
    def getattr_protected(self, cls, name):
        assert name.startswith('_')
        env = self.get_class(cls)
        return getattr(env, name)

    def setattr(self, cls, name, value):
        env = self.get_class(cls)
        setattr(env, name, value)

    def get_class(self, cls):
        env = self
        try:
            while not isinstance(env, cls):
                env = env.env
        except:
            raise ValueError('env is not in class: {}'.format(cls))
        return env

    def is_class(self, cls):
        try:
            self.get_class(cls)
        except ValueError:
            return False
        return True

    def assert_class(self, cls):
        self.get_class(cls)


class ReturnState(Wrapper):
    def step(self, action):
        res = self.env.step(action)
        return res + (self.state,)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        return ob, self.state


class ObWithTime(Wrapper):
    def __init__(self, env, t_state):
        # `t_state`: a function that maps time to desired features
        super().__init__(env)
        # Change the observation space.
        assert isinstance(self.observation_space, Box)
        low, high = self.observation_space.low, self.observation_space.high
        assert len(low.shape) == len(high.shape) == 1
        low, high = np.hstack([low, 0.0]), np.hstack([high, 1.0])
        self.observation_space = Box(low, high)
        self.t_state = t_state

    def append_ob(self, ob):
        t = self.getattr_protected(TimeLimit, '_elapsed_steps')
        return np.concatenate([ob.flatten(), (self.t_state(t),)])

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        return self.append_ob(ob)

    def step(self, action):
        res = list(self.env.step(action))
        res[0] = self.append_ob(res[0])
        return tuple(res)


class AugDartEnv(Wrapper):
    # Augmented DartEnv
    def __init__(self, env, bias=None):
        # No perturb is bias is .0 or None.
        super().__init__(env)
        self.assert_class(DartEnv)
        self.bias = bias
        if not (bias is None or np.isclose(self.bias, 0.0)):
            self._perturb_physcial_params(bias)
        self.get_obs = self.getattr_protected(DartEnv, '_get_obs')

    @property
    def state(self):
        return self.state_vector()

    def reset(self, state=None, tm=None):
        ob = self.env.reset()
        if state is not None:
            self.set_state_vector(state)
            ob = self.get_obs()
        if tm is not None:
            self.setattr(TimeLimit, '_elapsed_steps', tm)
        return ob

    def _perturb_physcial_params(self, bias):
        if bias is None or np.isclose(bias, 0.0):
            return
        # Mass.
        for body in self.robot_skeleton.bodynodes:
            body.set_mass(body.m * self._rand_ratio(bias, self.np_random))
        # Damping coeff for revolute joints.
        for j in self.robot_skeleton.joints:
            if isinstance(j, pydart.joint.RevoluteJoint):
                coeff = j.damping_coefficient(0) * self._rand_ratio(bias, self.np_random)
                j.set_damping_coefficient(0, coeff)

    @staticmethod
    def _rand_ratio(bias, np_rand):
        """Helper function to be used in _perturb_physcial_params."""
        assert 1.0 > bias >= 0.0
        return 1.0 + bias * (np_rand.choice(2) * 2.0 - 1.0)


class LearnDyn(Wrapper):
    # Currently only works for DartEnv, due to the access to get_obs method.
    # XXX step returns naive obs
    def __init__(self, env, dyn_sup):
        super().__init__(env)
        assert isinstance(dyn_sup, SupervisedLearner)
        self.dyn_sup = dyn_sup  # predicts next state given current state and action

    def step(self, action):
        # Assume rw is a function of st and ac.
        _, rw, dn, info = self.env.step(action)
        st = self.dyn_sup(np.hstack([self.state, action]))
        self.set_state_vector(st)
        return self.get_obs(), rw, dn, info
