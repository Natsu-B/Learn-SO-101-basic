"""Policy representations above the unchanged SI absolute-position environment.

Compose as NormalizedJointPositionWrapper(RateLimitWrapper(RobotEnv(...)))
or DeltaJointPositionWrapper(RateLimitWrapper(RobotEnv(...))). Limits and
max_delta/max_change are radians per Gym step, independent of robot dimension.
"""
import gymnasium as gym
import numpy as np


def _action(value, space):
    value = np.asarray(value)
    if value.shape != space.shape or value.dtype.kind not in "fiu":
        raise ValueError(f"expected numeric action of shape {space.shape}")
    if not np.isfinite(value).all() or np.any(value < space.low) or np.any(value > space.high):
        raise ValueError("action must be finite and inside action_space")
    return value.astype(np.float64)


def _limits(env):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("joint wrappers require a Box action space")
    lo, hi = env.action_space.low.copy(), env.action_space.high.copy()
    base = env.unwrapped.action_space
    if not np.array_equal(lo, base.low) or not np.array_equal(hi, base.high):
        raise ValueError("wrap an absolute SI action environment; put policy mapping outermost")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi):
        raise ValueError("joint wrappers require finite, nonempty safe ranges")
    return lo, hi


def _positive_vector(value, shape):
    try:
        result = np.broadcast_to(np.asarray(value, np.float64), shape).copy()
    except ValueError as error:
        raise ValueError("expected a scalar or one limit per joint") from error
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("limits must be finite and positive")
    return result


class NormalizedJointPositionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low, self.high = _limits(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, self.low.shape, np.float32)

    def action(self, action):
        action = _action(action, self.action_space)
        target = self.low.astype(np.float64) + (action + 1) * 0.5 * (self.high.astype(np.float64) - self.low)
        return np.clip(target, self.low, self.high).astype(np.float32)


class DeltaJointPositionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_delta=0.05):
        super().__init__(env)
        self.low, self.high = _limits(env)
        self.max_delta = _positive_vector(max_delta, self.low.shape)
        self.action_space = gym.spaces.Box(-1.0, 1.0, self.low.shape, np.float32)

    def action(self, action):
        action = _action(action, self.action_space)
        if self.unwrapped.last_observation is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        current = np.asarray(self.unwrapped.last_observation["joint_position"], np.float64)
        return np.clip(current + action * self.max_delta, self.low, self.high).astype(np.float32)


class RateLimitWrapper(gym.ActionWrapper):
    def __init__(self, env, max_change=0.05):
        super().__init__(env)
        self.low, self.high = _limits(env)
        self.max_change = _positive_vector(max_change, self.low.shape)
        self.previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = np.clip(obs["joint_position"], self.low, self.high).astype(np.float64)
        return obs, info

    def action(self, action):
        requested = _action(action, self.action_space)
        if self.previous is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        limited = self.previous + np.clip(requested - self.previous, -self.max_change, self.max_change)
        return np.clip(limited, self.low, self.high).astype(np.float32)

    def step(self, action):
        target = self.action(action)
        result = self.env.step(target)
        self.previous = target.astype(np.float64)
        result[4]["joint_target"] = target.copy()
        return result


class ResidualActionWrapper(gym.Wrapper):
    """Bounded normalized residual on a frozen callable returning absolute SI targets.

    base_policy(observation) must return radians, NOT PPO's normalized/delta action.
    Adapt that representation explicitly before using this interface. Safety always
    goes through our owned RateLimitWrapper and the underlying SI range validation.
    """
    def __init__(self, env, base_policy, max_residual=0.01, alpha=1.0, max_change=0.05):
        low, high = _limits(env)
        if not callable(base_policy) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('base_policy must be callable and alpha must be in [0,1]')
        super().__init__(RateLimitWrapper(env,max_change=max_change))
        self.low,self.high=low,high
        self.base_policy=base_policy
        self.max_residual=_positive_vector(max_residual,low.shape)
        self.alpha=float(alpha)
        self.action_space=gym.spaces.Box(-1.,1.,low.shape,np.float32)
        self._observation=None

    def reset(self, **kwargs):
        obs,info=self.env.reset(**kwargs)
        self._observation={k:v.copy() for k,v in obs.items()}
        return obs,info

    def step(self, action):
        residual=_action(action,self.action_space)*self.max_residual*self.alpha
        if self._observation is None: raise gym.error.ResetNeeded('call reset() before step()')
        base=np.asarray(self.base_policy({k:v.copy() for k,v in self._observation.items()}),np.float64)
        if base.shape!=self.low.shape or not np.isfinite(base).all(): raise ValueError('base policy must return finite SI joint positions')
        target=np.clip(base+residual,self.low,self.high).astype(np.float32)
        result=self.env.step(target)
        self._observation={k:v.copy() for k,v in result[0].items()}
        result[4]['residual_joint_position']=residual.copy()
        return result
