"""GoalEnv conventions for proprioceptive Reach (SI metres/radians).

Fixed-horizon training avoids incorrectly treating HER's relabelled goals as
terminal. Evaluation may stop on success. Hidden FK witness configurations are
never returned to Python. One native transaction produces each step snapshot.
"""
from dataclasses import replace
import gymnasium as gym
import numpy as np
from .env import RobotEnv
from .config import TaskConfig
from .selection import reach_actuators


class GoalReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, robot="so101", *, max_delta=0.05, distance_scale=0.1,
                 terminate_on_success=False, task_config=None, control_frequency=30.0,
                 rate_limit=0.1, domain_randomization=None):
        super().__init__()
        if not np.isfinite(max_delta) or max_delta <= 0:
            raise ValueError("max_delta must be finite and positive (radians)")
        if not np.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive (metres)")
        if not np.isfinite(rate_limit) or rate_limit <= 0:
            raise ValueError("rate_limit must be finite and positive (radians/control step)")
        self.robot = robot
        self.max_delta = float(max_delta)
        self.distance_scale = float(distance_scale)
        self.rate_limit = float(rate_limit)
        self.terminate_on_success = bool(terminate_on_success)
        config = task_config or TaskConfig(success_threshold=.005, max_episode_steps=100)
        self.base = RobotEnv(robot, task_config=config, control_frequency=control_frequency,
                             domain_randomization=domain_randomization)
        self.sim = self.base.sim
        self.actuator_names = reach_actuators(robot, self.sim.actuator_names)
        self.indices = [self.sim.actuator_names.index(n) for n in self.actuator_names]
        self.sim.configure_reach_selection(self.actuator_names)
        self.bounds = np.asarray(self.sim.action_bounds, np.float64)
        self.home = np.asarray(self.sim.home_position, np.float64)
        self.action_space = gym.spaces.Box(-1., 1., (len(self.indices),), np.float32)
        n = len(self.indices)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-np.inf, np.inf, (2*n+3,), np.float32),
            "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "desired_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        self._needs_reset = True
        self.last_result = None

    @property
    def task_config(self):
        return self.base.task_config

    def configure_task(self, **changes):
        from dataclasses import asdict
        config = replace(self.task_config, **changes)
        self.sim.configure_reach(**asdict(config))
        self.base.task_config = config
        self._needs_reset = True

    def _observation(self, result):
        q = np.asarray(result['joint_position'], np.float32)[self.indices]
        v = np.asarray(result['joint_velocity'], np.float32)[self.indices]
        ee = np.asarray(result['end_effector_position'], np.float32)
        goal = np.asarray(result['target_position'], np.float32)
        return dict(observation=np.concatenate((q,v,ee)), achieved_goal=ee, desired_goal=goal)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal, np.float64)-np.asarray(desired_goal, np.float64), axis=-1)
        return (-d/self.distance_scale + (d <= self.task_config.success_threshold)*self.task_config.success_bonus).astype(np.float32)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal, np.float64)-np.asarray(desired_goal, np.float64), axis=-1)
        return (d <= self.task_config.success_threshold) & self.terminate_on_success

    def compute_truncated(self, achieved_goal, desired_goal, info):
        if isinstance(info, dict):
            steps = np.asarray(info.get('episode_step', 0))
        else:
            steps = np.asarray([x.get('episode_step', 0) for x in info])
        return steps >= self.task_config.max_episode_steps

    def _info(self, result):
        info = self.base._info(result)
        # Success uses the native f64 distance; the reward uses float32 goals as
        # stored by HER. Both use <=, including the exact 5 mm boundary.
        info['is_success'] = result['distance'] <= self.task_config.success_threshold
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base.reset(seed=seed, options=options)
        self.last_result = self.sim.task_state()
        self.previous_command = np.asarray(self.last_result['joint_position'], np.float64)
        self._needs_reset = False
        return self._observation(self.last_result), self._info(self.last_result)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded('call reset() before step()')
        action = np.asarray(action)
        if action.shape != self.action_space.shape or action.dtype.kind not in 'fiu' or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError(f'expected finite normalized action {self.action_space.shape} in [-1,1]')
        current = np.asarray(self.last_result['joint_position'], np.float64)
        command = self.home.copy()
        command[self.indices] = current[self.indices] + action*self.max_delta
        command = np.clip(command, self.previous_command-self.rate_limit, self.previous_command+self.rate_limit)
        command = np.clip(command, self.bounds[:,0], self.bounds[:,1])
        self.last_result = self.sim.step_task(command.tolist(), self.base._period)
        self.previous_command = command
        obs = self._observation(self.last_result)
        info = self._info(self.last_result)
        terminated = bool(info['is_success'] and self.terminate_on_success)
        truncated = self.last_result['episode_step'] >= self.task_config.max_episode_steps
        self._needs_reset = terminated or truncated
        return obs, float(self.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)), terminated, truncated, info

    def close(self):
        self._needs_reset = True
        self.base.close()
        self.sim = None
