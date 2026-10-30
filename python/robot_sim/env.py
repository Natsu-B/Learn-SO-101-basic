"""Gymnasium transport wrapper; physics, reset and Reach reward live in Rust."""
import gymnasium as gym
import numpy as np
from ._native import Simulator
from .config import TaskConfig
from dataclasses import asdict, replace
from .randomization import load_profile, load_parameters
import json
import copy


class RobotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, robot="so101", task="reach", control_frequency=30.0,
                 max_episode_steps=None, render_mode=None, task_config=None, domain_randomization=None, model_parameters=None):
        if task != "reach":
            raise ValueError("only task='reach' is supported")
        if render_mode is not None:
            raise ValueError("camera rendering is not implemented; use CLI viewers")
        if not np.isfinite(control_frequency) or control_frequency <= 0:
            raise ValueError("control_frequency must be finite and positive")
        self.task_config = task_config or TaskConfig()
        if max_episode_steps is not None:
            self.task_config = replace(self.task_config, max_episode_steps=max_episode_steps)
        if not isinstance(self.task_config.max_episode_steps, int) or self.task_config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        self.sim = Simulator(robot=robot)
        try:
            self.sim.configure_randomization(json.dumps(load_profile(domain_randomization),allow_nan=False))
            parameters=load_parameters(model_parameters)
            if parameters is not None: self.sim.set_parameter_center(json.dumps(parameters,allow_nan=False))
            self.sim.configure_reach(**asdict(self.task_config))
        except Exception:
            self.sim.close()
            raise
        self._domain_parameters = None
        self._actuator_names = self.sim.actuator_names
        self.last_observation = None
        self.render_mode = None
        self._period = 1.0 / control_frequency
        self._needs_reset = True
        bounds = np.asarray(self.sim.action_bounds, dtype=np.float64)
        low, high = bounds[:, 0].astype(np.float32), bounds[:, 1].astype(np.float32)
        # Round inward: float32 Box samples must stay inside the native f64 limits.
        low = np.where(low.astype(np.float64) < bounds[:, 0], np.nextafter(low, np.float32(np.inf)), low)
        high = np.where(high.astype(np.float64) > bounds[:, 1], np.nextafter(high, np.float32(-np.inf)), high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        n = len(bounds)
        self.observation_space = gym.spaces.Dict({
            "joint_position": gym.spaces.Box(-np.inf, np.inf, (n,), np.float32),
            "joint_velocity": gym.spaces.Box(-np.inf, np.inf, (n,), np.float32),
            "end_effector_position": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "target_position": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })

    def _observation(self, result):
        self.last_observation = {key: np.asarray(result[key], np.float32)
                                 for key in self.observation_space.spaces}
        return {key: value.copy() for key, value in self.last_observation.items()}

    def _info(self, result):
        return {"observation_time": result["observation_time"], "time": result["time"], "distance": result["distance"],
                "is_success": result["terminated"], "episode_step": result["episode_step"],
                "actuator_names": list(self._actuator_names),
                "domain_randomization": copy.deepcopy(self._domain_parameters)}

    def reset(self, *, seed=None, options=None):
        if self.sim is None:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        if seed is not None and seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        task_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63))
        self._needs_reset = True
        self.sim.reset_reach(task_seed)
        self._domain_parameters=json.loads(self.sim.parameters_json())
        self._needs_reset = False
        result = self.sim.task_state()
        return self._observation(result), self._info(result)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before step()")
        action = np.asarray(action)
        if action.shape != self.action_space.shape or action.dtype.kind not in "fiu":
            raise ValueError(f"expected numeric action of shape {self.action_space.shape}")
        if not np.isfinite(action).all() or np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("joint targets must be finite and inside action_space")
        result = self.sim.step_task(action.astype(np.float64).tolist(), self._period)
        self._needs_reset = result["terminated"] or result["truncated"]
        return (self._observation(result), result["reward"], result["terminated"],
                result["truncated"], self._info(result))

    def close(self):
        self._needs_reset = True
        if self.sim is not None:
            self.sim.close()
        self.sim = None
