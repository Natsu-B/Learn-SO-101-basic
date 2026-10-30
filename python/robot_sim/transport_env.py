"""Goal-conditioned transport from the real scripted Grasp end state.

SI units throughout. Training always runs to the horizon, including after a
drop or success, so HER goals cannot change transition termination semantics.
Only the Rust predicates decide physical success; reward is purely geometric.
"""
import logging

import gymnasium as gym
import numpy as np

from ._native import CONTROL_PERIOD, GRIPPER_CLOSED, PickPlaceSimulator

logger = logging.getLogger(__name__)


def transport_observation(state, carry_z):
    """Build the policy input from a native snapshot (also used at inference)."""
    q = np.asarray(state["qpos"], np.float32)
    v = np.asarray(state["qvel"], np.float32)
    center = np.asarray(state["grasp_center"], np.float32)
    cube = np.asarray(state["cube_pos"], np.float32)
    velocity = np.asarray(state["cube_vel"][:3], np.float32)
    offset = (np.asarray(state["cube_pos"]) - state["grasp_center"]).astype(np.float32)
    goal = np.asarray([*state["target"][:2], carry_z], np.float32)
    return dict(observation=np.concatenate((q[:5], v[:5], center, cube,
                                            velocity, q[5:6], offset)),
                achieved_goal=cube, desired_goal=goal)


def transport_command(state, action, previous_command, bounds, *, max_delta, rate_limit):
    """Measured-q delta, command slew limit and native bounds; gripper stays closed."""
    action = np.asarray(action)
    if (action.shape != (5,) or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all() or np.any(np.abs(action) > 1)):
        raise ValueError("expected finite normalized action (5,) in [-1,1]")
    command = np.asarray(state["qpos"], np.float64).copy()
    command[:5] += action * max_delta
    command[:5] = np.clip(command[:5], previous_command[:5] - rate_limit,
                         previous_command[:5] + rate_limit)
    command = np.clip(command, bounds[:, 0], bounds[:, 1])
    command[5] = GRIPPER_CLOSED
    return command


class GoalTransportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, max_delta=0.05, rate_limit=0.1, distance_scale=0.1,
                 horizon=150, terminate_on_success=False, terminate_on_drop=False,
                 max_grasp_retries=3):
        super().__init__()
        for name, value in (("max_delta", max_delta), ("rate_limit", rate_limit),
                            ("distance_scale", distance_scale)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value, minimum in (("horizon", horizon, 1),
                                     ("max_grasp_retries", max_grasp_retries, 0)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_delta = float(max_delta)
        self.rate_limit = float(rate_limit)
        self.distance_scale = float(distance_scale)
        self.horizon = int(horizon)
        # These options are for evaluation only; the training runner leaves both off.
        self.terminate_on_success = bool(terminate_on_success)
        self.terminate_on_drop = bool(terminate_on_drop)
        self.max_grasp_retries = int(max_grasp_retries)
        self.sim = PickPlaceSimulator()
        self.bounds = np.asarray(self.sim.action_bounds(), np.float64)
        self.criteria = self.sim.criteria()
        self.carry_z = self.criteria["cube_half"] + self.criteria["carry_height"]
        self.action_space = gym.spaces.Box(-1., 1., (5,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-np.inf, np.inf, (23,), np.float32),
            "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "desired_goal": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        self.grasp_attempts_total = 0
        self.grasp_failures_total = 0
        self._needs_reset = True

    def _observation(self):
        return transport_observation(self.last_result, self.carry_z)

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(np.asarray(achieved_goal, np.float64)
                                  - np.asarray(desired_goal, np.float64), axis=-1)
        return (-distance / self.distance_scale).astype(np.float32)

    def _info(self):
        predicates = self.sim.predicates()
        cube = np.asarray(self.last_result["cube_pos"], np.float64)
        target = np.asarray([*self.last_result["target"][:2], self.carry_z])
        return dict(is_success=predicates["transported"], dropped=predicates["dropped"],
                    distance=float(np.linalg.norm(cube - target)),
                    xy_error=float(np.linalg.norm(cube[:2] - target[:2])),
                    episode_step=self.episode_step, seed=self.episode_seed,
                    requested_seed=self.requested_seed, grasp_retries=self.grasp_retries,
                    grasp_report=self.grasp_report.copy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True
        # Explicit seeds map directly to the native seed. Unseeded resets advance
        # Gymnasium's seeded generator, including SB3's automatic episode resets.
        self.requested_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**63))
        for attempt in range(self.max_grasp_retries + 1):
            self.episode_seed = (self.requested_seed + attempt) % 2**64
            self.sim.reset(self.episode_seed)
            self.grasp_attempts_total += 1
            self.grasp_report = self.sim.run_skill("grasp")
            if self.grasp_report["success"]:
                break
            self.grasp_failures_total += 1
            logger.warning("scripted Grasp failed: seed=%d attempt=%d reason=%s",
                           self.episode_seed, attempt + 1, self.grasp_report["reason"])
        else:
            raise RuntimeError(f"Grasp failed after {self.max_grasp_retries + 1} attempts "
                               f"starting at seed {self.requested_seed}")
        self.grasp_retries = attempt
        self.episode_step = 0
        self.last_result = self.sim.state()
        self.previous_command = np.asarray(self.last_result["qpos"], np.float64)
        self.previous_command[5] = GRIPPER_CLOSED
        self._needs_reset = False
        return self._observation(), self._info()

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before step()")
        command = transport_command(self.last_result, action, self.previous_command,
                                    self.bounds, max_delta=self.max_delta,
                                    rate_limit=self.rate_limit)
        self.last_result = self.sim.step(command.tolist(), CONTROL_PERIOD)
        self.previous_command = command
        self.episode_step += 1
        obs, info = self._observation(), self._info()
        terminated = bool((self.terminate_on_success and info["is_success"])
                          or (self.terminate_on_drop and info["dropped"]))
        truncated = self.episode_step >= self.horizon
        self._needs_reset = terminated or truncated
        return obs, float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)), terminated, truncated, info

    def close(self):
        self._needs_reset = True
        self.sim.close()
