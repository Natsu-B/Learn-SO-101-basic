"""Residual Grasp from the scripted pre-close state, under unknown object physics.

The scripted (IK) Grasp approaches and descends around the cube; the episode
starts at its Close phase boundary with the jaws open around the cube. Each
step the scripted skill proposes its command (close, lift, hold), and the
policy adds a bounded residual:

    command = clip(a_IK + [arm_scale * a[:5], gripper_scale * a[5]], bounds)

The IK skill is the base controller inside the policy; only the residual is
learned. Object mass and friction are randomized at reset by Rust and are never
part of the observation, reward or episode identity. Success is Rust's
``lifted`` predicate (both jaws, off the table, 30 mm up) held for 0.3 s
continuously. SI units throughout.
"""
from collections import deque
import logging
import math

import gymnasium as gym
import numpy as np

from ._native import CONTROL_PERIOD, PickPlaceSimulator

logger = logging.getLogger(__name__)

OBS_DIM = 41
ACTION_DIM = 6


def grasp_observation(state, base_command, progress):
    """Policy input from a native snapshot. Contains no mass or friction."""
    c = state["contacts"]
    cube = np.asarray(state["cube_pos"], np.float64)
    center = np.asarray(state["grasp_center"], np.float64)
    return np.concatenate((
        state["qpos"], state["qvel"], center, cube, state["cube_quat"], state["cube_vel"],
        cube - center, [float(c["fixed_jaw"]), float(c["moving_jaw"]), float(c["table"])],
        base_command, [progress])).astype(np.float32)


class ObservationHistory:
    """Stack the current observation with the previous ``length`` (obs, action)
    frames. ``length=0`` is the plain current observation. Shared by training
    and inference so both build identical inputs."""

    def __init__(self, length):
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 0:
            raise ValueError("history length must be a non-negative integer")
        self.length = int(length)
        self.frames = deque(maxlen=self.length)
        self.last = None

    @property
    def size(self):
        return OBS_DIM + self.length * (OBS_DIM + ACTION_DIM)

    def reset(self, obs):
        self.last = obs
        self.frames.clear()
        for _ in range(self.length):
            self.frames.append(np.concatenate((obs, np.zeros(ACTION_DIM, np.float32))))
        return self.build()

    def push(self, action, obs):
        """Record the action applied to the previous observation, then obs."""
        if self.length:
            self.frames.append(np.concatenate((self.last, np.asarray(action, np.float32))))
        self.last = obs
        return self.build()

    def build(self):
        return np.concatenate((self.last, *self.frames)).astype(np.float32)


def residual_command(base, action, bounds, *, arm_scale, gripper_scale):
    action = np.asarray(action)
    if (action.shape != (ACTION_DIM,) or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all() or np.any(np.abs(action) > 1)):
        raise ValueError("expected finite normalized action (6,) in [-1,1]")
    command = np.asarray(base, np.float64).copy()
    command[:5] += arm_scale * action[:5]
    command[5] += gripper_scale * action[5]
    return np.clip(command, bounds[:, 0], bounds[:, 1])


def approach_to_pre_close(sim, on_step=None):
    """From the current state, start the scripted Grasp and run its Approach and
    Descend phases. Returns True when it reached Close still running."""
    sim.start_skill("grasp")
    status = sim.skill_report()["status"]
    while status == "running" and sim.skill_phase() in ("approach", "descend"):
        status = sim.skill_step()
        if on_step is not None:
            on_step(sim.state(), "grasp", False)
    return status == "running" and sim.skill_phase() == "close"


def start_pre_close(sim, seed, *, max_retries=3):
    """Reset and run the scripted Grasp up to its Close phase (training only).
    Returns the native seed actually used and the number of retries."""
    for attempt in range(max_retries + 1):
        used = (int(seed) + attempt) % 2**64
        sim.reset(used)
        if approach_to_pre_close(sim):
            return used, attempt
        logger.warning("scripted approach failed: seed=%d reason=%s", used,
                       sim.skill_report()["reason"])
    raise RuntimeError(f"approach failed {max_retries + 1} times from seed {seed}")


def cube_tilt(quat):
    """Angle (rad) between the cube's local z axis and world z."""
    w, x, y, z = quat
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


class GraspTracker:
    """Success, drop and slip bookkeeping from Rust predicates and true motion."""

    def __init__(self, criteria):
        self.hold_steps = math.ceil(criteria["hold_seconds"] / CONTROL_PERIOD - 1e-9)
        self.lift_height = criteria["lift_height"]
        self.rest_z = criteria["cube_half"]
        self.held = 0
        self.was_lifted = False
        self.reference = None
        self.slip = 0.0
        self.max_slip = 0.0
        self.success = False
        self.ever_success = False
        self.dropped = False
        self.tilt = 0.0
        self.max_tilt = 0.0

    def update(self, state, predicates):
        offset = np.subtract(state["cube_pos"], state["grasp_center"])
        # Closing pushes the cube against the fixed jaw by design; slip is
        # measured from the moment it is pinched and off the table.
        if state["contacts"]["pinched"] and (self.reference is not None
                                             or not state["contacts"]["table"]):
            if self.reference is None:
                self.reference = offset
            self.slip = float(np.linalg.norm(offset - self.reference))
            self.max_slip = max(self.max_slip, self.slip)
        self.tilt = cube_tilt(state["cube_quat"])
        self.max_tilt = max(self.max_tilt, self.tilt)
        self.held = self.held + 1 if predicates["lifted"] else 0
        self.was_lifted |= predicates["lifted"]
        self.success = self.held >= self.hold_steps
        self.ever_success |= self.success
        # Dropped only counts after the cube had been lifted at least once.
        self.dropped = self.was_lifted and predicates["dropped"]
        return self


class ResidualGrasp:
    """One residual Grasp episode from the pre-close state. Training (the env)
    and inference (the pipeline) both step through this class, so observation
    history, residual mapping and termination are identical."""

    def __init__(self, sim, *, history, horizon, arm_scale, gripper_scale, criteria, bounds,
                 end_at="success"):
        if end_at not in ("success", "base"):
            raise ValueError("end_at must be success or base")
        # "success": stop as soon as the lift has been held 0.3 s (v1).
        # "base": run until the scripted base skill finishes its own Hold at
        # carry height, so the hand-off matches the IK Grasp's (v3).
        self.end_at = end_at
        self.sim = sim
        self.horizon = int(horizon)
        self.arm_scale = float(arm_scale)
        self.gripper_scale = float(gripper_scale)
        self.criteria = criteria
        self.bounds = bounds
        self.history = ObservationHistory(history)
        self.state = sim.state()
        self.tracker = GraspTracker(criteria)
        self.step_count = 0
        self.done = False
        self.base = self._base()
        self.obs = self.history.reset(self._observation(self.base))

    def _base(self):
        """Scripted proposal for this step, or None once the IK skill ended."""
        if self.sim.skill_report()["status"] != "running":
            return None
        return np.asarray(self.sim.skill_act(), np.float64)

    def _observation(self, base):
        hold = self.base if base is None else base
        return grasp_observation(self.state, hold, self.step_count / self.horizon)

    def step(self, action):
        """Apply a_IK + residual; returns (predicates, terminated, truncated)."""
        if self.done:
            raise RuntimeError("residual grasp episode has ended")
        command = self._command(action)
        self.state = self.sim.skill_apply(command.tolist())
        self._adjust(action)
        self.step_count += 1
        predicates = self.sim.predicates()
        t = self.tracker.update(self.state, predicates)
        if self.end_at == "success":
            terminated = t.success or t.dropped
            base = None if terminated else self._base()
            # The scripted skill can end on its own (its success or a hold failure);
            # without a base command the episode ends there and is judged by the tracker.
            truncated = not terminated and (base is None or self.step_count >= self.horizon)
        else:
            base = None if t.dropped else self._base()
            # The base skill ending (its Hold success or hold failure) is a true end.
            terminated = t.dropped or base is None
            truncated = not terminated and self.step_count >= self.horizon
        if base is not None:
            self.base = base
        self.predicates = predicates
        self.done = terminated or truncated
        self.obs = self.history.push(action, self._observation(base))
        return predicates, terminated, truncated

    def _command(self, action):
        return residual_command(self.base, action, self.bounds, arm_scale=self.arm_scale,
                                gripper_scale=self.gripper_scale)

    def _adjust(self, action):
        """Change the scripted skill's targets before its next proposal."""

    @property
    def success(self):
        """Grasp success for the hand-off: the 0.3 s lift was achieved, nothing
        dropped, and (for end_at="base") the cube is still lifted at the end."""
        t = self.tracker
        if self.end_at == "success":
            return bool(t.success)
        return bool(t.ever_success and not t.dropped and self.predicates["lifted"])


# Low-dimensional residual modes: number of policy outputs.
LOW_DIM_MODES = {"wrist": 1, "wrist_lift": 3}


class LowDimResidualGrasp(ResidualGrasp):
    """Residual on top of the IK Grasp plus a fixed wrist_roll offset.

    a[0] -> wrist_roll = base_wrist + wrist_scale * a[0] (rad, added to a_IK).
    With mode "wrist_lift" also a[1] -> lift target shifted by lift_scale * a[1]
    (m) along the jaw closing direction and a[2] -> lift speed scaled by
    1 + speed_scale * a[2]; both set the Rust Grasp's targets for its next
    proposals (Skill::adjust). The gripper keeps the IK command: it is at its
    force limit, so a gripper residual has no authority."""

    def __init__(self, sim, *, mode, base_wrist, wrist_scale, lift_scale=0.01,
                 speed_scale=0.5, **kwargs):
        if mode not in LOW_DIM_MODES:
            raise ValueError(f"mode must be one of {sorted(LOW_DIM_MODES)}")
        if kwargs.get("history"):
            raise ValueError("low-dimensional residuals use history 0")
        self.mode = mode
        self.base_wrist = float(base_wrist)
        self.wrist_scale = float(wrist_scale)
        self.lift_scale = float(lift_scale)
        self.speed_scale = float(speed_scale)
        super().__init__(sim, **kwargs)

    def _command(self, action):
        action = np.asarray(action, np.float64)
        if (action.shape != (LOW_DIM_MODES[self.mode],) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1)):
            raise ValueError("expected finite normalized low-dimensional action in [-1,1]")
        command = np.asarray(self.base, np.float64).copy()
        command[4] += self.base_wrist + self.wrist_scale * action[0]
        return np.clip(command, self.bounds[:, 0], self.bounds[:, 1])

    def _adjust(self, action):
        if self.mode != "wrist_lift" or self.sim.skill_report()["status"] != "running":
            return
        offset, speed = self.adjustment(action)
        self.sim.skill_adjust([offset, speed])

    def adjustment(self, action):
        """The (lift offset m, speed scale) an action requests, or None."""
        if self.mode != "wrist_lift":
            return None
        return self.lift_scale * float(action[1]), 1.0 + self.speed_scale * float(action[2])


def residual_episode(sim, config, *, criteria, bounds):
    """Build the residual Grasp episode a checkpoint's env config describes
    (shared by GoalGraspEnv and pipeline inference)."""
    end_at = config.get("end_at") or ("base" if config.get("reward") == "v3" else "success")
    common = dict(history=config["history"], horizon=config["horizon"],
                  arm_scale=config["arm_scale"], gripper_scale=config["gripper_scale"],
                  criteria=criteria, bounds=bounds, end_at=end_at)
    mode = config.get("action_mode", "joint6")
    if mode == "joint6":
        return ResidualGrasp(sim, **common)
    return LowDimResidualGrasp(sim, mode=mode, base_wrist=config["base_wrist"],
                               wrist_scale=config["wrist_scale"],
                               lift_scale=config.get("lift_scale", 0.01),
                               speed_scale=config.get("speed_scale", 0.5), **common)


class GoalGraspEnv(gym.Env):
    metadata = {"render_modes": []}
    # Scripted Grasp carry height (skills.rs ScriptedGrasp::carry_height).
    CARRY_HEIGHT = 0.08

    def __init__(self, *, history=0, horizon=75, arm_scale=0.03, gripper_scale=0.35,
                 mass_range=(0.02, 0.05), friction_range=(0.8, 1.5), friction_sampling="uniform",
                 action_penalty=0.05, slip_penalty=50.0, tilt_penalty=0.0, max_retries=3,
                 shape="cube", reward="v1", end_at=None, downstream=False,
                 action_mode="joint6", base_wrist=0.0, wrist_scale=0.1, lift_scale=0.01,
                 speed_scale=0.5, contact_model="legacy", floor_friction=None):
        super().__init__()
        for name, value in (("arm_scale", arm_scale), ("gripper_scale", gripper_scale)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = int(horizon)
        self.arm_scale = float(arm_scale)
        self.gripper_scale = float(gripper_scale)
        self.action_penalty = float(action_penalty)
        self.slip_penalty = float(slip_penalty)
        # Hand-off quality: a cube tilted in the jaws later topples at Place.
        self.tilt_penalty = float(tilt_penalty)
        self.max_retries = int(max_retries)
        if reward not in ("v1", "v3"):
            raise ValueError("reward must be v1 or v3")
        self.reward_version = reward
        self.end_at = end_at or ("base" if reward == "v3" else "success")
        # v3: after a successful Grasp, run IK Transport and Place in Rust and
        # reward the whole pick-and-place (the Grasp's downstream consequence).
        self.downstream = bool(downstream)
        self.carry_height = self.CARRY_HEIGHT
        if friction_sampling not in ("uniform", "log"):
            raise ValueError("friction_sampling must be uniform or log")
        if friction_sampling == "log" and friction_range is None:
            raise ValueError("log friction sampling needs a friction range")
        self.history_length = int(history)
        self.mass_range = None if mass_range is None else tuple(mass_range)
        self.friction_range = None if friction_range is None else tuple(friction_range)
        self.friction_sampling = friction_sampling
        if action_mode != "joint6" and action_mode not in LOW_DIM_MODES:
            raise ValueError("unknown action_mode")
        self.action_mode = action_mode
        self.low_dim = dict(base_wrist=float(base_wrist), wrist_scale=float(wrist_scale),
                            lift_scale=float(lift_scale), speed_scale=float(speed_scale))
        # pairwise: friction_range is the jaw friction; floor_friction "same"
        # gives the table the same value, None keeps the scene's 1.2.
        if floor_friction not in (None, "same"):
            raise ValueError("floor_friction must be None or 'same'")
        if floor_friction == "same" and contact_model != "pairwise":
            raise ValueError("floor_friction='same' needs contact_model='pairwise'")
        self.contact_model = contact_model
        self.floor_friction = floor_friction
        self.sim = PickPlaceSimulator(shape=shape, contact_model=contact_model)
        self.set_physics_ranges(self.mass_range, self.friction_range)
        self.bounds = np.asarray(self.sim.action_bounds(), np.float64)
        self.criteria = self.sim.criteria()
        dim = ACTION_DIM if action_mode == "joint6" else LOW_DIM_MODES[action_mode]
        self.action_space = gym.spaces.Box(-1., 1., (dim,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (ObservationHistory(history).size,), np.float32)
        self._needs_reset = True

    def set_physics_ranges(self, mass, friction):
        """Hidden-physics ranges for later resets (evaluation fixes them)."""
        self.mass_range, self.friction_range = mass, friction
        floor = dict(floor_friction=friction) if self.floor_friction == "same" else {}
        self.sim.set_object_randomization(mass=mass, friction=friction, **floor)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True
        requested = int(seed) if seed is not None else int(self.np_random.integers(0, 2**63))
        if self.friction_sampling == "log":
            # Log-uniform friction so the slippery regime is well covered. The
            # value goes to Rust as a fixed range and never into the observation.
            lo, hi = np.log(self.friction_range)
            mu = float(np.exp(self.np_random.uniform(lo, hi)))
            floor = dict(floor_friction=(mu, mu)) if self.floor_friction == "same" else {}
            self.sim.set_object_randomization(mass=self.mass_range, friction=(mu, mu), **floor)
        self.episode_seed, self.retries = start_pre_close(self.sim, requested,
                                                          max_retries=self.max_retries)
        self.requested_seed = requested
        config = dict(history=self.history_length, horizon=self.horizon,
                      arm_scale=self.arm_scale, gripper_scale=self.gripper_scale,
                      end_at=self.end_at, action_mode=self.action_mode, **self.low_dim)
        self.episode = residual_episode(self.sim, config, criteria=self.criteria,
                                        bounds=self.bounds)
        self.downstream_result = None
        self._potential = 0.0
        self._needs_reset = False
        return self.episode.obs, self._info()

    def _info(self):
        e = self.episode
        r = self.downstream_result
        return dict(is_success=e.success if e.step_count else False, dropped=e.tracker.dropped,
                    pipeline_success=None if r is None else r["placed"],
                    transport_success=None if r is None else r["transport"],
                    max_tilt=e.tracker.max_tilt,
                    max_slip=e.tracker.max_slip, seed=self.episode_seed,
                    requested_seed=self.requested_seed, retries=self.retries,
                    episode_step=e.step_count)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before step()")
        e = self.episode
        predicates, terminated, truncated = e.step(action)
        if self.reward_version == "v3":
            reward = self._reward_v3(action, predicates, terminated or truncated)
            self._needs_reset = e.done
            return e.obs, float(reward), bool(terminated), bool(truncated), self._info()
        t = e.tracker
        z = e.state["cube_pos"][2] - self.criteria["cube_half"]
        pinched = e.state["contacts"]["pinched"]
        reward = (0.25 * pinched
                  + pinched * float(np.clip(z / self.criteria["lift_height"], 0., 1.))
                  + 1.0 * predicates["lifted"]
                  - self.slip_penalty * t.slip
                  - self.tilt_penalty * t.tilt
                  - self.action_penalty * float(np.square(action).sum()))
        if t.success:
            reward += 10.
        elif t.dropped:
            reward -= 10.
        self._needs_reset = e.done
        return e.obs, float(reward), bool(terminated), bool(truncated), self._info()

    def _reward_v3(self, action, predicates, done):
        """Lift progress (potential-based), action cost, and terminal terms:
        drop, stable grasp at the hand-off, and downstream pick-and-place."""
        e = self.episode
        z = e.state["cube_pos"][2] - self.criteria["cube_half"]
        pinched = e.state["contacts"]["pinched"]
        potential = float(pinched) * float(np.clip(z / self.carry_height, 0., 1.))
        reward = 5.0 * (potential - self._potential)
        self._potential = potential
        reward -= self.action_penalty * float(np.square(action).sum())
        if e.tracker.dropped:
            return reward - 10.
        if not done:
            return reward
        if e.success:
            velocity = np.asarray(e.state["cube_vel"][:3])
            stable = (np.linalg.norm(velocity) < 0.05
                      and e.tracker.tilt < np.radians(5.0))
            reward += 2.0 + 3.0 * stable
            if self.downstream:
                self.downstream_result = self.run_downstream()
                reward += 3.0 * self.downstream_result["transport"]
                reward += 10.0 * self.downstream_result["placed"]
        return reward

    def run_downstream(self):
        """IK Transport and Place from the hand-off state (Rust, no policy)."""
        transport = self.sim.run_skill("transport")["success"]
        placed = False
        if transport:
            placed = self.sim.run_skill("place")["success"] and self.sim.predicates()["placed"]
        return dict(transport=bool(transport), placed=bool(placed))

    def close(self):
        self._needs_reset = True
        self.sim.close()
