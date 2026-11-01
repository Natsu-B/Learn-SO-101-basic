"""Grasp-force control environment on the SO-101 (docs/FORCE_CONTROL.md §6).

Reproduction of the tactile force-control MDP of Lach et al. (IROS 2024,
arXiv:2311.07245) on the Rust ``ForceGraspSimulator``; reimplemented from the
paper, no code from github.com/llach/learning_fc is copied.

Observation (one frame, stacked ``frame_stack`` times, newest last):
gripper angle, gripper velocity, jaw normal force [fixed, moving],
force error ``f_goal - f_i``, goal force, previous applied action and the
had-contact flags. Forces/angles carry Gaussian sensor noise. Object
stiffness, damping, friction, mass, width and gap are never observed.

Action: one value in [-1, 1]; the gripper target is ``q_des = q + phi * a *
dq_max`` (paper: position delta added to the measured position), clipped to
[gripper lower bound, GRIPPER_OPEN]. ``phi`` is the contact-state inductive
bias adapted to one moving jaw (docs/FORCE_CONTROL.md §1.2).

Reward ``"paper"``: ``a1 r_force + a2 r_obj + a3 r_act`` with
``r_force = 1 - tanh(sum_i |df_i| / f_goal)`` (public-code normalization),
``r_obj = -1`` if the object speed along the closing axis exceeds
``obj_velocity_max``, ``r_act = -|a_t - a_{t-1}|``. ``"paper_peak"`` adds
an overload penalty on the per-physics-step peak force (separate experiment).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover
    raise ImportError("ForceGraspEnv needs gymnasium (uv sync --extra learning)") from error

from ._native import CONTROL_PERIOD, GRIPPER_CLOSED, GRIPPER_OPEN, ForceGraspSimulator

FRAME_SIZE = 10
REWARDS = ("paper", "paper_peak")


def _lerp(a, b, t):
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


@dataclass
class ForceGraspConfig:
    """Everything that defines the MDP; saved next to each model."""

    max_steps: int = 150
    frame_stack: int = 3
    # "teleport": arm placed at the scripted descent pose by one IK solve
    # (same gap geometry, ~8x faster reset); "scripted": run the approach.
    positioning: str = "teleport"
    # Gripper opening (rad) reached open-loop before the episode starts. The
    # first moving-jaw contact over the default ranges is at 0.146-0.38 rad,
    # so contact is 2-14 full actions away, as in the paper (fully open
    # fingers a few dq_max from the object). From GRIPPER_OPEN (0.6) it would
    # be 11-23 and a zero-mean random policy almost never touches.
    initial_opening: float = 0.42
    # 0.02 rad/period = 0.6 rad/s, about 1.6 mm per period at the moving
    # jaw tip (paper: 3 mm per finger at 25 Hz). Near contact the jaw force
    # changes by ~0.4 N per mrad on the hardest object.
    dq_max: float = 0.02
    # 0 keeps commands continuous; 2*pi/4096 would mimic the STS3215 encoder.
    command_resolution: float = 0.0
    force_scale: float = 10.0
    noise_q: float = 4.4e-4  # STS3215 quantization (1.53 mrad / sqrt(12))
    noise_f: float = 0.01
    contact_threshold: float = 0.05
    goal_min: float = 0.3
    goal_cap: float = 8.0
    inductive_bias: bool = True
    # phi while only the moving jaw has touched (paper: 0.1 for the touching
    # finger; here it slows the only actuator while it pushes the object).
    bias_single_moving: float = 0.1
    reward: str = "paper"
    force_reward_norm: str = "goal"  # "goal" (public code) or "none" (paper eq. 8)
    alpha_force: float = 1.0
    alpha_act: float = 0.2
    # Curriculum (paper Table 2), interpolated by set_progress(0..1).
    alpha_obj: tuple[float, float] = (0.0, 1.0)
    # Paper: 2e-4 -> 5e-5 m per 25 Hz step = 5.0 -> 1.25 mm/s.
    obj_velocity_max: tuple[float, float] = (5.0e-3, 1.25e-3)
    width: tuple[tuple[float, float], tuple[float, float]] = ((0.025, 0.027), (0.022, 0.030))
    gap: tuple[tuple[float, float], tuple[float, float]] = ((0.004, 0.005), (0.004, 0.008))
    # Fixed ranges (not annealed).
    kappa: tuple[float, float] = (0.0, 1.0)
    friction: tuple[float, float] = (0.3, 1.5)
    mass: tuple[float, float] = (0.01, 0.1)
    gripper_damping_scale: tuple[float, float] = (0.7, 1.4)
    # paper_peak only: penalty on the physics-step peak face force above
    # (1 + peak_margin) * f_goal, normalized by f_goal and clipped.
    peak_weight: float = 1.0
    peak_margin: float = 0.5
    peak_clip: float = 5.0
    # "step": penalize the overload of every period (run p4: the policy
    # learned to avoid contact); "max": penalize only increases of the
    # episode's running maximum overload, so an episode costs at most
    # peak_weight * peak_clip and the term targets the largest force.
    peak_mode: str = "step"
    # If set, the peak weight is annealed from this value to peak_weight
    # with the curriculum (as alpha2), so the policy first learns to grasp.
    peak_weight_start: float | None = None

    def validate(self) -> None:
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}")
        if self.peak_mode not in ("step", "max"):
            raise ValueError("peak_mode must be step or max")
        if self.force_reward_norm not in ("goal", "none"):
            raise ValueError("force_reward_norm must be goal or none")
        if not (self.max_steps > 0 and self.frame_stack > 0 and self.dq_max > 0):
            raise ValueError("max_steps, frame_stack and dq_max must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForceGraspConfig":
        conv = {}
        for k, v in d.items():
            conv[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
        return cls(**conv)


@dataclass
class EpisodeMetrics:
    """Per-episode evaluation from true (noise-free) forces and per-physics-step extremes."""

    goal: float = 0.0
    forces: list = field(default_factory=list)  # true [fixed, moving] per period
    stats: list = field(default_factory=list)
    displacement: list = field(default_factory=list)
    face_shift: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    torques: list = field(default_factory=list)
    travel: float = 1.0

    def summary(self, threshold: float, penetration_limit: float, window: int = 30) -> dict[str, Any]:
        f = np.asarray(self.forces, dtype=float).reshape(-1, 2)
        n = len(f)
        both = (f > threshold).all(axis=1)
        contact = int(np.argmax(both)) if both.any() else None
        g = self.goal
        tol = max(0.1 * g, 0.05)
        tail = f[-window:]
        err = np.abs(tail - g)
        per_step_peak = [max(s["max_jaw_force"]) for s in self.stats]
        peak_face = max(per_step_peak, default=0.0)
        peak_step = int(np.argmax(per_step_peak)) if per_step_peak else None
        # Where the peak happens relative to the first period with both jaws
        # in contact: "push" (only the moving jaw), "contact" (within 3
        # periods of both-contact: the impact on the fixed jaw), "hold".
        if peak_step is None or contact is None:
            peak_phase = "push" if peak_step is not None and peak_face > 0 else "none"
        elif peak_step < contact - 1:
            peak_phase = "push"
        elif peak_step <= contact + 3:
            peak_phase = "contact"
        else:
            peak_phase = "hold"
        hold = per_step_peak[contact + 4:] if contact is not None else []
        peak_hold = max(hold, default=0.0)
        peak_rigid = max((max(s["max_rigid_force"]) for s in self.stats), default=0.0)
        peak_period_end = float(f.max()) if n else 0.0
        max_def = np.max([s["max_deformation"] for s in self.stats], axis=0) if self.stats else np.zeros(2)
        crushed = any(any(s["bottomed"]) for s in self.stats)
        invalid = any(s["max_penetration"] > penetration_limit for s in self.stats)
        mean_err_t = np.abs(f.mean(axis=1) - g)
        stable = None
        ok = (mean_err_t <= tol) & both
        if n and ok[-1]:
            k = n - 1
            while k > 0 and ok[k - 1]:
                k -= 1
            stable = k
        ratio = float(both[contact:].mean()) if contact is not None else 0.0
        lost_final = contact is not None and not bool(both[-window:].all())
        mean_abs = float(err.mean()) if n else float("nan")
        success = (
            contact is not None and not invalid and not crushed and not lost_final and mean_abs <= tol
        )
        return {
            "goal": g,
            "steps": n,
            "success": bool(success),
            "contact_step": contact,
            "mean_abs_error": mean_abs,
            "max_abs_error": float(err.max()) if n else float("nan"),
            "relative_error": mean_abs / g if g else float("nan"),
            "final_force": tail.mean(axis=0).tolist() if n else [0.0, 0.0],
            "peak_force_physics": float(peak_face),
            "peak_step": peak_step,
            "peak_phase": peak_phase,
            "peak_force_hold": float(peak_hold),
            "peak_rigid_force_physics": float(peak_rigid),
            "peak_force_period_end": peak_period_end,
            "peak_over_goal": float(peak_face / g) if g else float("nan"),
            "max_compression": (max_def / self.travel).tolist(),
            "max_compression_mm": (max_def * 1e3).tolist(),
            "crushed": bool(crushed),
            "invalid_contact": bool(invalid),
            "lost_contact_final": bool(lost_final),
            "both_contact_ratio": ratio,
            "time_to_stable_s": None if stable is None else stable * CONTROL_PERIOD,
            "object_displacement_mm": 1e3 * float(self.displacement[-1]) if self.displacement else 0.0,
            "max_object_displacement_mm": 1e3 * float(max(self.displacement, default=0.0)),
            "fixed_face_shift_mm": 1e3 * float(self.face_shift[-1]) if self.face_shift else 0.0,
            "return": float(np.sum(self.rewards)),
            "action_saturation": float(np.mean(np.abs(self.actions) >= 0.999)) if self.actions else 0.0,
            "torque_saturation": float(np.mean(self.torques)) if self.torques else 0.0,
        }


class ForceGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ForceGraspConfig | None = None, sim: ForceGraspSimulator | None = None):
        super().__init__()
        self.config = config or ForceGraspConfig()
        self.config.validate()
        self.sim = sim or ForceGraspSimulator(positioning=self.config.positioning)
        c = self.config
        dim = FRAME_SIZE * c.frame_stack
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.q_low = float(self.sim.gripper_bounds()[0])
        self.q_high = GRIPPER_OPEN
        self.penetration_limit = ForceGraspSimulator.penetration_limit
        self.progress = 0.0
        self._frames: deque = deque(maxlen=c.frame_stack)
        self._apply_progress()

    # ---- curriculum -------------------------------------------------
    def set_progress(self, progress: float) -> None:
        """Curriculum position in [0, 1]; ranges apply from the next reset."""
        self.progress = float(np.clip(progress, 0.0, 1.0))
        self._apply_progress()

    def _apply_progress(self) -> None:
        c, t = self.config, self.progress
        self.alpha_obj = float(_lerp(*c.alpha_obj, t))
        self.obj_velocity_max = float(_lerp(*c.obj_velocity_max, t))
        start = c.peak_weight if c.peak_weight_start is None else c.peak_weight_start
        self.peak_weight = float(_lerp(start, c.peak_weight, t))
        self.sim.set_ranges(
            width=list(_lerp(c.width[0], c.width[1], t)),
            gap=list(_lerp(c.gap[0], c.gap[1], t)),
            kappa=list(c.kappa),
            friction=list(c.friction),
            mass=list(c.mass),
            gripper_damping_scale=list(c.gripper_damping_scale),
        )

    # ---- gym API ------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        episode_seed = options.get("episode_seed")
        if episode_seed is None:
            episode_seed = int(self.np_random.integers(0, 2**62))
        self.episode_seed = int(episode_seed)
        # Goal and sensor noise derive from the episode seed so that every
        # controller sees the same goal and noise sequence for that seed.
        self._noise = np.random.default_rng([self.episode_seed, 0xF0])
        reading = self.sim.reset(self.episode_seed)
        if options.get("kappa") is not None:
            reading = self.sim.reset(self.episode_seed, self._with_kappa(float(options["kappa"])))
        reading = self._preclose(reading)
        lo, hi = self.sim.goal_range(self.config.goal_min, self.config.goal_cap)
        goal_rng = np.random.default_rng([self.episode_seed, 0x60A1])
        self.goal = float(options.get("goal", goal_rng.uniform(lo, hi)))
        self.object = self.sim.object()
        self.had_contact = np.zeros(2, dtype=bool)
        self.prev_action = 0.0  # applied (after the inductive bias)
        self.prev_raw_action = None
        self._max_over = 0.0
        self.steps = 0
        self._reading = reading
        self._last_pos = np.asarray(reading["object_position"])
        self._sense(reading)
        self.metrics = EpisodeMetrics(goal=self.goal, travel=self.object["travel"])
        frame = self._frame()
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)
        return self._obs(), self._info(reading, None)

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        phi = self.bias() if self.config.inductive_bias else 1.0
        applied = phi * a
        q_des = self.q_obs + applied * self.config.dq_max
        return self._advance(q_des, a, applied)

    def step_command(self, q_des: float, raw_action: float = 0.0):
        """Baselines: advance with an absolute gripper target through the same
        bookkeeping (observation, reward, metrics) as ``step``."""
        return self._advance(float(q_des), raw_action, raw_action)

    # ---- internals ----------------------------------------------------
    def _with_kappa(self, kappa: float) -> dict:
        """The seed's sampled object with stiffness set by kappa (paper's
        evaluation sweeps kappa with everything else sampled); damping keeps
        its ratio to critical damping."""
        o = dict(self.sim.object())
        hard, soft = self.sim.ranges()["stiffness"]
        k = float(np.exp(np.log(hard) + kappa * (np.log(soft) - np.log(hard))))
        o["damping"] *= np.sqrt(k / o["stiffness"])
        o["stiffness"] = k
        return o

    def _preclose(self, reading):
        """Close open-loop to ``initial_opening`` (not part of the episode)."""
        target = self.config.initial_opening
        q = reading["gripper_q"]
        self.precontact = False
        while q > target + 1e-9:
            q = max(target, q - 0.05)
            reading = self.sim.step(q)["reading"]
            self.precontact |= max(reading["jaw_force"]) > 0.0
        for _ in range(3):
            reading = self.sim.step(target)["reading"]
            self.precontact |= max(reading["jaw_force"]) > 0.0
        return reading

    def bias(self) -> float:
        """Inductive bias phi (paper eq. 7) for one moving jaw: 1 before any
        contact; 0.1 while only the moving jaw has touched (it is pushing the
        object across the gap); 1 if only the fixed jaw has touched (the
        moving jaw must keep closing); max(0.9, 1 - mean|df|/f_goal) once both
        have touched."""
        h_fixed, h_moving = self.had_contact
        if h_fixed and h_moving:
            return max(0.9, 1.0 - float(np.mean(np.abs(self.goal - self.f_obs))) / self.goal)
        if h_moving:
            return self.config.bias_single_moving
        return 1.0

    def _sense(self, reading) -> None:
        c = self.config
        self.q_obs = reading["gripper_q"] + self._noise.normal(0.0, c.noise_q)
        self.qvel_obs = reading["gripper_qvel"]
        self.f_obs = np.asarray(reading["jaw_force"]) + self._noise.normal(0.0, c.noise_f, 2)
        self.had_contact |= self.f_obs > c.contact_threshold

    def _frame(self) -> np.ndarray:
        c = self.config
        q = 2.0 * (self.q_obs - self.q_low) / (self.q_high - self.q_low) - 1.0
        v = self.qvel_obs / (c.dq_max / CONTROL_PERIOD)
        f = self.f_obs / c.force_scale
        df = (self.goal - self.f_obs) / c.force_scale
        return np.array(
            [q, v, f[0], f[1], df[0], df[1], self.goal / c.force_scale, self.prev_action,
             *self.had_contact.astype(float)],
            dtype=np.float32,
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)

    def _advance(self, q_des: float, raw: float, applied: float):
        c = self.config
        if c.command_resolution > 0:
            q_des = round(q_des / c.command_resolution) * c.command_resolution
        q_des = float(np.clip(q_des, self.q_low, self.q_high))
        try:
            out = self.sim.step(q_des)
        except RuntimeError as error:
            info = {"error": str(error), "episode_seed": self.episode_seed}
            return self._obs(), -1.0, True, False, info
        reading, stats = out["reading"], out["stats"]
        self._reading = reading
        self._sense(reading)
        pos = np.asarray(reading["object_position"])
        # Paper: object velocity along the grasp axis. The arm is frozen, so
        # horizontal motion of the core is along the pinch.
        speed = float(np.hypot(*(pos - self._last_pos)[:2])) / CONTROL_PERIOD
        self._last_pos = pos
        reward, terms = self._reward(raw, stats, speed)
        self.prev_action = applied
        self.prev_raw_action = raw
        self.steps += 1
        self._frames.append(self._frame())
        m = self.metrics
        m.forces.append(list(reading["jaw_force"]))
        m.stats.append(stats)
        m.displacement.append(reading["object_displacement"])
        m.face_shift.append(reading["fixed_face_shift"])
        m.rewards.append(reward)
        m.actions.append(raw)
        m.commands.append(q_des)
        m.torques.append(stats["max_actuator_torque"] >= 0.999 * reading["torque_limit"])
        truncated = self.steps >= c.max_steps
        info = self._info(reading, stats)
        info["reward_terms"] = terms
        if truncated:
            info["episode_metrics"] = m.summary(c.contact_threshold, self.penetration_limit)
        return self._obs(), reward, False, truncated, info

    def _reward(self, raw: float, stats, speed: float):
        c = self.config
        err = np.abs(self.goal - self.f_obs).sum()
        norm = self.goal if c.force_reward_norm == "goal" else 1.0
        r_force = 1.0 - np.tanh(err / norm)
        r_obj = -1.0 if speed > self.obj_velocity_max else 0.0
        prev = raw if self.prev_raw_action is None else self.prev_raw_action
        r_act = -abs(raw - prev)
        terms = {"force": float(r_force), "obj": r_obj, "act": float(r_act)}
        reward = c.alpha_force * r_force + self.alpha_obj * r_obj + c.alpha_act * r_act
        if c.reward == "paper_peak":
            peak = max(stats["max_jaw_force"])
            over = min(max(0.0, peak - (1.0 + c.peak_margin) * self.goal) / self.goal, c.peak_clip)
            if c.peak_mode == "max":
                r_peak = -max(0.0, over - self._max_over)
                self._max_over = max(self._max_over, over)
            else:
                r_peak = -over
            terms["peak"] = float(r_peak)
            reward += self.peak_weight * r_peak
        return float(reward), terms

    def _info(self, reading, stats) -> dict[str, Any]:
        return {
            "episode_seed": self.episode_seed,
            "precontact": self.precontact,
            "goal": self.goal,
            "jaw_force": list(reading["jaw_force"]),
            "had_contact": self.had_contact.tolist(),
            "bias": self.bias() if self.config.inductive_bias else 1.0,
            "stats": stats,
        }

    def close(self):
        self.sim.close()


def fixed_command_controller(env: ForceGraspEnv):
    """Existing Grasp close command held for the whole episode."""
    return lambda: GRIPPER_CLOSED


PI_DELTA_LIMIT = 0.03  # rad per period, as in force_grasp_probe (docs §4)


def pi_controller(env: ForceGraspEnv, pi):
    """Force PI on the same noisy observations as the policy (no inductive bias)."""
    def act():
        d = pi.act(list(env.f_obs), env.goal, CONTROL_PERIOD)
        return env.q_obs + float(np.clip(d, -PI_DELTA_LIMIT, PI_DELTA_LIMIT))
    return act
