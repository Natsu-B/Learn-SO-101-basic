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
REWARDS = ("paper", "paper_peak", "contact_safe", "coop_safe")
ACTION_MODES = ("gripper", "arm_gripper", "arm_pi")
ARM_FRAME_EXTRA = 6


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
    # contact_safe (P9, docs §10.4): +contact_bonus once when both jaws first
    # touch, -contact_bonus at truncation if they never did, so avoiding
    # contact never pays; overload above max(1.5 f_goal, f_goal + 1 N)
    # normalized by a fixed safe_force_scale (not by f_goal) and charged on
    # increases of the running maximum, at most safe_overload_weight *
    # safe_overload_clip per episode; -crush_penalty once at the first
    # bottoming (reward only; the observation is unchanged).
    contact_bonus: float = 20.0
    safe_force_scale: float = 5.0
    safe_overload_weight: float = 5.0
    safe_overload_clip: float = 2.0
    crush_penalty: float = 5.0
    # Arm + gripper cooperation (docs §11). "gripper": the paper's 1-D task
    # (default, unchanged). "arm_gripper": 2-D [gripper delta, arm shift
    # delta]. "arm_pi": 1-D arm shift delta, gripper by the scripted force
    # law of ArmAssistPi. The arm shift moves the pinch target horizontally
    # along the closing direction (ForceGraspSimulator.shift_arm), limited
    # to arm_step_max per period and arm_shift_range in total; the target
    # height is unchanged (no table approach) and the IK keeps joint limits.
    action_mode: str = "gripper"
    arm_step_max: float = 0.0004  # m per period = 12 mm/s
    arm_shift_range: tuple[float, float] = (0.0, 0.012)
    # coop_safe reward (docs §11.2): tracking only while both jaws touch,
    # hold bonus, one-time contact bonus / crush / invalid penalties,
    # P9's absolute overload term, terminal success bonus and no-contact
    # penalty.
    hold_bonus: float = 0.5
    invalid_penalty: float = 10.0
    coop_crush_penalty: float = 10.0
    success_bonus: float = 50.0

    def validate(self) -> None:
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        if self.action_mode != "gripper" and (self.positioning != "teleport" or self.inductive_bias):
            raise ValueError("arm action modes need teleport positioning and no inductive bias")
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
    term_sums: dict = field(default_factory=dict)
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
        # Decomposition (docs/FORCE_CONTROL.md §10): which jaw touched first,
        # when the goal was first met, and when/where a face first bottomed.
        touched = f > threshold
        any_touch = touched.any(axis=1)
        first_touch = int(np.argmax(any_touch)) if any_touch.any() else None
        if first_touch is None:
            first_jaw = None
        elif touched[first_touch].all():
            first_jaw = "both"
        else:
            first_jaw = "fixed" if touched[first_touch][0] else "moving"
        goal_step = int(np.argmax(ok)) if ok.any() else None
        bottom_steps = [k for k, s in enumerate(self.stats) if any(s["bottomed"])]
        first_bottom = bottom_steps[0] if bottom_steps else None
        if first_bottom is None:
            bottom_phase = None
        elif contact is None or first_bottom < contact - 1:
            bottom_phase = "push"
        elif first_bottom <= contact + 3:
            bottom_phase = "contact"
        elif goal_step is None or first_bottom < goal_step:
            bottom_phase = "before_goal"
        else:
            bottom_phase = "after_goal"
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
            "any_contact": first_touch is not None,
            "first_contact_jaw": first_jaw,
            "goal_reached_step": goal_step,
            "first_bottom_step": first_bottom,
            "bottom_phase": bottom_phase,
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
            "reward_terms": dict(self.term_sums),
        }


class ForceGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ForceGraspConfig | None = None, sim: ForceGraspSimulator | None = None):
        super().__init__()
        self.config = config or ForceGraspConfig()
        self.config.validate()
        self.sim = sim or ForceGraspSimulator(positioning=self.config.positioning)
        c = self.config
        self.arm = c.action_mode != "gripper"
        frame = FRAME_SIZE + (ARM_FRAME_EXTRA if self.arm else 0)
        dim = frame * c.frame_stack
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        n_act = 2 if c.action_mode == "arm_gripper" else 1
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self._gripper_law = None
        if c.action_mode == "arm_pi":
            self._gripper_law = ArmAssistPi(drive_arm=False, push_ramp=0.5, push_speed=0.3,
                                            push_ratio=0.8)
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
        self.arm_shift = 0.0
        self.prev_arm_action = 0.0
        self._arm_raw = 0.0
        self._last_pinch = reading["pinch_offset"]
        self.pinch_speed = 0.0
        self._once: set = set()
        if self._gripper_law is not None:
            self._gripper_law.reset()
        self._contact_rewarded = False
        self._crush_charged = False
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
        if self.arm:
            return self._step_arm(action)
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        phi = self.bias() if self.config.inductive_bias else 1.0
        applied = phi * a
        q_des = self.q_obs + applied * self.config.dq_max
        return self._advance(q_des, a, applied)

    def _step_arm(self, action):
        c = self.config
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        a_arm = float(a[-1])
        lo, hi = c.arm_shift_range
        shift = float(np.clip(self.arm_shift + a_arm * c.arm_step_max, lo, hi))
        if abs(shift - self.arm_shift) > 1e-9:
            self.sim.shift_arm(shift)
            self.arm_shift = shift
        self._arm_raw = a_arm
        if c.action_mode == "arm_gripper":
            raw = float(a[0])
            q_des = self.q_obs + raw * c.dq_max
        else:
            raw = 0.0
            q_des = self._gripper_law(self)
        return self._advance(q_des, raw, raw)

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
        base = np.array(
            [q, v, f[0], f[1], df[0], df[1], self.goal / c.force_scale, self.prev_action,
             *self.had_contact.astype(float)],
            dtype=np.float32,
        )
        return np.concatenate([base, self._arm_frame()]) if self.arm else base

    def _arm_frame(self) -> np.ndarray:
        c = self.config
        span = c.arm_shift_range[1]
        now = (self.f_obs > c.contact_threshold).astype(float)
        return np.array(
            [self.arm_shift / span, self._reading["pinch_offset"] / span,
             self.pinch_speed / (c.arm_step_max / CONTROL_PERIOD), self.prev_arm_action, *now],
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
        if self.arm:
            pinch = reading["pinch_offset"]
            self.pinch_speed = (pinch - self._last_pinch) / CONTROL_PERIOD
            self._last_pinch = pinch
        reward, terms = self._reward(raw, stats, speed)
        if self.arm:
            self.prev_arm_action = self._arm_raw
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
        if truncated and c.reward == "coop_safe":
            # Terminal terms from the same summary the evaluation uses.
            summary = m.summary(c.contact_threshold, self.penetration_limit)
            r_end = c.success_bonus if summary["success"] else 0.0
            if summary["contact_step"] is None:
                r_end -= c.contact_bonus
            terms["terminal"] = r_end
            reward += r_end
            m.rewards[-1] = reward
        for k, v in terms.items():
            m.term_sums[k] = m.term_sums.get(k, 0.0) + v
        info = self._info(reading, stats)
        info["reward_terms"] = terms
        if truncated:
            info["episode_metrics"] = m.summary(c.contact_threshold, self.penetration_limit)
            if self.arm:
                info["episode_metrics"]["final_arm_shift_mm"] = 1e3 * self.arm_shift
                info["episode_metrics"]["final_pinch_offset_mm"] = 1e3 * reading["pinch_offset"]
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
        if c.reward == "coop_safe":
            both_now = bool((self.f_obs > c.contact_threshold).all())
            tol = max(0.1 * self.goal, 0.05)
            r_track = float(r_force) if both_now else 0.0
            r_hold = c.hold_bonus if both_now and abs(self.f_obs.mean() - self.goal) <= tol else 0.0
            r_act2 = -abs(self._arm_raw - self.prev_arm_action) if self.arm else 0.0
            peak = max(stats["max_jaw_force"])
            limit = max(1.5 * self.goal, self.goal + 1.0)
            over = min(max(0.0, peak - limit) / c.safe_force_scale, c.safe_overload_clip)
            r_over = -max(0.0, over - self._max_over)
            self._max_over = max(self._max_over, over)
            r_contact = 0.0
            if self.had_contact.all() and "contact" not in self._once:
                self._once.add("contact")
                r_contact = c.contact_bonus
            r_crush = r_invalid = 0.0
            if any(stats["bottomed"]) and "crush" not in self._once:
                self._once.add("crush")
                r_crush = -c.coop_crush_penalty
            if stats["max_penetration"] > self.penetration_limit and "invalid" not in self._once:
                self._once.add("invalid")
                r_invalid = -c.invalid_penalty
            # Weighted contributions (they sum to the reward).
            terms = {"track": c.alpha_force * r_track, "hold": r_hold,
                     "act": c.alpha_act * float(r_act), "arm_act": c.alpha_act * r_act2,
                     "overload": c.safe_overload_weight * float(r_over), "contact": r_contact,
                     "crush": r_crush, "invalid": r_invalid}
            return float(sum(terms.values())), terms
        if c.reward == "contact_safe":
            peak = max(stats["max_jaw_force"])
            limit = max(1.5 * self.goal, self.goal + 1.0)
            over = min(max(0.0, peak - limit) / c.safe_force_scale, c.safe_overload_clip)
            r_over = -max(0.0, over - self._max_over)
            self._max_over = max(self._max_over, over)
            r_contact = 0.0
            if self.had_contact.all() and not self._contact_rewarded:
                self._contact_rewarded = True
                r_contact = c.contact_bonus
            if self.steps + 1 >= c.max_steps and not self._contact_rewarded:
                r_contact = -c.contact_bonus
            r_crush = 0.0
            if any(stats["bottomed"]) and not self._crush_charged:
                self._crush_charged = True
                r_crush = -c.crush_penalty
            terms.update(overload=float(r_over), contact=r_contact, crush=r_crush)
            reward += c.safe_overload_weight * r_over + r_contact + r_crush
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


class TwoStagePi:
    name = "two_stage_pi"
    """Strong scripted baseline (docs/FORCE_CONTROL.md §10.2).

    Stage 1, no contact: close at ``free_speed``. Stage 2, one jaw touching
    (the moving jaw pushes the object across the unobserved gap): close only
    as far as the touching jaw's force stays below a push limit that starts
    at ``push_ratio * f_goal`` and rises by ``push_ramp`` N/s while only one
    jaw touches (table friction can exceed f_goal; the object must still be
    slid to the fixed jaw), never faster than ``push_speed``. The speed cap
    limits the impact on the fixed jaw.
    Stage 3, both jaws touching: force PI (Rust ``ForcePi`` law) on the mean
    force. Uses the policy's noisy observations and |dq| <= env dq_max.
    """

    def __init__(self, free_speed=0.6, push_speed=0.15, push_ratio=0.8, push_gain=1.0,
                 push_ramp=1.0, kp=0.5, ki=2.0, stiffness=100.0):
        self.free_speed, self.push_speed, self.push_ramp = free_speed, push_speed, push_ramp
        self.push_ratio, self.push_gain = push_ratio, push_gain
        self.gains = (kp, ki, stiffness)

    def reset(self):
        from ._native import ForcePi

        kp, ki, k = self.gains
        # Contact threshold 0: stage 3 starts from this controller's own
        # contact flags, not the PI's.
        self.pi = ForcePi(kp=kp, ki=ki, stiffness=k, closing_speed=0.0, threshold=-1.0)
        self.push_time = 0.0

    def __call__(self, env: ForceGraspEnv) -> float:
        dq_max = env.config.dq_max
        h = env.had_contact
        if h.all():
            d = self.pi.act(list(env.f_obs), env.goal, CONTROL_PERIOD)
        elif h.any():
            f = float(env.f_obs[1] if h[1] else env.f_obs[0])
            ref = self.push_ratio * env.goal + self.push_ramp * self.push_time
            self.push_time += CONTROL_PERIOD
            scale = float(np.clip(self.push_gain * (ref - f) / ref, -1.0, 1.0))
            d = -scale * self.push_speed * CONTROL_PERIOD
        else:
            d = -self.free_speed * CONTROL_PERIOD
        return env.q_obs + float(np.clip(d, -dq_max, dq_max))


class ArmAssistPi(TwoStagePi):
    name = "arm_assist_pi"
    """Separate experiment beyond the gripper-only task (docs §10.5).

    While only the moving jaw touches, instead of pushing the object across
    the gap it keeps that jaw's force near ``light_ratio * f_goal`` (at most
    ``light_max`` N) and shifts the arm along the closing direction at
    ``shift_speed`` m/s, so the fixed jaw comes to the object (the second
    finger of the paper's gripper). Once both jaws touch the shift is held
    and the force PI takes over. Needs teleport positioning.
    """

    def __init__(self, shift_speed=0.006, light_ratio=0.5, light_max=0.3, drive_arm=True, **kw):
        super().__init__(**kw)
        self.shift_speed, self.light_ratio, self.light_max = shift_speed, light_ratio, light_max
        # drive_arm=False: gripper law only (the arm_pi action mode moves the arm).
        self.drive_arm = drive_arm

    def reset(self):
        super().reset()
        self.shift = 0.0

    def __call__(self, env: ForceGraspEnv) -> float:
        h = env.had_contact
        if h[1] and not h[0]:
            if self.drive_arm:
                limit = 0.012
                self.shift = min(limit, self.shift + self.shift_speed * CONTROL_PERIOD)
                env.sim.shift_arm(self.shift)
            ref = min(self.light_ratio * env.goal, self.light_max)
            scale = float(np.clip((ref - float(env.f_obs[1])) / ref, -1.0, 1.0))
            d = -scale * self.push_speed * CONTROL_PERIOD
            return env.q_obs + float(np.clip(d, -env.config.dq_max, env.config.dq_max))
        return super().__call__(env)
