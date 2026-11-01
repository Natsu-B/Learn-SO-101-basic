"""Grasp-force Gymnasium environment and PPO helpers (docs/FORCE_CONTROL.md §6)."""
import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from robot_sim import GRIPPER_CLOSED, ForceGraspSimulator, ForcePi
from robot_sim.force_grasp_env import FRAME_SIZE, ForceGraspConfig, ForceGraspEnv
from robot_sim.force_grasp_learning import aggregate, evaluate, run_episode


@pytest.fixture
def env():
    e = ForceGraspEnv(ForceGraspConfig(max_steps=40))
    e.set_progress(1.0)
    try:
        yield e
    finally:
        e.close()


def test_spaces_and_api():
    # Without the bias full closing reaches both jaws within 40 periods
    # (with it, phi = 0.1 slows the push across the gap ten-fold).
    env = ForceGraspEnv(ForceGraspConfig(max_steps=40, inductive_bias=False))
    env.set_progress(1.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (FRAME_SIZE * 3,)
    assert env.action_space.shape == (1,)
    assert not info["precontact"], "pre-closing to the initial opening never touches"
    lo, hi = env.sim.goal_range(env.config.goal_min, env.config.goal_cap)
    assert lo <= env.goal <= hi
    for k in range(40):
        obs, reward, terminated, truncated, info = env.step([-1.0])
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated
        assert truncated == (k == 39)
    m = info["episode_metrics"]
    assert m["steps"] == 40 and m["contact_step"] is not None
    assert m["peak_force_physics"] >= max(m["final_force"]) - 1e-9
    env.close()


def test_observation_hides_object_properties(env):
    """Same seed, different stiffness/mass/friction: the first observation is
    identical before contact (no ground truth in the observation)."""
    obs_a, _ = env.reset(options={"episode_seed": 5})
    o = env.sim.object()
    env.sim.reset(5, {**o, "stiffness": 300.0, "mass": 0.09, "friction": 0.4})
    env.sim.step(env.config.initial_opening)
    other = ForceGraspEnv(ForceGraspConfig(max_steps=40))
    other.set_progress(1.0)
    try:
        obs_b, _ = other.reset(options={"episode_seed": 5})
        assert np.allclose(obs_a, obs_b)
        frame = obs_a[-FRAME_SIZE:]
        assert frame[6] == pytest.approx(other.goal / other.config.force_scale)
    finally:
        other.close()


def test_same_episode_seed_same_episode(env):
    a = run_episode(env, 8_000_003, "pi")
    b = run_episode(env, 8_000_003, "pi")
    assert a["goal"] == b["goal"] and a["mean_abs_error"] == b["mean_abs_error"]
    assert a["peak_force_physics"] == b["peak_force_physics"]


def test_inductive_bias_cases(env):
    env.reset(options={"episode_seed": 1})
    env.had_contact[:] = [False, False]
    assert env.bias() == 1.0
    env.had_contact[:] = [False, True]
    assert env.bias() == env.config.bias_single_moving
    env.had_contact[:] = [True, False]
    assert env.bias() == 1.0
    env.had_contact[:] = [True, True]
    env.f_obs = np.array([env.goal, env.goal])
    assert env.bias() == 1.0
    env.f_obs = np.array([0.0, 0.0])
    assert env.bias() == 0.9


def test_baselines_through_the_same_bookkeeping(env):
    fixed = run_episode(env, 8_000_001, "fixed")
    pi = run_episode(env, 8_000_001, "pi")
    assert fixed["goal"] == pi["goal"]
    assert fixed["crushed"], "the stall-torque close command crushes the object"
    assert pi["mean_abs_error"] < 0.2 * fixed["mean_abs_error"]
    agg = aggregate([fixed, pi])
    assert agg["episodes"] == 2 and 0 <= agg["valid_episodes"] <= 2


def test_physics_step_peaks_are_at_least_the_period_end(env):
    env.reset(options={"episode_seed": 2})
    for _ in range(25):
        _, _, _, _, info = env.step_command(GRIPPER_CLOSED)
        s = info["stats"]
        assert s["physics_steps"] in (6, 7), "30 Hz control over 5 ms physics"
        assert max(s["max_jaw_force"]) >= max(info["jaw_force"]) - 1e-12


def test_config_round_trip_and_reward_terms():
    c = ForceGraspConfig(reward="paper_peak", alpha_obj=(0.0, 0.5))
    assert ForceGraspConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    with pytest.raises(ValueError):
        ForceGraspConfig(reward="nope").validate()
    e = ForceGraspEnv(c)
    try:
        e.set_progress(1.0)
        e.reset(options={"episode_seed": 4})
        _, _, _, _, info = e.step([0.0])
        assert set(info["reward_terms"]) == {"force", "obj", "act", "peak"}
        assert e.alpha_obj == 0.5
    finally:
        e.close()


def test_kappa_override_sets_stiffness(env):
    env.reset(options={"episode_seed": 6, "kappa": 0.0})
    assert env.object["stiffness"] == pytest.approx(5000.0)
    env.reset(options={"episode_seed": 6, "kappa": 1.0})
    assert env.object["stiffness"] == pytest.approx(200.0)


def test_force_pi_binding_and_simulator_limits():
    pi = ForcePi()
    assert pi.gains == (0.5, 2.0, 100.0, 0.6, 0.05)
    assert pi.act([0.0, 0.0], 1.0, 1 / 30) == pytest.approx(-0.02)
    assert ForceGraspSimulator.penetration_limit == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        ForceGraspSimulator(positioning="nope")


def test_evaluate_runs_ppo_interface(env):
    class Zero:
        def predict(self, obs, deterministic=True):
            return np.zeros(1, dtype=np.float32), None

    rows = evaluate(env, [8_000_000], "ppo", Zero())
    assert rows[0]["contact_step"] is None, "a zero policy never closes"
