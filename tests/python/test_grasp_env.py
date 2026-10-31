"""Residual Grasp environment: hidden physics, residual semantics, history, parity."""
import numpy as np
import pytest

from robot_sim import PickPlaceSimulator
from robot_sim.grasp_env import (ACTION_DIM, OBS_DIM, GoalGraspEnv, ObservationHistory,
                                 approach_to_pre_close, residual_command)
from robot_sim.pick_place_pipeline import run_pick_place_episode


@pytest.fixture
def env():
    e = GoalGraspEnv()
    try:
        yield e
    finally:
        e.close()


def run_zero(env, seed):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, np.float32))
        done = terminated or truncated
    return info, terminated


def test_zero_residual_reproduces_scripted_grasp(env):
    for seed in (0, 1):
        info, terminated = run_zero(env, seed)
        assert info["is_success"] and terminated and not info["dropped"]
        assert info["episode_step"] < env.horizon


def test_reset_starts_open_around_cube_with_randomized_hidden_physics(env):
    obs, info = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    state = env.sim.state()
    assert env.sim.skill_phase() == "close"
    assert not state["contacts"]["pinched"] and state["contacts"]["table"]
    p = env.sim.object_physics()
    assert 0.02 <= p["mass"] <= 0.05 and 0.8 <= p["friction"] <= 1.5
    # Neither the true mass nor friction appears anywhere in the observation.
    assert not np.isclose(obs, p["mass"]).any() and not np.isclose(obs, p["friction"]).any()


def test_residual_is_bounded_and_clipped():
    bounds = np.asarray([[-1., 1.]] * 6)
    base = np.zeros(6)
    out = residual_command(base, np.ones(6), bounds, arm_scale=0.03, gripper_scale=0.35)
    assert np.allclose(out, [0.03] * 5 + [0.35])
    out = residual_command(np.full(6, 0.99), np.ones(6), bounds, arm_scale=0.03,
                           gripper_scale=0.35)
    assert np.allclose(out, 1.0)
    with pytest.raises(ValueError):
        residual_command(base, np.full(6, 1.5), bounds, arm_scale=0.03, gripper_scale=0.35)


def test_history_stacks_previous_observations_and_actions():
    h = ObservationHistory(2)
    o0, o1, o2 = (np.full(OBS_DIM, v, np.float32) for v in (0, 1, 2))
    assert h.reset(o0).shape == (h.size,) == (OBS_DIM + 2 * (OBS_DIM + ACTION_DIM),)
    a = np.full(ACTION_DIM, 0.5, np.float32)
    h.push(a, o1)
    x = h.push(-a, o2)
    assert np.all(x[:OBS_DIM] == 2)
    newest = x[OBS_DIM + (OBS_DIM + ACTION_DIM):]
    assert np.all(newest[:OBS_DIM] == 1) and np.all(newest[OBS_DIM:] == -0.5)
    assert ObservationHistory(0).size == OBS_DIM


def test_history_env_space_matches_observation():
    e = GoalGraspEnv(history=3)
    try:
        obs, _ = e.reset(seed=0)
        assert e.observation_space.shape == obs.shape == (ObservationHistory(3).size,)
    finally:
        e.close()


class ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros(ACTION_DIM, np.float32), None


def test_pipeline_learned_grasp_matches_env_and_completes_pick_place(env):
    config = dict(history=0, horizon=env.horizon, arm_scale=env.arm_scale,
                  gripper_scale=env.gripper_scale)
    env.sim.set_object_randomization(mass=(0.03, 0.03), friction=(1.0, 1.0))
    info, _ = run_zero(env, 5)
    sim = PickPlaceSimulator()
    try:
        sim.set_object_randomization(mass=(0.03, 0.03), friction=(1.0, 1.0))
        row = run_pick_place_episode(5, transport="scripted", sim=sim, grasp="learned",
                                     grasp_model=ZeroPolicy(), grasp_config=config)
    finally:
        sim.close()
    grasp = row["skills"][0]
    assert grasp["learned"] and grasp["success"] == info["is_success"]
    assert grasp["steps"] - grasp["approach_steps"] == info["episode_step"]
    assert row["success"], row["skills"]


def test_scripted_approach_stops_at_close_phase():
    sim = PickPlaceSimulator()
    try:
        sim.reset(0)
        assert approach_to_pre_close(sim)
        assert sim.skill_phase() == "close"
    finally:
        sim.close()


def test_cube_tilt_is_zero_upright_and_penalty_defaults_off():
    from robot_sim.grasp_env import cube_tilt
    assert cube_tilt([1, 0, 0, 0]) == 0
    assert np.isclose(cube_tilt([np.cos(0.25), np.sin(0.25), 0, 0]), 0.5)
    e = GoalGraspEnv()
    try:
        assert e.tilt_penalty == 0.0
    finally:
        e.close()
