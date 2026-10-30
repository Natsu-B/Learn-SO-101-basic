"""Transport's native physics, GoalEnv/HER contract and episode semantics."""
import gymnasium as gym
import numpy as np
import pytest

from robot_sim import CONTROL_PERIOD, GRIPPER_CLOSED, GoalTransportEnv


@pytest.fixture
def env():
    instance = GoalTransportEnv(horizon=6)
    try:
        yield instance
    finally:
        instance.close()


def test_spaces_observation_and_real_grasp(env):
    obs, info = env.reset(seed=42)
    s = env.sim.state()
    assert env.action_space.shape == (5,)
    assert obs["observation"].shape == (23,)
    assert obs["achieved_goal"].shape == obs["desired_goal"].shape == (3,)
    assert env.observation_space.contains(obs)
    assert all(v.dtype == np.float32 for v in obs.values())
    expected = (s["qpos"][:5] + s["qvel"][:5] + s["grasp_center"] + s["cube_pos"]
                + s["cube_vel"][:3] + [s["gripper"]]
                + (np.asarray(s["cube_pos"]) - s["grasp_center"]).tolist())
    np.testing.assert_allclose(obs["observation"], expected, atol=1e-8)
    np.testing.assert_array_equal(obs["achieved_goal"], np.float32(s["cube_pos"]))
    np.testing.assert_array_equal(obs["desired_goal"], np.float32([
        *s["target"][:2], env.criteria["cube_half"] + env.criteria["carry_height"]]))
    assert info["seed"] == info["requested_seed"] == 42
    assert info["grasp_report"]["success"] and not info["grasp_report"]["learned"]
    assert env.sim.predicates()["lifted"] and s["contacts"]["pinched"]
    assert not s["contacts"]["table"]


def test_reset_reproducibility_including_unseeded_sequence(env):
    a, info_a = env.reset(seed=7)
    env.step(np.ones(5, np.float32))
    next_a, next_info_a = env.reset()
    b, info_b = env.reset(seed=7)
    next_b, next_info_b = env.reset()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(next_a[key], next_b[key])
    assert info_a == info_b
    assert next_info_a == next_info_b
    assert next_info_a["seed"] != 7


def test_reward_vectorization_relabel_and_step_consistency(env):
    obs, _ = env.reset(seed=0)
    samples = []
    goal = obs["desired_goal"].copy()
    for _ in range(env.horizon):
        obs, reward, terminated, truncated, info = env.step(np.zeros(5, np.float32))
        samples.append((obs["achieved_goal"], reward))
        assert info["is_success"] == env.sim.predicates()["transported"]
        assert info["dropped"] == env.sim.predicates()["dropped"]
        assert reward == env.compute_reward(obs["achieved_goal"], goal, info)
        np.testing.assert_array_equal(obs["desired_goal"], goal)
        assert not terminated
    assert truncated
    achieved = np.stack([a for a, _ in samples])
    rewards = env.compute_reward(achieved, goal, [{}] * len(samples))
    assert rewards.dtype == np.float32
    np.testing.assert_array_equal(rewards, np.float32([r for _, r in samples]))
    best = env.compute_reward(achieved, achieved, None)
    np.testing.assert_array_equal(best, np.zeros(len(samples), np.float32))
    assert np.all(rewards <= best)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(5))


def test_gripper_closed_and_command_slew_bounds(env):
    env.max_delta = .2
    env.rate_limit = .01
    env.reset(seed=25)
    for _ in range(env.horizon):
        previous = env.previous_command.copy()
        q = np.asarray(env.last_result["qpos"])
        before_time = env.last_result["time"]
        env.step(np.ones(5, np.float32))
        expected = np.clip(q[:5] + env.max_delta, previous[:5] - env.rate_limit,
                           previous[:5] + env.rate_limit)
        expected = np.clip(expected, env.bounds[:5, 0], env.bounds[:5, 1])
        np.testing.assert_array_equal(env.previous_command[:5], expected)
        assert env.previous_command[5] == GRIPPER_CLOSED
        assert np.all(env.previous_command >= env.bounds[:, 0])
        assert np.all(env.previous_command <= env.bounds[:, 1])
        dt = env.last_result["time"] - before_time
        # 30 Hz on a 200 Hz physics clock alternates 6/7 substeps.
        assert abs(dt - CONTROL_PERIOD) < env.sim.timestep() + 1e-10


def test_short_random_rollout(env):
    env.action_space.seed(19)
    env.reset(seed=19)
    for _ in range(env.horizon):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated
    assert truncated


@pytest.mark.parametrize("action", [[0] * 4, [0] * 6, [np.nan] * 5, [1.01] * 5, ["x"] * 5])
def test_invalid_action_leaves_physics_unchanged(env, action):
    env.reset(seed=0)
    state = env.sim.state()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.sim.state() == state


class _Proxy:
    def __init__(self, sim):
        self.sim = sim

    def __getattr__(self, name):
        return getattr(self.sim, name)


def test_grasp_retry_seed_logging_and_bound(env, caplog):
    class FailGrasp(_Proxy):
        fail = True

        def run_skill(self, name):
            if self.fail:
                self.fail = False
                return dict(success=False, reason="injected grasp failure")
            return self.sim.run_skill(name)

    env.sim = FailGrasp(env.sim)
    obs, info = env.reset(seed=8)
    assert info["requested_seed"] == 8 and info["seed"] == 9
    assert info["grasp_retries"] == 1
    assert env.grasp_failures_total == 1 and env.grasp_attempts_total == 2
    assert "seed=8" in caplog.text and "injected grasp failure" in caplog.text
    reference, _ = env.reset(seed=9)
    for key in obs:
        np.testing.assert_array_equal(obs[key], reference[key])
    env.max_grasp_retries = 0
    env.sim.fail = True
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        env.reset(seed=8)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(5))


@pytest.mark.parametrize("success,drop", [(True, False), (False, True)])
def test_training_fixed_horizon_eval_native_termination(env, success, drop):
    class Predicates(_Proxy):
        def predicates(self):
            return dict(transported=success, dropped=drop)

    env.sim = Predicates(env.sim)
    env.reset(seed=0)
    for _ in range(env.horizon):
        _, _, terminated, truncated, info = env.step(np.zeros(5))
        assert not terminated  # Neither physical condition changes training horizon.
        assert info["is_success"] == success and info["dropped"] == drop
    assert truncated
    env.terminate_on_success = env.terminate_on_drop = True
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(5))
    assert terminated and not truncated


def test_sb3_her_replay_optimization_and_eval_cli(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import json
    from pathlib import Path
    import subprocess
    import sys
    import torch
    from stable_baselines3.common.env_checker import check_env

    torch.set_num_threads(1)
    env = GoalTransportEnv(horizon=4)
    try:
        check_env(env, warn=True)
        model = sb3.SAC("MultiInputPolicy", env, replay_buffer_class=sb3.HerReplayBuffer,
                        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
                        learning_starts=4, train_freq=4, gradient_steps=1, batch_size=8,
                        buffer_size=64, policy_kwargs=dict(net_arch=[16, 16]), device="cpu", seed=42)
        model.learn(24)
        assert model._n_updates > 0
        batch = model.replay_buffer.sample(8)
        assert batch.rewards.shape == (8, 1)
        assert np.isfinite(batch.rewards.cpu().numpy()).all()
        model.save(tmp_path / "final")
        (tmp_path / "config.json").write_text(json.dumps(dict(
            environment_version="goal-transport-v1", env=dict(horizon=4))))
        script = Path(__file__).resolve().parents[2] / "examples/python/eval_transport.py"
        result = subprocess.run([sys.executable, str(script), "--model", str(tmp_path / "final.zip"),
                                 "--episodes", "1", "--seed", "7"],
                                check=True, capture_output=True, text=True, timeout=30)
        episode, summary = [json.loads(line) for line in result.stdout.splitlines()]
        assert episode["type"] == "episode" and episode["seed"] == 7
        assert [skill["learned"] for skill in episode["skills"]] == [False, True]
        assert summary["type"] == "summary" and summary["episodes"] == 1
        assert summary["training_steps"] == 24
    finally:
        env.close()
