"""Full pipeline physics, shared policy inputs and deterministic inference wiring."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robot_sim import GoalTransportEnv, PickPlaceSimulator
from robot_sim.pick_place_pipeline import run_pick_place_episode, summarize_pick_place
from robot_sim.transport_env import transport_command, transport_observation


@pytest.mark.parametrize("seed", [0, 1, 25])
def test_scripted_pipeline_matches_rust_episode(seed):
    sim = PickPlaceSimulator()
    frames = []
    try:
        reference = sim.run_episode(seed)
        report = run_pick_place_episode(seed, transport="scripted", sim=sim,
                                        on_step=lambda *frame: frames.append(frame))
        assert {k: report[k] for k in reference} == reference
        assert report["success"] and sim.predicates()["placed"]
        assert not report["dropped"]
        assert [r["learned"] for r in report["skills"]] == [False] * 3
        assert len(frames) == report["control_steps"]
        assert all(not learned for _, _, learned in frames)
        times = np.asarray([state["time"] for state, _, _ in frames])
        assert np.all(np.diff(times) > 0)
        for skill in report["skills"]:
            assert sum(name == skill["name"] for _, name, _ in frames) == skill["steps"]
        assert frames[-1][0] == sim.state()
        # Bulk execution of Grasp/Place must produce the same report as callbacks.
        assert run_pick_place_episode(seed, transport="scripted", sim=sim) == report
    finally:
        sim.close()


def test_skill_report_is_read_only_and_preserves_running_status():
    sim = PickPlaceSimulator()
    try:
        with pytest.raises(ValueError, match="start_skill first"):
            sim.skill_report()
        sim.reset(0)
        sim.start_skill("grasp")
        state = sim.state()
        report = sim.skill_report()
        assert report["status"] == "running" and report["steps"] == 0
        assert sim.state() == state
        while sim.skill_step() == "running":
            pass
        state = sim.state()
        report = sim.skill_report()
        assert report["success"] and report["reason"] is None
        assert report["steps"] > 0 and sim.state() == state
        assert sim.skill_step() == "success" and sim.state() == state
        sim.reset(0)
        failure = sim.run_skill("grasp", max_steps=1)
        assert sim.skill_report() == failure
        assert failure["reason"].startswith("timeout in ")
    finally:
        sim.close()


def test_shared_observation_and_commands_match_env():
    env = GoalTransportEnv(horizon=4, max_delta=.2, rate_limit=.01)
    try:
        obs, _ = env.reset(seed=25)
        for action in (np.zeros(5, np.float32), np.ones(5, np.float32),
                       -np.ones(5, np.float32), np.linspace(-1, 1, 5, dtype=np.float32)):
            shared = transport_observation(env.sim.state(), env.carry_z)
            for key in obs:
                np.testing.assert_array_equal(shared[key], obs[key])
                assert shared[key].dtype == obs[key].dtype
            expected = transport_command(env.last_result, action, env.previous_command,
                                         env.bounds, max_delta=env.max_delta, rate_limit=env.rate_limit)
            obs, *_ = env.step(action)
            np.testing.assert_array_equal(env.previous_command, expected)
    finally:
        env.close()


def test_tiny_sac_pipeline_and_cli(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import torch

    torch.set_num_threads(1)
    env = GoalTransportEnv(horizon=3, max_delta=.02, rate_limit=.01)
    try:
        # Construction only: no learning or dependency on a local checkpoint.
        model = sb3.SAC("MultiInputPolicy", env, seed=42, device="cpu",
                        policy_kwargs=dict(net_arch=[8, 8]), buffer_size=16)
        frames = []

        class RecordingPolicy:
            calls = []

            def predict(self, obs, *, deterministic):
                assert deterministic is True
                self.calls.append(obs)
                return model.predict(obs, deterministic=deterministic)

        policy = RecordingPolicy()
        report = run_pick_place_episode(7, model=policy, max_transport_steps=env.horizon,
                                        max_delta=env.max_delta, rate_limit=env.rate_limit,
                                        on_step=lambda *frame: frames.append(frame))
        assert not report["success"] and not report["dropped"]
        assert [s["name"] for s in report["skills"]] == ["grasp", "transport"]
        assert [s["learned"] for s in report["skills"]] == [False, True]
        transport = report["skills"][1]
        assert transport["reason"] == "timeout" and transport["steps"] == 3
        assert sum(learned for _, _, learned in frames) == 3
        assert len(frames) == report["control_steps"]
        # Replay the same seed through the environment: inputs and every resulting
        # transport state must match the pipeline, including custom action limits.
        obs, _ = env.reset(seed=7)
        transport_frames = [s for s, name, _ in frames if name == "transport"]
        for recorded, state in zip(policy.calls, transport_frames, strict=True):
            for key in obs:
                np.testing.assert_array_equal(recorded[key], obs[key])
            action, _ = model.predict(obs, deterministic=True)
            obs, *_ = env.step(action)
            assert env.sim.state() == state
        summary = summarize_pick_place([report], transport="learned")
        assert summary["learned_skills"] == ["transport"]
        assert summary["skills"]["place"]["attempts"] == 0
        assert summary["skills"]["place"]["success_rate"] == 0
        assert summary["skills"]["place"]["success_rate_when_attempted"] is None

        model.save(tmp_path / "tiny")
        (tmp_path / "config.json").write_text(json.dumps(dict(
            environment_version="goal-transport-v1",
            env=dict(horizon=3, max_delta=.02, rate_limit=.01))))
        script = Path(__file__).resolve().parents[2] / "examples/python/eval_pick_place.py"
        for mode in ("learned", "scripted"):
            result = subprocess.run([sys.executable, str(script), "--model", str(tmp_path / "tiny.zip"),
                                     "--episodes", "1", "--seed", "7", "--transport", mode],
                                    check=True, capture_output=True, text=True, timeout=30)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            assert len(rows) == 1 and rows[0]["type"] == "episode" and rows[0]["seed"] == 7
            aggregate = json.loads(result.stderr)
            assert aggregate["type"] == "summary" and aggregate["episodes"] == 1
            assert aggregate["learned_skills"] == (["transport"] if mode == "learned" else [])
            assert rows[0]["success"] == (mode == "scripted")
            if mode == "learned":
                assert {k: rows[0][k] for k in report} == report
    finally:
        env.close()
