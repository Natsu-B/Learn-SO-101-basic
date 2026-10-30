"""Native binding contracts and physical scripted skill outcomes."""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from robot_sim import CONTROL_PERIOD, GRIPPER_CLOSED, GRIPPER_OPEN, PickPlaceSimulator
from robot_sim import _native


@pytest.fixture
def sim():
    simulator = PickPlaceSimulator()
    try:
        yield simulator
    finally:
        simulator.close()


def test_state_and_metadata(sim):
    assert _native.PickPlaceSimulator is PickPlaceSimulator
    sim.reset(42)
    state = sim.state()
    shapes = {
        "qpos": 6, "qvel": 6, "cube_pos": 3, "cube_quat": 4,
        "cube_vel": 6, "target": 3, "grasp_center": 3,
    }
    assert set(state) == set(shapes) | {"time", "gripper", "contacts"}
    for key, size in shapes.items():
        assert isinstance(state[key], list)
        assert len(state[key]) == size
        assert all(math.isfinite(v) for v in state[key])
    assert state["time"] == 0
    assert state["gripper"] == state["qpos"][5]
    assert set(state["contacts"]) == {
        "fixed_jaw", "moving_jaw", "table", "other_robot", "pinched",
    }
    assert all(isinstance(v, bool) for v in state["contacts"].values())
    assert sim.joint_names() == _native.ACTUATOR_NAMES
    assert len(sim.action_bounds()) == 6
    assert all(lo < hi for lo, hi in sim.action_bounds())
    assert sim.timestep() == 0.005
    assert CONTROL_PERIOD == 1 / 30
    assert sim.action_bounds()[5][0] <= GRIPPER_CLOSED < GRIPPER_OPEN <= sim.action_bounds()[5][1]
    assert set(sim.predicates()) == {"lifted", "transported", "placed", "dropped"}
    assert all(isinstance(v, bool) for v in sim.predicates().values())
    criteria = sim.criteria()
    assert set(criteria) == {
        "lift_height", "hold_seconds", "transport_tolerance", "place_tolerance",
        "rest_speed", "cube_half", "carry_height",
    }
    assert all(v > 0 for v in criteria.values())
    assert state["target"][2] == criteria["cube_half"]
    state["qpos"][0] = 999
    assert sim.state()["qpos"][0] != 999


def test_reset_reproducibility(sim):
    sim.reset(7)
    initial = sim.state()
    expected = [sim.step(initial["qpos"]) for _ in range(5)]
    sim.reset(8)
    assert sim.state()["cube_pos"] != initial["cube_pos"]
    sim.reset(7)
    assert sim.state() == initial
    assert [sim.step(initial["qpos"]) for _ in range(5)] == expected


def test_step_uses_control_period_and_returns_snapshot(sim):
    sim.reset(0)
    state = sim.step(sim.state()["qpos"])
    assert state == sim.state()
    assert CONTROL_PERIOD <= state["time"] < CONTROL_PERIOD + sim.timestep()
    state = sim.step(state["qpos"], period=0.05)
    assert state == sim.state()
    assert CONTROL_PERIOD + 0.05 <= state["time"] < CONTROL_PERIOD + 0.05 + sim.timestep()


@pytest.mark.parametrize("command", [
    [0.] * 5, [0.] * 7, [float("nan")] * 6, [float("inf")] * 6, [100.] * 6,
])
def test_invalid_command_does_not_advance(sim, command):
    sim.reset(0)
    initial = sim.state()
    with pytest.raises(ValueError):
        sim.step(command)
    assert sim.state() == initial


@pytest.mark.parametrize("period", [0, -1, float("nan"), float("inf")])
def test_invalid_period_does_not_advance(sim, period):
    initial = sim.state()
    with pytest.raises(ValueError):
        sim.step(initial["qpos"], period)
    assert sim.state() == initial


def test_bounds_are_exactly_enforced(sim):
    for i, (lo, hi) in enumerate(sim.action_bounds()):
        for bound, direction in [(lo, -math.inf), (hi, math.inf)]:
            sim.reset(0)
            command = sim.state()["qpos"]
            command[i] = math.nextafter(bound, direction)
            with pytest.raises(ValueError):
                sim.step(command)
            command[i] = bound
            sim.step(command)


@pytest.mark.parametrize("seed", range(5))
def test_scripted_episode_places_cube(sim, seed):
    report = sim.run_episode(seed)
    assert report["seed"] == seed
    assert report["success"]
    assert [r["name"] for r in report["skills"]] == ["grasp", "transport", "place"]
    assert all(r["success"] and r["status"] == "success" and not r["learned"]
               and r["reason"] is None for r in report["skills"])
    assert report["control_steps"] == sum(r["steps"] for r in report["skills"])
    assert report["cube_final"] == sim.state()["cube_pos"]
    assert report["target"] == sim.state()["target"]
    assert report["final_xy_error"] == pytest.approx(math.dist(report["cube_final"][:2], report["target"][:2]))
    assert report["final_xy_error"] < sim.criteria()["place_tolerance"]
    assert sim.predicates()["placed"]


def test_grasp_lifts_and_pinches(sim):
    sim.reset(11)
    report = sim.run_skill("grasp")
    assert report["name"] == "grasp"
    assert report["success"] and report["status"] == "success"
    assert report["steps"] > 0 and not report["learned"]
    assert sim.predicates()["lifted"]
    assert sim.state()["contacts"]["pinched"]
    assert not sim.state()["contacts"]["table"]


def test_stepwise_skills_match_batch_and_episode_exactly(sim):
    sim.reset(25)
    expected = []
    for name in ["grasp", "transport", "place"]:
        report = sim.run_skill(name)
        assert report["success"]
        expected.append((report, sim.state(), sim.predicates()))
    sim.reset(25)
    for report, state, predicates in expected:
        sim.start_skill(report["name"])
        for step in range(report["steps"]):
            status = sim.skill_step()
            assert status == ("success" if step == report["steps"] - 1 else "running")
        assert sim.state() == state
        assert sim.predicates() == predicates
        assert sim.skill_step() == "success"
        assert sim.state() == state  # Terminal calls are idempotent.
    episode = sim.run_episode(25)
    assert episode["skills"] == [r for r, _, _ in expected]
    assert sim.state() == expected[-1][1]


@pytest.mark.parametrize("max_steps", [0, 1, 3])
def test_skill_step_budget_reports_failure(sim, max_steps):
    sim.reset(0)
    report = sim.run_skill("grasp", max_steps=max_steps)
    assert report["status"] == "failure" and not report["success"]
    assert report["steps"] == max_steps
    assert "timeout" in report["reason"]
    state = sim.state()
    assert sim.skill_step() == "failure"
    assert sim.state() == state


def test_skill_lifecycle_and_invalid_name(sim):
    with pytest.raises(ValueError, match="start_skill"):
        sim.skill_step()
    for method in [sim.run_skill, sim.start_skill]:
        with pytest.raises(ValueError, match="skill must be"):
            method("unknown")
    sim.start_skill("grasp")
    sim.reset(0)
    with pytest.raises(ValueError, match="start_skill"):
        sim.skill_step()
    sim.start_skill("grasp")
    sim.step(sim.state()["qpos"])
    with pytest.raises(ValueError, match="start_skill"):
        sim.skill_step()
    sim.close()
    sim.close()
    with pytest.raises(RuntimeError, match="closed"):
        sim.state()


def test_constructor_resolution(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="mutually exclusive"):
        PickPlaceSimulator(tmp_path, robot="so101")
    with pytest.raises(ValueError, match="only so101"):
        PickPlaceSimulator(robot="panda")
    with pytest.raises(FileNotFoundError):
        PickPlaceSimulator(tmp_path)
    directory = os.environ["SO101_MODEL_DIR"]
    monkeypatch.delenv("SO101_MODEL_DIR")
    with pytest.raises(RuntimeError, match="SO101_MODEL_DIR"):
        PickPlaceSimulator()
    explicit = PickPlaceSimulator(model_dir=directory)
    try:
        explicit.reset(0)
        assert len(explicit.state()["qpos"]) == 6
    finally:
        explicit.close()


def test_access_from_another_python_thread(sim):
    sim.reset(0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        report = pool.submit(sim.run_skill, "grasp").result(timeout=30)
    assert report["success"]
    assert sim.predicates()["lifted"]
