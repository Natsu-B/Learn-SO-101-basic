import json
import math
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
import so101_mujoco
from so101_mujoco import Simulator

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def assert_finite(state):
    assert set(state) == {"position", "velocity"}
    for values in state.values():
        assert len(values) == 6
        assert all(math.isfinite(value) for value in values)


def test_import_and_create():
    assert so101_mujoco.__version__ == "0.1.0"
    sim = Simulator()
    assert sim.actuator_names == NAMES
    assert so101_mujoco.ACTUATOR_NAMES == NAMES
    assert sim.time == 0.0
    assert sim.timestep == 0.005


def test_state():
    sim = Simulator()
    state = sim.state()
    assert state == {"position": [0.0] * 6, "velocity": [0.0] * 6}
    assert_finite(state)
    state["position"][0] = 999
    assert sim.state()["position"][0] == 0.0


def test_step():
    sim = Simulator()
    assert_finite(sim.step([0.2] * 6))
    assert sim.time == sim.timestep
    assert sim.state()["position"] != [0.0] * 6


def test_1000_steps_are_finite():
    sim = Simulator()
    for _ in range(1000):
        command = [0.3 * math.sin(math.tau * 0.25 * sim.time), 0.1, -0.2, 0.1, 0.1, 0.2]
        assert_finite(sim.step(command))
    assert sim.time == pytest.approx(5.0)


def test_reset_and_reproducibility():
    sim = Simulator()
    initial = sim.state()
    commands = [[0.2 * math.sin(i * 0.04), 0.1, -0.2, 0.1, 0.0, 0.3] for i in range(200)]
    expected = [sim.step(c) for c in commands]
    sim.reset()
    assert sim.time == 0.0
    assert sim.state() == initial
    assert [sim.step(c) for c in commands] == expected


@pytest.mark.parametrize("command", [[0.0] * 5, [0.0] * 7, [float("nan")] * 6, [float("inf")] * 6, [100.0] * 6])
def test_invalid_command_is_value_error(command):
    sim = Simulator()
    initial = sim.state()
    with pytest.raises(ValueError):
        sim.step(command)
    assert sim.time == 0.0
    assert sim.state() == initial


def test_missing_model_is_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Simulator(tmp_path)


def test_missing_environment_is_runtime_error(monkeypatch):
    monkeypatch.delenv("SO101_MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="SO101_MODEL_DIR"):
        Simulator()


def test_calls_from_another_python_thread():
    sim = Simulator()
    with ThreadPoolExecutor(max_workers=1) as pool:
        state = pool.submit(sim.step, [0.0] * 6).result()
    assert_finite(state)
    assert sim.time == sim.timestep


@pytest.mark.parametrize("sine", [False, True])
def test_python_result_matches_rust_cli_exactly(sine):
    root = Path(__file__).resolve().parents[2]
    args = ["cargo", "run", "--quiet", "-p", "so101-cli", "--", "--json"]
    if sine:
        args.append("--sine")
    rust = json.loads(subprocess.check_output(args, cwd=root, text=True))
    sim = Simulator()
    for _ in range(1000):
        command = [0.0] * 6
        if sine:
            command[0] = 0.3 * math.sin(math.tau * 0.25 * sim.time)
        state = sim.step(command)
    assert sim.time == rust["time"]
    assert state["position"] == rust["position"]
    assert state["velocity"] == rust["velocity"]


def test_independent_simulators_on_two_python_threads():
    from threading import Barrier
    barrier = Barrier(2)
    def run(robot):
        sim = Simulator(robot=robot)
        command = sim.state()["position"]
        barrier.wait(timeout=10)
        for _ in range(100):
            state = sim.step_control(command, 1 / 30)
        assert all(math.isfinite(v) for values in state.values() for v in values)
        time = sim.time
        sim.close()
        return time
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, robot) for robot in ["panda", "ur5e"]]
        assert all(f.result(timeout=30) > 3 for f in futures)


def test_ambiguous_model_selection_rejected():
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        Simulator(robot="panda", model_dir="/unused")


def test_task_transaction_matches_snapshot():
    sim = Simulator(robot="panda")
    sim.reset_reach(42)
    result = sim.step_task(sim.state()["position"], 1/30)
    assert result == sim.task_state()
    assert result["time"] == sim.time
    assert result["reward"] == sim.reward
    assert result["episode_step"] == 1
    sim.close()


def test_gym_step_makes_one_worker_request():
    from robot_sim import make_env
    import numpy as np
    env=make_env('so101');env.reset(seed=42)
    native=env.unwrapped.sim
    class StepOnly:
        calls=0
        def step_task(self,*args):
            self.calls+=1
            return native.step_task(*args)
        def __getattr__(self,name):
            raise AssertionError(f'extra worker request: {name}')
    proxy=StepOnly();env.unwrapped.sim=proxy
    try:
        env.step(np.zeros(env.action_space.shape))
        assert proxy.calls==1
    finally:
        env.unwrapped.sim=native
        env.close()
