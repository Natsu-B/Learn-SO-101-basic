import subprocess
from robot_sim import Simulator


def test_native_telemetry_is_optional_and_headless(tmp_path):
    sim = Simulator()
    sim.reset_reach(42)
    path = tmp_path / "reach.rrd"
    try:
        sim.start_rerun(record=path)
    except RuntimeError as error:
        assert "sim-python/rerun" in str(error)
        assert sim.time == 0
        return
    command = sim.state()["position"]
    for _ in range(10): sim.step_control(command, 1/30)
    sim.reset_reach(43)
    for _ in range(3): sim.step_control(command, 1/30)
    sim.flush_telemetry()
    sim.close()
    del sim
    assert path.stat().st_size > 100
    text = subprocess.check_output(["rerun", "rrd", "print", "-vv", str(path)], text=True)
    for entity in ["simulation_time", "episode_step", "global_step", "metrics/episode","simulation/time", "metrics/reward", "metrics/distance", "metrics/success", "robot/end_effector", "task/target", "robot/joints/shoulder_pan", "robot/joint_targets/shoulder_pan", "control/requested/shoulder_pan", "control/applied/shoulder_pan", "state/true/shoulder_pan", "state/observed/shoulder_pan"]:
        assert entity in text
