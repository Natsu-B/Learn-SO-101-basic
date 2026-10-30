"""All telemetry is logged by Rust; --record saves without opening a GUI."""
import argparse
from robot_sim import RobotEnv

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="so101", choices=["so101", "panda", "ur5e"])
parser.add_argument("--record")
args = parser.parse_args()
env = RobotEnv(robot=args.robot)
obs, info = env.reset(seed=42)
env.sim.start_rerun(record=args.record)
action = obs["joint_position"].copy()
for _ in range(300):
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.sim.flush_telemetry()
print(info)
env.close()
