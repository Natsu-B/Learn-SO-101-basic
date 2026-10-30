"""Random normalized delta actions; optional Rust Rerun telemetry."""
import argparse
from robot_sim import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["so101", "panda", "ur5e"], default="so101")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--rerun", action="store_true")
parser.add_argument("--record")
args = parser.parse_args()
env = make_env(args.robot, seed=42)
try:
    if args.rerun or args.record:
        env.unwrapped.sim.start_rerun(record=args.record)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    if args.rerun or args.record:
        env.unwrapped.sim.flush_telemetry()
    print(info)
finally:
    env.close()
