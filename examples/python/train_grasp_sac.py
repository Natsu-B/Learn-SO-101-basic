"""Train the residual Grasp with SAC under randomized, unobserved object physics.

Model A: --history 0 (current observation). Model B: --history K (K past
observation/action frames stacked). Checkpoints: best.zip, latest.zip, final.zip.
"""
import argparse
import json

from robot_sim.grasp_learning import train_grasp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--horizon", type=int, default=75)
    p.add_argument("--arm-scale", type=float, default=0.06)
    p.add_argument("--gripper-scale", type=float, default=0.35)
    p.add_argument("--mass-range", type=float, nargs=2, default=(0.02, 0.1))
    p.add_argument("--friction-range", type=float, nargs=2, default=(0.05, 1.5))
    p.add_argument("--friction-sampling", choices=("uniform", "log"), default="log")
    p.add_argument("--tilt-penalty", type=float, default=0.0,
                   help="reward penalty per rad of cube tilt in the jaws (hand-off quality)")
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=6)
    a = p.parse_args()
    summary = train_grasp(a.output, steps=a.steps, seed=a.seed, history=a.history,
                          horizon=a.horizon, arm_scale=a.arm_scale,
                          gripper_scale=a.gripper_scale, mass_range=a.mass_range,
                          friction_range=a.friction_range,
                          friction_sampling=a.friction_sampling,
                          tilt_penalty=a.tilt_penalty, eval_interval=a.eval_interval,
                          eval_episodes=a.eval_episodes)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
