"""Train the SO-101 grasp-force policy with PPO (docs/FORCE_CONTROL.md §6–7).

Refuses a non-empty --output (never overwrites earlier runs). Saves
config.json, progress.csv (SB3 logger), selection.jsonl (checkpoint
selection on seeds 7,000,000+), checkpoints/step_N.zip, best.zip, final.zip.
"""
import argparse
import json

from robot_sim.force_grasp_env import ForceGraspConfig
from robot_sim.force_grasp_learning import train_force_grasp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--curriculum-fraction", type=float, default=1 / 3)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--checkpoint-freq", type=int, default=10_000)
    p.add_argument("--reward", choices=("paper", "paper_peak", "contact_safe"), default="paper")
    p.add_argument("--no-bias", action="store_true", help="disable the inductive bias")
    p.add_argument("--peak-weight", type=float, default=1.0)
    p.add_argument("--peak-margin", type=float, default=0.5)
    p.add_argument("--peak-mode", choices=("step", "max"), default="step")
    p.add_argument("--peak-weight-start", type=float,
                   help="anneal the peak weight from this value with the curriculum")
    p.add_argument("--alpha-obj", type=float, nargs=2, default=(0.0, 1.0),
                   help="object-movement penalty weight at curriculum start/end (paper 0 -> 1; "
                        "public scripts 0 0)")
    p.add_argument("--bias-single-moving", type=float, default=0.1)
    p.add_argument("--resume", help="model zip to continue from (written to a new --output)")
    a = p.parse_args()
    config = ForceGraspConfig(reward=a.reward, inductive_bias=not a.no_bias,
                              peak_weight=a.peak_weight, peak_margin=a.peak_margin,
                              peak_mode=a.peak_mode, peak_weight_start=a.peak_weight_start,
                              alpha_obj=tuple(a.alpha_obj),
                              bias_single_moving=a.bias_single_moving)
    summary = train_force_grasp(
        a.output, steps=a.steps, seed=a.seed, n_envs=a.n_envs, n_steps=a.n_steps,
        batch_size=a.batch_size, learning_rate=a.learning_rate,
        curriculum_fraction=a.curriculum_fraction, eval_freq=a.eval_freq,
        eval_episodes=a.eval_episodes, checkpoint_freq=a.checkpoint_freq, config=config,
        resume=a.resume)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
