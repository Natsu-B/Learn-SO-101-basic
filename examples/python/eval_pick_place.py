"""Evaluate full pick-and-place: episode JSON lines on stdout, summary on stderr."""
import argparse
import json
from pathlib import Path
import sys

from robot_sim import PickPlaceSimulator
from robot_sim.pick_place_pipeline import run_pick_place_episode, summarize_pick_place


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--transport", choices=("learned", "scripted"), default="learned")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or args.seed + args.episodes > 2**64:
        parser.error("episodes must be positive and seeds must fit in uint64")
    if args.transport == "learned" and args.model is None:
        parser.error("--model is required for learned transport")

    model = None
    load_env = None
    options = {}
    sim = None
    try:
        if args.transport == "learned":
            import torch
            from stable_baselines3 import SAC
            from robot_sim.transport_env import GoalTransportEnv

            config = json.loads((args.model.parent / "config.json").read_text())
            if config.get("environment_version") != "goal-transport-v1":
                parser.error("--model must be a GoalTransportEnv checkpoint")
            torch.set_num_threads(1)
            # HER checkpoints require an environment on load, even for inference.
            # Reuse its simulator; only the pipeline resets/steps it, without retries.
            load_env = GoalTransportEnv(**config["env"])
            sim = load_env.sim
            model = SAC.load(args.model, env=load_env, device="cpu")
            options = dict(max_transport_steps=load_env.horizon,
                           max_delta=load_env.max_delta, rate_limit=load_env.rate_limit)
        else:
            sim = PickPlaceSimulator()
        outcomes = []
        for seed in range(args.seed, args.seed + args.episodes):
            row = run_pick_place_episode(seed, model=model, transport=args.transport,
                                         sim=sim, **options)
            row["type"] = "episode"
            outcomes.append(row)
            print(json.dumps(row), flush=True)
        summary = summarize_pick_place(outcomes, transport=args.transport)
        if model is not None:
            summary.update(model=str(args.model), training_steps=model.num_timesteps)
        print(json.dumps(summary), file=sys.stderr, flush=True)
    finally:
        if load_env is not None:
            load_env.close()
        elif sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
