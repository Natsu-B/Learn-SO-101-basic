"""Evaluate learned Transport only: episode JSON lines and an aggregate summary."""
import argparse
import json
from pathlib import Path

from robot_sim.transport_learning import evaluate_transport
from robot_sim.transport_env import GoalTransportEnv


def main():
    from stable_baselines3 import SAC
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000000)
    args = parser.parse_args()
    config = json.loads((args.model.parent / "config.json").read_text())
    if config.get("environment_version") != "goal-transport-v1":
        parser.error("--model must be a GoalTransportEnv checkpoint")
    torch.set_num_threads(1)
    env = GoalTransportEnv(**config["env"], terminate_on_success=True, terminate_on_drop=True)
    try:
        # HER requires an env even for inference when restoring the SAC model.
        model = SAC.load(args.model, env=env, device="cpu")
        summary = evaluate_transport(model, config, episodes=args.episodes, seed=args.seed, env=env,
                                     emit=lambda row: print(json.dumps(row), flush=True))
        summary.pop("outcomes")
        summary.update(model=str(args.model), training_steps=model.num_timesteps)
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
