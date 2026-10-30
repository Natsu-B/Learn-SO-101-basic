"""Install optional dependencies: uv sync --extra learning --no-install-project."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from robot_sim import TaskConfig
from robot_sim.learning import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["so101", "panda", "ur5e"], default="so101")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--action-mode", choices=["delta", "normalized"], default="delta")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--domain-randomization", nargs="?", const="moderate", default="none", help="none/light/moderate or TOML/JSON path")
    parser.add_argument("--identified-parameters", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / args.robot / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = train(args.robot, args.steps, output, seed=args.seed, rollout_steps=args.rollout_steps,
                   batch_size=args.batch_size, n_epochs=args.epochs, action_mode=args.action_mode,
                   wandb_enabled=args.wandb, wandb_mode=args.wandb_mode,
                   domain_randomization=args.domain_randomization,model_parameters=args.identified_parameters,
                   task_config=TaskConfig(success_bonus=1.0, max_episode_steps=args.episode_steps))
    print({k: v for k, v in result.items() if k != "episodes"})


if __name__ == "__main__":
    main()
