"""Train SO-101 Transport; resume latest.zip/final.zip with their replay and config."""
import argparse
import json
from pathlib import Path

from robot_sim.transport_learning import train_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="default: resume directory or runs/so101-transport/seed-<seed>")
    parser.add_argument("--steps", type=int, default=250000, help="additional steps when resuming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="restores saved environment/SAC settings, weights and replay")
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--max-delta", type=float, default=.05)
    parser.add_argument("--rate-limit", type=float, default=.1)
    parser.add_argument("--distance-scale", type=float, default=.1)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000000)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--train-freq", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1000)
    args = parser.parse_args()
    options = vars(args)
    output = options.pop("output") or (args.resume.parent if args.resume else
                                       Path("runs/so101-transport") / f"seed-{args.seed}")
    print(json.dumps(train_transport(output, **options)), flush=True)


if __name__ == "__main__":
    main()
