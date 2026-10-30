"""Evaluate PPO or Goal SAC/HER against random actions on fixed held-out seeds."""
import argparse
import json
from pathlib import Path
from robot_sim.learning import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["so101", "panda", "ur5e"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000000)
    parser.add_argument("--threshold", type=float, default=.005)
    parser.add_argument("--hold-seconds", type=float, default=0., help="require continuous in-range control samples while the policy keeps acting")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--record", type=Path, help="save trained episodes to .rrd without a Viewer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domain-randomization", help="override saved profile: none/light/moderate or file")
    args = parser.parse_args()
    directory = args.model if args.model.is_dir() else args.model.parent
    config = json.loads((directory / "config.json").read_text())
    options = dict(robot=args.robot, episodes=args.episodes, seed=args.seed,
                   rerun=args.rerun, record=args.record, domain_randomization=args.domain_randomization)
    if config.get("goal_environment"):
        from robot_sim.precision import evaluate_goal
        result = evaluate_goal(args.model, threshold=args.threshold, hold_seconds=args.hold_seconds, **options)
    else:
        if args.hold_seconds:
            parser.error("--hold-seconds currently requires a Goal SAC/HER checkpoint")
        result = evaluate(args.model, **options)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "outcomes"} if isinstance(v, dict) else v
                      for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
