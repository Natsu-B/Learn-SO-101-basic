"""Pick-and-place robustness over fixed object mass/friction conditions.

Grasp is the scripted IK skill, or with --grasp-model the residual SAC policy
(IK approach + learned close/lift). Transport and Place are scripted IK.

Each condition fixes mass and friction (equal-ended ranges) and runs the same
evaluation seeds. Episode JSON lines go to stdout, one summary per condition to
stderr. The true physics is logged; no policy sees it.

Slip is the largest change of the cube position relative to the pinch centre
after Grasp succeeded (during Transport), in metres.
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np

from robot_sim import PickPlaceSimulator
from robot_sim.pick_place_pipeline import run_pick_place_episode

IN_RANGE = dict(mass=[0.02, 0.035, 0.05], friction=[0.8, 1.15, 1.5])
OUT_OF_RANGE = dict(mass=[0.02, 0.05, 0.1, 0.2], friction=[0.6, 0.4, 0.2, 0.1, 0.05])


def run_condition(sim, mass, friction, seeds, grasp_options):
    sim.set_object_randomization(mass=(mass, mass), friction=(friction, friction))
    rows = []
    for seed in seeds:
        track = dict(offset=None, slip=0.0, min_z=np.inf)

        def on_step(state, name, learned):
            if name != "transport":
                return
            offset = np.subtract(state["cube_pos"], state["grasp_center"])
            if track["offset"] is None:
                track["offset"] = offset
            track["slip"] = max(track["slip"], float(np.linalg.norm(offset - track["offset"])))
            track["min_z"] = min(track["min_z"], state["cube_pos"][2])

        row = run_pick_place_episode(seed, transport="scripted", sim=sim, on_step=on_step,
                                     **grasp_options)
        physics = sim.object_physics()
        skills = {r["name"]: r for r in row["skills"]}
        first = row["skills"][0]
        row.update(type="episode", mass=physics["mass"], friction=physics["friction"],
                   grasp=first["success"], grasp_learned=first["learned"],
                   fallback_used=any(r.get("fallback") for r in row["skills"]),
                   held=skills.get("transport", {}).get("success", False),
                   transport_slip_m=track["slip"],
                   failure=next((f'{r["name"]}: {r["reason"]}' for r in row["skills"]
                                 if not r["success"]), None))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def summarize(rows, mass, friction, region, controller):
    n = len(rows)
    grasped = [r for r in rows if r["grasp"]]
    slips = np.asarray([r["transport_slip_m"] for r in grasped]) if grasped else np.zeros(1)
    errors = np.asarray([r["final_xy_error"] for r in rows])
    reasons = {}
    for r in rows:
        if r["failure"]:
            reasons[r["failure"]] = reasons.get(r["failure"], 0) + 1
    return dict(type="summary", region=region, mass=mass, friction=friction, episodes=n,
                first_seed=rows[0]["seed"], controller=controller,
                grasp_rate=len(grasped) / n,
                fallback_rate=sum(r["fallback_used"] for r in rows) / n,
                hold_rate=sum(r["held"] for r in rows) / n,
                drop_rate=sum(r["dropped"] for r in rows) / n,
                success_rate=sum(r["success"] for r in rows) / n,
                transport_slip_mm=dict(mean=float(slips.mean() * 1e3), max=float(slips.max() * 1e3)),
                final_xy_error_mm=dict(mean=float(errors.mean() * 1e3),
                                       p95=float(np.percentile(errors, 95) * 1e3)),
                failures=reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3000000,
                        help="first evaluation seed (training uses other seeds)")
    parser.add_argument("--region", choices=("in", "out", "both"), default="both")
    parser.add_argument("--mass", type=float, nargs="*", help="override the mass grid (kg)")
    parser.add_argument("--friction", type=float, nargs="*", help="override the friction grid")
    parser.add_argument("--conditions", type=float, nargs="*",
                        help="explicit mass friction pairs (instead of a grid)")
    parser.add_argument("--label", default="custom", help="region label for --conditions")
    parser.add_argument("--shape", choices=("cube", "sphere"), default="cube")
    parser.add_argument("--grasp-model", help="residual Grasp checkpoint (best.zip)")
    parser.add_argument("--fixed-wrist-roll", type=float,
                        help="non-learned baseline: IK Grasp + constant wrist_roll offset (rad)")
    parser.add_argument("--grasp-fallback", action="store_true",
                        help="run the IK Grasp after a failed learned Grasp")
    args = parser.parse_args()
    grasp_options, controller = {}, "scripted IK (no learned skill)"
    if args.grasp_model and args.fixed_wrist_roll is not None:
        parser.error("--grasp-model and --fixed-wrist-roll are exclusive")
    if args.fixed_wrist_roll is not None:
        from robot_sim.grasp_learning import wrist_roll_policy
        policy = wrist_roll_policy(args.fixed_wrist_roll)
        grasp_options = dict(grasp="learned", grasp_model=policy, grasp_config=policy.config,
                             grasp_fallback=args.grasp_fallback)
        controller = f"IK Grasp + fixed wrist_roll {args.fixed_wrist_roll:+.3f} rad (not learned)"
    if args.grasp_model:
        from robot_sim.grasp_learning import load_grasp_policy
        model, config = load_grasp_policy(args.grasp_model)
        grasp_options = dict(grasp="learned", grasp_model=model, grasp_config=config,
                             grasp_fallback=args.grasp_fallback)
        controller = (f"learned residual Grasp (SAC, history={config['history']}) "
                      f"from {args.grasp_model}" + (" + IK fallback" if args.grasp_fallback else ""))
    grids = []
    if args.conditions:
        if len(args.conditions) % 2:
            parser.error("--conditions takes mass friction pairs")
        grids.append((args.label, list(zip(args.conditions[::2], args.conditions[1::2]))))
    elif args.mass or args.friction:
        grids.append(("custom", dict(mass=args.mass or IN_RANGE["mass"],
                                     friction=args.friction or IN_RANGE["friction"])))
    else:
        if args.region in ("in", "both"):
            grids.append(("in", IN_RANGE))
        if args.region in ("out", "both"):
            grids.append(("out", OUT_OF_RANGE))
    seeds = range(args.seed, args.seed + args.episodes)
    sim = PickPlaceSimulator(shape=args.shape)
    try:
        for region, grid in grids:
            pairs = grid if isinstance(grid, list) else itertools.product(grid["mass"],
                                                                          grid["friction"])
            for mass, friction in pairs:
                started = time.perf_counter()
                rows = run_condition(sim, mass, friction, seeds, grasp_options)
                summary = summarize(rows, mass, friction, region, controller)
                summary.update(seconds=time.perf_counter() - started, shape=args.shape)
                print(json.dumps(summary), file=sys.stderr, flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
