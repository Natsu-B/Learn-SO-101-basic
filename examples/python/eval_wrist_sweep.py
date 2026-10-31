"""Fixed wrist_roll offsets (or a Grasp checkpoint) over fixed mass/friction
conditions, with hand-off and pose metrics (docs/PHYSICS_AUDIT.md §3).

Grasp = IK approach, then IK Close/Lift/Hold plus a constant wrist_roll offset
(not learned, reported learned=False) or a residual checkpoint; Transport and
Place are scripted IK. The contact environment is chosen with --contact-model:

- legacy: the original friction DR (one value for jaw and table contacts, the
  object's priority/condim/solref take over all its contacts).
- pairwise: jaw-pad friction = the condition's friction; table friction is
  --floor same (equal to the jaw value), nominal (scene value, 1.2) or a number.
  condim/solref stay those of the unrandomized scene.

Episode JSON lines go to stdout and one summary per (controller, condition) to
stderr. The true physics is logged; no controller sees it.
"""
import argparse
import json
import sys
import time

import numpy as np

from robot_sim import PickPlaceSimulator
from robot_sim.grasp_env import cube_tilt
from robot_sim.pick_place_pipeline import run_pick_place_episode


def physics_ranges(contact_model, mass, friction, floor):
    ranges = dict(mass=(mass, mass), friction=(friction, friction))
    if contact_model == "pairwise" and floor != "nominal":
        value = friction if floor == "same" else float(floor)
        ranges["floor_friction"] = (value, value)
    return ranges


def episode(sim, seed, options):
    track = dict(start=None, pinch=None, close_xy=None, push=0.0, slip=0.0, first_pinch=None,
                 lift_tilt=0.0, phase=None)

    def on_step(state, name, learned):
        cube = np.asarray(state["cube_pos"])
        if track["start"] is None:
            track["start"] = cube.copy()
        if name != "grasp":
            return
        # Residual steps (after the IK approach) are flagged learned=True by
        # run_learned_grasp even for a fixed offset; plain IK never is.
        offset = cube - np.asarray(state["grasp_center"])
        if state["contacts"]["pinched"] and track["pinch"] is None:
            track["pinch"] = offset
        if track["pinch"] is not None:
            track["slip"] = max(track["slip"], float(np.linalg.norm(offset - track["pinch"])))
        if cube[2] < sim_half + 0.002:
            # Horizontal push of the cube along the table before it is lifted.
            track["push"] = max(track["push"], float(np.linalg.norm(cube[:2] - track["start"][:2])))
        else:
            track["lift_tilt"] = max(track["lift_tilt"], cube_tilt(state["cube_quat"]))

    sim_half = sim.criteria()["cube_half"]
    row = run_pick_place_episode(seed, transport="scripted", sim=sim, on_step=on_step, **options)
    skills = {r["name"]: r for r in row["skills"]}
    ends = {e["name"]: e for e in row["skill_end_states"]}
    physics = sim.object_physics()
    grasp_end = ends.get("grasp", {})
    place_start = ends.get("transport", {})
    row.update(type="episode", mass=physics["mass"], jaw_friction=physics["jaw_friction"],
               floor_friction=physics["floor_friction"], contact_model=physics["contact_model"],
               grasp=bool(skills["grasp"]["success"]),
               held=bool(skills.get("transport", {}).get("success", False)),
               table_push_mm=track["push"] * 1e3, grasp_slip_mm=track["slip"] * 1e3,
               max_lifted_tilt_deg=float(np.degrees(track["lift_tilt"])),
               grasp_end_tilt_deg=grasp_end.get("tilt_deg"),
               grasp_end_height_mm=None if not grasp_end else grasp_end["height_m"] * 1e3,
               place_start_tilt_deg=place_start.get("tilt_deg"),
               place_start_speed_mps=place_start.get("speed_mps"),
               final_tilt_deg=float(np.degrees(cube_tilt(sim.state()["cube_quat"]))),
               failure=next((f'{r["name"]}: {r["reason"]}' for r in row["skills"]
                             if not r["success"]), None))
    return row


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(rows):
    n = len(rows)
    reasons = {}
    for r in rows:
        if r["failure"]:
            reasons[r["failure"]] = reasons.get(r["failure"], 0) + 1
    ok = [r for r in rows if r["success"]]
    errors = np.asarray([r["final_xy_error"] for r in ok]) if ok else np.full(1, np.nan)
    return dict(type="summary", episodes=n, grasp=sum(r["grasp"] for r in rows),
                held=sum(r["held"] for r in rows), success=sum(r["success"] for r in rows),
                table_push_mm=mean([r["table_push_mm"] for r in rows]),
                grasp_slip_mm=mean([r["grasp_slip_mm"] for r in rows if r["grasp"]]),
                max_lifted_tilt_deg=mean([r["max_lifted_tilt_deg"] for r in rows if r["grasp"]]),
                grasp_end_tilt_deg=mean([r["grasp_end_tilt_deg"] for r in rows if r["grasp"]]),
                place_start_tilt_deg=mean([r["place_start_tilt_deg"] for r in rows]),
                place_start_tilt_success_deg=mean([r["place_start_tilt_deg"] for r in ok]),
                place_start_tilt_failure_deg=mean([r["place_start_tilt_deg"] for r in rows
                                                   if r["held"] and not r["success"]]),
                final_xy_error_success_mm=dict(mean=float(np.mean(errors) * 1e3),
                                               max=float(np.max(errors) * 1e3)),
                failures=reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4000000,
                        help="first seed (4,000,000+: selection block, 3,000,000+: final block)")
    parser.add_argument("--conditions", type=float, nargs="+", required=True,
                        help="mass friction pairs")
    parser.add_argument("--label", default="custom")
    parser.add_argument("--contact-model", choices=("legacy", "pairwise"), default="legacy")
    parser.add_argument("--floor", default="same",
                        help="pairwise table friction: same | nominal | <value>")
    parser.add_argument("--wrist", type=float, nargs="*", default=[],
                        help="fixed wrist_roll offsets (rad); 'ik' is always plain IK")
    parser.add_argument("--ik", action="store_true", help="also run the plain IK Grasp")
    parser.add_argument("--grasp-model", action="append", default=[],
                        help="residual Grasp checkpoint(s) (best.zip)")
    parser.add_argument("--end-at", choices=("base", "success"), default="base",
                        help="fixed-offset hand-off: base = after the IK Hold (v3), "
                             "success = at the 0.3 s lift (original baseline)")
    args = parser.parse_args()
    if len(args.conditions) % 2:
        parser.error("--conditions takes mass friction pairs")
    from robot_sim.grasp_learning import load_grasp_policy, wrist_roll_policy
    controllers = []
    if args.ik:
        controllers.append(("ik", {}))
    for w in args.wrist:
        policy = wrist_roll_policy(w, end_at=args.end_at)
        controllers.append((f"wrist{w:+.2f}-{args.end_at}",
                            dict(grasp="learned", grasp_model=policy, grasp_config=policy.config)))
    for path in args.grasp_model:
        model, config = load_grasp_policy(path)
        controllers.append((path, dict(grasp="learned", grasp_model=model, grasp_config=config)))
    sim = PickPlaceSimulator(contact_model=args.contact_model)
    seeds = range(args.seed, args.seed + args.episodes)
    try:
        for mass, friction in zip(args.conditions[::2], args.conditions[1::2]):
            sim.set_object_randomization(**physics_ranges(args.contact_model, mass, friction,
                                                          args.floor))
            for name, options in controllers:
                started = time.perf_counter()
                rows = []
                for seed in seeds:
                    row = episode(sim, seed, options)
                    row.update(controller=name, region=args.label, condition_mass=mass,
                               condition_friction=friction, floor=args.floor)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                s = summarize(rows)
                s.update(controller=name, region=args.label, mass=mass, friction=friction,
                         floor_friction=rows[0]["floor_friction"],
                         contact_model=args.contact_model, first_seed=args.seed,
                         seconds=time.perf_counter() - started)
                print(json.dumps(s), file=sys.stderr, flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
