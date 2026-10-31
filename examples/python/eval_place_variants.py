"""Place-failure isolation (docs/PHYSICS_AUDIT.md §4).

For every seed the same Grasp and IK Transport are replayed (the simulation is
deterministic, so each variant starts Place from the identical physical state;
this is checked and reported as ``same_place_start``) and only the step after
Transport changes:

- place: the current IK Place (lower to 2 mm above the table, open).
- settle: best-effort IK Stabilize (stop, wait until still), then Place.
- level: rotate the hand about the jaw closing axis while lowering so the
  cube's up face is level at release.
- contact: keep lowering until the cube touches the table, then open.
- level_contact: both.
- gentle: open only to 0.25 rad (just past the cube) before retreating.
- gentle_contact: gentle + contact.

During Place it records whether the cube was ever resting on the table at the
goal after release (Rust ``placed``) and the largest jaw force on the cube
after the jaws first let go, so a failure can be classed as "never placed"
(release/tipping) or "disturbed after placement" (re-contact while opening or
retreating).

Episode JSON lines to stdout, one summary per (variant, condition) to stderr.
"""
import argparse
import json
import sys

import numpy as np

from robot_sim import PickPlaceSimulator
from robot_sim.grasp_env import cube_tilt
from robot_sim.pick_place_pipeline import run_pick_place_episode

VARIANTS = dict(place=dict(place="place"), settle=dict(place="place", settle_before_place=True),
                level=dict(place="place_level"), contact=dict(place="place_contact"),
                level_contact=dict(place="place_level_contact"),
                gentle=dict(place="place_gentle"),
                gentle_contact=dict(place="place_gentle_contact"))


def grasp_options(spec, end_at):
    if spec == "ik":
        return {}
    from robot_sim.grasp_learning import load_grasp_policy, wrist_roll_policy
    try:
        policy = wrist_roll_policy(float(spec), end_at=end_at)
        return dict(grasp="learned", grasp_model=policy, grasp_config=policy.config)
    except ValueError:
        model, config = load_grasp_policy(spec)
        return dict(grasp="learned", grasp_model=model, grasp_config=config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", default="ik", help="ik | fixed wrist_roll offset | checkpoint")
    parser.add_argument("--end-at", choices=("base", "success"), default="base")
    parser.add_argument("--conditions", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3000000)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--contact-model", choices=("legacy", "pairwise"), default="legacy")
    parser.add_argument("--floor", default="same")
    parser.add_argument("--label", default="custom")
    args = parser.parse_args()
    from eval_wrist_sweep import physics_ranges
    options = grasp_options(args.grasp, args.end_at)
    sim = PickPlaceSimulator(contact_model=args.contact_model)
    try:
        for mass, friction in zip(args.conditions[::2], args.conditions[1::2]):
            sim.set_object_randomization(**physics_ranges(args.contact_model, mass, friction,
                                                          args.floor))
            rows = {v: [] for v in args.variants}
            for seed in range(args.seed, args.seed + args.episodes):
                starts = []
                for variant in args.variants:
                    track = dict(released=False, placed_once=False, recontact_n=0.0,
                                 release_z=None, release_tilt=None, grip=None)

                    def on_step(state, name, learned):
                        if name != "place":
                            return
                        jaw = [c for c in sim.contact_details() if c["partner"].endswith("jaw")]
                        if track["grip"] is None:
                            track["grip"] = state["gripper"]
                        if not track["released"]:
                            # Released once the jaws have opened 0.05 rad past the
                            # holding angle (a light touch may remain after that).
                            if state["gripper"] > track["grip"] + 0.05:
                                track["released"] = True
                                track["release_z"] = state["cube_pos"][2]
                                track["release_tilt"] = float(np.degrees(
                                    cube_tilt(state["cube_quat"])))
                            return
                        track["placed_once"] |= sim.predicates()["placed"]
                        track["recontact_n"] = max(track["recontact_n"],
                                                   sum(c["normal_force"] for c in jaw))

                    row = run_pick_place_episode(seed, transport="scripted", sim=sim,
                                                 on_step=on_step, **options, **VARIANTS[variant])
                    ends = {e["name"]: e for e in row["skill_end_states"]}
                    start = ends.get("transport")
                    starts.append(None if start is None else json.dumps(start, sort_keys=True))
                    place = next((r for r in row["skills"] if r["name"] == "place"), None)
                    out = dict(type="episode", variant=variant, grasp=args.grasp, seed=seed,
                               mass=mass, friction=friction, label=args.label,
                               contact_model=args.contact_model, success=row["success"],
                               reached_place=place is not None,
                               place_success=None if place is None else place["success"],
                               place_reason=None if place is None else place["reason"],
                               place_start_tilt_deg=None if start is None else start["tilt_deg"],
                               place_start_speed_mps=None if start is None else start["speed_mps"],
                               settle_end=ends.get("settle"), place_end=ends.get("place"),
                               final_xy_error=row["final_xy_error"],
                               release_height_mm=None if track["release_z"] is None
                               else (track["release_z"] - sim.criteria()["cube_half"]) * 1e3,
                               release_tilt_deg=track["release_tilt"],
                               placed_after_release=track["placed_once"],
                               recontact_max_n=track["recontact_n"],
                               final_tilt_deg=float(np.degrees(cube_tilt(sim.state()["cube_quat"]))),
                               failure=next((f'{r["name"]}: {r["reason"]}' for r in row["skills"]
                                             if not r["success"]), None))
                    rows[variant].append(out)
                    print(json.dumps(out), flush=True)
                same = len(set(starts)) == 1
                for variant in args.variants:
                    rows[variant][-1]["same_place_start"] = same
            for variant, rs in rows.items():
                reached = [r for r in rs if r["reached_place"]]
                tilt = np.asarray([r["place_start_tilt_deg"] for r in reached])

                def rate(mask):
                    sel = [r for r, m in zip(reached, mask) if m]
                    return [sum(r["place_success"] for r in sel), len(sel)]
                print(json.dumps(dict(
                    type="summary", variant=variant, grasp=args.grasp, label=args.label,
                    mass=mass, friction=friction, contact_model=args.contact_model,
                    episodes=len(rs), success=sum(r["success"] for r in rs),
                    reached_place=len(reached),
                    place_success=sum(r["place_success"] for r in reached),
                    place_failed_never_placed=sum(not r["place_success"] and not r["placed_after_release"]
                                                  for r in reached),
                    place_failed_disturbed=sum(not r["place_success"] and r["placed_after_release"]
                                               for r in reached),
                    recontact_over_1n=sum(r["recontact_max_n"] > 1.0 for r in reached),
                    release_height_mm=float(np.mean([r["release_height_mm"] for r in reached
                                                     if r["release_height_mm"] is not None]
                                                    or [np.nan])),
                    place_by_start_tilt={"<3deg": rate(tilt < 3), "3-6deg": rate((tilt >= 3) & (tilt < 6)),
                                         ">=6deg": rate(tilt >= 6)},
                    same_place_start=all(r.get("same_place_start", False) for r in rs),
                    final_xy_error_mm=float(np.mean([r["final_xy_error"] for r in rs if r["success"]]
                                                    or [np.nan]) * 1e3))), file=sys.stderr,
                    flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
