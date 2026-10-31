"""State handed from Grasp to Transport: cube tilt, lift height and gripper
angle at the end of Grasp, versus full pick-and-place success.

Usage: analyze_grasp_handoff.py [MODEL_DIR ...]   ("ik" = scripted Grasp)
Defaults reproduce docs/OBJECT_PHYSICS.md §6 (0.02 kg, friction 0.8, 30 seeds).
"""
import argparse

import numpy as np

from robot_sim import PickPlaceSimulator
from robot_sim.grasp_env import cube_tilt
from robot_sim.grasp_learning import load_grasp_policy
from robot_sim.pick_place_pipeline import run_pick_place_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("controllers", nargs="*",
                   default=["ik", "runs/so101-grasp/h0-seed-7", "runs/so101-grasp/h4-seed-42"])
    p.add_argument("--mass", type=float, default=0.02)
    p.add_argument("--friction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=3000000)
    p.add_argument("--episodes", type=int, default=30)
    a = p.parse_args()
    sim = PickPlaceSimulator()
    sim.set_object_randomization(mass=(a.mass, a.mass), friction=(a.friction, a.friction))
    half = sim.criteria()["cube_half"]
    try:
        for name in a.controllers:
            options = {}
            if name != "ik":
                model, config = load_grasp_policy(f"{name}/best.zip")
                options = dict(grasp="learned", grasp_model=model, grasp_config=config)
            rows = []
            for seed in range(a.seed, a.seed + a.episodes):
                last = {}
                run = run_pick_place_episode(
                    seed, transport="scripted", sim=sim, **options,
                    on_step=lambda s, skill, learned: last.__setitem__(skill, s))
                g = last["grasp"]
                final = last.get("place", g)
                rows.append((np.degrees(cube_tilt(g["cube_quat"])), g["cube_pos"][2] - half,
                             g["gripper"], np.degrees(cube_tilt(final["cube_quat"])), run["success"]))
            x = np.asarray(rows, float)
            failed = x[x[:, 4] == 0]
            print(f"{name}: success {int(x[:, 4].sum())}/{len(x)}, grasp-end tilt {x[:, 0].mean():.1f} deg, "
                  f"lift {x[:, 1].mean() * 1e3:.1f} mm, gripper {x[:, 2].mean():.3f} rad, "
                  f"final tilt of failures {failed[:, 3].mean() if len(failed) else float('nan'):.1f} deg",
                  flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
