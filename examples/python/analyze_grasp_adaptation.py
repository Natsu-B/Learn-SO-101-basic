"""Does the residual Grasp policy change its behaviour with the hidden physics?

For each seed the cube/goal placement is identical across conditions, and the
policy never observes mass or friction. Any difference in its actions between
conditions therefore comes from the observed physical response. Reports, per
condition, the mean residual per action dimension, and versus the reference
condition the first step at which the actions differ and their mean distance
over the common steps. JSON lines on stdout.
"""
import argparse
import json

import numpy as np

from robot_sim.grasp_env import GoalGraspEnv
from robot_sim.grasp_learning import load_grasp_policy

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def rollout(env, model, seed):
    obs, info = env.reset(seed=seed)
    actions, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return np.asarray(actions), info["is_success"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed", type=int, default=3000000)
    p.add_argument("--conditions", type=float, nargs="+",
                   default=[0.035, 1.15, 0.035, 0.1, 0.035, 0.05, 0.1, 1.15, 0.1, 0.1],
                   help="mass friction pairs; the first pair is the reference")
    a = p.parse_args()
    if len(a.conditions) % 2:
        p.error("conditions are mass/friction pairs")
    conditions = list(zip(a.conditions[::2], a.conditions[1::2]))
    model, config = load_grasp_policy(a.model)
    env = GoalGraspEnv(**{**config, "friction_sampling": "uniform"})
    try:
        runs = {}
        for mass, friction in conditions:
            env.set_physics_ranges((mass, mass), (friction, friction))
            runs[(mass, friction)] = [rollout(env, model, s) for s in range(a.seed, a.seed + a.seeds)]
        reference = runs[conditions[0]]
        for (mass, friction), rows in runs.items():
            stacked = np.concatenate([r[0] for r in rows])
            out = dict(type="condition", model=a.model, history=config["history"], mass=mass,
                       friction=friction, episodes=len(rows),
                       success_rate=float(np.mean([r[1] for r in rows])),
                       mean_steps=float(np.mean([len(r[0]) for r in rows])),
                       mean_action={n: float(v) for n, v in zip(NAMES, stacked.mean(0))},
                       mean_abs_action={n: float(v) for n, v in zip(NAMES, np.abs(stacked).mean(0))})
            if (mass, friction) != conditions[0]:
                first, dist = [], []
                for (x, _), (y, _) in zip(rows, reference):
                    n = min(len(x), len(y))
                    d = np.linalg.norm(x[:n] - y[:n], axis=1)
                    idx = np.flatnonzero(d > 1e-3)
                    first.append(int(idx[0]) + 1 if idx.size else None)
                    dist.append(float(d.mean()))
                diverged = [f for f in first if f is not None]
                out.update(reference=list(conditions[0]),
                           diverged_episodes=len(diverged),
                           median_first_divergence_step=float(np.median(diverged)) if diverged else None,
                           mean_action_distance=float(np.mean(dist)))
            print(json.dumps(out), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
