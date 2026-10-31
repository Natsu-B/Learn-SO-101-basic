"""Does a residual Grasp policy change its correction with the hidden physics?

For each seed the cube/goal placement and the pre-contact motion are identical
across conditions, and the policy never observes mass or friction. Before the
jaws touch the cube, physics cannot differ, so actions must agree; any later
difference comes from the observed response (cube motion, contacts, joints).

Per condition and versus the reference condition (first pair) this reports:
the step of first jaw contact, the first step at which actions differ, how
many episodes diverge only after contact, and the mean residual (in rad) per
joint before and after contact, plus the post-contact action distance.
JSON lines on stdout.
"""
import argparse
import json

import numpy as np

from robot_sim.grasp_env import GoalGraspEnv
from robot_sim.grasp_learning import load_grasp_policy

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def rollout(env, model, seed):
    obs, info = env.reset(seed=seed)
    actions, contact, done = [], None, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.asarray(action, np.float64))
        obs, _, terminated, truncated, info = env.step(action)
        c = env.episode.state["contacts"]
        if contact is None and (c["fixed_jaw"] or c["moving_jaw"]):
            contact = len(actions)  # 1-based step after which a jaw first touched
        done = terminated or truncated
    return dict(actions=np.asarray(actions), contact=contact, success=info["is_success"])


def residual_rad(actions, config):
    scale = np.r_[[config["arm_scale"]] * 5, config["gripper_scale"]]
    return actions * scale


def mean_by_joint(rows, config, part):
    chunks = []
    for r in rows:
        k = r["contact"] if r["contact"] is not None else len(r["actions"])
        chunk = r["actions"][:k] if part == "pre" else r["actions"][k:]
        if len(chunk):
            chunks.append(residual_rad(chunk, config))
    if not chunks:
        return None
    return {n: float(v) for n, v in zip(NAMES, np.concatenate(chunks).mean(0))}


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
            runs[(mass, friction)] = [rollout(env, model, s)
                                      for s in range(a.seed, a.seed + a.seeds)]
        reference = runs[conditions[0]]
        for (mass, friction), rows in runs.items():
            contacts = [r["contact"] for r in rows if r["contact"] is not None]
            out = dict(type="condition", model=a.model, history=config["history"], mass=mass,
                       friction=friction, episodes=len(rows),
                       success_rate=float(np.mean([r["success"] for r in rows])),
                       mean_steps=float(np.mean([len(r["actions"]) for r in rows])),
                       median_contact_step=float(np.median(contacts)) if contacts else None,
                       residual_pre_contact_rad=mean_by_joint(rows, config, "pre"),
                       residual_post_contact_rad=mean_by_joint(rows, config, "post"))
            if (mass, friction) != conditions[0]:
                first, after_contact, dist, per_joint = [], 0, [], []
                for x, y in zip(rows, reference):
                    n = min(len(x["actions"]), len(y["actions"]))
                    diff = residual_rad(x["actions"][:n], config) - residual_rad(
                        y["actions"][:n], config)
                    d = np.linalg.norm(x["actions"][:n] - y["actions"][:n], axis=1)
                    idx = np.flatnonzero(d > 1e-3)
                    if idx.size:
                        step = int(idx[0]) + 1
                        first.append(step)
                        contact = min(c for c in (x["contact"], y["contact"]) if c is not None) \
                            if (x["contact"] or y["contact"]) else None
                        after_contact += contact is not None and step > contact
                    k = x["contact"] or 0
                    if n > k:
                        dist.append(float(d[k:n].mean()))
                        per_joint.append(np.abs(diff[k:n]).mean(0))
                out.update(reference=list(conditions[0]), diverged_episodes=len(first),
                           diverged_after_contact=int(after_contact),
                           median_first_divergence_step=float(np.median(first)) if first else None,
                           post_contact_action_distance=float(np.mean(dist)) if dist else None,
                           post_contact_abs_difference_rad={
                               n: float(v) for n, v in zip(NAMES, np.mean(per_joint, 0))}
                           if per_joint else None)
            print(json.dumps(out), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
