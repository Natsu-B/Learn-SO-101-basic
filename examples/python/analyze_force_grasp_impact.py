"""Is the fixed-jaw impact avoidable with one moving jaw? (docs/FORCE_CONTROL.md §8)

Scripted variant of the force PI: while only the moving jaw has touched (it
pushes the object across the unobserved gap) it closes at --push-speed
rad/s instead of the PI's 0.6 rad/s; after both jaws touched it is the
same PI. Runs on the final-evaluation seeds and prints the same aggregate
as eval_force_grasp.py, plus the action profile of PPO policies during the
push phase.

  python examples/python/analyze_force_grasp_impact.py --output DIR \
      --push-speeds 0.6 0.15 0.06 0.03 --ppo P1=runs/.../best.zip
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robot_sim import CONTROL_PERIOD, ForcePi
from robot_sim.force_grasp_env import ForceGraspConfig, ForceGraspEnv
from robot_sim.force_grasp_learning import FINAL_SEEDS, aggregate

PI_DELTA_LIMIT = 0.03


def slow_push_episode(env, seed, push_speed):
    env.reset(options={"episode_seed": seed})
    pi = ForcePi()
    while True:
        h_fixed, h_moving = env.had_contact
        if h_moving and not h_fixed:
            d = -push_speed * CONTROL_PERIOD
        else:
            d = float(np.clip(pi.act(list(env.f_obs), env.goal, CONTROL_PERIOD),
                              -PI_DELTA_LIMIT, PI_DELTA_LIMIT))
        _, _, terminated, truncated, info = env.step_command(env.q_obs + d)
        if terminated or truncated:
            m = info.get("episode_metrics") or env.metrics.summary(
                env.config.contact_threshold, env.penetration_limit)
            m.update(seed=seed, push_speed=push_speed)
            return m


def ppo_push_profile(env, model, seeds):
    """Applied gripper speed (rad/s) per contact phase and the peak force."""
    phases = {"free": [], "push": [], "both": []}
    for s in seeds:
        obs, _ = env.reset(options={"episode_seed": s})
        while True:
            h = env.had_contact.copy()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(action)
            speed = -float(np.clip(action[0], -1, 1)) * (env.bias() if env.config.inductive_bias else 1)
            key = "both" if h.all() else ("push" if h[1] else "free")
            phases[key].append(speed * env.config.dq_max / CONTROL_PERIOD)
            if te or tr:
                break
    return {k: {"mean_closing_speed_rad_s": float(np.mean(v)) if v else None, "periods": len(v)}
            for k, v in phases.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--push-speeds", type=float, nargs="+", default=[0.6, 0.15, 0.06, 0.03])
    p.add_argument("--ppo", nargs="*", default=[])
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    env = ForceGraspEnv(ForceGraspConfig())
    env.set_progress(1.0)
    seeds = range(FINAL_SEEDS, FINAL_SEEDS + a.episodes)
    result = {"slow_push": {}, "ppo_push_profile": {}}
    with (out / "episodes.jsonl").open("w") as f:
        for v in a.push_speeds:
            rows = [slow_push_episode(env, s, v) for s in seeds]
            for r in rows:
                f.write(json.dumps(r) + "\n")
            agg = aggregate(rows)
            result["slow_push"][str(v)] = agg
            print(f"push {v:5.2f} rad/s", json.dumps({k: round(agg[k], 3) if isinstance(agg[k], float)
                  else agg[k] for k in ("success", "mean_abs_error", "peak_force_physics",
                  "peak_force_valid_mean", "valid_episodes", "crushed", "invalid_contact",
                  "lost_contact_final", "time_to_stable_mean_s", "peak_phase_contact")}))
    from stable_baselines3 import PPO

    for spec in a.ppo:
        name, path = spec.split("=", 1)
        cfg = ForceGraspConfig.from_dict(json.loads((Path(path).parent / "config.json").read_text())["env"])
        penv = ForceGraspEnv(cfg)
        penv.set_progress(1.0)
        prof = ppo_push_profile(penv, PPO.load(path, device="cpu"), list(seeds)[:50])
        result["ppo_push_profile"][name] = prof
        print(name, json.dumps(prof))
    (out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
