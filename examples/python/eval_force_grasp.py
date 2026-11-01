"""Compare fixed command, force PI and PPO policies on identical episodes
(docs/FORCE_CONTROL.md §7). Every controller sees the same episode seeds,
objects, goal forces and sensor-noise sequences. Final-evaluation seeds
start at 8,000,000 (checkpoint selection used 7,000,000+).

  python examples/python/eval_force_grasp.py --output runs/force-grasp-eval/NAME \
      --controllers fixed pi ppo:p1=runs/force-grasp-ppo/.../best.zip --episodes 200
  --kappa-grid 11 evaluates the paper's protocol (11 stiffness values,
  --episodes per value, everything else sampled).

Writes episodes.jsonl, summary.json and summary.md; refuses a non-empty output.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robot_sim.force_grasp_env import ForceGraspConfig, ForceGraspEnv
from robot_sim.force_grasp_learning import FINAL_SEEDS, aggregate, run_episode

COLUMNS = [
    ("success", "成功率", "{:.1%}"),
    ("mean_abs_error", "平均誤差 N", "{:.3f}"),
    ("max_abs_error", "最大誤差 N", "{:.3f}"),
    ("relative_error", "相対誤差", "{:.3f}"),
    ("peak_force_physics", "ピーク力 N（物理step）", "{:.2f}"),
    ("peak_force_max", "ピーク力 最大 N", "{:.1f}"),
    ("peak_over_goal", "ピーク/目標", "{:.1f}"),
    ("valid_episodes", "有効接触の試行数", "{}"),
    ("peak_force_valid_mean", "ピーク力 N（有効試行の平均）", "{:.2f}"),
    ("peak_force_valid_median", "ピーク力 N（有効試行の中央値）", "{:.2f}"),
    ("peak_force_valid_max", "ピーク力 N（有効試行の最大）", "{:.1f}"),
    ("peak_force_hold", "保持中（接触4周期後〜）のピーク N", "{:.2f}"),
    ("peak_phase_push", "ピーク: 押し渡し中", "{}"),
    ("peak_phase_contact", "ピーク: 固定ジョー接触時", "{}"),
    ("peak_phase_hold", "ピーク: 保持中", "{}"),
    ("crushed", "潰れ率", "{:.1%}"),
    ("invalid_contact", "無効接触率", "{:.1%}"),
    ("lost_contact_final", "接触喪失率", "{:.1%}"),
    ("both_contact_ratio", "両ジョー接触維持", "{:.1%}"),
    ("time_to_stable_mean_s", "安定まで s", "{:.2f}"),
    ("stable_reached", "安定到達数", "{}"),
    ("object_displacement_mm", "物体移動 mm", "{:.1f}"),
]


def load_ppo(path):
    from stable_baselines3 import PPO

    config_path = Path(path).parent / "config.json"
    cfg = ForceGraspConfig.from_dict(json.loads(config_path.read_text())["env"])
    return PPO.load(path, device="cpu"), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--controllers", nargs="+", default=["fixed", "pi"],
                   help="fixed, pi, or ppo:NAME=PATH/best.zip")
    p.add_argument("--first-seed", type=int, default=FINAL_SEEDS)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--kappa-grid", type=int, default=0)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    kappas = list(np.linspace(0, 1, a.kappa_grid)) if a.kappa_grid else [None]
    seeds = list(range(a.first_seed, a.first_seed + a.episodes))
    rows, summary = [], {}
    base = ForceGraspEnv(ForceGraspConfig())
    for spec in a.controllers:
        if spec.startswith("ppo:"):
            name, path = spec[4:].split("=", 1)
            model, cfg = load_ppo(path)
            env = ForceGraspEnv(cfg)
            controller = "ppo"
        else:
            name, model, env, controller = spec, None, base, spec
        env.set_progress(1.0)
        mine = []
        for kappa in kappas:
            for s in seeds:
                r = run_episode(env, s, controller, model, kappa=kappa)
                r["name"] = name
                mine.append(r)
        rows += mine
        summary[name] = aggregate(mine)
        if kappas[0] is not None:
            summary[name]["by_kappa"] = {
                f"{k:.1f}": aggregate([r for r in mine if r["kappa"] == k]) for k in kappas}
        print(name, json.dumps({k: summary[name][k] for k in ("success", "mean_abs_error",
                                "peak_force_physics", "crushed", "invalid_contact")}))
    with (out / "episodes.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"controllers": a.controllers, "seeds": [seeds[0], seeds[-1]], "kappas": kappas,
            "summary": summary}
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))
    lines = ["| 指標 | " + " | ".join(summary) + " |", "|---" * (len(summary) + 1) + "|"]
    for key, label, fmt in COLUMNS:
        cells = []
        for name in summary:
            v = summary[name].get(key)
            cells.append("—" if v is None else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
