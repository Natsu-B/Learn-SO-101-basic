"""Learning curves of the grasp-force PPO runs (docs/FORCE_CONTROL.md §7).

Small multiples over training timesteps, from each run's selection.jsonl
(deterministic policy on the 24 checkpoint-selection seeds 7,000,000+).
Usage: python scripts/plot_force_grasp_curves.py OUT.png [--coop] NAME=RUN_DIR ...
--coop plots the final safe-grasp success rate and valid-contact peak force
(docs §11) instead of the return and all-episode peak.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (dataviz reference palette, light); markers give
# a second encoding because three slots are below 3:1 against the surface.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
PANELS = [
    ("return", "累積報酬（選択用seed）", None),
    ("mean_abs_error", "力の平均誤差 [N]（最後の1 s）", None),
    ("peak_force_physics", "ピーク法線力 [N]（物理step）", None),
    ("crushed", "潰れ率", (0, 1)),
]


COOP_PANELS = [
    ("success", "安全把持の成功率（選択用seed）", (0, 1)),
    ("mean_abs_error", "力の平均誤差 [N]（最後の1 s）", None),
    ("peak_force_valid_mean", "ピーク法線力 [N]（有効接触の試行）", None),
    ("crushed", "潰れ率", (0, 1)),
]


def main():
    out = sys.argv[1]
    args = sys.argv[2:]
    panels = PANELS
    if args and args[0] == "--coop":
        panels, args = COOP_PANELS, args[1:]
    runs = [a.split("=", 1) for a in args]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    for family in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP"):
        if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            break
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), facecolor="#fcfcfb")
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for i, (name, run) in enumerate(runs):
            rows = [json.loads(l) for l in (Path(run) / "selection.jsonl").read_text().splitlines()]
            seen, xs, ys = set(), [], []
            for r in rows:  # resumed runs may repeat a timestep; keep the first
                if r["timesteps"] in seen:
                    continue
                seen.add(r["timesteps"])
                xs.append(r["timesteps"] / 1e6)
                ys.append(r[key])
            ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=4, linewidth=1.6,
                    label=name)
        ax.set_title(title, loc="left", color="#0b0b0b")
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("学習ステップ [M]")
    axes[0, 0].legend(frameon=False, loc="lower right" if panels is PANELS else "upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
