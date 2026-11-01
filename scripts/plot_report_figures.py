"""Comparison figure for the final-assignment report (docs/report).

Reads the 200-episode evaluation summaries (final seeds 8,000,000-199) and
draws safe-grasp success and crush rates per controller as two small
multiples sharing the controller axis.

  uv run --no-project --with matplotlib python scripts/plot_report_figures.py docs/report/figures
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCES = [
    "runs/force-grasp-eval/final-200/summary.json",
    "runs/force-grasp-eval/diag2-200/summary.json",
    "runs/force-grasp-coop/eval-final-200/summary.json",
    "runs/force-grasp-coop/eval-final-200-p10b/summary.json",
]
# (label, summary key prefix, group)
ROWS = [
    ("固定指令（既存Grasp）", "fixed", "gripper"),
    ("力PI（論文のベースライン）", "pi", "gripper"),
    ("二段階PI", "tspi", "gripper"),
    ("PPO P1（公開コード相当）", "P1", "gripper"),
    ("PPO P2（論文相当）", "P2", "gripper"),
    ("PPO P8（ピーク罰）", "P8", "gripper"),
    ("PPO P9（接触獲得報酬）", "P9", "gripper"),
    ("腕補正PI（スクリプト）", "armpi", "arm"),
    ("PPO P11（腕のみ学習＋PI）", "P11", "arm"),
    ("PPO P10（腕＋グリッパ，50万step）", "P10", "arm"),
    ("PPO P10b（腕＋グリッパ，100万step）", "P10b", "arm"),
]
# Validated categorical slots 1-2 of the dataviz reference palette (light).
COLORS = {"gripper": "#eb6834", "arm": "#2a78d6"}
LEGEND = {"gripper": "グリッパ単独", "arm": "腕の補正あり"}


def lookup(summary, key):
    for name, value in summary.items():
        if name == key or (key in ("tspi", "armpi") and name.startswith(key + ":")):
            return value
    raise KeyError(key)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in SOURCES:
        summary.update(json.loads(Path(path).read_text())["summary"])
    rows = [(label, lookup(summary, key), group) for label, key, group in ROWS][::-1]
    for family in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP"):
        if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True, facecolor="#fcfcfb",
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, key, title in ((axes[0], "success", "安全把持の成功率"), (axes[1], "crushed", "潰れ率")):
        ax.set_facecolor("#fcfcfb")
        y = range(len(rows))
        values = [100 * r[1][key] for r in rows]
        ax.barh(list(y), values, height=0.62, color=[COLORS[r[2]] for r in rows],
                edgecolor="#fcfcfb", linewidth=2)
        for i, v in zip(y, values):
            ax.text(v + 1.2, i, f"{v:.1f}%", va="center", color="#0b0b0b", fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=10)
        ax.grid(True, axis="x", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("%（最終評価seed 200試行）", color="#52514e")
    axes[0].set_yticks(list(range(len(rows))), [r[0] for r in rows])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[g]) for g in ("gripper", "arm")]
    axes[1].legend(handles, [LEGEND[g] for g in ("gripper", "arm")], frameon=False,
                   loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "force_grasp_comparison.png", dpi=180, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
