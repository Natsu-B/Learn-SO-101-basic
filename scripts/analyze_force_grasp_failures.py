"""Decompose grasp-force evaluations (docs/FORCE_CONTROL.md §10).

Reads an eval_force_grasp.py episodes.jsonl and prints markdown tables:
the success funnel, the exclusive first failure cause, valid-contact peak
force distributions, and success / failure cause / crush phase by kappa and
by face capacity k·travel.

  python scripts/analyze_force_grasp_failures.py runs/force-grasp-eval/diag-200/episodes.jsonl
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

HARD, SOFT = 5000.0, 200.0
CAUSES = ("no_contact", "single_jaw_only", "invalid_contact", "crushed", "lost_contact",
          "goal_not_met", "success")


def kappa(r):
    return math.log(HARD / r["object"]["stiffness"]) / math.log(HARD / SOFT)


def cause(r):
    """First failing stage, in the order the grasp progresses."""
    if not r["any_contact"]:
        return "no_contact"
    if r["contact_step"] is None:
        return "single_jaw_only"
    if r["invalid_contact"]:
        return "invalid_contact"
    if r["crushed"]:
        return "crushed"
    if r["lost_contact_final"]:
        return "lost_contact"
    if not r["success"]:
        return "goal_not_met"
    return "success"


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "—"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    names = list(by)

    print("### 成功までの段階（累積．各段は前の段をすべて満たした試行数）\n")
    stages = [
        ("いずれかのジョーが接触", lambda r: r["any_contact"]),
        ("両ジョーの接触を獲得", lambda r: r["contact_step"] is not None),
        ("最後の1 sも両ジョー接触を維持", lambda r: not r["lost_contact_final"]),
        ("目標力に到達（一度でも）", lambda r: r["goal_reached_step"] is not None),
        ("無効接触なし", lambda r: not r["invalid_contact"]),
        ("潰れなし", lambda r: not r["crushed"]),
        ("最終成功", lambda r: r["success"]),
    ]
    out = []
    for label, _ in stages:
        out.append([label])
    for n in names:
        alive = by[n]
        for i, (_, test) in enumerate(stages):
            alive = [r for r in alive if test(r)]
            out[i].append(pct(len(alive), len(by[n])))
    print(table(["段階"] + names, out))

    print("\n### 段階ごとの単独の通過率（他の段に関係なく）\n")
    out = []
    for label, test in stages:
        out.append([label] + [pct(sum(bool(test(r)) for r in by[n]), len(by[n])) for n in names])
    print(table(["条件"] + names, out))

    print("\n### 最初に失敗した段階（排他的）\n")
    out = []
    for c in CAUSES:
        out.append([c] + [pct(sum(cause(r) == c for r in by[n]), len(by[n])) for n in names])
    print(table(["原因"] + names, out))

    print("\n### ピーク法線力の分布（物理ステップ）\n")
    out = []
    for n in names:
        for label, sel in (
            ("両ジョー接触あり・有効接触", lambda r: r["contact_step"] is not None and not r["invalid_contact"]),
            ("接触あり（片側含む）・有効接触", lambda r: r["any_contact"] and not r["invalid_contact"]),
            ("全試行", lambda r: True),
        ):
            v = np.asarray([r["peak_force_physics"] for r in by[n] if sel(r)])
            if len(v):
                out.append([n, label, len(v), f"{np.median(v):.2f}", f"{np.percentile(v, 95):.2f}",
                            f"{v.max():.1f}"])
    print(table(["制御", "対象", "試行数", "中央値 N", "P95 N", "最大 N"], out))

    print("\n### κごとの成功率と失敗原因（κ = ln(5000/k)/ln(25)，5区間）\n")
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]
    for n in names:
        out = []
        for lo, hi in bins:
            sub = [r for r in by[n] if lo <= kappa(r) < hi]
            c = Counter(cause(r) for r in sub)
            ph = Counter(r["bottom_phase"] for r in sub if r["crushed"])
            out.append([f"{lo:.1f}–{min(hi, 1):.1f}", len(sub), pct(c["success"], len(sub)),
                        ", ".join(f"{k} {v}" for k, v in c.most_common() if k != "success"),
                        ", ".join(f"{k} {v}" for k, v in ph.most_common())])
        print(f"**{n}**\n")
        print(table(["κ", "試行数", "成功", "失敗原因", "潰れた局面"], out))
        print()

    print("### 面の耐力 k·travel と潰れ（最大圧縮に達した局面）\n")
    cbins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    for n in names:
        out = []
        for lo, hi in cbins:
            sub = [r for r in by[n] if lo <= r["object"]["stiffness"] * r["object"]["travel"] < hi]
            if not sub:
                continue
            cr = [r for r in sub if r["crushed"]]
            ph = Counter(r["bottom_phase"] for r in cr)
            g = np.mean([r["goal"] for r in sub])
            out.append([f"{lo}–{hi} N", len(sub), f"{g:.2f}", pct(len(cr), len(sub)),
                        ", ".join(f"{k} {v}" for k, v in ph.most_common()),
                        pct(sum(r["success"] for r in sub), len(sub))])
        print(f"**{n}**\n")
        print(table(["k·travel", "試行数", "目標力平均 N", "潰れ", "潰れた局面", "成功"], out))
        print()


if __name__ == "__main__":
    main()
