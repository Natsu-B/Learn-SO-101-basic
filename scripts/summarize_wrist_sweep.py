"""Tables for the fixed wrist_roll sweep (docs/PHYSICS_AUDIT.md §3).

Reads OUT_DIR/*-summary.jsonl from scripts/run_wrist_sweep.sh and prints, per
environment and seed block: Grasp / held / overall success per condition and
controller, region totals, the best offset per condition (most overall
successes, ties -> more Grasp successes, then the offset closest to 0), and the
mean pose metrics per controller.

Usage: python scripts/summarize_wrist_sweep.py runs/wrist-sweep [BLOCK]
"""
import collections
import glob
import json
import os
import re
import sys

REGIONS = ("easy", "hard", "out_of_range")


def load(directory, block):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*-summary.jsonl"))):
        env = os.path.basename(path).split("-4000000-")[0].split("-3000000-")[0]
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("type") != "summary" or str(r["first_seed"]) != block:
                    continue
                r["env"] = env
                rows.append(r)
    return rows


def offset(name):
    m = re.match(r"wrist([+-]\d+\.\d+)", name)
    return float(m.group(1)) if m else 0.0


def controller_key(name):
    if name == "ik":
        return (-1, 0.0, name)
    if name.startswith("wrist"):
        return (0, offset(name), name)
    return (1, 0.0, name)


def main(directory, block="4000000"):
    rows = load(directory, block)
    by_env = collections.defaultdict(list)
    for r in rows:
        by_env[r["env"]].append(r)
    for env, rs in sorted(by_env.items()):
        controllers = sorted({r["controller"] for r in rs}, key=controller_key)
        print(f"\n### {env}, seeds {block}+ (Grasp / held / overall success, of 30)\n")
        print("| region | mass | friction | " + " | ".join(controllers) + " | best |")
        print("|---" * (len(controllers) + 4) + "|")
        cells = {(r["region"], r["mass"], r["friction"], r["controller"]): r for r in rs}
        conditions = sorted({(r["region"], r["mass"], r["friction"]) for r in rs},
                            key=lambda c: (REGIONS.index(c[0]), c[1], c[2]))
        totals = collections.defaultdict(lambda: [0, 0, 0, 0])
        best_counts = collections.Counter()
        for cond in conditions:
            line, scored = [], []
            for c in controllers:
                r = cells.get(cond + (c,))
                if r is None:
                    line.append("—")
                    continue
                line.append(f'{r["grasp"]}/{r["held"]}/{r["success"]}')
                t = totals[(cond[0], c)]
                t[0] += r["grasp"]; t[1] += r["held"]; t[2] += r["success"]; t[3] += r["episodes"]
                if c.startswith("wrist") and c.endswith("-base"):
                    scored.append((r["success"], r["grasp"], -abs(offset(c)), c))
            best = max(scored)[3] if scored else "—"
            if scored:
                top = max(scored)[:2]
                ties = [s[3] for s in scored if s[:2] == top]
                best_counts.update(ties)
                best = ", ".join(f"{offset(t):+.2f}" for t in ties)
            print(f"| {cond[0]} | {cond[1]} | {cond[2]} | " + " | ".join(line) + f" | {best} |")
        for region in REGIONS + ("all",):
            parts = []
            for c in controllers:
                if region == "all":
                    t = [sum(totals[(g, c)][i] for g in REGIONS) for i in range(4)]
                else:
                    t = totals[(region, c)]
                parts.append(f"**{t[0]}/{t[1]}/{t[2]}** of {t[3]}" if t[3] else "—")
            print(f"| **{region} total** | | | " + " | ".join(parts) + " | |")
        print("\nconditions where each offset is (co-)best:",
              dict(sorted(best_counts.items(), key=lambda kv: controller_key(kv[0]))))
        print("\nmean metrics per controller (episode-weighted over conditions):\n")
        keys = ("table_push_mm", "grasp_slip_mm", "max_lifted_tilt_deg", "grasp_end_tilt_deg",
                "place_start_tilt_deg", "place_start_tilt_success_deg",
                "place_start_tilt_failure_deg")
        print("| controller | region | " + " | ".join(keys) + " | final XY err (success) mm |")
        print("|---" * (len(keys) + 3) + "|")
        for c in controllers:
            for region in REGIONS:
                sel = [r for r in rs if r["controller"] == c and r["region"] == region]
                vals = []
                for k in keys:
                    v = [r[k] for r in sel if r.get(k) is not None]
                    vals.append(f"{sum(v) / len(v):.2f}" if v else "—")
                e = [r["final_xy_error_success_mm"]["mean"] for r in sel
                     if r["final_xy_error_success_mm"]["mean"] == r["final_xy_error_success_mm"]["mean"]]
                vals.append(f"{sum(e) / len(e):.2f}" if e else "—")
                print(f"| {c} | {region} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
