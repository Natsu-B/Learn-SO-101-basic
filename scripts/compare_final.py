"""Phase 7 table: IK, fixed offsets, v3 and low-dimensional residual SAC on the
final seeds (3,000,000+), both contact environments (docs/PHYSICS_AUDIT.md §7).

Totals per region (Grasp / held / overall success) and, for each controller,
an exact two-sided sign test (McNemar) of overall success against a reference
controller on the same (condition, seed) episodes.
Usage: python scripts/compare_final.py runs/wrist-sweep [REFERENCE]
"""
import collections
import glob
import json
import math
import os
import sys

REGIONS = ("easy", "hard", "out_of_range")
NAMES = {"ik": "IK", "wrist-0.15-base": "IK + wrist −0.15 (hand-off after Hold)",
         "wrist-0.05-base": "IK + wrist −0.05 (hand-off after Hold)",
         "wrist-0.10-success": "IK + wrist −0.10 (hand-off at 0.3 s lift, original)",
         "wrist+0.00-success": "IK + wrist 0 (hand-off at 0.3 s lift)",
         "runs/so101-grasp/v3-h0-seed-42/best.zip": "v3 A seed 42 (SAC, 6-D)",
         "runs/so101-grasp/lowdim-wrist-seed-42/best.zip": "low-dim SAC: wrist (1-D)",
         "runs/so101-grasp/lowdim-wrist-lift-seed-42/best.zip": "low-dim SAC: wrist+lift (3-D)"}


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(directory, reference="wrist-0.15-base"):
    episodes = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(directory, "*-3000000-*.jsonl")):
        if path.endswith("-summary.jsonl"):
            continue
        env = os.path.basename(path).split("-3000000-")[0]
        for line in open(path):
            r = json.loads(line)
            if r["controller"] not in NAMES:
                continue
            key = (r["region"], r["condition_mass"], r["condition_friction"], r["seed"])
            episodes[(env, r["controller"])][key] = r
    for env in ("legacy", "pairwise-same", "pairwise-nominal"):
        ctrls = [c for c in NAMES if (env, c) in episodes]
        if not ctrls:
            continue
        print(f"\n### {env}, seeds 3,000,000+ (Grasp / held / overall success)\n")
        print("| controller | learned | easy /150 | hard /300 | out of range /120 | overall /570 |"
              f" vs {NAMES.get(reference, reference)}: +/− (p) |")
        print("|---|---|---|---|---|---|---|")
        ref = episodes.get((env, reference), {})
        for c in ctrls:
            eps = episodes[(env, c)]
            cells = []
            for region in REGIONS:
                rs = [r for k, r in eps.items() if k[0] == region]
                n = {"easy": 150, "hard": 300, "out_of_range": 120}[region]
                cells.append(f"{sum(r['grasp'] for r in rs)}/{sum(r['held'] for r in rs)}/"
                             f"{sum(r['success'] for r in rs)}" if len(rs) == n else "—")
            total = sum(r["success"] for r in eps.values()) if len(eps) == 570 else "—"
            shared = [k for k in eps if k in ref]
            b = sum(eps[k]["success"] and not ref[k]["success"] for k in shared)
            d = sum(ref[k]["success"] and not eps[k]["success"] for k in shared)
            learned = any(s.get("learned") for r in eps.values() for s in r["skills"]
                          if s["name"] == "grasp")
            cmp = f"+{b}/−{d} ({sign_test(b, d):.2f}, n={len(shared)})" if c != reference else "—"
            print(f"| {NAMES[c]} | {'yes' if learned else 'no'} | " + " | ".join(cells)
                  + f" | {total} | {cmp} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
