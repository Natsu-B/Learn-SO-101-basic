"""Summarize eval_place_variants.py episode lines (docs/PHYSICS_AUDIT.md §4).

Per variant: overall and Place success, failures split into "never placed"
(the cube did not come to rest at the goal after release) and "disturbed"
(it was placed, then moved by a jaw while opening or retreating), jaw
re-contact after release, release height and Place success by Place-start
tilt. Usage: python scripts/summarize_place_variants.py FILE.jsonl [...]
"""
import collections
import json
import sys

import numpy as np


def main(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    rows = [r for r in rows if r.get("type") == "episode"]
    order = list(dict.fromkeys(r["variant"] for r in rows))
    print("| variant | episodes | success | reached Place | Place ok | fail: never placed |"
          " fail: disturbed | jaw re-contact > 1 N | release height mm | Place ok by start tilt"
          " <3° / 3–6° / ≥6° |")
    print("|---" * 10 + "|")
    for v in order:
        rs = [r for r in rows if r["variant"] == v]
        reached = [r for r in rs if r["reached_place"]]
        ok = [r for r in reached if r["place_success"]]
        never = sum(not r["place_success"] and not r["placed_after_release"] for r in reached)
        disturbed = sum(not r["place_success"] and r["placed_after_release"] for r in reached)
        recontact = sum(r["recontact_max_n"] > 1.0 for r in reached)
        heights = [r["release_height_mm"] for r in reached if r["release_height_mm"] is not None]
        bins = collections.OrderedDict((k, [0, 0]) for k in ("<3", "3-6", ">=6"))
        for r in reached:
            t = r["place_start_tilt_deg"]
            k = "<3" if t < 3 else "3-6" if t < 6 else ">=6"
            bins[k][0] += r["place_success"]
            bins[k][1] += 1
        tilt = " / ".join(f"{a}/{b}" for a, b in bins.values())
        print(f"| {v} | {len(rs)} | {sum(r['success'] for r in rs)} | {len(reached)} | {len(ok)} |"
              f" {never} | {disturbed} | {recontact} |"
              f" {np.mean(heights) if heights else float('nan'):.1f} | {tilt} |")
    same = all(r.get("same_place_start", True) for r in rows)
    print(f"\nidentical Place-start state across variants for every seed: {same}")


if __name__ == "__main__":
    main(sys.argv[1:])
