"""Tabulate scripts/eval_grasp_comparison.sh outputs: one row per condition,
one column group per controller (grasp / hold / full success out of N)."""
import json
import sys
from pathlib import Path

REGIONS = ("easy", "hard", "out_of_range")


def load(directory, name):
    rows = {}
    for region in REGIONS:
        path = Path(directory) / f"{name}-{region}-summary.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                rows[(region, d["mass"], d["friction"])] = d
    return rows


def main():
    directory, names = sys.argv[1], sys.argv[2:]
    data = {n: load(directory, n) for n in names}
    keys = sorted({k for d in data.values() for k in d},
                  key=lambda k: (REGIONS.index(k[0]), k[1], k[2]))
    print("| region | mass kg | friction | " + " | ".join(
        f"{n} grasp/hold/all" for n in names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    totals = {n: {r: [0, 0, 0, 0] for r in REGIONS} for n in names}
    for k in keys:
        cells = []
        for n in names:
            d = data[n].get(k)
            if d is None:
                cells.append("—")
                continue
            e = d["episodes"]
            g, h, s = (round(d[x] * e) for x in ("grasp_rate", "hold_rate", "success_rate"))
            cells.append(f"{g}/{h}/{s}")
            t = totals[n][k[0]]
            t[0] += g; t[1] += h; t[2] += s; t[3] += e
        print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")
    for r in REGIONS:
        cells = []
        for n in names:
            g, h, s, e = totals[n][r]
            cells.append(f"**{g}/{h}/{s}** of {e}" if e else "—")
        print(f"| **{r} total** | | | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
