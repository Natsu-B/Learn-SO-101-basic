"""Hand-off states per controller from eval_object_physics.py episode logs.

For each NAME, reads DIR/NAME-REGION.jsonl and reports, for the end of each
skill (Grasp, Stabilize, Transport = Place start, Place), the mean height
above rest (mm), linear speed (m/s), angular speed (rad/s) and cube tilt
(deg), split by full success; plus overall counts.
Usage: summarize_handoff.py DIR REGION NAME [NAME ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = (("height_m", 1e3, "h mm"), ("speed_mps", 1.0, "v m/s"),
          ("angular_speed_rps", 1.0, "w rad/s"), ("tilt_deg", 1.0, "tilt deg"))


def main():
    directory, region, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    for name in names:
        rows = [json.loads(line) for line in (directory / f"{name}-{region}.jsonl").read_text()
                .splitlines() if line.startswith("{")]
        ok = sum(r["success"] for r in rows)
        print(f"## {name} ({region}): success {ok}/{len(rows)}")
        print("| skill end | outcome | n | " + " | ".join(f[2] for f in FIELDS) + " |")
        print("|---|---|---|" + "---|" * len(FIELDS))
        for skill in ("grasp", "stabilize", "transport", "place"):
            for label, group in (("success", [r for r in rows if r["success"]]),
                                 ("failure", [r for r in rows if not r["success"]])):
                ends = [e for r in group for e in r["skill_end_states"][-4:]
                        if e["name"] == skill]
                if not ends:
                    continue
                cells = [f"{np.mean([e[k] for e in ends]) * s:.3g}" for k, s, _ in FIELDS]
                print(f"| {skill} | {label} | {len(ends)} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
