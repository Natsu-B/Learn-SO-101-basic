"""Summarize examples/rust/contact_audit.rs traces (docs/PHYSICS_AUDIT.md).

Per setting and seed, at the end of the Close phase and over Lift: cube offset
and contact normals in the gripperframe (x = approach / fingertip direction,
z = closing direction fixed -> moving jaw), jaw and table normal forces,
penetration, cube tilt and in-plane rotation, lateral acceleration, and a
Newton check (m a = sum of contact forces + m g).

Usage: python scripts/analyze_contact_audit.py runs/physics-audit/audit.jsonl
"""
import collections
import json
import sys

import numpy as np

G = np.array([0.0, 0.0, -9.81])


def frame(row):
    x = np.asarray(row["approach"])
    z = np.asarray(row["closing"])
    y = np.cross(z, x)
    return np.stack((x, y, z))  # rows: site axes in world


def yaw_in_jaws(row):
    """Angle (deg) between the cube face normal nearest the closing axis and
    that axis, projected on the table plane."""
    w, x, y, z = row["cube_quat"]
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    c = np.asarray(row["closing"])[:2]
    c /= np.linalg.norm(c) + 1e-12
    best = 90.0
    for k in range(2):
        n = r[:2, k] / (np.linalg.norm(r[:2, k]) + 1e-12)
        best = min(best, np.degrees(np.arccos(np.clip(abs(n @ c), 0, 1))))
    return best


def jaw_summary(row):
    out = {}
    f = frame(row)
    for jaw in ("fixed_jaw", "moving_jaw", "table", "other"):
        cs = [c for c in row["contacts"] if c["partner"] == jaw]
        normal_n = sum(c["frame_force"][0] for c in cs)
        force = np.sum([c["force"] for c in cs], axis=0) if cs else np.zeros(3)
        weighted = np.sum([c["frame_force"][0] * np.asarray(c["normal"]) for c in cs],
                          axis=0) if cs else np.zeros(3)
        n = weighted / (np.linalg.norm(weighted) + 1e-12)
        out[jaw] = dict(count=len(cs), normal_force=normal_n, force_world=force,
                        force_site=f @ force, normal_site=f @ n,
                        penetration=max([-c["dist"] for c in cs], default=0.0),
                        geoms=sorted({c["geom"] or f"mesh#{c['geom_id']}" for c in cs}))
    return out


def episode(rows):
    rows = sorted(rows, key=lambda r: r["t"])
    close = [r for r in rows if r["phase"] == "close"]
    lift = [r for r in rows if r["phase"] == "lift"]
    end = close[-1]
    f = frame(end)
    offset = f @ (np.asarray(end["cube_pos"]) - np.asarray(end["grasp_center"]))
    j = jaw_summary(end)
    t0 = lift[0]["t"] if lift else end["t"]
    window = [r for r in lift if r["t"] - t0 <= 0.2]
    lat = [np.linalg.norm(np.asarray(r["cube_acc"][:2])) for r in window] or [0.0]
    start_off = offset
    if window:
        last = window[-1]
        moved = frame(last) @ (np.asarray(last["cube_pos"]) - np.asarray(last["grasp_center"]))
    else:
        moved = start_off
    newton = []
    for r in rows:
        if not r["contacts"]:
            continue
        total = np.sum([c["force"] for c in r["contacts"]], axis=0) + r["mass"] * G
        ma = r["mass"] * np.asarray(r["cube_acc"][:3])
        newton.append(np.linalg.norm(ma - total) / max(np.linalg.norm(total), 1e-3))
    pen = collections.defaultdict(float)
    peak = collections.defaultdict(float)
    for r in rows:
        for c in r["contacts"]:
            pen[c["partner"]] = max(pen[c["partner"]], -c["dist"])
            peak[c["partner"]] = max(peak[c["partner"]], c["frame_force"][0])
    lift_forces = [jaw_summary(r) for r in window]
    sat = np.mean([abs(r["gripper_force"]) >= 0.99 * 2.94 for r in lift]) if lift else 0.0
    return dict(
        close_offset_mm=offset * 1e3, close_tilt_deg=np.degrees(end["tilt"]),
        close_yaw_deg=yaw_in_jaws(end), close_gripper_q=end["gripper_q"],
        close_jaw=j, lift_lateral_acc_max=max(lat),
        lift_offset_change_mm=(moved - start_off) * 1e3,
        lift_fixed_normal_mean=np.mean([s["fixed_jaw"]["normal_force"] for s in lift_forces])
        if lift_forces else 0.0,
        lift_moving_normal_mean=np.mean([s["moving_jaw"]["normal_force"] for s in lift_forces])
        if lift_forces else 0.0,
        lift_jaw_force_site=np.mean([s["fixed_jaw"]["force_site"] + s["moving_jaw"]["force_site"]
                                     for s in lift_forces], axis=0) if lift_forces else np.zeros(3),
        newton_rel_median=float(np.median(newton)) if newton else float("nan"),
        penetration_mm={k: v * 1e3 for k, v in pen.items()},
        peak_normal_n=dict(peak), gripper_saturated_lift=sat,
        close_table_slide_mm=None)


def main(path):
    traces = collections.defaultdict(list)
    status = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["type"] == "trace":
                traces[(r["setting"], r["seed"])].append(r)
            elif r["type"] == "equivalence":
                status[(r["setting"], r["seed"])] = (r["status"].startswith("Success"),
                                                      r["cube_final_max_abs_diff"])
    by_setting = collections.defaultdict(list)
    for key, rows in traces.items():
        e = episode(rows)
        e["success"], e["equivalence_diff"] = status[key]
        by_setting[key[0]].append((key[1], e))
    np.set_printoptions(precision=2, suppress=True)
    for setting, eps in by_setting.items():
        print(f"## {setting}")
        for seed, e in sorted(eps):
            j = e["close_jaw"]
            print(f"seed {seed} success={e['success']} diff={e['equivalence_diff']:.1e} "
                  f"newton_rel={e['newton_rel_median']:.3f}")
            print(f"  close end: offset(site x,y,z)={e['close_offset_mm']} mm "
                  f"tilt={e['close_tilt_deg']:.1f} deg yaw_in_jaws={e['close_yaw_deg']:.1f} deg "
                  f"gripper={e['close_gripper_q']:.3f}")
            for jaw in ("fixed_jaw", "moving_jaw", "table"):
                s = j[jaw]
                print(f"    {jaw:10s} n={s['count']:2d} N={s['normal_force']:6.2f} "
                      f"normal_site={s['normal_site']} force_site={s['force_site']} "
                      f"pen={s['penetration'] * 1e3:.3f} mm geoms={s['geoms']}")
            print(f"  lift (0.2 s): max lateral acc={e['lift_lateral_acc_max']:.1f} m/s2 "
                  f"offset change={e['lift_offset_change_mm']} mm "
                  f"N fixed/moving={e['lift_fixed_normal_mean']:.1f}/"
                  f"{e['lift_moving_normal_mean']:.1f} jaw force(site)={e['lift_jaw_force_site']}"
                  f" saturated={e['gripper_saturated_lift']:.2f}")
            print(f"  max penetration mm={ {k: round(v, 3) for k, v in e['penetration_mm'].items()} }"
                  f" peak normal N={ {k: round(v, 1) for k, v in e['peak_normal_n'].items()} }")


if __name__ == "__main__":
    main(sys.argv[1])
