"""Is the choice of fixed wrist_roll offset condition-dependent beyond noise?

For each condition, the spread of success counts across the offsets is
compared with the spread expected if every offset had the same success
probability (pooled over offsets): parametric bootstrap of the summed
across-offset variance (chi-square-like statistic). Also reports, per
environment, the gain of the best offset per condition over the single best
offset (in-sample, i.e. optimistic) and its null distribution.

Every offset runs the same seeds, so outcomes are correlated; the paired
test permutes the offset labels within each (condition, seed) and is the
one to rely on (the independent-binomial version overstates the noise).

Usage: python scripts/wrist_noise_test.py runs/wrist-sweep [BLOCK]
"""
import collections
import glob
import json
import os
import sys

import numpy as np


def main(directory, block="4000000", draws=20000):
    rng = np.random.default_rng(0)
    data = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(directory, f"*-{block}-*-summary.jsonl")):
        env = os.path.basename(path).split(f"-{block}-")[0]
        for line in open(path):
            r = json.loads(line)
            if r["controller"].startswith("wrist") and r["controller"].endswith("-base"):
                data[env][(r["region"], r["mass"], r["friction"], r["controller"])] = r
    for env, cells in sorted(data.items()):
        controllers = sorted({k[3] for k in cells})
        if len(controllers) < 2:
            continue
        conds = sorted(c for c in {k[:3] for k in cells}
                       if all(c + (k,) in cells for k in controllers))
        print(f"\n## {env}, seeds {block}+, {len(controllers)} offsets, {len(conds)} conditions")
        for metric in ("grasp", "success"):
            obs = np.array([[cells[c + (k,)][metric] for k in controllers] for c in conds])
            n = 30
            p = obs.mean(axis=1, keepdims=True) / n
            stat = ((obs - obs.mean(axis=1, keepdims=True)) ** 2).sum()
            sims = rng.binomial(n, np.broadcast_to(p, (draws,) + obs.shape))
            null = ((sims - sims.mean(axis=2, keepdims=True)) ** 2).sum(axis=(1, 2))
            pval = (null >= stat).mean()
            gain = obs.max(axis=1).sum() - obs.sum(axis=0).max()
            null_gain = sims.max(axis=2).sum(axis=1) - sims.sum(axis=1).max(axis=1)
            print(f"{metric}: across-offset variance {stat:.0f} vs null mean {null.mean():.0f} "
                  f"(p = {pval:.3f}); per-condition best minus single best = {gain} "
                  f"(null mean {null_gain.mean():.1f}, 95% {np.percentile(null_gain, 95):.0f})")


def paired(directory, block="4000000", draws=5000):
    rng = np.random.default_rng(1)
    outcome = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(directory, f"*-{block}-*-wrist*.jsonl")):
        if path.endswith("-summary.jsonl"):
            continue
        env = os.path.basename(path).split(f"-{block}-")[0]
        for line in open(path):
            r = json.loads(line)
            if not r["controller"].endswith("-base"):
                continue
            key = (r["region"], r["condition_mass"], r["condition_friction"], r["seed"])
            outcome[env].setdefault(key, {})[r["controller"]] = (r["grasp"], r["success"])
    for env, rows in sorted(outcome.items()):
        controllers = sorted({c for v in rows.values() for c in v})
        keys = sorted(k for k, v in rows.items() if len(v) == len(controllers))
        conds = sorted({k[:3] for k in keys})
        print(f"\n## paired permutation, {env}: {len(conds)} conditions x seeds, "
              f"{len(controllers)} offsets")
        for m, metric in enumerate(("grasp", "success")):
            x = np.array([[rows[k][c][m] for c in controllers] for k in keys], float)
            cidx = np.array([conds.index(k[:3]) for k in keys])

            def stat(arr):
                counts = np.zeros((len(conds), arr.shape[1]))
                np.add.at(counts, cidx, arr)
                return ((counts - counts.mean(axis=1, keepdims=True)) ** 2).sum(), counts

            observed, counts = stat(x)
            null = np.empty(draws)
            for i in range(draws):
                perm = np.argsort(rng.random(x.shape), axis=1)
                null[i] = stat(np.take_along_axis(x, perm, axis=1))[0]
            gain = counts.max(axis=1).sum() - counts.sum(axis=0).max()
            print(f"{metric}: statistic {observed:.0f}, null mean {null.mean():.0f}, "
                  f"p = {(null >= observed).mean():.3f}; totals per offset "
                  f"{dict(zip([c[5:10] for c in controllers], counts.sum(axis=0).astype(int)))}; "
                  f"per-condition best - single best = {gain:.0f} (in-sample)")


if __name__ == "__main__":
    main(*sys.argv[1:])
    paired(*sys.argv[1:])
