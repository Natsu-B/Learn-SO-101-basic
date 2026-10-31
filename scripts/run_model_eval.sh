#!/usr/bin/env bash
# Final-seed (3,000,000+) evaluation of Grasp checkpoints in the standard 19
# conditions, legacy and pairwise (floor=same) contacts, same protocol as
# scripts/run_wrist_sweep.sh. Run inside `nix develop`.
# Usage: scripts/run_model_eval.sh OUT_DIR JOBS MODEL_NAME... (runs/so101-grasp/NAME/best.zip)
set -euo pipefail
out=$1; jobs=$2; shift 2
export out
export easy="0.02 0.8 0.02 1.5 0.05 0.8 0.05 1.5 0.035 1.15"
export hard="0.02 0.05 0.02 0.1 0.05 0.05 0.05 0.1 0.1 0.05 0.1 0.1 0.03 0.07 0.07 0.07 0.03 0.15 0.07 0.15"
export out_of_range="0.2 0.05 0.2 0.1 0.02 0.03 0.05 0.03"
for env in legacy pairwise-same; do for m in "$@"; do for region in easy hard out_of_range; do
  echo "$env $m $region"; done; done; done | xargs -P "$jobs" -L 1 bash -c '
  env=$0; m=$1; region=$2; model=${env%%-*}; conds=${!region}
  f=$out/$env-3000000-$region-$m
  [ -s "$f-summary.jsonl" ] && [ "$(wc -l < "$f-summary.jsonl")" -ge $(( $(wc -w <<< "$conds") / 2 )) ] && exit 0
  python examples/python/eval_wrist_sweep.py --episodes 30 --seed 3000000 --label "$region" \
    --contact-model "$model" --floor same --conditions $conds \
    --grasp-model "runs/so101-grasp/$m/best.zip" > "$f.jsonl" 2> "$f-summary.jsonl"'
