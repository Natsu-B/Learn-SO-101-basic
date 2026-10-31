#!/usr/bin/env bash
# Final Grasp comparison under hidden mass/friction (docs/OBJECT_PHYSICS.md).
# Same fixed conditions and evaluation seeds (3,000,000+, disjoint from the
# checkpoint-selection seeds 4,000,000+) for every controller; Transport and
# Place are scripted IK. Run inside `nix develop` after `maturin develop`.
# Usage: scripts/eval_grasp_comparison.sh OUT_DIR CONTROLLER_NAME [eval args...]
#   e.g. scripts/eval_grasp_comparison.sh runs/grasp-eval ik
#        scripts/eval_grasp_comparison.sh runs/grasp-eval wrist --fixed-wrist-roll -0.1
#        scripts/eval_grasp_comparison.sh runs/grasp-eval h0-seed-42 \
#            --grasp-model runs/so101-grasp/h0-seed-42/best.zip
set -euo pipefail
out=$1; name=$2; shift 2
mkdir -p "$out"
episodes=${EPISODES:-30}
seed=${SEED:-3000000}
# E: the originally specified easy range (regression check).
easy=(0.02 0.8 0.02 1.5 0.05 0.8 0.05 1.5 0.035 1.15)
# H: hard conditions inside the training range (mass 0.02-0.1, friction 0.05-1.5).
hard=(0.02 0.05 0.02 0.1 0.05 0.05 0.05 0.1 0.1 0.05 0.1 0.1 0.03 0.07 0.07 0.07 0.03 0.15 0.07 0.15)
# O: outside the training range.
out_of_range=(0.2 0.05 0.2 0.1 0.02 0.03 0.05 0.03)
run() {
  local region=$1; shift
  python examples/python/eval_object_physics.py --episodes "$episodes" --seed "$seed" \
    --label "$region" "$@" \
    > "$out/$name-$region.jsonl" 2> "$out/$name-$region-summary.jsonl"
}
run easy --conditions "${easy[@]}" "$@"
run hard --conditions "${hard[@]}" "$@"
run out_of_range --conditions "${out_of_range[@]}" "$@"
