#!/usr/bin/env bash
# Phase 2/3 grid (docs/PHYSICS_AUDIT.md): fixed wrist_roll offsets and IK over
# the standard conditions, both contact environments, selection (4,000,000+)
# and final (3,000,000+) seed blocks. One job per (env, block, region,
# controller); at most JOBS processes. Finished jobs are skipped on rerun. Run inside `nix develop`.
# Usage: scripts/run_wrist_sweep.sh OUT_DIR JOBS selection|final
set -euo pipefail
out=$1; jobs=$2; stage=$3
mkdir -p "$out"
easy="0.02 0.8 0.02 1.5 0.05 0.8 0.05 1.5 0.035 1.15"
hard="0.02 0.05 0.02 0.1 0.05 0.05 0.05 0.1 0.1 0.05 0.1 0.1 0.03 0.07 0.07 0.07 0.03 0.15 0.07 0.15"
out_of_range="0.2 0.05 0.2 0.1 0.02 0.03 0.05 0.03"
wrists="-0.20 -0.15 -0.10 -0.05 0.00 0.05 0.10"
# selection: every offset on the selection seeds; final: IK and the offsets in
# FINAL_WRISTS (chosen from the selection block) on the final seeds.
list() {
  case $stage in
    selection)
      for region in easy hard out_of_range; do
        for env in legacy pairwise-same; do
          for w in $wrists; do echo "$env 4000000 $region wrist $w"; done
        done
        echo "pairwise-nominal 4000000 $region ik -"
      done ;;
    final)
      for region in easy hard out_of_range; do
        for env in legacy pairwise-same pairwise-nominal; do
          echo "$env 3000000 $region ik -"
          for w in ${FINAL_WRISTS:-}; do echo "$env 3000000 $region wrist $w"; done
        done
        # The original baseline hand-off (end at the 0.3 s lift), for continuity.
        echo "legacy 3000000 $region wrist-success -0.10"
      done ;;
  esac
}
run() {
  local env=$1 block=$2 region=$3 kind=$4 value=$5
  local model=${env%%-*} floor=${env#*-}
  [ "$model" = legacy ] && floor=same
  local conds=${!region}
  local args=(--episodes 30 --seed "$block" --label "$region" --contact-model "$model"
              --floor "$floor" --conditions $conds)
  local name
  case $kind in
    wrist) args+=(--wrist="$value"); name=wrist$value ;;
    wrist-success) args+=(--wrist="$value" --end-at success); name=wrist$value-success ;;
    ik) args+=(--ik); name=ik ;;
  esac
  local f="$out/$env-$block-$region-$name"
  # Done when every condition has its summary line.
  local n=$(( $(wc -w <<< "$conds") / 2 ))
  [ -f "$f-summary.jsonl" ] && [ "$(wc -l < "$f-summary.jsonl")" -ge "$n" ] && return 0
  python examples/python/eval_wrist_sweep.py "${args[@]}" > "$f.jsonl" 2> "$f-summary.jsonl"
}
export -f run
export out easy hard out_of_range
list | xargs -P "$jobs" -L 1 bash -c 'run "$@"' _
