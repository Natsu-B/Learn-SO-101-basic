# Reach hold evaluation

Implemented in `python/robot_sim/precision.py` (`HoldTracker`, `rollout`,
`evaluate_goal`, `log_selected`) and `examples/python/eval_reach.py`.
No changes to Rust physics, policy weights, HER training or action scaling.

## Semantics

`--hold-seconds 0.5` keeps the deterministic policy active after first reaching.
A continuous sequence of in-range control endpoints spanning at least 0.5
simulation seconds is required. An out-of-range endpoint resets the timer.
The existing 100-control-step deadline includes both approach and hold. A
successful last endpoint alone is insufficient. `--hold-seconds 0` preserves
first-entry termination. Both random and trained policies use the same rule.

The test uses native true end-effector distance and MuJoCo simulation timestamps,
not delayed policy observation or host clock. It samples control endpoints (~30Hz),
so excursions between sampled endpoints are not detected. This is not a physical
continuous-time safety guarantee. Rerun replay follows the same hold rule.

Per-episode data includes first reach time (episode simulation time), longest
continuous in-range duration, hold success and mean/max error after first reach.
Post-reach statistics include excursions and retries; they are not limited to the
final successful window. Aggregate `success_rate` means hold success when holding
is enabled; `reach_success_rate` remains the first-entry rate. `within_m` describes
final endpoint distances, and is not a hold-success statistic.

## Actual evaluation

Frozen checkpoint: `runs/so101-reach-5mm/base-policy/best.zip`, SAC+HER,
425,136 training steps. No additional learning performed. Nominal SO-101,
500 new targets per evaluation, seeds 3,000,000–3,000,499, 5–25cm initial distance.
Each run also evaluated a random-policy baseline on the same seeds.

| Threshold | First reach | Hold >=0.5s | Mean final error | P95 final error |
|---|---:|---:|---:|---:|
| 5mm | 479/500 (95.8%) | 354/500 (70.8%) | 4.342mm | 9.932mm |
| 2mm | 330/500 (66.0%) | 50/500 (10.0%) | 5.181mm | 10.318mm |

Random-policy hold success: 0/500 for both thresholds.
Raw episode outcomes: `docs/results/reach-hold-5mm.json` and
`docs/results/reach-hold-2mm.json`. Each run stops on completed hold or timeout;
therefore final errors are not fixed-horizon steady-state estimates. The two
thresholds produce different stop times. The old 96.8% 5mm first-entry result
used different seeds and cannot be directly treated as a paired comparison.

These results expose a remaining policy stability problem. Adding a hold
criterion does not itself train a better holding controller.

## Commands and regression checks

Before changes: `nix develop --command sh -c 'cargo fmt --check && cargo clippy
--workspace --all-targets && cargo test --workspace && maturin develop && pytest -q'`
passed (Rust 37 tests; Python 83 tests).

Evaluation:

```bash
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/base-policy/best.zip \
  --episodes 500 --seed 3000000 --hold-seconds 0.5 \
  --output docs/results/reach-hold-5mm.json
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/base-policy/best.zip \
  --episodes 500 --seed 3000000 --threshold 0.002 --hold-seconds 0.5 \
  --output docs/results/reach-hold-2mm.json
```

After changes: `cargo fmt --check`, `cargo clippy --workspace --all-targets`,
`cargo test --workspace`, `pytest -q`, all inside Nix. Hold regression tests
include timer reset, exact boundary, zero duration, invalid configuration,
summary semantics and successful/timeout rollouts on SO-101/Panda/UR5e.

Final regression result: Rust 37 tests and Python 95 tests passed. Existing
warnings remain: dirty Git tree, Nix C compiler target naming and nine Gymnasium
checker warnings for the legacy absolute action/unbounded observation spaces.
No new unsafe blocks or dependencies were added.

## Holding fine-tune (actual subsequent run)

The preserved base policy was warm-started into `runs/so101-reach-hold`.
SAC+HER trained for **100,032 additional transitions**, then stopped at the
configured >=95% validation hold-success criterion (500 validation episodes,
seeds 1,000,000–1,000,499). Validation reached 97.8%. Runtime including validation
and saves was 127.825 seconds. Original pretraining was 425,136 transitions,
so the lineage contains 525,168 transitions. No domain randomization was used.

Training retains joint-delta .05rad, command rate limit .1rad/control-step,
30Hz control, .005s physics, 100-step episodes and network [128,128].
Training reward is `-distance/0.1 + 0.2 * (distance <= 0.003)` each step.
Training stays fixed-horizon, so maintaining the target is rewarded repeatedly
and HER can recompute rewards for replacement goals. This does not introduce a
stateful holding termination into HER. No privileged FK witness or simulator
parameter was added to observation.

Learning rate .0003, replay capacity 300,000, batch 256, gamma .98, tau .005,
train frequency 16, gradient steps 4, learning starts 1,000, training seed 73.
Fresh replay avoids mixing the old reward into the new reward problem.
Warm-start load now explicitly applies requested SAC parameters instead of
silently retaining the source checkpoint's learning-rate schedule. Weights and
optimizer state are retained; new training transition count starts at zero.
A regression test verifies the loaded schedule and new goal-recomputable reward.

### Independent paired comparison

Final comparison uses seeds **4,000,000–4,000,499**, which were not used for
training or checkpoint selection. Both policies use identical goals, unchanged
100-step horizon and the same .5s holding condition. Final evaluations were
performed after selecting the checkpoint; no subsequent selection/training used
these outcomes.

| Criterion | Base | Fine-tuned |
|---|---:|---:|
| 5mm first reach | 484/500 (96.8%) | 499/500 (99.8%) |
| 5mm hold >=0.5s | 365/500 (73.0%) | 490/500 (98.0%) |
| 2mm first reach | 350/500 (70.0%) | 450/500 (90.0%) |
| 2mm hold >=0.5s | 51/500 (10.2%) | 351/500 (70.2%) |
| 5mm-run mean final error | 4.165mm | 2.063mm |
| 5mm-run median final error | 3.475mm | 1.755mm |
| 5mm-run P95 final error | 9.240mm | 4.115mm |

Random policy holds succeeded in 0/500 episodes for all four evaluations.
Raw results: `reach-hold-base-4m.json`, `reach-hold-trained-4m.json`,
`reach-hold-base-2mm-4m.json`, `reach-hold-trained-2mm-4m.json` in `docs/results`.
Training configuration and validation history: `docs/results/reach-hold-training.json`.
Episode reward values should not be compared directly between these checkpoints:
the fine-tuned configuration includes a success bonus, unlike the original.
Comparison above uses geometric distance and holding outcomes.

Checkpoint: `runs/so101-reach-hold/best.zip`; resumable state:
`latest.zip`, `latest-replay.pkl`, `training_state.json`, `config.json`.
Original base files were not overwritten. SHA256:

- Base: `27e4eee989a6c21cf70280bffd0f797366b04560b825d83c20002a56a2f3b5ca`
- Holding policy: `c3c755735460b238c748b599d97d34ab59f8328bc77db7cd05d29b950e6ef432`

Rerun records for five best/five worst 5mm holding episodes are under
`runs/so101-reach-hold/evaluation-rerun`; all ten .rrd files were written and
verified to exist. No GUI inspection or real-robot performance is claimed.

```bash
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-hold/best.zip \
  --episodes 500 --seed 4000000 --hold-seconds 0.5 \
  --record runs/so101-reach-hold/evaluation-rerun \
  --output docs/results/reach-hold-trained-4m.json
# Add --threshold 0.002 for 2mm, or --rerun to open the Viewer.
```

Regression commands after fine-tune support changes (inside Nix):
`cargo fmt --check`, `cargo clippy --workspace --all-targets`,
`cargo test --workspace`: passed, Rust 37 tests. `pytest -q`: 96 passed with
Rerun disabled; `maturin develop --features sim-python/rerun` then `pytest -q`:
96 passed with Rerun enabled. The same nine existing Gymnasium warnings remain.
No unsafe blocks, native dependencies, or physical model parameters changed.

Remaining limitation: the fine-tuned policy is not yet >=90% at **2mm holding**,
even though its 2mm first-entry rate is 90%. Hold validation samples control
endpoints, and all improvements reported here are nominal simulation only.
