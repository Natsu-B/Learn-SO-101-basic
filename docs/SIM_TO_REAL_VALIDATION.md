# Sim-to-Real validation — 2026-09-30

All commands ran locally in the existing aarch64-darwin Nix devShell. MuJoCo3.10.0,
rusty_mujoco0.4.0, Menagerie commit and native Nix pins were preserved. No real robot
was connected. All “real” test trajectories are explicitly synthetic fixtures.

## Baseline / regression

Before editing: `cargo fmt --check`, workspace clippy, workspace tests, `maturin develop`,
`pytest` all passed: Rust28, Python48, including four PPO/W&B smoke tests.
Final: Rust35 + optional Rerun SDK1; Python66, with both default and Rerun-enabled extension.
No new clippy warnings remain. Tests do not require GUI, network, W&B login or hardware.
The learning extra is installed; PPO/SAC tests are not skipped in these results.

Phases were built/tested incrementally (logs `/tmp/sim-real-baseline.log`,
`/tmp/sim-real-phase2.log` through `phase14.log`, with comparison phases combined in
`phase6-7.log`; final `/tmp/sim-real-final.log`). A Rerun Python API mismatch (`rr.flush`
removed) was caught in phase6/7 and fixed to `RecordingStream.flush` before continuing.
New clippy style/unused-import warnings were corrected before final verification.

## Actual training / robustness results

Matched nominal and moderate-DR PPO training: seed42; same MultiInputPolicy [64,64] pi/vf,
rollout256, batch64, 5epochs, learning rate0.0003, CPU, delta/rate0.05rad per30Hz control step,
200-step horizon, threshold0.02m, success bonus1. SO-10132768 steps each; Panda/UR5e2048 each.
Evaluation seed20000 onward: same20 targets for SO-101, same5 for Panda/UR5e in every condition.
All are **smoke/short training only**, not converged robust policies.

| Robot | Train DR | Train steps | Eval DR | Episodes | Success | Mean reward | Mean final distance (m) |
|---|---|---:|---|---:|---:|---:|---:|
| so101 | none | 32768 | none | 20 | 0.0% | -36.860190 | 0.187593 |
| so101 | none | 32768 | light | 20 | 0.0% | -38.661742 | 0.195481 |
| so101 | none | 32768 | moderate | 20 | 0.0% | -40.297887 | 0.229949 |
| so101 | moderate | 32768 | none | 20 | 0.0% | -57.006991 | 0.410918 |
| so101 | moderate | 32768 | light | 20 | 10.0% | -64.209466 | 0.282409 |
| so101 | moderate | 32768 | moderate | 20 | 5.0% | -44.155582 | 0.263176 |
| panda | none | 2048 | none | 5 | 0.0% | -44.122344 | 0.309913 |
| panda | none | 2048 | light | 5 | 0.0% | -42.455513 | 0.293526 |
| panda | none | 2048 | moderate | 5 | 0.0% | -39.995904 | 0.273186 |
| panda | moderate | 2048 | none | 5 | 0.0% | -47.063565 | 0.345417 |
| panda | moderate | 2048 | light | 5 | 0.0% | -44.671824 | 0.317009 |
| panda | moderate | 2048 | moderate | 5 | 0.0% | -41.182327 | 0.276277 |
| ur5e | none | 2048 | none | 5 | 0.0% | -44.163398 | 0.310156 |
| ur5e | none | 2048 | light | 5 | 0.0% | -43.308297 | 0.293475 |
| ur5e | none | 2048 | moderate | 5 | 0.0% | -42.470600 | 0.270418 |
| ur5e | moderate | 2048 | none | 5 | 0.0% | -47.481691 | 0.339996 |
| ur5e | moderate | 2048 | light | 5 | 0.0% | -46.486005 | 0.324593 |
| ur5e | moderate | 2048 | moderate | 5 | 0.0% | -45.118480 | 0.300327 |

No robustness improvement is established. SO-101 DR policy occasionally succeeds under
perturbation, but mean reward/distance are generally worse than the nominal-trained policy.
The low episode counts and single training seed do not justify statistical claims.
Panda/UR5e tests verify functional compatibility, not policy quality. No real-world transfer
or system-identification accuracy was measured.

Machine-readable full metrics, including random-policy baselines and per-episode outcomes:
`docs/results/sim-real-{so101,panda,ur5e}-{none,moderate}.json`.
Checkpoints/config/training logs are gitignored at `runs/sim-real-<robot>-<profile>/`.
One artificial identified center (damping1.05, latency10ms) was loaded, randomized with light,
and trained256 steps with actual W&B **offline** logging. Parameter means/latency and episode
metrics were recorded; no online upload was performed. SB3 SAC performed32 steps with
optimization in a separate optional smoke test; no SAC training framework was added.

## Measurement and Rerun checks

- SO-101 shoulder_pan, Panda joint1, UR5e shoulder_pan_joint: 2-second excitation at30Hz,
  amplitude0.1rad, 60samples each after settling. Saved/reloaded with correct joint dimensions.
- Identical traces: RMSE0. Synthetic shoulder_pan offset0.02rad: position RMSE0.02,
  MAE0.020000000000000004, max0.02000000000000001, estimated lag0.
- Unit fixture with artificial80ms delay: lag estimate within11ms; identity and constant-offset
  cases pass. Timestamp alignment uses interpolation and does not assume equal index/time.
- Rust recorder roundtrip also passes, with missing velocity/Cartesian fields allowed.
- Rerun comparison and DR recordings were inspected with `rerun rrd print -vv` and required
  command/position/error/parameter paths verified. No GUI was required. Viewer layout was
  not visually inspected in this update. The comparison CLI with `--rerun` was actually
  executed: Viewer spawn/data send returned exit0 without SDK error (log `/tmp/sim-real-viewer-spawn.log`).
- Rust CLI Panda `--trajectory` generated the same schema for10 steps.
- Built a wheel and inspected ZIP contents: none/light/moderate TOML profiles are included.
  This is a Nix-local wheel; maturin warns native libraries must be repaired/bundled for
  distribution outside Nix. Portable wheel distribution was not attempted.

Local artifacts: `data/sim-real-validation/` (gitignored), including a directory named
`synthetic-real` whose metadata explicitly says it is NOT hardware data.

## Commands executed

Within `nix develop` (invoked noninteractively with `nix develop --command bash -e -c ...`):

```bash
cargo fmt --check
cargo fmt
bash scripts/check-phase.sh  # fmt --all --check; clippy --workspace --all-targets; test --workspace
maturin develop
pytest
pytest -q tests/python/test_trajectory.py
pytest -q tests/python/test_learning.py
pytest -q tests/python/test_residual.py tests/python/test_learning.py
maturin develop --features sim-python/rerun
bash scripts/verify.sh --learning
```

The final verify script also runs:

```bash
uv sync --locked --extra learning --no-install-project
cargo clippy -p sim-cli --all-targets --features rerun,viewer
cargo clippy -p sim-python --all-targets --features rerun
cargo test -p telemetry-rerun --features sdk
# maturin develop + pytest before AND after enabling native Rerun
```

Training/evaluation (each robot/profile pair executed):

```bash
python examples/python/train_reach_ppo.py --robot so101 --domain-randomization moderate --steps 2048 --output runs/sim-real-phase12
python examples/python/eval_robustness.py --robot so101 --model runs/sim-real-phase12 --episodes 3 --output /tmp/sim-real-phase13.json
# Final matched runs: robot=so101/panda/ur5e, profile=none/moderate;
# steps=32768 for so101, 2048 for panda/ur5e; episodes=20 / 5 respectively.
python examples/python/train_reach_ppo.py --robot "$robot" --domain-randomization "$profile" --steps "$steps" --output "runs/sim-real-$robot-$profile"
python examples/python/eval_robustness.py --robot "$robot" --model "runs/sim-real-$robot-$profile" --episodes "$episodes" --output "docs/results/sim-real-$robot-$profile.json"
python examples/python/train_reach_ppo.py --robot so101 --domain-randomization light --identified-parameters models/identified/so101-example.json --steps 256 --wandb --wandb-mode offline --output runs/sim-real-identified-smoke
```

Measurement/integration:

```bash
python examples/python/record_identification_trajectory.py --duration 1 --output /tmp/sim-real-sine
python examples/python/compare_trajectories.py --sim /tmp/sim-real-sine --real /tmp/sim-real-sine --record /tmp/sim-real-comparison.rrd
# executed for three robot/joint combinations listed above:
python examples/python/record_identification_trajectory.py --robot "$robot" --joint "$joint" --duration 2 --amplitude .1 --output "data/sim-real-validation/$robot"
python examples/python/compare_trajectories.py --sim data/sim-real-validation/so101 --real data/sim-real-validation/synthetic-real --record data/sim-real-validation/comparison.rrd
cargo run -p sim-cli -- --robot panda --steps 10 --trajectory data/sim-real-validation/cli-panda
maturin build --out /tmp/sim-real-wheels
rerun rrd print -vv data/sim-real-validation/comparison.rrd
rerun rrd print -vv data/sim-real-validation/domain.rrd
python examples/python/compare_trajectories.py --sim data/sim-real-validation/so101 --real data/sim-real-validation/synthetic-real --rerun
```

## Warnings / safety boundaries

Remaining pre-existing warnings: Nix cc wrapper target spelling and nine Gym checker Box
recommendations (bare SI action and unbounded observations); dirty working tree warning.
New code compiles/clippy-checks without additional Rust warnings.

Four unsafe blocks, all in mujoco-runtime:

1. model_access.rs::actuator_joint: existing checked C transmission lookup.
2. model_access.rs::parameters: reads seven values into a sized output buffer; C validates
   actuator/hinge/DOF indices, retains no pointer.
3. model_access.rs::set_parameters: exclusive mutable model access at reset; finite seven-value
   buffer, bounds/type checks, no changes to allocation/topology; data reset before stepping.
4. simulation.rs::site_position: rusty_mujoco site_xpos requires unsafe; model and data are
   simulator-owned matching allocations, including scratch FK data.

Two unsafe extern declaration blocks declare the C functions. No unsafe Send/Sync.
The physics loop remains single-owner worker code with GIL-detached requests.

Unmodeled: mass/inertia/payload variation, geom/contact friction randomization, backlash,
deadband, quantization/noise, thermal/voltage effects, packet jitter/loss, clock drift, cameras.
Range/latency caps are input guards, not real-hardware safety certification.

Next real SO-101 step: a minimal calibrated RobotIo adapter with explicit unit mapping,
command/sensor monotonic timestamps, stop/limits and shared trajectory export. Collect low-amplitude
responses, compare them, estimate a parameter JSON externally, validate held-out trajectories,
then retrain around that center. No actual driver, hardware reset, optimizer or residual RL
was implemented in this update.

## Files changed / added

- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
- `README.md`
- `configs/domain_randomization/so101-moderate.toml`
- `crates/mujoco-runtime/Cargo.toml`
- `crates/mujoco-runtime/src/lib.rs`
- `crates/mujoco-runtime/src/model_access.c`
- `crates/mujoco-runtime/src/model_access.rs`
- `crates/mujoco-runtime/src/presets.rs`
- `crates/mujoco-runtime/src/randomization.rs`
- `crates/mujoco-runtime/src/simulation.rs`
- `crates/mujoco-runtime/tests/latency.rs`
- `crates/mujoco-runtime/tests/randomization.rs`
- `crates/mujoco-runtime/tests/reach.rs`
- `crates/mujoco-runtime/tests/validation.rs`
- `crates/robotics-core/Cargo.toml`
- `crates/robotics-core/src/controller.rs`
- `crates/robotics-core/src/lib.rs`
- `crates/robotics-core/src/spec.rs`
- `crates/robotics-core/src/task.rs`
- `crates/robotics-core/src/trajectory.rs`
- `crates/sim-cli/Cargo.toml`
- `crates/sim-cli/src/main.rs`
- `crates/sim-python/Cargo.toml`
- `crates/sim-python/src/lib.rs`
- `crates/sim-python/src/worker.rs`
- `crates/trajectory-io/Cargo.toml`
- `crates/trajectory-io/src/lib.rs`
- `docs/ARCHITECTURE.md`
- `docs/SIM_TO_REAL.md`
- `docs/SIM_TO_REAL_VALIDATION.md`
- `docs/VALIDATION.md`
- `docs/results/sim-real-panda-moderate.json`
- `docs/results/sim-real-panda-none.json`
- `docs/results/sim-real-so101-moderate.json`
- `docs/results/sim-real-so101-none.json`
- `docs/results/sim-real-ur5e-moderate.json`
- `docs/results/sim-real-ur5e-none.json`
- `examples/python/compare_trajectories.py`
- `examples/python/eval_reach.py`
- `examples/python/eval_robustness.py`
- `examples/python/record_identification_trajectory.py`
- `examples/python/train_reach_ppo.py`
- `models/identified/README.md`
- `models/identified/so101-example.json`
- `python/robot_sim/__init__.py`
- `python/robot_sim/env.py`
- `python/robot_sim/factory.py`
- `python/robot_sim/learning.py`
- `python/robot_sim/profiles/light.toml`
- `python/robot_sim/profiles/moderate.toml`
- `python/robot_sim/profiles/none.toml`
- `python/robot_sim/randomization.py`
- `python/robot_sim/trajectory.py`
- `python/robot_sim/wrappers.py`
- `tests/python/test_learning.py`
- `tests/python/test_randomization.py`
- `tests/python/test_residual.py`
- `tests/python/test_simulator.py`
- `tests/python/test_trajectory.py`
