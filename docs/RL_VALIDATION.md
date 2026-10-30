# Reach RL implementation validation

2026-09-27 baseline: Nix development shell; cargo fmt --check,
cargo clippy --workspace --all-targets, cargo test --workspace,
maturin develop, pytest all passed. Rust: 21 tests. Python: 33 tests.
Existing warnings: Nix cc target spelling; 9 Gymnasium Box recommendations.
Full local baseline log: /tmp/reach-rl-baseline.log.

## Implemented phases

2–9: validated home/sine, detached Python worker, CameraSpec, FK targets, action wrappers,
Rust episode configuration, Rerun0.38.1 and SyncVectorEnv. Each phase was checked with
Rust fmt/clippy/tests and relevant Python tests.
10–13: optional PPO and W&B, deterministic evaluation, three-robot training/recording.
Final: Rust28 + optional SDK1; Python48 with/without SDK. See VALIDATION.md for measured results.

## Files changed or added in this update

- `.gitignore`
- `Cargo.lock`
- `README.md`
- `crates/mujoco-runtime/src/presets.rs`
- `crates/mujoco-runtime/src/simulation.rs`
- `crates/mujoco-runtime/tests/controllers.rs`
- `crates/mujoco-runtime/tests/reach.rs`
- `crates/mujoco-runtime/tests/robots.rs`
- `crates/mujoco-runtime/tests/validation.rs`
- `crates/robotics-core/src/controller.rs`
- `crates/robotics-core/src/error.rs`
- `crates/robotics-core/src/robot.rs`
- `crates/robotics-core/src/spec.rs`
- `crates/robotics-core/src/task.rs`
- `crates/robotics-core/src/telemetry.rs`
- `crates/sim-cli/src/main.rs`
- `crates/sim-python/src/lib.rs`
- `crates/sim-python/src/worker.rs`
- `crates/so101-cli/src/main.rs`
- `crates/telemetry-rerun/Cargo.toml`
- `crates/telemetry-rerun/src/lib.rs`
- `docs/ARCHITECTURE.md`
- `docs/RL_VALIDATION.md`
- `docs/PREVIOUS_VALIDATION.md`
- `docs/VALIDATION.md`
- `docs/results/panda-evaluation.json`
- `docs/results/so101-evaluation.json`
- `docs/results/ur5e-evaluation.json`
- `examples/python/eval_reach.py`
- `examples/python/gym_random_policy.py`
- `examples/python/train_reach_ppo.py`
- `examples/rust/common/reach.rs`
- `examples/rust/sine.rs`
- `pyproject.toml`
- `python/robot_sim/__init__.py`
- `python/robot_sim/config.py`
- `python/robot_sim/env.py`
- `python/robot_sim/factory.py`
- `python/robot_sim/learning.py`
- `python/robot_sim/wrappers.py`
- `scripts/verify.sh`
- `tests/python/test_gym.py`
- `tests/python/test_learning.py`
- `tests/python/test_simulator.py`
- `tests/python/test_telemetry.py`
- `tests/python/test_vector.py`
- `tests/python/test_wrappers.py`
- `uv.lock`
