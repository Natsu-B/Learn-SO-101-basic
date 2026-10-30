Latest SO-101 5mm work: [2026-10-02 measured results](SO101_REACH_5MM.md).

Latest Sim-to-Real work: [2026-09-30 validation](SIM_TO_REAL_VALIDATION.md).

# Validation — 2026-09-27

Apple Silicon / aarch64-darwin, local Nix devShell. MuJoCo 3.10.0,
rusty_mujoco 0.4.0, Python 3.12.14, PyO3 0.27.2, Rust 1.98.1.
Rerun Rust/Python 0.38.1, SB3 2.9.0, torch 2.14.0, W&B 0.30.0.
MuJoCo and Menagerie pins unchanged. Earlier validation is preserved in
[PREVIOUS_VALIDATION.md](PREVIOUS_VALIDATION.md); it describes the previous implementation.

## Baseline and final checks

Before changes: fmt/clippy successful, Rust **21 passed**, Python **33 passed**.
After changes: fmt/clippy successful, Rust **28 passed**, Python **48 passed** with
Rerun disabled, and **48 passed** with Rerun enabled. Rerun SDK **1 passed** separately.
No tests start a GUI or require W&B login/network. Optional learning dependencies must
be installed for the four learning tests; absent SB3 causes that module to skip.

Coverage includes three robot models/discovery, cached nonconsecutive IDs, ranges and
invalid input, home/sine, FK target witness/reproducibility/state preservation, task
reward/success/horizon, long finite runs and control timing. Python covers Dict spaces,
wrappers, two independent simulator threads, two-environment SyncVectorEnv for each
robot, three PPO optimization/save/load/evaluation tests and mocked W&B metrics.
A Rust binding test verifies Python can be acquired on another thread while a worker
request waits. The default and optional SDK builds use the same runtime.

## Actual learning and evaluation

PPO MultiInputPolicy, CPU, seed42, pi/vf [64,64], n_steps256, batch64, epochs5,
learning rate0.0003, gamma0.99, GAE0.95, clip0.2. Delta + rate limit 0.05rad/Gym step,
30Hz control, horizon200, success threshold0.02m, bonus1. FK targets 0.05–0.25m from
initial end effector with5% joint margin. Evaluation uses the **same 20 seeds10000–10019**
for random and deterministic trained policies; no seed selection after seeing results.

| Robot | Policy | Training steps | Success rate | Mean episode reward | Mean final distance (m) |
|---|---|---:|---:|---:|---:|
| so101 | random | 0 | 0.0% | -50.322952 | 0.274416 |
| so101 | trained | 100096 | 0.0% | -27.214437 | 0.131007 |
| panda | random | 0 | 0.0% | -56.117852 | 0.409750 |
| panda | trained | 8192 | 0.0% | -51.526860 | 0.394699 |
| ur5e | random | 0 | 5.0% | -44.620555 | 0.284338 |
| ur5e | trained | 8192 | 0.0% | -48.461240 | 0.342796 |

SO-101 requested100000 and completed100096 steps (PPO full rollout), 1955 optimization
epoch updates, training time55.94s. Panda/UR5e each8192 steps, 160 updates, 5.47s/4.82s.
Panda/UR5e are **smoke training only**. SO-101 distance improved but success remained0%;
this run does not establish a learned successful reaching policy. UR5e was worse than
random. No convergence or real-hardware performance claim is made. These are single-seed
training runs and20 evaluation episodes, not statistical evidence of comparative performance.

Full episode metrics: [SO-101](results/so101-evaluation.json),
[Panda](results/panda-evaluation.json), [UR5e](results/ur5e-evaluation.json).
Local checkpoints/config/training records are under `runs/validation-so101-100k`,
`runs/validation-panda-smoke`, `runs/validation-ur5e-smoke` (gitignored).
Each robot completed env creation/reset/rollout/optimization/save/load/evaluation and
saved trained episodes to `evaluation.rrd` in its run directory.
All three recordings were inspected with `rerun rrd print -vv`; required metric, joint,
point and timeline names passed. Sizes: SO-101 3103372, Panda 3643944, UR5e 3148153 bytes.

W&B was also run with the real SDK in **offline** mode for256 steps; episode reward,
length, final distance, rolling success rate and learning rate were saved locally.
No online upload or dashboard verification was performed.

## Commands actually executed

All project build/test/training commands used `nix develop --command bash ...`
(the explicit Nix binary with `--extra-experimental-features 'nix-command flakes'`).

```bash
cargo fmt --check
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
maturin develop
pytest
# Repeated after Rust phases; Python tests repeated after binding/wrapper changes.
bash scripts/check-phase.sh
uv sync --locked --extra learning --no-install-project
maturin develop --features sim-python/rerun
pytest -q tests/python/test_learning.py
cargo clippy -p sim-cli --all-targets --features rerun,viewer
cargo clippy -p sim-python --all-targets --features rerun
cargo test -p telemetry-rerun --features sdk
bash scripts/verify.sh --learning
python examples/python/gym_random_policy.py --robot so101 --steps 1000
python examples/python/train_reach_ppo.py --robot so101 --steps 1024 --output runs/phase10-so101
python examples/python/train_reach_ppo.py --robot so101 --steps 256 --episode-steps 32 --wandb --wandb-mode offline --output runs/wandb-offline-smoke
python examples/python/eval_reach.py --robot so101 --model runs/phase10-so101 --episodes 2 --record /tmp/phase12-eval.rrd
python examples/python/train_reach_ppo.py --robot so101 --steps 100000 --output runs/validation-so101-100k
python examples/python/train_reach_ppo.py --robot panda --steps 8192 --output runs/validation-panda-smoke
python examples/python/train_reach_ppo.py --robot ur5e --steps 8192 --output runs/validation-ur5e-smoke
```

Evaluation for each above run:

```bash
python examples/python/eval_reach.py --robot so101 --model runs/validation-so101-100k --episodes 20 --record runs/validation-so101-100k/evaluation.rrd --output docs/results/so101-evaluation.json
python examples/python/eval_reach.py --robot panda --model runs/validation-panda-smoke --episodes 20 --record runs/validation-panda-smoke/evaluation.rrd --output docs/results/panda-evaluation.json
python examples/python/eval_reach.py --robot ur5e --model runs/validation-ur5e-smoke --episodes 20 --record runs/validation-ur5e-smoke/evaluation.rrd --output docs/results/ur5e-evaluation.json
```

Logs locally: `/tmp/reach-rl-baseline.log`, `/tmp/reach-rl-phase2.log` through
`/tmp/reach-rl-phase13.log` where applicable, `/tmp/reach-rl-final.log`.

## Remaining warnings and incomplete GUI verification

- Nix cc wrapper warns about target spelling `arm64-apple-macosx` versus
  `arm64-apple-darwin`. Native compilation/linking/tests passed.
- Gym checker emits9 existing recommendations: unnormalized bare SI action spaces and
  infinite observation bounds. Learning wrappers normalize policy actions; observation
  bounds remain unbounded because physics can cross soft limits slightly.
- Nix warns that the working tree is dirty; no commit was created.
- Rerun0.38.1 recording and `rerun rrd print -vv` verification pass, including all required
  metric/point entities and timelines. GUI screenshot attempt with `--screenshot-to`
  failed in wgpu: requested Surface40000x989 exceeds maximum16384, despite window-size1280x900.
  `--headless --screenshot-to` exited0 but produced no screenshot. Native UI inspection
  was blocked awaiting macOS Accessibility/Screen Recording permissions. **Viewer visual
  confirmation remains incomplete**; do not treat an RRD file as proof of correct GUI layout.

## Unsafe and next work

Two existing unsafe blocks remain: `mujoco-runtime/src/model_access.rs` calls a bounds/type
checked read-only C shim to compensate for rusty_mujoco's actuator_trnid accessor;
`simulation.rs` reads site_xpos with matching owned model/data (including scratch FK data).
The C function has one unsafe extern declaration. No unsafe Send/Sync implementation.

Next: verify Viewer interactively; investigate PPO curriculum/normalization/hyperparameters
with held-out seeds and longer runs before claiming successful Reach learning. Then add a
separate camera renderer using logical CameraSpec, test calibration/timestamps, map named
RobotObservation/action through JointCodec in a minimal LeRobot adapter, and only later add
real motor I/O and image imitation learning. FK reachability does not guarantee a collision-free path.
