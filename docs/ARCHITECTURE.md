# Architecture

`robotics-core` defines dynamic joint state/action, RobotSpec, CameraSpec, controllers,
ReachTask/TaskConfig, JointCodec and TelemetrySink. No MuJoCo, PyO3, Gymnasium or GUI dependencies.
`mujoco-runtime -> robotics-core` owns MuJoCo model/data, cached IDs, limits and stepping.
`sim-cli` and `sim-python` share that implementation. `telemetry-rerun -> robotics-core`
provides the optional Rust Rerun SDK sink. Existing so101-core/cli remain compatibility layers.
Python Gymnasium wrappers and optional SB3/W&B are outside the Rust core.

## Robot, task and I/O

RobotSpec selects MJCF, ordered controlled actuator names, default/logical end-effector sites,
logical/source CameraSpec, initial keyframe and optional explicit home position.
Home selection is explicit valid position, otherwise valid model qpos0, otherwise valid
simulator initial state. Sine varies only a caller-selected joint around this home.
All home/sine extrema must fit the model-derived safe ranges.

RobotIo exposes observation/send_action without reset or advancing time. Existing
RobotBackend remains compatible. ReachTask uses the narrow ReachGeometry interface
(bounds, end effector, FK); it does not reset the robot. The simulation environment
explicitly resets physics before task reset. Hardware reset policy belongs to a future adapter.

Reach samples joint configurations inside safe bounds with fractional margin and uses
FK to obtain targets within configurable min/max distance of the initial end effector.
A separate scratch mjData is reset to the model's initial state before FK. Live state,
control scheduler and time are untouched. Seeded SplitMix64 and bounded rejection sampling
make results reproducible; exhaustion is an error. Kinematic reachability does not imply
a collision-free path. The FK configuration is a test witness, not a policy observation.
Reward is negative distance plus configurable success bonus. Success uses inclusive distance
<= threshold; horizon truncates only if success did not terminate. Counters live in Rust.

## Three clocks

MuJoCo model timestep is authoritative: SO-101 0.005s, Panda/UR5e 0.002s.
`step_physics` executes one mj_step. `step_control` holds targets until an accumulated
absolute deadline, quantized to integer physics ticks. Overshoot is below one physics tick;
30Hz control on 200Hz physics alternates tick counts without accumulating fixed-seven-tick drift.
Explicit physics stepping rebases the control deadline; reset clears clocks.

One Gymnasium step is one control step (default 1/30s), not one physics tick. Episode
length and delta/rate limits count Gym steps. PPO rollout length counts Gym steps as well.
The native one-tick compatibility step advances a Reach episode counter once when used.

## Units and learning

Runtime uses SI: radian, rad/s, meter. Controlled actuators must be supported unit-gear
hinge position servos. Limits intersect enabled joint range and actuator ctrlrange;
unlimited sides contribute infinity. No duplicated robot-specific limit tables.
Panda's tendon gripper is not part of the action vector.

Bare RobotEnv accepts absolute SI positions. Policy wrappers map [-1,1] into absolute
safe ranges or bounded deltas from measured joint position. RateLimitWrapper then limits
change from the previous accepted command and clamps safe bounds. Mapping wraps the
rate limiter, which wraps RobotEnv. Default training uses delta + rate limit, both 0.05rad
per Gym step. Rejected steps do not update command history. Float32 bounds round inward.
Future real-robot degree/normalized/gripper-percent units pass through JointCodec, with
calibrated per-joint scale/offset, before SI validation. These are distinct boundaries.

Dict observations contain joint_position, joint_velocity, end_effector_position,
target_position. Gym FlattenObservation is optional. Same-robot SyncVectorEnv batches are
supported; mixed 6/7-axis spaces need separate batches. SB3 MultiInputPolicy/PPO is optional.
Saved config.json and model.zip together define the trained policy's environment contract.
Evaluation compares deterministic policy and random baseline with identical held-out seeds.

## Python ownership

Each native Python object owns a channel to a std worker which creates, uses and destroys
its Simulator on that thread. Worker jobs/results use Rust errors and values, no PyErr/PyResult.
All request waits, worker startup and explicit close/join run in PyO3 0.27 Python::detach.
Error conversion happens after reattachment. There is no unsafe Send/Sync implementation.
Drop closes the sender without joining under the GIL; explicit close detaches and joins.
Worker panic stops the session and becomes RuntimeError. Independent Python threads own
independent Simulators. Tests verify another thread can acquire Python during a worker wait.

## Telemetry and rendering

Rerun 0.38.1 inspects episode behavior; W&B compares training runs. Neither is mandatory.
Rust logs reward/distance/success, positions/targets, end-effector/target points and episode.
Timelines: simulation_time, episode_step, episode and monotonic global_step. Use global_step
across resets. RRD save/inspection tests need no Viewer; GUI is outside core.
W&B is only imported for explicitly enabled Python training; offline mode works without login.
MuJoCo GLFW Viewer remains an optional CLI feature running on macOS's main thread.

CameraSpec separates logical `wrist` from MuJoCo source `wrist_cam`. Future
render_camera(name,width,height) belongs to a renderer owning the offscreen context;
RGB, meshes and image policies are deferred.

## Extension boundary and safety

Supported scope: fixed-base position-controlled SO-101/Panda arm/UR5e. No floating base,
torque generalization, tendon control, mobile base or humanoid abstraction is claimed.
A minimal LeRobot adapter maps named RobotObservation fields to features and external action
through JointCodec to RobotIo. Task observations/reset/reward stay out of real hardware I/O.
Camera rendering/calibration and system identification precede image imitation learning.

The four runtime unsafe blocks are listed below. No unsafe ownership/thread trait implementations.

## Sim-to-Real extension (2026-09-30)

Policy → Gymnasium → RobotAction → MuJoCo / future RobotIo backend → Observation
→ common TrajectoryRecorder → sim/real CSV → comparison / future SysID optimizer
→ ModelParameters → reset-time Domain Randomization → retraining.
The implementation is measurement infrastructure, not a hardware driver or optimizer.

`robotics-core::trajectory` defines the serde schema and recorder trait; `trajectory-io`
implements file I/O. Python `trajectory.py` imports/exports that same schema and computes
metrics. Physics is independent of file writers. ModelParameters/DR live in mujoco-runtime,
not in PPO. Immutable nominal arrays are captured once; reset applies nominal × center ×
random scale. Gains change kp and matching bias together. Only supported scalar servo/DOF
fields are mutated; body mass/inertia and geometry are unchanged.

Each Gym step now calls `step_task` once: Rust steps, computes a coherent StepResult and
returns it in one detached worker transaction. Delta mapping uses the last returned observation,
not another channel request. `simulation_time` is current physics time. With sensor latency,
joint state/end-effector are from `observation_time`; reward/termination use current true state.
Target is task state, never a randomization parameter. Initial history supplies warmup observations.

Command due times are seconds on the integer physics clock. Delivery occurs before the first
physics tick at/after due time, so rounding adds less than one physics tick. Multiple submissions
at the same physics instant coalesce to the latest command. History stores physics snapshots and
retains the nearest previous sample plus newer samples, bounded by latency/timestep. No interpolation
is performed for policy observations. Encoder zero bias is added only to delivered position.
Reset clears queues/history; a new seed samples independent DR and Reach random streams.

Default end effector comes from RobotSpec.default_end_effector; no task/telemetry literal name.
Python model_dir is a legacy SO-101-directory route, exclusive with an explicitly supplied robot.
Custom Python RobotSpec/model_path loading is still deferred; no parameter is silently ignored.

Parameters are diagnostic info, Rerun episode telemetry and W&B run/episode data, never policy
observation features. Residual composition is Python-only and requires SI base targets, followed
by independent correction bounds, joint range clamping and an owned rate limiter.

Four unsafe blocks now exist: the existing checked actuator-joint shim and matched site_xpos,
plus bounded C parameter read/write wrappers. Writes require exclusive model access at reset;
arrays retain size/topology, pointers are never stored, and data is reset before stepping.
There are two unsafe extern declaration blocks. No unsafe Send/Sync or Python-dependent worker.
See SIM_TO_REAL.md for schemas, clock semantics, profiles and deliberately unsupported gaps.

## Goal Reach / precision learning (2026-10-02)

Existing RobotEnv and PPO remain compatible. GoalReachEnv adds the GoalEnv convention:
`observation` (selected q, qvel, end-effector XYZ), `achieved_goal`, `desired_goal`.
SO-101 Reach selects five arm actuators by name and holds the omitted gripper at home.
RobotSpec still exposes all six actuators; Panda/UR5e select their existing arm actuators.
Selection is a task setting, not a fixed dimension in core. FK targets use the same omitted
joint values. Hidden FK witness q is only accessible to Rust tests, never a policy feature.

Policy actions are normalized joint deltas, then measured q + action * max_delta,
previous-command rate limiting, safe range clamping, native absolute SI targets.
DLS uses scratch FK and a numerical Jacobian; it is an independent validation controller,
not part of joint-space SAC/HER action generation. Physics gains, limits, friction and
collision geometry are unchanged. Target XYZ stays fixed until reset.

HER recomputes dense reward from goals only: -distance/0.1 (+ optional success bonus),
with inclusive 0.005m success. Training uses fixed-horizon timeout episodes with no success
termination. SB3 2.9.0 HER relabels rewards but retains transition dones; early success
termination would produce incorrect relabelled terminal transitions. Evaluation terminates
on true native success, or truncates at the horizon. Under sensor latency the policy's
achieved_goal is delayed, while diagnostic distance/success describe the current true state.

SAC/HER, replay, curriculum, W&B and checkpoint handling stay in Python. Stage transitions
keep actor/critic/optimizer and by default reset replay; the optional retained-replay ablation
truncates unfinished HER trajectories before resetting the environment. Resume restores
model/optimizer, replay, target RNG, NumPy and Torch RNG state, starts a fresh physical
episode, and truncates the unfinished saved trajectory. It is not a mid-physics-state restore.
Final-stage validation defaults to 500 separate held-out seeds; final reporting uses another
fixed 500-seed set. The runner has an explicit <=5M-step cap.

Trajectory V2 records command issue, actual application, sensor delivery and original
sensor sample times separately. Requested/applied targets are distinct vectors. V1 remains
readable; its missing command/application times cannot be recovered and are treated as its
legacy timestamp. Command comparison uses zero-order hold; measured positions use linear
interpolation on the delivery clock. `sensor_zero_offset_rad` changes delivered encoder
position only; `actuator_zero_offset` (command/physical coordinate offset) remains TODO.
The old `joint_zero_offset_rad` is accepted as a read-only configuration alias.

Rust Rerun logs control/requested, control/applied, state/true, state/observed per actuator,
in addition to reward/distance/success and target/end-effector points. Best/worst evaluation
seeds can be replayed to separate .rrd files without opening a GUI. No new unsafe code
or new core dependency on a learning framework was introduced.
