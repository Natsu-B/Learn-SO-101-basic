# Robot simulation / Reach learning

Apple Silicon Mac上でRustのMuJoCo runtimeをPython/Gymnasiumから共用します。
対応範囲は **position-controlled articulated robot arms**: SO-101、Panda arm、UR5eです。
MuJoCo 3.10.0 / rusty_mujoco 0.4.0 / Menagerieの既存pinを維持しています。
Docker、Homebrew、システムPythonは不要です。Nix自体は事前にインストールしてください。

## 1. Simulationを動かす

このrepositoryで実行します（Nixの `nix-command flakes` を有効にしてください）。

```bash
nix develop
cargo test --workspace
cargo run -p sim-cli -- --robot so101 --task reach
cargo run -p sim-cli -- --robot panda --task reach
cargo run -p sim-cli -- --robot ur5e --task reach

uv venv --python "$NIX_PYTHON" --seed
source .venv/bin/activate
uv sync --locked --no-install-project
maturin develop
pytest
python examples/python/so101_reach.py
```

NixがRust stable、cargo、clang、GLFW、MuJoCo、pkg-config、Python 3.12、uv、maturinを供給します。
`MUJOCO_LIB`、`MENAGERIE_DIR`、`SO101_MODEL_DIR`とruntime rpathをdevShellで設定します。
次回の `nix develop` は既存 `.venv` をactivateします。Python依存はuv、native extensionはmaturinで
installします。`flake.lock` / `Cargo.lock` / `uv.lock` を保持してください。

## 2. Rerunを見る

Rust SDK / Python SDK / Viewerは **0.38.1** で一致しています。

```bash
maturin develop --features sim-python/rerun
python examples/python/gym_random_policy.py --robot so101 --rerun
# GUIを開かず保存、後から表示
python examples/python/gym_random_policy.py --robot panda --record reach.rrd
rerun reach.rrd
# Rust CLIからも利用可能
cargo run -p sim-cli --features rerun -- --robot ur5e --task reach --telemetry rerun
```

`metrics/reward`, `metrics/distance`, `metrics/success`, `robot/joints/*`,
`robot/joint_targets/*`, `robot/end_effector`, `task/target`をRustから記録します。
`simulation_time` / `episode_step` / `episode` / `global_step`のtimelineを持ちます。
複数episodeを連続して見るときは単調増加する **global_step** を選択してください。
`metrics/episode`でresetの境界を確認できます。3D点群と時系列を表示し、robot meshは未実装です。
Rerunなしのbuildと全headless testも維持しています。

## 3. Gymnasiumを使う

```python
from robot_sim import make_env, TaskConfig

env = make_env("panda", task="reach", action_mode="delta",
               task_config=TaskConfig(success_bonus=1, max_episode_steps=200))
obs, info = env.reset(seed=42)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

`RobotEnv`のactionは絶対joint position（radian）、`make_env`のdefaultは正規化deltaです。
SO-101 / UR5eは6軸、Pandaは7軸で、同じコードを使えます。
Dict observationは `joint_position`, `joint_velocity`, `end_effector_position`,
`target_position`（すべてfloat32）。必要ならGymnasiumの `FlattenObservation` を使用してください。

- `NormalizedJointPositionWrapper`: [-1,1] → joint range ∩ actuator ctrlrange。
- `DeltaJointPositionWrapper`: current position + action × max_delta（default 0.05 rad/step）、safe rangeへclamp。
- `RateLimitWrapper`: 前回commandとの差をmax_change以下に制限し、safe rangeへclamp。resetで履歴を初期化。

合成順は `DeltaJointPositionWrapper(RateLimitWrapper(RobotEnv(...)))` です。
`make_env(..., rate_limit=0.05)`でも同じ構成になります。制限値は1 Gym stepあたりのradianです。
invalid shape、NaN/Inf、action_space外の入力はValueErrorになります。
`gymnasium.vector.SyncVectorEnv([lambda: make_env("so101"), lambda: make_env("so101")])`
も使用できます。通常の同一space vector batchに6軸と7軸を混在させないでください。

Reach targetはsafe joint range内の乱数configurationをFKへ通して生成します。
実際のrobot stateは変更せず、同じseedで再現します。margin（rangeの5%）、初期姿勢からの距離
0.05–0.25m、最大4096試行はTaskConfigで変更できます。これは**運動学的な到達可能性**であり、
衝突のない経路の保証ではありません。rewardは-distance + success bonus（成功時のみ）。
距離 < success_thresholdでterminated、horizonでtruncated。成功とhorizonが重なった場合は成功を優先します。

Bare `Simulator`も維持しています。`step`はphysics 1 tick、`step_control`はcontrol periodを進めます。
`reset_reach`, `configure_reach`, `state`, `time`, `reward`, `distance`, `terminated`, `truncated`,
`episode_step`, `action_bounds`, `actuator_names`, `camera_names`を公開します。
Python raw pointerは公開しません。旧 `so101_mujoco` importとSO-101 CLIも維持しています。

## 4. PPOを学習する

```bash
uv sync --locked --extra learning --no-install-project
maturin develop --features sim-python/rerun
python examples/python/train_reach_ppo.py --robot so101 --steps 100000
# 同じscriptで --robot panda / --robot ur5e も指定可能
```

Stable-Baselines3 PPO / MultiInputPolicy、CPU、2層64unit、rollout256、batch64、5epochs、
learning rate 3e-4を使用します。defaultはdelta action + 0.05rad/step rate limit、200step episode、
success bonus 1です。`runs/<robot>/<timestamp>/`に `model.zip`, `config.json`, `training.json`を保存します。
`--output`で場所を指定できます。PPOはrollout単位で学習するため100000要求時は100096stepになります。
チェックポイントは学習時のaction/task設定と組で保持してください。

## 5. W&Bを見る（optional）

```bash
python examples/python/train_reach_ppo.py --robot so101 --steps 100000 --wandb
# login/networkなしでローカル記録のみ
python examples/python/train_reach_ppo.py --robot so101 --steps 256 --wandb --wandb-mode offline
```

episode reward、success rate（直近100episode）、episode length、final distance、learning rateを記録します。
online使用時は自分のW&B認証を設定してください。W&Bなしでも学習できます。
Rerunはepisode内部の挙動、W&Bはtraining run同士の比較を担当します。両方ともRust coreの依存ではありません。

## 6. 学習済みpolicyを評価する

```bash
python examples/python/eval_reach.py --robot so101 --model runs/<robot>/<timestamp>/model.zip
python examples/python/eval_reach.py --robot so101 --model runs/<robot>/<timestamp>/model.zip --rerun
# headless記録と機械可読の結果
python examples/python/eval_reach.py --model runs/<robot>/<timestamp> --record evaluation.rrd --output evaluation.json
```

20episode（`--episodes`で変更）、default seed10000から連続seedを使用し、random policyと
**deterministic** trained policyを同じtarget集合で比較します。success rate / mean reward /
mean final distance / episode lengthを出力します。Rerunはtrained policyのepisodeを記録します。
Panda / UR5eも同じscriptです。robot指定とcheckpoint設定が違う場合は拒否します。
実測値と制約は [docs/VALIDATION.md](docs/VALIDATION.md) を参照してください。
短い学習の成功を収束・実機性能の証拠とは扱いません。

## Models and controlled joints

Pinned Menagerie: `c96a32d28fb5da84da38c1da4d749e7a13212855`。

| Robot | MJCF | Controlled actuators (API order) | Logical gripper site |
|---|---|---|---|
| SO-101 | robotstudio_so101/so101.xml | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | gripperframe |
| Panda | franka_emika_panda/panda.xml | actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7 | panda_hand_site (adapter) |
| UR5e | universal_robots_ur5e/ur5e.xml | shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 | attachment_site |

Pandaのpinned modelにはsiteがありません。既存body `hand`の原点にrepository-owned
`panda_hand_site`をtemporary includeへ追加します。これはhand frameであり指先TCPではありません。
Menagerie原本は変更しません。Panda actuator8はtendon gripper / 0..255表現で、radianの
JointPosition APIには含めずnative home keyframeのctrl=255を保持します。Panda/UR5eはnative
home keyframe、SO-101はqpos0でresetします。SO-101はlogical camera `wrist` → source `wrist_cam`を持ちます。
actuator/gain/force/friction/collision/timestepはmodelを使用し、独自PIDはありません。


## MuJoCo GLFW Viewer

```bash
cargo run -p sim-cli --features viewer -- --robot so101 --task reach --viewer
cargo run -p sim-cli --features viewer -- --robot panda --task reach --viewer
cargo run -p sim-cli --features viewer -- --robot ur5e --task reach --viewer
# 既存のSO-101 CLI
cargo run -p so101-cli --features viewer -- --viewer --sine
```

macOS main threadで実行します。dragでorbit、scrollでzoom、Spaceでpause、Rでphysics reset、Escで終了。
Viewerの`--steps`はphysics tick数、headless CLIではcontrol step数です。ViewerのRはphysicsのみ
resetし、task targetは維持します。RerunとGLFW Viewerは今回個別に利用します。

## Tests / architecture

```bash
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace
maturin develop
pytest
# optional SDKとlearningも含む検証（GUI/login/networkはtestに不要）
bash scripts/verify.sh --learning
```

最初のdependency取得にはnetworkが必要です。learning extraなしではPPO testsをskipします。
[Architecture](docs/ARCHITECTURE.md) / [validation](docs/VALIDATION.md) / [変更前の基準](docs/RL_VALIDATION.md)。
HomeControllerはRobotSpec明示home → 合法なqpos0 → simulator初期状態の順で選択し、
SineControllerはhomeを中心として指定jointのみを動かします。

## 7. 次の段階

camera RGB renderer → Pick / Place → imitation learning / ACT → LeRobot adapter →
real SO-101 → system identificationを段階的に追加します。今回cameraはCameraSpec metadataのみです。
将来の `lerobot_robot_*` adapterはRobotObservationを名前付きfeatureへ、LeRobot actionをJointCodecで
SI単位へ変換してRobotIoへ渡します。実機をTaskから勝手にresetしない構造です。
将来のLeRobot学習では `lerobot-train ... --wandb.enable=true` を想定します。

floating-base / quadruped / humanoid / mobile base / torque control / tendon gripperは未対応です。
Panda gripperは既存keyframeの状態を保持し、学習actionには含めません。
ACT、image policy、serial/Feetech driver、full LeRobot plugin、dataset recorder、Gazebo、Isaac Sim、ROS、
独自neural network / multiprocessing frameworkは実装していません。


## Sim-to-Real計測・再学習

実機driverは追加せず、共通trajectory形式、比較、identified parameter、Domain Randomizationを追加しました。
まず [SIM_TO_REAL.md](docs/SIM_TO_REAL.md) と [実測検証](docs/SIM_TO_REAL_VALIDATION.md) を参照してください。

```bash
nix develop
uv sync --locked --extra learning --no-install-project
maturin develop --features sim-python/rerun

python examples/python/record_identification_trajectory.py \
  --robot so101 --joint shoulder_pan --duration 10 --output data/sim
python examples/python/compare_trajectories.py --sim data/sim --real data/real --rerun

python examples/python/train_reach_ppo.py \
  --robot so101 --domain-randomization moderate --steps 500000
python examples/python/eval_robustness.py \
  --robot so101 --model runs/<run>/model.zip --episodes 100

# Rust CLIでも同じmeasurement schema
cargo run -p sim-cli -- --robot panda --steps 100 --trajectory data/panda
```

`data/real`は実測データを共通schemaへ変換して用意します。比較のために実機接続は不要で、
まずsim同士でも実行できます。profileはnone/light/moderateまたはTOML/JSONファイルです。
`--identified-parameters <JSON>`でDRの中心を指定できます。添付so101-example.jsonは人工例です。
範囲は実機の実測toleranceではありません。body mass・geom friction DR、serial通信、実機RL、
SysID optimizerは未実装です。

Gym stepは観測・reward・終了判定を1回のRust worker transactionで取得します。
`RobotEnv(..., domain_randomization="light")` / `make_env(..., domain_randomization="moderate")`
からも利用でき、parameter情報はinfoにだけ入り、policy observationには入りません。
legacy `Simulator(model_dir=...)` と `Simulator(robot=...)` は排他的です。

## SO-101 Reachを5mmで学習・検証する

既存PPOとは別に、Goal-conditioned **SAC + HER**を使用します。
RobotSpecの6 actuatorは維持し、Reach actionはgripperを除いた5 arm actuatorです。
目標はFKで生成したXYZのみがpolicyへ渡り、生成時のjoint configurationは非公開です。
Panda/UR5eにも同じGoalReachEnv／scriptを使えます。

```bash
nix develop
uv sync --locked --extra learning --no-install-project
maturin develop --features sim-python/rerun

# 先に物理的な到達精度を検証（学習policyではない）
python examples/python/eval_reach_ik.py --robot so101 --episodes 500 --threshold 0.005

# 最初の500k step。5mm達成前はDRを無効にする
python examples/python/train_reach_sac_her.py --robot so101 --curriculum \
  --steps 500000 --max-steps 5000000 --net-arch 128 128 --wandb --wandb-mode offline

# 追加step数を指定して再開。replayも保存する
python examples/python/train_reach_sac_her.py --robot so101 \
  --resume runs/so101-reach-5mm/latest.zip --steps 500000

# 固定500 held-out episodes、random baselineとの比較
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/best.zip --episodes 500 --threshold 0.005

# best/worst各5 episodeを保存（GUI不要）
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/best.zip --episodes 500 --threshold 0.005 \
  --record runs/so101-reach-5mm/evaluation-rerun
# --rerun なら保存後Viewerを起動
```

```python
from robot_sim import GoalReachEnv

env = GoalReachEnv(robot="so101", max_delta=0.05)
obs, info = env.reset(seed=42)
# obs: observation / achieved_goal / desired_goal (XYZはmeter)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

`--algorithm ppo|sac|sac-her`で同じGoal環境を比較できます。
`--max-delta 0.05|0.02|0.01`、learning rate、batch、networkなどはCLI引数です。
A→B→C→D curriculumはheld-out success 90%で進行し、Dは0.05–0.25m／5mmです。
trainingはHERのため固定horizon、evaluationは5mm到達時に終了します。
`best.zip` / `latest.zip` / `latest-replay.pkl` / `artifact.json`を同じdirectoryで保持してください。
W&Bは実際のSB3 actor/critic loss、entropy coefficient、learning rateと評価値を記録します。

nominal 5mm達成後のlight DR fine-tuning（元checkpointは保存）:

```bash
python examples/python/train_reach_sac_her.py --robot so101 \
  --initialize-from runs/so101-reach-5mm/best.zip --domain-randomization light \
  --output runs/so101-reach-light --steps 100000
python examples/python/eval_robustness.py --robot so101 \
  --model runs/so101-reach-light/best.zip --episodes 100
```

実測結果、失敗例、checkpointと全設定は [SO101_REACH_5MM.md](docs/SO101_REACH_5MM.md) に記録します。
Trajectory V2はcommand発行／適用／測定受信／元測定時刻を分離し、Rerunも
`control/requested/*`、`control/applied/*`、`state/true/*`、`state/observed/*`を分けます。
旧trajectory V1は読み込めます。詳細は [SIM_TO_REAL.md](docs/SIM_TO_REAL.md) を参照してください。

2026-10-02実測: SO-101 nominal 500 episode中484件（96.8%）が5mm以内。
採用checkpointは `runs/so101-reach-5mm/base-policy/best.zip`、主学習425,136 step。
light fine-tune候補はnominal性能を落としたため採用していません。

### 到達後の保持評価

既存のGoal SAC/HER policyを再学習せず、到達後もpolicyを動かして保持能力を測れます。

```bash
nix develop
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/base-policy/best.zip \
  --episodes 500 --seed 3000000 --threshold 0.005 --hold-seconds 0.5 \
  --output docs/results/reach-hold-5mm.json
```

`--hold-seconds 0`（default）は従来の到達時終了です。正の値では、
指定した誤差以内の状態をsimulation timeで連続して維持した場合に成功とします。
範囲外に出るとタイマーをリセットし、再到達から数え直します。
既存のepisode上限100 control stepsは維持するため、期限直前の到達は保持成功になりません。
到達率 `reach_success_rate` と保持成功率 `hold_success_rate` を分けて出力します。
各episodeには初回到達時刻、最長連続保持時間、初回到達以降の平均・最大誤差も保存します。
判定は約30Hzのcontrol step末端の真の手先距離を使います。physics step間も含む厳密な
連続時間保証ではありません。観測遅延がある場合も判定は遅延したobservationではありません。
`--threshold 0.002`で2mm保持、`--record <directory>` / `--rerun`で保持中も含めた
best/worst episodeの記録を利用できます（native Rerun featureが必要）。
保持は評価層だけへ追加し、HERの固定horizon training・報酬・action・checkpointは変更していません。

### 保持精度の追加学習

既存の成功済みpolicyから、action/physicsを変更せず追加学習できます。
報酬は従来の距離報酬に、training threshold以内にいる各stepのbonusを加えます。
HERのgoal置換時にも再計算できる報酬です。trainingは固定horizonを維持し、
checkpoint選択に保持評価を使います。

```bash
python examples/python/train_reach_sac_her.py --robot so101 \
  --initialize-from runs/so101-reach-5mm/base-policy/best.zip \
  --output runs/so101-reach-hold --steps 250000 --max-steps 300000 \
  --seed 73 --target-threshold 0.003 --validation-threshold 0.005 \
  --success-bonus 0.2 --hold-seconds 0.5 --validation-success 0.95 \
  --learning-rate 0.0003 --buffer-size 300000 --horizon 100 \
  --train-freq 16 --gradient-steps 4 --eval-interval 25000 \
  --final-eval-episodes 500
```

元policyは上書きしません。新しいrewardに合わせreplayを新規作成し、
`--initialize-from`では指定した学習率等をロードしたSACへ適用します。
途中再開は`--resume runs/so101-reach-hold/latest.zip`を使用してください。
保存した設定とreplayを引き継ぎます。
`--hold-seconds`は学習の終了条件ではなく検証基準です。

この追加学習の実測結果は[保持評価レポート](docs/REACH_HOLD.md)を参照してください。
新しい500目標で、5mm・0.5秒保持が73.0%→98.0%、2mm・0.5秒保持が10.2%→70.2%となりました。
改善済みpolicyは`runs/so101-reach-hold/best.zip`です。

```bash
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-hold/best.zip \
  --episodes 500 --seed 4000000 --hold-seconds 0.5
# 保持中の挙動を見る（native Rerun feature有効時）
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-hold/best.zip --hold-seconds 0.5 --rerun
```
