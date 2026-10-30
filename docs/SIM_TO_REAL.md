# Sim-to-Real: measurement first

この実装は、実機で計測したresponseをsimulationと同じ形式で比較し、推定parameterを
MuJoCoへ戻し、その周辺で学習をやり直すための基盤です。実機driver、実機reset、実機RL、
SysID optimizerは実装していません。SO-101 / Panda arm / UR5eのposition-control範囲を維持します。

## 1. Reality Gap

simulationと実機の差には、摩擦、粘性減衰、servo応答、トルク上限、通信／sensor遅延、
原点ずれ、慣性、backlash、温度、電源、取付け誤差があります。今回実装したのは最初の7種類の
近似モデルです。実機との一致や安全性を確認したモデルではありません。

## 2. System Identification

既知の安全なcommandを与え、timestamp付きの応答を測り、model parameterを調整します。
`crates/robotics-core/src/controller.rs::SineTrajectory`はhome基準で指定jointのみを励振し、
全振幅がsafe range内か検証します。`SineController`と同じRust実装をPythonから使用します。
step/chirp、複数jointの自動連続励振、parameter最適化は未実装です。

```bash
nix develop
uv sync --locked --extra learning --no-install-project
maturin develop --features sim-python/rerun
python examples/python/record_identification_trajectory.py \
  --robot so101 --joint shoulder_pan --duration 10 --output data/sim
# Panda: --robot panda --joint joint1
# UR5e: --robot ur5e --joint shoulder_pan_joint
```

exampleはhomeで2秒settleした後に記録します。これはsimulation専用で、実機へそのまま
reset/settleを送る設計ではありません。amplitude/frequency/duration/control-hzを指定できます。

## 3. Domain Randomization

`crates/mujoco-runtime/src/randomization.rs`がseed付きのepisode parameterを生成し、
`simulation.rs::reset_seed`が適用します。PPOやGymnasiumへの依存はありません。

| Parameter | light | moderate | Meaning |
|---|---|---|---|
| joint damping scale | 0.95–1.05 | 0.90–1.10 | controlled hinge DOF damping |
| frictionloss scale | 0.95–1.05 | 0.90–1.10 | controlled hinge Coulomb loss |
| actuator gain scale | 0.95–1.05 | 0.90–1.10 | kpと位置bias -kpを同倍率、kvは保持 |
| actuator force scale | 0.95–1.05 | 0.90–1.10 | enabled actuator forcerange両端 |
| added control latency | 0–10ms | 0–40ms | command delivery delay |
| added observation latency | 0–5ms | 0–20ms | sensor delivery delay |
| added sensor zero offset | ±0.1deg相当 | ±0.3deg相当 | encoder bias、内部表現はradian |

`none`は空設定です。各fieldのNone/省略はperturbationなし、min==maxは固定値。
scaleはjointごとに独立、latencyはrobot全体でepisodeごとに1つです。stepごとのjitterはありません。
zero offsetは測定positionだけに加算し、物理geometryやcommandは変更しません。velocityはbiasなし。
nominalが0のdamping/frictionlossは倍率を掛けても0のままです。
force-limitedでないactuatorに非1倍のforce scaleを要求するとerrorになります。
ctrlrange、joint range、joint-level force limitsは変更しません。

**範囲は実測toleranceではなく、小さめのengineering sensitivity-test priorです。**
[LeRobot SO-101公式資料](https://huggingface.co/docs/lerobot/en/so101)でfollowerのSTS3215構成と
個体ごとのcalibrationの必要性を確認しましたが、damping/gain/latencyの個体差分布は示されていません。
そのため±10%や40msを実機の既知仕様とは扱いません。測定で範囲を置換してください。
[MuJoCo 3.10のmodel変更規則](https://github.com/google-deepmind/mujoco/blob/3.10.0/doc/programming/simulation.rst#model-changes)
とpinned headersに基づき、reset時に実数parameterのみ変更します。model構造は変更しません。
body mass/inertiaとgeom friction randomizationは今回見送りました。

`python/robot_sim/profiles/{none,light,moderate}.toml`が同梱profile、
`configs/domain_randomization/so101-moderate.toml`が編集用exampleです。
Rustは非有限／逆転range、負latency等を拒否します。設定上限はscale10倍、latency1秒、
|offset|0.2radであり、これは実機の安全上限や安定性保証ではありません。

## 4. Trajectory recording / interchange

`robotics-core::trajectory`にschemaとTrajectoryRecorder/NullRecorder、
`crates/trajectory-io`にCSV保存・再読込を配置しました。simulation本体はfile writerを所有しません。
CLI `--trajectory <directory>`、Python `sim.start_trajectory(path, control_period, seed)` /
`sim.finish_trajectory()`で有効化します。明示finishでIO errorを確認してください。
resetは進行中の記録をfinishします。1 directoryは1回の連続captureです。
既存directoryは上書きしません。recording periodはphysics timestep以上で、stepのperiodと一致が必要です。

`metadata.json`（schema_version=2）:

- robot、backend (`mujoco` / `real`)、model_version、joint_names（順序付き）。
- control_period、physics_timestep（実機はnull）、timestamp（capture作成時のUnix秒）。
- seed、randomization_parameters（そのcaptureで実際に適用したModelParameters）。

`samples.csv` V2は `measurement_time, command_time, applied_time, source_measurement_time`
の後、各jointの `command/<name>`、`applied/<name>`、`position/<name>`、
`velocity/<name>`、最後に `end_effector/x,y,z` の順です。

| Field | Meaning |
|---|---|
| command_time | 最後の要求targetをworkerへ発行したbackend時刻 |
| applied_time | 記録されたapplied targetをactuatorへ実際に適用した時刻 |
| measurement_time | workerが測定値を受け取った現在backend時刻（単調増加） |
| source_measurement_time | 遅延history内の元sensor sample時刻 |
| command | 最新の要求target。遅延queue内で未適用の場合もある |
| applied | 実際のactuator target。要求とは独立に記録 |
| position / velocity | observation latency / encoder bias適用後の測定 |

単位は秒、radian、rad/s、meterです。captureのUnix秒とbackendのelapsed秒は別です。
command_time <= measurement_time、source_measurement_time <= measurement_timeを検証します。
velocityとend-effectorはsample単位で全成分空欄にできます。V1も読み込めますが、
失われたcommand/application時刻は復元できず、legacy timestampで代用します。

`Trajectory.from_csv()` / Rust loaderはheader・次元・有限性・timestampを検証します。
comparisonのposition/stateはmeasurement delivery clockで線形補間、command/appliedは
それぞれのissue/application時刻でZero-Order Holdします。sensor acquisition clockで
比較したい場合はsource_measurement_timeを使用した明示的な再alignが必要です。

`joint_zero_offset_rad`の曖昧な名称は `sensor_zero_offset_rad` に変更しました。
旧名はconfig読込aliasだけを維持します。これはencoder observation biasであり、
physical jointやcommand座標は変わりません。**actuator_zero_offset: TODO**。

## 5. Sim-real comparison

```bash
python examples/python/compare_trajectories.py --sim data/sim --real data/real
python examples/python/compare_trajectories.py --sim data/sim --real data/real --rerun
# GUI不要
python examples/python/compare_trajectories.py --sim data/sim --real data/real --record comparison.rrd
rerun comparison.rrd
```

`python/robot_sim/trajectory.py`はjoint名・順序を確認し、重なるtimestamp範囲へ線形補間します。
defaultでは各traceの先頭時刻を0とします。同期済み時計には`--absolute-timestamps`を指定します。
比較sampling periodは2つのmedian periodの大きい方です。範囲外へ外挿しません。
各jointについてposition RMSE / MAE / 最大絶対誤差と、両方にvelocityがあればvelocity RMSEを出します。

±`--max-lag`（default0.2s）の範囲をsampling period刻みで探索し、mean-centered correlationが
最大のshiftを推定します。正のlagはrealがsimより遅れる意味です。定数信号ではnullです。
周期信号では曖昧性があるため、励振周波数と探索範囲を適切に選んでください。
lag補正後RMSEも別に報告し、生のtimestamp-aligned誤差を隠しません。
最低8sampleと十分な重複区間が必要です。clock drift/jitterの推定は未実装です。

Rerunは共通`elapsed_seconds` timelineに`sim/joints/*`, `real/joints/*`, `error/joints/*`,
`sim/commands/*`, `real/commands/*`を記録します。画面には未補正の応答差を表示します。

## 6. Identified model

`ModelParameters`はschema_version、robot、joint_namesと各jointのscale/offset配列、
control/observation latencyを持つJSONです。Rust `load/save`、Python
`sim.load_parameters(path)` / `sim.save_parameters(path)`に対応します。
名前・順序・robotが違えば拒否します。parameter centerの変更は**次のreset**から反映します。
seedなしのbare `reset()` はseed0を使用します。episodeを変えるときは `reset_reach(seed)` / Rust `reset_seed(seed)` を使用します。
`save_parameters`は現在有効なparameter setを保存し、上書きしません。

`models/identified/so101-example.json`は**人工のAPI検証用**です。実機identified値ではありません。
実際の推定値で置換する前提です。`nominal × identified_scale × random_scale`から毎episode作り直し、
過去episodeの値に倍率を掛けません。latency/offsetのperturbationはidentified値へ加算します。
`none`でもidentified centerを指定していればその固定モデルです。

## 7. Robust RL

```bash
python examples/python/train_reach_ppo.py --robot so101 --domain-randomization moderate --steps 500000
python examples/python/train_reach_ppo.py --robot so101 \
  --domain-randomization configs/domain_randomization/so101-moderate.toml \
  --identified-parameters models/identified/so101-example.json --steps 2048
python examples/python/eval_robustness.py --robot so101 --model runs/<run>/model.zip --episodes 100
```

同じscriptでPanda/UR5eも使用できます。profileとidentified setの内容をcheckpoint隣のconfig.jsonに
埋め込みます。後で元ファイルを編集しても実験設定は変わりません。評価は同じtarget seed集合を
none/light/moderateで使用します。identified centerがある場合は3条件とも同じcenterです。
`eval_reach.py --domain-randomization none`でも条件をoverrideできます。

policy observationは既存4項目のDictのままです。parameter値は`info["domain_randomization"]`にのみ
追加します。Rust Rerunはepisode開始時にjointごとのscale/offsetとlatencyを記録し、W&Bは
run configとepisode終了時のscale平均／latencyを記録します。W&Bなしでも実行可能です。
実測評価は[SIM_TO_REAL_VALIDATION.md](SIM_TO_REAL_VALIDATION.md)。短い学習では頑健性を証明できません。

## 8. Residual RL boundary

`ResidualActionWrapper`は凍結したcallableの**absolute SI target**に小さい補正を加えます。
policy trainingや実機I/Oは行いません。

```python
from robot_sim import RobotEnv, ResidualActionWrapper
base = RobotEnv(robot="so101")
home = base.sim.home_position
# 例はhome保持。実際は凍結したbase policyの出力をSI positionへ変換して渡す。
env = ResidualActionWrapper(base, lambda observation: home,
                           max_residual=0.01, alpha=1.0, max_change=0.05)
```

residual actionは[-1,1]で、補正量はalpha × max_residual以下です。alphaは[0,1]。
最終targetをsafe joint rangeへclampし、内蔵RateLimitWrapperを必ず通します。
base policyがnormalized/deltaなら、その出力をabsolute radianへ明示変換してください。
residual=0はbase targetを維持しますが、range/rate safetyは常に適用します。
これは実機の非常停止、接触安全性、motor thermal limitの代替ではありません。

## 9. 実機で次に計測するもの

最小adapterは、校正済みjoint名とSI変換、monotonic command送信／sensor受信時刻、
positionと可能ならvelocityを取得し、同schemaへ保存するものです。
取付け姿勢、payload、電源電圧、温度、firmware、calibration versionをmodel_versionや別の計測ノートで
管理してください。安全な低振幅sine、home近傍、低速から実測し、latencyとtracking誤差を比較します。
Taskから実機をresetせず、停止・許可・hardware limitsは実機adapterが管理する必要があります。

未対応: mass/inertia uncertainty、geom/contact friction、backlash、deadband、量子化、thermal drift、
電圧依存、payload変化、packet loss/jitter、clock drift、camera、force sensing、実機driver／optimizer／RL。
今回作ったloopは measurement → comparison → parameter file → randomized retraining までです。
