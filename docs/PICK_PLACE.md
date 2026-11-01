# SO-101 Pick & Place：現状確認・実装・計画

2026-10-31 時点。数値はすべてこのリポジトリで実際に実行した結果のみ記載する。

## 1. 前提認識と実コードの照合

| 認識 | 実コード | 判定 |
|---|---|---|
| Rust中心 / MuJoCo 3.10.0 / rusty_mujoco 0.4.0 | `Simulator::from_spec` が `mj_versionString()=="3.10.0"` を強制 | 一致 |
| Menagerie SO-101 | `flake.nix` の menagerie input、`presets::so101` | 一致 |
| DLS IK | `mujoco-runtime/src/ik.rs::dls_delta`（位置3次元のみ・1ステップ分） | 一致（姿勢拘束なし） |
| Domain Randomization | `randomization.rs`（関節減衰/摩擦/ゲイン/力/遅延） | 一致（質量・摩擦DRなし） |
| SAC+HER Reach, 5 arm関節, グリッパ固定 | `goal_env.py` + `selection.reach_actuators`（gripper除外） | 一致 |
| 6番目のgripperアクチュエータ | `SO101_ACTUATORS[5]="gripper"`、範囲 -0.1745..1.745 rad | 一致 |
| 立方体の把持・運搬は未実装 | 既存シーン `scenes/robot.xml` は床のみ | 一致（今回追加） |
| RGBカメラはメタデータのみ | `CameraSpec{wrist→wrist_cam}` の存在確認のみ | 一致 |
| 学習済みチェックポイントは未同梱 | **`runs/so101-reach-5mm/base-policy/best.zip`、`runs/so101-reach-hold/best.zip` が存在** | **相違** |

追加で判明した点:
- 既存 `.venv` はリポジトリ移動前のパスを参照しており `maturin develop` が失敗した。
  `.venv-old-moved-path/` に退避し、`uv venv` + `uv sync --locked --extra learning` で再作成した。
- 既存Reach checkpoint（base-policy）は再作成した環境で読み込み可能。100 episode・5mmで到達率 98%（seed 10000〜）。
- `gripperframe` は挟持中心ではない。実測で固定ジョー先端は site 座標 (+2.9, 0.2, −20.1) mm、
  25mm立方体を挟んだときの中心は約 (−6, 0.2, −8) mm。site の +x が指先方向、+z が固定→可動ジョー方向。
- SO-101は position servo（kp≈998、forcerange ±2.94）。グリッパを接触点より奥（−0.17 rad）へ指令すると
  力制限で押し付ける形になり、25mm立方体では約 0.10 rad で停止する。

## 2. 再利用した機能と今回の変更

再利用: `Simulator`（モデル読込・関節写像・安全範囲・`step_control` の30Hz制御周期・DR）、
`presets::so101`、`robotics-core` の `JointCommand`/`Result`、既存のテスト基盤・Nix環境。

| ファイル | 内容 |
|---|---|
| `scenes/so101_pick_place.xml`（新規） | テーブル（z=0平面）、25mm/30g 立方体（freejoint, condim 4, 摩擦1.2）、目標マーカー（mocap・非接触） |
| `crates/mujoco-runtime/src/simulation.rs` | `from_spec_with_scene` を追加。`from_spec` は従来どおり `robot.xml` を使う（挙動不変） |
| `crates/mujoco-runtime/src/pick_place.rs`（新規） | `PickPlaceSim`：seed付きランダム配置、立方体/目標の状態、ジョー接触判定、挟持中心FK、位置＋指先方向＋閉じ方向のDLS IK |
| `crates/mujoco-runtime/src/skills.rs`（新規） | `Skill` trait、成功判定 `predicates`、IKによる `ScriptedGrasp/Transport/Place`、`run_episode` |
| `crates/mujoco-runtime/tests/pick_place.rs`（新規） | 物理検証テスト6件 |
| `examples/rust/so101_pick_place.rs`、`pick_place_probe.rs`（新規） | 評価（JSON lines）とジョー形状の計測 |

立方体の姿勢を書き込むのは `reset` 時の `set_cube` のみ。以後は接触物理だけで動く。

### 成功判定（`skills::predicates`、学習環境でも同じ定義を使う）

- **Grasp**: 両ジョーが立方体に接触、テーブル非接触、立方体中心が静止高さ+30mm以上、を0.3秒連続。
- **Transport**: 両ジョー接触のまま、立方体が15mm以上の高さで目標XY 15mm以内。
- **Place**: ジョー非接触・テーブル接触、高さ誤差3mm未満、速度0.01 m/s未満、目標XY 20mm以内（解放・後退の後0.33秒で判定）。
- **Drop**: 両ジョー接触が失われ、立方体がテーブル付近に戻った。

## 3. 検証結果（従来制御＝IK＋状態機械、学習なし）

```bash
nix develop
cargo run --release -p mujoco-runtime --example so101_pick_place -- 200   # seeds 0..199
```

| 項目 | 結果 |
|---|---|
| 配置範囲 | 立方体・目標とも半径 0.15–0.25 m、方位 ±0.8 rad、相互距離 ≥ 0.08 m、yaw=0 |
| Grasp / Transport / Place | 200/200, 200/200, 200/200 |
| 最終XY誤差 | 平均 2.4 mm、P95 4.1 mm、最大 6.9 mm |

途中の失敗と対策（記録として残す）:
1. IK姿勢誤差に外積を使い、指先「真上」の解に収束 → 方向ベクトル差に変更。
2. 固定ジョー内面と立方体側面のすき間が約1mmで、下降中に上面へ衝突（200中1件）→ 下降時だけ固定ジョーを4mm離し、閉じる動作で押し付ける。
   最初は符号を逆にして 95/200 に悪化したため、ログで確認して修正した。

テスト（`tests/pick_place.rs`）はコンパイルとは別に物理挙動を確認する:
静止時ドリフト < 0.2mm、seed再現性、IK精度、**両ジョー接触で30mm以上持ち上がること**、
**同じ動作でグリッパを閉じなければ持ち上がらないこと（対照）**、5 seedでのPick & Place成功とテレポートなし。

**この時点の成果にAI（学習済みモデル）は含まれない。** 課題要件を満たすには次節の学習が必要。

## 4. 2日間の最小構成（優先順）

締切 2026-11-02 10:00。動作する成果物を優先し、学習が間に合わない部分は従来制御のまま明示する。

1. **Python binding**（`sim-python`）: `PickPlaceSimulator`（reset(seed), state, step, 各predicate, scripted skill 実行）。
2. **Transport を SAC+HER で学習（第1候補）**
   - 理由: 把持後は「立方体位置を目標へ運ぶ」目標条件付き問題で、既存Reach（96.8%）と構造が同じ。HERが使える。
   - obs: 5関節 q/qvel、挟持中心、立方体位置・速度、グリッパ角。achieved_goal=立方体位置、desired_goal=目標上方（+carry高さ）。
   - action: 5 arm関節の正規化delta（既存と同じ max_delta/rate limit）。グリッパは閉指令で固定。
   - 初期状態: 毎エピソード `ScriptedGrasp` を実行して作る（数十ms）。報酬は achieved_goal のみの関数なのでHER再計算可能。落下すると立方体が床に残り自然に大きな距離になる。
   - 予算: 200–300k step を目安。学習中は `best.zip` を随時保存し、時間切れでも途中モデルで評価できるようにする。
3. **統合評価**: Grasp(IK) → Transport(**学習**) → Place(IK)。`SkillReport.learned` で区別し、全IK版と同じ seed で成功率・誤差を比較。
4. **動画・レポート**: オフスクリーン描画でフレーム保存 → ffmpeg。学習済み区間を画面上に表示。

### 状態分布の不一致への対策
- Transportの初期状態は学習時も実行時も同じ `ScriptedGrasp` の終端から作る（分布一致）。
- 加えて学習時に初期関節へ小さなノイズを入れ、把持位置ずれに対して頑健にする。
- Place は Transport の成功判定（目標XY 15mm以内）を入口条件にし、学習Transportの終端誤差を実測して許容範囲を確認する。

## 5. その後の拡張
- Place の学習（下降＋解放、報酬: XY誤差・静止・ジョー非接触）。
- Grasp の学習（接触報酬・段階報酬、またはIKを基準にした residual policy）。
- 立方体yawのランダム化（IKの閉じ方向拘束と `PickPlaceConfig::yaw_range` は実装済みだが yaw≠0 は未検証）、質量・摩擦DR。

## 6. Python binding（2026-10-30追加）

`PickPlaceSimulator`を`robot_sim._native`と`robot_sim`から公開した．
既存の`Worker`をセッション型について汎用化し，シミュレータと実行中のSkillを専用スレッドで所有する．
Pythonからの待機中は`py.detach`でGILを解放する．
ワーカー内でpanicが発生すると`RuntimeError`を返し，以後の状態取得や操作を拒否する．

| API | 内容 |
|---|---|
| `PickPlaceSimulator(model_dir=None, *, robot=None)` | 既定はSO-101．`model_dir`と`robot`は排他．環境変数による解決は既存`Simulator`と同じ．専用シーンの対象外のrobotは拒否 |
| `reset(seed)` | 物理状態とSkill実行状態の初期化 |
| `state()` | `time`, `qpos[6]`, `qvel[6]`, `gripper`, `cube_pos[3]`, `cube_quat[4]`, `cube_vel[6]`, `target[3]`, `grasp_center[3]`, `contacts`のコピー |
| `step(position, period=CONTROL_PERIOD)` | 6関節の位置指令で実行し，状態を返す．範囲・周期の検証は`PickPlaceSim::step`に委譲．手動実行後は実行中のSkillを解除 |
| `action_bounds()`, `joint_names()`, `timestep()` | Rustが保持する関節の許容範囲，順序，物理刻み |
| `predicates()` | Rustの`lifted`, `transported`, `placed`, `dropped`を既定の`Criteria`で評価 |
| `criteria()` | 既定の全`Criteria`フィールドに加え，シーンの`cube_half`と`ScriptedTransport`の`carry_height` |
| `run_skill(name, max_steps=None)` | `grasp` / `transport` / `place`を現在の状態から実行．上限省略時はSkill既定値を使用．`name`, `status`, `success`, `steps`, `learned`, `reason`を返す |
| `start_skill(name)`, `skill_step()` | 現在の状態から開始し，1制御ステップずつ実行．状態文字列は`running` / `success` / `failure`．終端後の呼び出しでは物理状態を進めない |
| `skill_report()` | 実行中または終了したSkillのレポートを状態変更なしで取得．実行中の`status`は`running`．未開始時は`ValueError` |
| `run_episode(seed)` | Rustの全IKパイプラインを実行し，`EpisodeReport`の全フィールドを辞書で返す．各Skillの辞書には`status`も付加 |
| `close()` | ワーカーの終了とjoin．複数回の呼び出しが可能 |

`contacts`は`fixed_jaw`, `moving_jaw`, `table`, `other_robot`, `pinched`を含む．
`grasp_center`はRustの挟持中心FK，`cube_quat`はMuJoCoのwxyz順，`cube_vel`はfreejointの並進3成分と回転3成分である．
`criteria()["carry_height"]`は静止高さからの増分なので，運搬目標の高さは`cube_half + carry_height`で求める．
`GRIPPER_OPEN`, `GRIPPER_CLOSED`, `CONTROL_PERIOD`も両モジュールから公開する．
すべてIK版であり，返す`learned`は`False`である．

```python
from robot_sim import PickPlaceSimulator

sim = PickPlaceSimulator()
try:
    sim.reset(0)
    grasp = sim.run_skill("grasp")
    assert grasp["success"] and sim.predicates()["lifted"]
    state = sim.state()  # 学習Transportの初期状態
finally:
    sim.close()
```

### 追加検証の実行結果

`cargo fmt --all --check`と`cargo clippy --workspace --all-targets`は終了コード0で完了した．
`cargo test --workspace`は45件成功，失敗0件で，追加したpanic処理とGIL解放の2件を含む．
`maturin develop`で拡張を再ビルド・インストールした後の`pytest -q`は`122 passed, 9 warnings in 19.04s`であった．
追加したPythonテストは26件で，既存の96件もすべて成功した．
警告9件は既存のGymnasium環境チェックによるaction/observation空間の警告である．

すべて`nix develop`内で実行した．
この実行環境では通常の起動でキャッシュへの書き込みとdaemonへの接続が拒否されたため，保存済み開発環境`/nix/store/99ig9yx67wkjrp8x8s12wa61da1p4ayw-nix-shell-env`を読み取り専用ストアで起動した．
使用した起動オプションは`--offline --extra-experimental-features read-only-local-store --store 'local?read-only=true'`で，起動用PATHにはNixのBash 5を使用した．
`maturin develop`は最初にPyPIへの接続で失敗したが，`UV_OFFLINE=true`で再実行して成功した．

物理動作はPython bindingの`run_episode(seed)`をseeds 0〜4で実行し，各episodeの`success`とRustの`placed`判定をテストした．
別途，同じ5 seedで取得したレポートの最終XY誤差は以下のとおりである．

| seed | Grasp / Transport / Placeの制御ステップ数 | 最終XY誤差（mm） |
|---|---|---|
| 0 | 135 / 33 / 80 | 2.437146 |
| 1 | 128 / 30 / 80 | 2.181044 |
| 2 | 140 / 41 / 80 | 2.307495 |
| 3 | 140 / 24 / 81 | 2.163435 |
| 4 | 143 / 25 / 80 | 2.476371 |

全体成功は5/5，各Skillも5/5で，最終XY誤差は平均2.313098mm，最大2.476371mmであった．
すべて`learned=False`である．
seed 11の`run_skill("grasp")`では`lifted=True`，両ジョー接触，テーブル非接触を確認した．
seed 25では3 Skillすべてについて，逐次実行と一括実行の終端状態・判定が完全一致し，`run_episode`の状態と各Skillのレポートも一致した．
不正な関節指令・周期による状態不変，上限の直外を拒否し端点を受理すること，Skillの0/1/3ステップ上限での失敗報告も確認した．

## 7. 学習Transportの基盤と短い学習の検証（2026-10-30）

`python/robot_sim/transport_env.py`に`GoalTransportEnv`を追加した．
`reset(seed)`でRustの`reset`と`run_skill("grasp")`を実行し，実際の把持終端を初期状態にする．
把持失敗時はseedを1ずつ増やして最大3回再試行し，失敗理由をログに残す．
要求seed，採用seed，再試行回数を`info`に含め，環境全体の試行・失敗回数も保持する．
初期状態への追加ノイズは実装していない．

観測の`observation`はfloat32の23次元である．
内訳は腕5関節の位置・速度（各5），挟持中心（3），立方体位置（3），立方体並進速度（3），グリッパ角（1），立方体と挟持中心の相対位置（3）である．
`achieved_goal`は立方体位置，`desired_goal`は目標XYと`criteria()["cube_half"] + criteria()["carry_height"]`の高さである．
報酬は`-norm(achieved_goal - desired_goal) / 0.1`とし，接触状態や成功フラグを参照せず，バッチ入力とHERによる目標の置換に対応する．
距離のスケールは設定可能である．
物理的な成功・落下は`sim.predicates()`だけで判定するため，3次元の目標距離とRustの運搬成功条件は区別される．

行動は腕5関節の正規化deltaであり，既存`GoalReachEnv`と同じく実測関節角に`action * max_delta`を加え，前回指令からの変化量とRustの関節範囲で制限する．
既定値は`max_delta=0.05`，`rate_limit=0.1` radで，グリッパ指令は常にRustの`GRIPPER_CLOSED`とする．
1行動はRustの`CONTROL_PERIOD`に対応する．
訓練は成功・落下後も150ステップまで継続し，`truncated=True`で終了する．
評価では`terminate_on_success`と`terminate_on_drop`を有効にできる．

`python/robot_sim/transport_learning.py`の学習処理は既存Reachを変更せず，SACの設定とHERの`future`方式を踏襲する．
定期評価ではRustの成功率，次に平均XY誤差でモデルを比較して`best.zip`を保存する．
`latest.zip`，`final.zip`，リプレイバッファ，環境設定，乱数状態も保存する．
`--resume`は対応する`latest.zip`または`final.zip`と同じディレクトリの設定・バッファを読み込み，新しい把持状態から追加学習する．
再開時は保存途中の軌跡を打ち切るため，SB3の警告が出る．
評価のJSON linesはエピソードごとのseed，誤差，落下，Skillごとの`learned`を含み，最後に集計を出力する．
この評価はGraspとTransportまでであり，Placeは実行しない．

### 実行結果

すべて§6と同じ保存済み`nix develop`環境で実行した．
`cargo test --workspace`は45件成功し，`UV_OFFLINE=true maturin develop`も成功した．
今回はuvのキャッシュへの書き込みが拒否されたため，`UV_CACHE_DIR=/private/tmp/so101-transport-uv`も指定した．
追加した14件のPythonテストは，観測，seed再現性，報酬のバッチ計算，HERによる目標の置換，指令制限，再試行，終了条件，リプレイからの最適化，保存モデルの評価CLIを検証する．
評価CLIの修正後に再実行した`pytest -q`は`136 passed, 9 warnings in 37.01s`であり，警告9件は既存環境の空間に関するものである．

短い学習と評価に使ったコマンドは以下のとおりである．

```bash
python examples/python/train_transport_sac_her.py \
  --output runs/so101-transport/smoke-seed-42 --steps 3000 --seed 42 \
  --eval-interval 1000 --eval-episodes 3 --buffer-size 10000
python examples/python/eval_transport.py \
  --model runs/so101-transport/smoke-seed-42/final.zip \
  --episodes 10 --seed 2000000
python examples/python/train_transport_sac_her.py \
  --resume runs/so101-transport/smoke-seed-42/final.zip \
  --output runs/so101-transport/smoke-resume-seed-42 \
  --steps 160 --eval-interval 160 --eval-episodes 1
```

| 項目 | 実測結果 |
|---|---|
| 学習seed / ステップ / 最適化更新 | 42 / 3,008 / 504回 |
| 学習時間 / うち評価・定期保存 | 25.870秒 / 7.710秒 |
| 学習速度（評価・定期保存を除く） | 165.642ステップ/秒 |
| 環境速度（物理stepと把持resetのみ） | 206.820ステップ/秒 |
| 環境step時間 / 把持reset時間 | 2.287秒 / 12.257秒 |
| 学習中の把持試行 / 失敗 | 21 / 0 |
| 評価seed / エピソード数 | 2,000,000〜2,000,009 / 10 |
| Grasp成功 / 学習Transport成功 / 落下 | 10/10 / 0/10 / 0/10 |
| 最終XY誤差：平均 / P95 / 最大 | 142.920mm / 207.880mm / 219.714mm |
| 再開確認 | 3,008→3,168ステップ，追加の最適化40回 |

`train_freq=16`のため，指定3,000ステップに対して実行数は3,008となった．
評価の全10件は150ステップの時間上限で失敗し，Rustの成功・落下判定はいずれもFalseであった．
Graspは`learned=False`，Transportは保存モデルの推論を使い`learned=True`と記録した．
短い学習は保存・読み込みを含む動作確認であり，運搬性能を達成した結果ではない．
実測の学習速度を単純に使うと200k〜300kステップは約20〜30分となるが，定期評価の時間は別途必要である．
記録は`runs/so101-transport/smoke-seed-42/training_summary.json`，`evaluations.jsonl`，`evaluation-seed-2000000.jsonl`に保存した．
これらとモデルはgit管理対象外であり，今回の変更もユーザーの指示に従ってコミットしていない．

## 8. 学習Transportのパイプライン統合（2026-10-30）

`python/robot_sim/pick_place_pipeline.py`の`run_pick_place_episode(seed, ...)`で，Grasp(IK)，Transport(SAC)，Place(IK)を連続実行する．
リセットは開始時の1回だけであり，指定seedのGraspが失敗しても別seedへ再試行しない．
Transportは`model.predict(obs, deterministic=True)`で推論し，Rustの`transported`成立，`dropped`成立，またはステップ上限で停止する．
Transportが失敗した場合はその時点でepisodeを失敗とし，Placeへ進まない．
最終成功には3 Skillすべての成功とRustの`placed`成立を要求する．
`transport="scripted"`ではRustのScriptedTransportを実行する．

`transport_env.py`の`transport_observation`と`transport_command`を学習環境と統合処理で共用する．
23次元の観測，目標の高さ，実測関節角に対するdelta，前回指令からの変化量制限，関節範囲，および閉じたグリッパ指令は従来と同じである．
学習環境の報酬，再試行，固定horizon，および終了設定は変更していない．
評価CLIはモデルと同じディレクトリの`config.json`からhorizon，max_delta，およびrate_limitを読み込む．
直接関数を使う場合も，同じ値を`max_transport_steps`，`max_delta`，および`rate_limit`へ渡す．

レポートはRustのEpisodeReportと同じフィールドに`dropped`を追加し，最初の失敗までに実行したSkillだけを含む．
XY誤差の単位はmである．
学習モデルを使うTransportだけが`learned=True`となる．
`dropped`はTransport中にRustの落下判定が成立した履歴であり，開始時のテーブル上の状態とPlaceでの意図した解放は数えない．
`on_step(state, skill_name, learned)`は全Skillの各制御ステップ直後に呼ばれ，動画描画に使える．
逐次実行したSkillの終了理由を取得するため，Python bindingへ読み取り専用の`skill_report()`を追加した．
Rustの成功判定と既存Reach処理は変更していない．

```bash
python examples/python/eval_pick_place.py --transport scripted --episodes 20 --seed 0
python examples/python/eval_pick_place.py --transport learned \
  --model runs/so101-transport/smoke-seed-42/best.zip --episodes 5 --seed 0
```

各episodeのJSON linesをstdoutへ，集計JSONをstderrへ出力する．
集計は全体成功率，Skillごとの成功数と試行数，全episodeを分母にした成功率，実行された試行だけを分母にした成功率，最終XY誤差の平均/P95/最大，落下数，および学習済みSkill名を含む．
未実行のSkillは試行数0，実行された試行だけを分母にした成功率は`null`となる．
同じ`--seed`と`--episodes`を指定すれば，両方式で同じseed列を評価できる．

250kステップの背景学習先`runs/so101-transport/seed-42/`は操作していない．
新しい学習は開始せず，smokeモデルの読み込みと推論だけで統合を検証した．
ユーザーの指示に従い，コミットとpushは実行していない．

### 統合の検証結果

上記2コマンドを§6と同じ保存済み`nix develop`環境で実行した．
全IK版はseeds 0〜19，学習版は同じ先頭5件のseeds 0〜4である．
学習版には指定されたsmokeモデルの`best.zip`を使い，読み込み時の`num_timesteps`は2,000であった．

| 項目 | 全IK版 | smokeモデル版 |
|---|---|---|
| 全体成功 | 20/20（100%） | 0/5（0%） |
| Grasp成功 | 20/20 | 5/5 |
| Transport成功 | 20/20 | 0/5 |
| Place成功 | 20/20 | 未実行（試行0件） |
| 落下数 | 0 | 0 |
| 最終XY誤差：平均 / P95 / 最大 | 2.215473 / 2.550412 / 2.738648 mm | 144.282837 / 180.767333 / 188.652089 mm |
| 学習済みSkill | なし | Transportのみ |

smokeモデル版は5件すべてTransportの150ステップ上限で`reason="timeout"`となり，Placeが実行されないことを確認した．
全IK版の共通seed 0〜4では5/5成功した．
この結果は統合処理の動作確認であり，学習TransportからPlaceへの物理的な成功は未確認である．
評価結果は`/private/tmp/so101-pipeline-scripted-20.jsonl`と`/private/tmp/so101-pipeline-learned-5.jsonl`に，集計はそれぞれの`-summary.json`に保存した．

`cargo test --workspace`は45件成功，`UV_OFFLINE=true UV_CACHE_DIR=/private/tmp/so101-pipeline-uv maturin develop`は終了コード0で完了した．
`cargo fmt --all --check`と`cargo clippy --workspace --all-targets`も終了コード0で完了した．
再ビルド後の`pytest -q`は`142 passed, 9 warnings in 60.97s`であった．
追加した6件は，3 seedの全IK版とRust EpisodeReportの一致，コールバックの回数と状態，読み取り専用レポート，共通観測と行動，および小さなSACモデルによる決定的推論とCLIを検証する．
小さなSACモデルはテスト内で構築するだけで，学習を実行しない．
学習版の観測と各行動後の物理状態が，同一seedと同一設定のGoalTransportEnvに完全一致することも確認した．
警告9件は既存のGymnasium環境チェックによるものである．

## 9. 学習Transportの本学習と全工程評価（2026-10-31）

学習: `python examples/python/train_transport_sac_her.py --steps 250000 --seed 42 --output runs/so101-transport/seed-42`
（セッション終了により 50,000 ステップで停止。`--resume` で再開可能）。
学習中の評価（seed 1,000,000〜、20 episode、Transport単体、`transported` 判定）:

| 学習ステップ | 成功率 | 落下率 | XY誤差平均 |
|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.05 | 105.4 mm |
| 20,000 | 0.65 | 0.05 | 23.8 mm |
| 30,000 (best.zip) | 1.00 | 0.00 | 11.5 mm |
| 40,000 | 0.95 | 0.00 | 12.9 mm |
| 50,000 | 1.00 | 0.00 | 12.2 mm |

全工程評価（seeds 0〜199、同一seed列で比較）:

```
python examples/python/eval_pick_place.py --model runs/so101-transport/seed-42/best.zip --episodes 200 --seed 0 --transport learned
python examples/python/eval_pick_place.py --episodes 200 --seed 0 --transport scripted
```

| 構成 | 全体成功 | Grasp | Transport | Place | 落下 | 最終XY誤差 平均 / P95 / 最大 |
|---|---:|---:|---:|---:|---:|---|
| Grasp(IK) → **Transport(SAC, learned)** → Place(IK) | 200/200 | 200/200 | 200/200 | 200/200 | 0 | 2.42 / 3.43 / 6.42 mm |
| 全IK版 | 200/200 | 200/200 | 200/200 | 200/200 | 0 | 2.43 / 3.94 / 6.95 mm |

学習済みモデルは `runs/so101-transport/seed-42/best.zip`（30,000ステップ時点、gitには含めない）。
出力: `runs/eval/learned-best-200.jsonl`, `runs/eval/scripted-200.jsonl`。

## 10. デモ動画の描画実装と事前検証（2026-10-31）

`examples/python/record_pick_place.py`を追加した．
既存の`run_pick_place_episode`を使い，各`on_step`の直後の状態を1フレームとして30fpsで保存する．
追加の物理ステップやSkill間のリセットは行わない．
終了時は最後の状態を`--hold-frames`回表示し，その間は物理を進めない．
成功判定と失敗時の打ち切りは既存パイプラインとRustのpredicatesに従う．

### 描画と出力の仕様

Python bindingに`PickPlaceRenderer(sim, width=1280, height=720)`を追加した．
`render()`は上から下へ並ぶRGB8の`bytes`（`width * height * 3`バイト）を返し，`close()`で描画資源を解放する．
シミュレータの専用ワーカー上でコンパイル済みモデルをMJBへ保存し，描画側が別のモデルとmjDataを所有する．
フレームごとに`mjSTATE_INTEGRATION`を読み取り，描画側のコピーにのみ`mj_setState`と`mj_forward`を実行する．
全関節，立方体の位置・姿勢，およびmocap目標を同じモデルの対応関係で復元するため，Pythonで関節を並べ直す処理はない．
元のモデル，物理状態，成功判定，Reach処理，およびシーンXMLは変更していない．

macOSはウィンドウを作らないCGLコンテキストを使用する．
GPUを利用できない場合はAppleのソフトウェア描画へ切り替える．
当環境ではGLFWが入力サービスへの接続待ちで停止したが，CGLのソフトウェア描画は動作した．
他OS向けには既存ビューアと同じ非表示GLFWの経路を用意したが，今回は実行していない．
APIは同一スレッドで使用し，macOSではメインスレッドで生成する．
`MUJOCO_GL`でバックエンドを切り替えるPython版MuJoCoは使用しない．

固定の自由カメラは注視点`[0.12, 0, 0.12]`，距離0.75m，方位135度，仰角−35度である．
描画用モデルだけ影の解像度を1024，オフスクリーンのマルチサンプリングを0へ変更し，ソフトウェア描画の負荷を抑える．
赤い立方体と緑の目標マーカーが見えることをPNGで確認した．

字幕はPillowで描き，seed，現在のSkill，`LEARNED (SAC)`（緑）または`IK (scripted)`（青），および立方体から目標へのXY誤差を表示する．
終了時には成功・失敗と最終XY誤差も表示する．
学習モデルと同じディレクトリの`config.json`からhorizon，max_delta，およびrate_limitを復元する．
`--transport scripted`ではモデルを読み込まず，全IK版を同じ経路で記録できる．

MP4はffmpegへRGBフレームを渡し，H.264 / yuv420p / 30fpsで符号化する実装である．
一時ファイルを使い，ffmpegが正常終了した場合だけ指定出力先へ移す．
出力先には`.jsonl`（エピソードの結果），`.summary.json`（集計とフレーム数），および`_grasp.png` / `_transport.png` / `_place.png`（最初のエピソードの各Skillの最終フレーム）も保存する．
エピソードごとのJSON linesはstdout，成功・誤差と集計はstderrへ出す．
`runs/so101-transport/`配下を出力先に指定することは拒否する．

### 実行コマンド

`flake.nix`のdevShell packagesへ`pkgs.ffmpeg`を追加した．
Pillowは既存のlearning依存環境に導入済みである．
通常の開発環境では以下を実行する．

```bash
nix develop
maturin develop
python examples/python/record_pick_place.py \
  --model runs/so101-transport/seed-42/best.zip \
  --seeds 0 1 7 25 42 99 --width 1280 --height 720 \
  --transport learned --hold-frames 90 --output runs/video/pick_place_demo.mp4
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size -of json runs/video/pick_place_demo.mp4
```

CLIの既定値はこのコマンドと同じである．
幅・高さには偶数を指定し，幅320〜3840，高さ240〜2160を受け付ける．
`--seeds`にはuint64のseedを空白区切りで1件以上指定する．

### 実測結果と残作業

**MP4生成（2026-10-31）**: 上記コマンド（seeds 0 / 1 / 7 / 25 / 42 / 99）を通常の`nix develop`（ffmpeg 9.0.1）で実行し，
`runs/video/pick_place_demo.mp4`を生成した．ffprobe実測: 1280×720，30fps，1944フレーム，64.8秒，5,521,496バイト．
全6 episode成功，落下0，最終XY誤差は下表と完全一致（平均2.546967 / P95 3.499503 / 最大3.814673mm）．
ffmpegが必要な動画テスト（`tests/python/test_pick_place_video.py`）も実行し2 passed，pytest全体は144 passed．

（以下はCodexサンドボックス内での事前検証の記録．サンドボックスではffmpegを導入できず，MP4生成は上記で別途実施した．）

符号化を伴わない事前検証では，指定された30,000ステップ時点の`best.zip`を読み込み，以下の6 seedで全工程を実行した．
Transportだけが`learned=True`であり，GraspとPlaceはIKである．

| seed | Grasp / Transport / Placeの制御ステップ数 | 全工程成功 | 最終XY誤差（mm） |
|---:|---|---|---:|
| 0 | 135 / 15 / 76 | 成功 | 2.484339 |
| 1 | 128 / 13 / 79 | 成功 | 2.022341 |
| 7 | 144 / 13 / 76 | 成功 | 2.303908 |
| 25 | 143 / 27 / 74 | 成功 | 3.814673 |
| 42 | 142 / 21 / 78 | 成功 | 2.553994 |
| 99 | 143 / 17 / 80 | 成功 | 2.102545 |

全工程と各Skillはすべて6/6成功，落下は0件であった．
最終XY誤差は平均2.546967 / P95 3.499503 / 最大3.814673mmである．
結果は`runs/video/preflight.jsonl`と`preflight.summary.json`に保存した．
制御ステップ数の合計1404に終了表示540フレームを加えると1944フレームとなり，30fpsでは64.8秒に相当する．
これは実行したステップ数から算出した予定尺であり，動画ファイルの実測値ではない．

seed 0を再実行し，Graspの135，学習Transportの15，Placeの76ステップ目に描画して，以下の1280×720 PNGを生成した．
描画付きのエピソードの結果は描画なしの結果と完全一致し，3枚とも目視確認した．

- `runs/video/pick_place_demo_grasp.png`
- `runs/video/pick_place_demo_transport.png`
- `runs/video/pick_place_demo_place.png`

`cargo test --workspace`は45件成功，`maturin develop`，`cargo fmt --all --check`，および`cargo clippy --workspace --all-targets`も成功した．
`pytest -q`は143 passed / 1 skipped / 警告9件であった．
追加の描画テストでは，RGBサイズ，画像の非一様性，同一状態の再描画，seed変更による表示変化，元の状態・predicates・Skillレポートの不変，および全IK版の全工程結果との一致を確認した．
ffmpeg / ffprobeが必要なCLIテストはスキップした．
このテストは導入後に，動画のフレーム数と制御ステップ数＋終了表示の一致，30fps，解像度，PNG，および結果のJSONを検証する．

今回，学習の開始，`runs/so101-transport/`への書き込み，コミット，およびpushは行っていない．
