# SO-101 Pick & Place：現状確認・実装・計画

2026-10-30 時点。数値はすべてこのリポジトリで実際に実行した結果のみ記載する。

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
- 既存 `.venv` はリポジトリ移動前のパス（`/Users/natsu/PhysicalAI/...`）を参照しており `maturin develop` が失敗した。
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
