# 接触物理の監査・摩擦の分離・手首補正の探索・Place失敗の切り分け

v3までの結果（[OBJECT_PHYSICS.md](OBJECT_PHYSICS.md)）を受けて，固定制御と学習制御の性能差がどこから来るかを調べた記録。
数値はすべて，記載したコマンドの出力から書いている。生成物（`runs/`）はgit管理外である。

**環境のバージョン**: この文書では接触環境を2つ使う。結果を混ぜないこと。

| 名前 | 内容 |
|---|---|
| legacy | 従来の摩擦DR。立方体geomのpriorityを2に上げ，1つのμがジョー接触とテーブル接触の両方を決める。ジョー接触はcondim 6→4，テーブル接触のsolrefは0.015→0.01に変わる（OBJECT_PHYSICS.md §1）。v1〜v3の学習・評価はすべてこの環境。 |
| pairwise | 今回追加（`ContactModel::Pairwise`）。ジョーとの滑り摩擦はジョーpad側のgeom（priority 1）に，テーブルとの滑り摩擦は立方体とテーブルの両geomに同じ値を書く。priority・condim・solrefは変えない。DRなしなら従来シーンとビット単位で同じ軌道になる（テスト済み）。 |

pairwiseの条件は「ジョーμ / テーブルμ」で書く。`floor=same`はテーブルμ＝ジョーμ（legacyと同じ組合せで，condim/solrefだけ固定），`floor=nominal`はテーブルμをシーン値1.2に固定した場合を表す。

## 実装と再現方法

| 目的 | 実装 |
|---|---|
| 接触環境 | `PickPlaceConfig::contact_model`（`legacy` / `pairwise`），`ContactOverrides`（pairwiseでジョー/床のcondim・solref），`ObjectRandomization::floor_friction`．Python: `PickPlaceSimulator(contact_model="pairwise", jaw_condim=…, …)`，`set_object_randomization(friction=, floor_friction=)`，`object_physics()`に`jaw_friction` / `floor_friction` / `contact_model` |
| 接触の監査 | `PickPlaceSim::contact_details()`（接触点・法線・力・めり込み），`solver_options()`，`geom_info()`．Python: `contact_details()`．例: `contact_audit`，`penetration_probe`，`grip_sensitivity`．集計: `scripts/analyze_contact_audit.py` |
| 手首補正の探索 | `examples/python/eval_wrist_sweep.py`（`--wrist`，`--end-at base/success`，`--contact-model`，`--floor`），一括実行 `scripts/run_wrist_sweep.sh OUT JOBS selection|final`，表 `scripts/summarize_wrist_sweep.py` |
| Placeの切り分け | `ScriptedPlace::level` / `release_on_contact` / `release_opening`，skill名 `place_level` / `place_contact` / `place_gentle`（組合せ可），`run_pick_place_episode(place=…, settle_before_place=…)`，`examples/python/eval_place_variants.py` |
| 低次元Residual RL | `LowDimResidualGrasp`（`action_mode="wrist"`: 1次元，`"wrist_lift"`: 手首＋持ち上げ方向＋速度の3次元），`Skill::adjust` / `skill_adjust`（Rust Graspの持ち上げ目標を変更），`train_grasp_sac.py --action-mode … --base-wrist …` |

テスト: Rust `crates/mujoco-runtime/tests/contact_model.rs`（5件），Python `tests/python/test_physics_audit.py`（6件）．

## 1. 物理モデルの監査（Phase 1）

### 1.1 静的な設定（`contact_audit`の`model`行，Menagerie `robotstudio_so101/so101.xml`）

```bash
cargo run --release -p mujoco-runtime --example contact_audit -- 3000000 4 > runs/physics-audit/audit.jsonl
python scripts/analyze_contact_audit.py runs/physics-audit/audit.jsonl > runs/physics-audit/audit_summary.txt
```

| 項目 | 値 |
|---|---|
| ソルバ | timestep 0.005 s，integrator implicitfast，Newton，iterations 10，ls_iterations 20，elliptic cone，impratio 10，noslip 0 |
| 6関節のサーボ（共通） | position actuator kp 998.22，kv 2.731，forcerange ±2.94 N·m，damping 0.6，frictionloss 0.052，armature 0.028 |
| グリッパ関節 | range −0.1745〜1.745 rad，ctrlrange 同じ．開 0.6，閉指令 −0.17（skills.rs） |
| ジョーの衝突geom | 固定ジョー: box×7，指先の球×3（半径0.75 mm），capsule×1，凸包mesh×1．可動ジョー: box×3，指先の球×3，凸包mesh×2．カメラマウントのbox×2．すべてclass `collision_gripper`: friction (1, 0.005, 0.0005)，condim 6，solref (0.01, 1)，priority 1，solimp (0.9, 0.95, 0.001) |
| 固定ジョーのpad | box5〜7 が x = −10.8 / −12.5 / −14.3 mm（z = −90.5 / −72.7 / −53.0 mm）の階段状．指先に向かって約5.3°開いた面になる |
| 立方体 | 25 mm，30 g（DRで変更），慣性 3.125e-6 kg·m²（= m a²/6），friction (1.2, 0.005, 1e-4)，condim 4，solref (0.01, 1)，priority 0 |
| テーブル | plane，friction (1, 0.005, 1e-4)，condim 3，solref (0.02, 1)（既定値） |
| DRなしの実効値 | ジョー接触: dim 6，μ 1.0，solref 0.01（priorityでジョー側）．テーブル接触: dim 4，μ 1.2（max），solref 0.015（平均） |

`model_access.c`の`robot_sim_geom_contacts`で`mjData.contact`と`mj_contactForce`を読み，接触点・法線・立方体に働く力（世界座標）・めり込み（`dist`）を5 ms（物理1ステップ）ごとに記録した．
5 msごとに進めても，制御周期ごとに進めた場合と最終状態がビット単位で一致することを全36 episodeで確認した（`equivalence`行）．
Newtonの検算（m a = Σ接触力 + m g）の相対残差は中央値0.000で，力の取り出し方に誤りはない．

### 1.2 39 Nの法線力の由来と実機との比較

- 位置サーボのkp = 998 N·m/radに対しforcerangeは2.94 N·mなので，**指令が接触角から3 mrad以上内側にあれば出力は飽和する**．閉指令−0.17 radに対し25 mmの立方体では約0.09 radで止まるため，Close以降は常に飽和している（OBJECT_PHYSICS.md §11.3と一致）．
  したがって，以前の「グリッパ指令を0.0〜0.08 radに弱める」実験も，実際には挟む力を変えていなかった（接触角0.09 radとの差が20 mrad以上あり，飽和したまま）．
- 法線力 ≈ 2.94 N·m ÷ 可動ジョー上の接触点までの距離（約0.07 m）≈ 40 N．実測は片側39〜49 N（Close終了時），Lift中は35〜41 Nだった．Closeで最初にぶつかった瞬間は85〜93 Nに達した．
- 2.94 N·mは，STS3215の12 V時のストールトルク30 kg·cmと同じ値である．販売店の仕様では定格トルクは約10 kg·cm（0.98 N·m）であり，LeRobotのSO-101設定はグリッパのトルク上限を50%，保護電流を50%，過負荷時のトルクを25%に下げると記載されている．
  **シミュレーションのグリッパは，実機が連続では出さない（出させない）ストールトルクで常に握っている**．実機相当の連続出力（0.98〜1.47 N·m）なら法線力は13〜20 N程度と見積もられる．
- 25 mm・20 gの立方体に対して39 Nは重さの約200倍で，μ ≥ 0.2なら摩擦で支えられる力に大きな余裕がある．低摩擦での失敗はこの力の不足では説明できない（§1.4）．

### 1.3 接触点とめり込み

- 把持中の接触は**指先のgeomだけ**だった．固定ジョーは`fixed_jaw_box3/box4/sph_tip1〜3`，可動ジョーは`moving_jaw_box2/box3/sph_tip1〜3`と指先側の凸包mesh．padの中ほど（box5〜7）にはほとんど触れない．
- Close終了時の法線は，閉じ方向（gripperframeの+z）から1°以内にそろっていた．把持開始時点でくさびのような斜めの力は出ていない．
- **めり込みが大きい**: 半径0.75 mmの指先球に対して，片側のめり込みは1.3〜2.6 mmだった．球の直径より深く，接触点と法線の位置が不正確になる．
- **めり込みは質量に依存する**（`penetration_probe`，pairwise・シーン摩擦・各10 seed，IK Grasp終了時の最大めり込み）:

| 質量 | 10 g | 20 g | 50 g | 100 g | 200 g |
|---|---|---|---|---|---|
| 2.94 N·m（法線力39 N） | 2.73 mm | 1.73 mm | 1.06 mm | 0.76 mm | 0.60 mm |
| 1.47 N·m（法線力19.5 N） | 1.73 mm | 1.17 mm | 0.75 mm | 0.57 mm | 0.42 mm |

同じ挟む力でも，軽い物体ほど深くめり込む．MuJoCoのsoft contactは`solref`（時定数）で加速度の応答を決めるので，力に対する剛性は物体の有効質量に比例する．
実際の接触では，めり込み（弾性変形）は質量ではなく力と材料で決まる．これはモデルの性質による人工的な依存であり，「軽いほど失敗しやすい」という以前の観測の一部はこれで説明できる可能性がある．
対策候補: 負の`solref`（剛性・減衰を直接指定）や`solimp`の調整．`ContactOverrides::jaw_solref`は負の値も受け付けるようにしたが，今回は既存結果との比較を優先し，評価には使っていない．

### 1.4 低摩擦での押し出しの経過（法線方向と横加速度）

20 g・μ0.1（pairwise，ジョーμ0.1・テーブルμ0.1，seed 3,000,000）の失敗を5 msごとに追った:

1. Close終了時の立方体は挟持中心から閉じ方向に約0.6 mmで，両側40 N強の力で挟まれ，傾き0.7°．
2. Lift中の0.45 sは保持したまま30〜54 mm持ち上がるが，立方体はジョーに対して**指先方向（gripperframe +x）へ0.8 → 2.4 mmずつ滑っていく**．
3. 54 mmの時点で，指先から抜けるように一気に外れる（1制御ステップで指先方向へ13 → 36 mm，傾き25°）．
   横加速度は外れる瞬間に190〜860 m/s²に達し，抜けた立方体はテーブルに落ちて5〜11 mmめり込んだ．

接触が指先の小さなgeomに限られ，摩擦が法線力の0.1倍しかないと，指先側へ開いたジョー面（固定ジョーのpadが約5°開いている）に沿って立方体が押し出される．
OBJECT_PHYSICS.mdの「くさび効果」の解釈は，**把持開始時の法線の傾きではなく，持ち上げ中の指先方向への滑りとその後の抜け**として観測された．

### 1.5 ジョー摩擦とテーブル摩擦の切り分け（pairwise，同じ4 seed）

| 設定 | IK Grasp成功 | Close終了時の立方体（指先方向 x） |
|---|---|---|
| legacy 20 g・μ0.1（ジョー・テーブルとも0.1，condim 4，solref 0.01） | 1/4 | +0.6〜+1.4 mm |
| pairwise 20 g・ジョー0.1・テーブル1.2 | **4/4** | **−1.6〜−8.2 mm（奥側）** |
| pairwise 20 g・ジョー0.1・テーブル0.1 | 0/4 | +0.7〜+1.0 mm |
| pairwise 20 g・ジョー0.1・テーブル1.2・jaw condim 4 | 4/4 | −1.6〜−8.2 mm |
| pairwise 200 g・ジョー0.1・テーブル1.2 | 4/4 | −1.8〜+0.6 mm |

- **低摩擦でのGrasp失敗は，ジョーの摩擦だけでは起きず，テーブルの摩擦も低いときに起きた．** テーブル摩擦が高いと，Closeでジョーが当たった瞬間に立方体が2〜3 mm持ち上がって奥（掌側）へ入り，その位置では持ち上げても抜けなかった．
- jawのcondim 6と4の違いは結果を変えなかった．
- legacy環境では1つのμがジョーとテーブルの両方を下げるので，この2つの効果が分離できていなかった．

### 1.6 物理モデルの問題点のまとめ

1. グリッパは実機の定格を超えるストールトルク（2.94 N·m）で常に飽和しており，指令による力の調整ができない．
2. 指先の小さなprimitiveだけで把持し，めり込みが球の直径を超える（1.3〜2.7 mm）．めり込みは質量に依存する（モデル上の人工的な依存）．
3. legacyの摩擦DRはジョーとテーブルの摩擦を連動させ，condimとsolrefも変える．
4. 押し出しは「指先に向かって開いたジョー面＋低いジョー摩擦」で起き，把持位置（奥か指先か）に強く依存する．把持位置はClose時のテーブル摩擦にも左右される．

既存の学習・評価との比較可能性を保つため，**シーン・グリッパの力・solref/solimpは変更していない**．pairwise環境は摩擦の書き込み方だけを変えたもので，DRなしでは従来と同じ物理である．

## 2. 摩擦だけを変える環境でのIK再評価（Phase 2）

同じ選択用seed（4,000,000〜029）・同じ19条件（容易5・難10・範囲外4，OBJECT_PHYSICS.md §6と同じ）で，IK（＝wrist 0の固定補正，終了はIKのHold後）の全工程成功を3つの接触設定で比べた（`runs/wrist-sweep/selection_tables.md`）．

| 環境 | 容易（/150） | 難（/300） | 範囲外（/120） | 合計（/570） |
|---|---|---|---|---|
| legacy（ジョーμ＝テーブルμ，condim 4・solref 0.01に変わる） | 149 | 66 | 23 | 238 |
| pairwise floor=same（ジョーμ＝テーブルμ，condim/solrefは元のまま） | 149 | 124 | 31 | 304 |
| pairwise floor=nominal（ジョーμのみ変更，テーブル1.2） | 150 | 151 | 38 | 339 |

各セルはGrasp / 保持 / 全工程の詳細を`selection_tables.md`に記録した．

- **同じ摩擦の値でも，legacyの副作用（ジョー接触のcondim 6→4，テーブル接触のsolref 0.015→0.01，立方体のpriority 2）だけで難条件の全工程が124 → 66に半減した．** 以前の難条件の結果（IK 68/300など）は，摩擦の効果とこの接触モデル変更の効果が混ざった値である．
- テーブル摩擦を元の値に保つと，さらに151/300まで上がる（特にGrasp: 難条件で242/300．legacyのIKは176/300）．
- `grip_sensitivity`（IK，pairwise，各20 seed，`runs/physics-audit/grip_sensitivity.jsonl`）の結果は§2.1に示す．

### 2.1 グリッパの力・ジョー摩擦・テーブル摩擦（IK，pairwise，seed 3,000,000〜019）

`cargo run --release -p mujoco-runtime --example grip_sensitivity -- 3000000 20`．グリッパのforcerangeをパラメータ中心経由で1.0倍（2.94 N·m）と0.5倍（1.47 N·m，LeRobotのトルク上限50%相当）にした．

| gripper limit N·m | mass kg | jaw μ | table μ | Grasp | hold | success (/20) | jaw normal N |
|---|---|---|---|---|---|---|---|
| 2.94 | 0.02 | 1.0 | 1.2 | 20 | 20 | 20 | 38.9 |
| 2.94 | 0.02 | 1.0 | 0.1 | 20 | 20 | 20 | 38.9 |
| 2.94 | 0.02 | 0.3 | 1.2 | 20 | 20 | 16 | 41.3 |
| 2.94 | 0.02 | 0.3 | 0.1 | 20 | 20 | 20 | 39.0 |
| 2.94 | 0.02 | 0.1 | 1.2 | 19 | 17 | 6 | 44.5 |
| 2.94 | 0.02 | 0.1 | 0.1 | 3 | 3 | 1 | 50.4 |
| 2.94 | 0.1 | 1.0 | 1.2 | 20 | 20 | 20 | 39.3 |
| 2.94 | 0.1 | 1.0 | 0.1 | 20 | 20 | 20 | 39.3 |
| 2.94 | 0.1 | 0.3 | 1.2 | 20 | 20 | 20 | 39.2 |
| 2.94 | 0.1 | 0.3 | 0.1 | 20 | 20 | 20 | 39.3 |
| 2.94 | 0.1 | 0.1 | 1.2 | 20 | 20 | 20 | 39.9 |
| 2.94 | 0.1 | 0.1 | 0.1 | 20 | 18 | 18 | 39.5 |
| 1.47 | 0.02 | 1.0 | 1.2 | 20 | 19 | 19 | 19.4 |
| 1.47 | 0.02 | 1.0 | 0.1 | 20 | 19 | 19 | 19.5 |
| 1.47 | 0.02 | 0.3 | 1.2 | 20 | 20 | 20 | 19.5 |
| 1.47 | 0.02 | 0.3 | 0.1 | 20 | 20 | 20 | 19.6 |
| 1.47 | 0.02 | 0.1 | 1.2 | 15 | 11 | 7 | 21.0 |
| 1.47 | 0.02 | 0.1 | 0.1 | 15 | 10 | 6 | 20.3 |
| 1.47 | 0.1 | 1.0 | 1.2 | 20 | 20 | 20 | 19.5 |
| 1.47 | 0.1 | 1.0 | 0.1 | 20 | 20 | 20 | 19.6 |
| 1.47 | 0.1 | 0.3 | 1.2 | 20 | 20 | 20 | 19.7 |
| 1.47 | 0.1 | 0.3 | 0.1 | 20 | 20 | 20 | 19.7 |
| 1.47 | 0.1 | 0.1 | 1.2 | 20 | 20 | 20 | 19.6 |
| 1.47 | 0.1 | 0.1 | 0.1 | 20 | 20 | 20 | 19.6 |

- **テーブル摩擦だけが低い（ジョー1.0・テーブル0.1）場合は全工程20/20**．以前「μ ≤ 0.1ではPlace後にテーブル上で滑る」とした失敗は，テーブル摩擦単独では起きなかった．
- **ジョー摩擦だけが低い（20 g・ジョー0.1・テーブル1.2）場合，Graspは19/20だがPlaceで失敗する**（全工程6/20）．立方体がジョー内で回り，傾いたまま渡されるため（§4.2）．
- **Grasp失敗はジョーとテーブルの両方が低いときに集中した**（20 g: 3/20）．
- **グリッパの力を半分にすると，20 g・ジョー0.1・テーブル0.1のGraspは3 → 15/20に改善した．** 力が強いほどめり込みが深く（§1.3），指先からの押し出しが起きやすい．一方，20 g・ジョー0.1・テーブル1.2ではGraspが19 → 15/20に下がった．力の上限の妥当性は成功率に直接影響する．
- 100 gではどの組合せでも18〜20/20だった．軽い物体ほど失敗しやすい傾向は，めり込みの質量依存（§1.3）と一致する．


**以下，legacyとpairwiseの結果は別の実験として扱う．** v1〜v3の学習モデルはlegacyで学習・選択されたので，pairwiseでの評価は学習時と異なる物理での評価になる．


## 3. 固定手首補正の網羅的評価（Phase 3）

`scripts/run_wrist_sweep.sh runs/wrist-sweep 6 selection`（wrist_roll = −0.20, −0.15, −0.10, −0.05, 0, +0.05, +0.10 rad．IK＋一定のオフセットで，終了はIK GraspのHold後＝v3と同じ受け渡し）．
各オフセットは同じseed・同じ初期位置・目標位置・質量・摩擦で評価した．選択用seed（4,000,000〜029）で探索し，最終評価seed（3,000,000〜029）とは分けた．

### 3.1 選択用seedの全工程成功（/570）

| 環境 | −0.20 | −0.15 | −0.10 | −0.05 | 0（IK） | +0.05 | +0.10 |
|---|---|---|---|---|---|---|---|
| legacy | 242 | **251** | 243 | 241 | 238 | 239 | 242 |
| pairwise floor=same | 306 | 310 | 315 | **317** | 304 | 311 | 311 |

Grasp成功（/570）: legacy 353〜380，pairwise 427〜450．容易な条件はすべてのオフセットで149〜150/150．

### 3.2 オフセットの効果と条件依存性はノイズと区別できない

`scripts/wrist_noise_test.py`（`runs/wrist-sweep/noise_test_selection.txt`）: 全オフセットが同じseedを使うので，各（条件，seed）の中でオフセットのラベルを並べ替える対応ありの並べ替え検定を行った．
統計量は条件ごとのオフセット間の成功数の分散の和で，オフセットの主効果と条件×オフセットの交互作用の両方を含む．

| 環境 | 指標 | 統計量 | 帰無分布の平均 | p値 | 条件ごとの最良 − 単一の最良（同じデータ，楽観的） |
|---|---|---|---|---|---|
| legacy | Grasp | 261 | 368 | 0.96 | +23 |
| legacy | 全工程 | 97 | 109 | 0.62 | +4 |
| pairwise | Grasp | 270 | 270 | 0.48 | +16 |
| pairwise | 全工程 | 113 | 170 | 0.95 | +11 |

- **どちらの環境でも，固定オフセットの違いによる成功数の差は，オフセットを入れ替えても生じる程度のばらつきに収まった．** 条件ごとの最良オフセットは7つの値に散らばり（legacyで各値が5〜11条件で最良），質量や摩擦との対応は見られなかった．
- 以前の「wrist_roll −0.1が最良」（OBJECT_PHYSICS.md §5，選択用28/30 vs IK 24/30，最終評価253 vs 240/570）も，この程度の差である．
- 手首の小さなオフセットは，接触の結果を系統的に良くするのではなく，カオス的な接触過程の初期条件を変えて結果を「引き直す」ように働いている．低摩擦での失敗は§1.4–1.5の把持位置（奥か指先か）でほぼ決まり，その位置はClose時の衝突で決まる．
- 姿勢の指標（Grasp終了時の傾き，Place開始時の傾き，ジョー内の滑り）にも，オフセットに対する系統的な違いはなかった（`selection_tables.md`の後半．難条件では立方体がジョー内で大きく回り，Place開始時の傾きの平均は60〜77°）．

### 3.3 最終評価seed（3,000,000〜029）での確認

`FINAL_WRISTS="-0.15 -0.05" scripts/run_wrist_sweep.sh runs/wrist-sweep 6 final`（legacyの難・範囲外では7つのオフセットをすべて実行）．表は`runs/wrist-sweep/final_tables.md`，検定は`noise_test_final.txt`．

- legacyのIK（240/570）と，元の固定補正（wrist −0.10，受け渡しは0.3 sの持ち上げ時点．253/570）は，OBJECT_PHYSICS.md §11.5の値と完全に一致した（再現性の確認）．
- **受け渡しのタイミングを揃えると，オフセットの差は消えた．** 難＋範囲外（/420）の全工程: IK 91，受け渡しをHold後にしたオフセット −0.20〜+0.10で88〜98．一方，受け渡しを0.3 sの持ち上げ時点にすると，**オフセット0でも103，−0.10でも103**だった．
  つまり，以前の「固定wrist_roll −0.1が最良」の差（+13/570）は**手首の角度ではなく受け渡しのタイミング**（IKのHoldを待たずにTransportへ移る）によるものだった．早い受け渡しは低摩擦でHold中に抜け落ちる前にTransportへ入れるためと考えられる（Grasp成功の判定は保持し続けることを要求しない）．
- 対応ありの並べ替え検定（legacy，7オフセット，14条件）: Grasp p = 0.19，全工程 p = 0.55．条件ごとに最良のオフセットを選ぶ利得（同じデータで選ぶ楽観的な値）は全工程で+12で，帰無分布の95%点（14）未満．選択用seedで選んだ条件ごとのオフセットは，最終seedでは別のオフセットに入れ替わった．

**Phase 3の結論: 質量・摩擦ごとの最適な手首補正角は存在するとは言えない．** 固定オフセットの選択はノイズの範囲で，手首角を状態に応じて変える学習で改善できる余地は小さいと予想した（Phase 5で確認）．



## 4. Place失敗の切り分け（Phase 4）

`examples/python/eval_place_variants.py`は，seedごとに同じGraspとIK Transportを再実行し（シミュレーションは決定的．Place開始時の状態が全変種で一致することを全seedで確認），Transport後の処理だけを変える．
状態の保存・復元は実装せず，決定的な再実行で同じ物理状態を作った．集計は`scripts/summarize_place_variants.py`．

| 変種 | 内容 |
|---|---|
| place | 現在のIK Place（テーブルの2 mm上まで下げて開く→0.6 radまで開いて上昇） |
| settle | Transport後にbest-effort IK Stabilize（停止・静止待ち）を挟んでからPlace |
| level | 下降中に閉じ方向の軸まわりに手を回し，立方体の上面を水平にしてから離す |
| contact | 立方体がテーブルに触れるまで下げてから開く |
| gentle | 0.25 rad（立方体幅をわずかに超える開き）までしか開かずに上昇する |

失敗の分類: 「disturbed」= 離した後に一度はゴール上で静止（Rustの`placed`）したが，開く・上昇するジョーに当たって動いた．「never placed」= 一度も静止しなかった（転倒・滑り）．

### 4.1 学習Grasp（v1 B seed 42）からのPlace，legacy，20 g / 50 g・μ0.8，seed 3,000,000〜029

（`runs/place-variants/v1-h4-seed-42-easy.jsonl`．placeの全工程成功16/30・23/30はOBJECT_PHYSICS.md §6の同じ条件の値と一致した）

| 変種 | 全工程成功（/60） | Placeまで到達 | Place成功 | 失敗: never placed | 失敗: disturbed | 開始時傾き <3° / 3–6° / ≥6° のPlace成功 |
|---|---|---|---|---|---|---|
| place | 39 | 55 | 39 | 5 | **11** | 34/45 / 4/9 / 1/1 |
| settle | 39 | 55 | 39 | 5 | 11 | 33/45 / 5/9 / 1/1 |
| level | 42 | 55 | 42 | 13 | 0 | 36/45 / 5/9 / 1/1 |
| contact | 25 | 55 | 25 | 30 | 0 | 20/45 / 4/9 / 1/1 |
| **gentle** | **49** | 55 | **49** | 6 | 0 | 40/45 / 8/9 / 1/1 |
| gentle_contact | 43 | 55 | 43 | 12 | 0 | 36/45 / 6/9 / 1/1 |

（level / contact / gentle系のnever/disturbedの分類は，変種のskill名を判定から漏らしていた不具合の修正前に実行したため，すべてneverに数えられている．成功数は有効．）

### 4.2 IK GraspからのPlace（IKのPlaceが失敗する条件）

| 環境・条件 | place | settle | level | contact | gentle |
|---|---|---|---|---|---|
| legacy 0.05 kg・μ0.1（/30） | 2 | 1 | 3 | 2 | 4 |
| legacy 0.1 kg・μ0.1（/30） | 6 | 6 | 10 | 11 | 9 |
| pairwise 0.02 kg・ジョー0.1 / テーブル1.2（/30） | 12 | 13 | 17 | 15 | **23** |
| pairwise 0.1 kg・ジョー0.1 / テーブル1.2（/30） | 28 | 28 | 25 | 25 | **29** |

- pairwise（ジョー0.1）では，Place開始時の立方体がジョー内で大きく回っていても（開始時傾き3〜6°で9/19 → gentleで18/19），gentleで52/56まで回復した．
- legacyの低μ（テーブルも0.1）では，どの変種でもPlace成功は31回中8〜13回にとどまった．Place開始時の傾きは31件中30件が6°以上で，離した後に低摩擦のテーブル上で静止しない失敗（never placed）が主である．

### 4.3 Place失敗の原因（結論）

1. **以前の解釈（ジョー内の傾きによる転倒）は一部しか正しくなかった．** 容易な条件での学習GraspのPlace失敗16件のうち11件は，立方体が一度ゴール上で水平に静止した後，**0.6 radまで開くジョーと上昇する手が立方体に当たって倒した**ものだった（5 msごとの追跡で，離した後の立方体は傾き0°・速度0で静止→ジョーと接触して最大27 N・テーブル反力40 N→0.85 m/sではじかれ90°横倒し）．
2. 学習Graspは立方体をジョーの奥・高い位置で渡すため，離す高さが高く（6.4 mm，IKは3.4 mm），開くジョーの軌道に立方体が残る．IK Graspでは同じseedで接触は起きなかった．
3. **静止待ち（settle）は効果がない**（39 → 39）．速度ではなく，離す姿勢と離し方の問題である．
4. **接地を確認してから離す（contact）は悪化した**（39 → 25）．押し付けたまま開くと，ジョーの弾性めり込み（§1.3）が解放されて立方体がはじかれる．
5. **開きを小さくする（gentle）が最も効いた**: 学習Grasp 39 → 49/55，IK（pairwise・ジョー0.1）40 → 52/56．水平化（level）は小さな改善（39 → 42）．
6. 低いテーブル摩擦では，離した後に滑って許容範囲外に出る失敗が残り，Place制御では解決できない（legacy 0.1/0.1）．

したがって，**学習Graspの「容易な条件での全工程低下」の大半はPlace制御側（開き方と退避）で吸収できる問題**であり，Grasp側を学習で直す必要はなかった．
これはTransport・Placeを変えると既存の評価プロトコル（IK Place）と比較できなくなるため，既定のPlaceは変更していない（`place="place_gentle"`で選べる）．

## 5. 低次元Residual RL（Phase 5）

Phase 3では改善余地を確認できなかったが，予想を直接確かめるため短い学習を行った（長時間の学習はしていない）．

- ベースライン: IK＋選択用seedで最良だった固定オフセット wrist_roll −0.15 rad（受け渡しはHold後）．
- 行動: (A) 1次元 `wrist`: wrist_roll = −0.15 + 0.1·a₀ rad．(B) 3次元 `wrist_lift`: (A)＋持ち上げ目標を閉じ方向に±10 mmずらす＋持ち上げ速度を0.5〜1.5倍（Rustの`ScriptedGrasp::adjust`経由．関節範囲・速度制限はIKのMotionがそのまま守る）．グリッパはIK指令のまま（力制限で飽和しており権限がないため）．
- 観測: v1〜v3と同じ41次元（質量・摩擦は含まない）．報酬・終了・チェックポイント選択はv3と同じ（持ち上げ進捗，Hold終了まで実行，安定した受け渡しと後段IK成功の終端報酬，選択は全工程成功率）．legacy環境で学習（v3と比較するため），質量0.02〜0.1 kg・摩擦0.05〜1.5（対数一様）．
- `python examples/python/train_grasp_sac.py --output runs/so101-grasp/lowdim-wrist[-lift]-seed-42 --steps 30000 --seed 42 --reward v3 --action-mode wrist[_lift] --base-wrist -0.15 --wrist-scale 0.1`

| 実行 | 学習時間 | ステップ/秒（評価除外，2並列＋評価6並列と同時） | 最適化回数 | episode | 選択用の全工程成功率（5k / 10k / 15k / 20k / 25k / 30k） | best |
|---|---|---|---|---|---|---|
| 1-D wrist | 3,871 s | 10.1 | 14,500 | 608 | 0.60 / 0.60 / 0.63 / 0.67 / 0.60 / 0.63 | 20k |
| 3-D wrist_lift | 3,810 s | 10.3 | 14,500 | 584 | 0.60 / 0.60 / 0.63 / 0.60 / 0.63 / 0.63 | 15k |

同じ選択用セットでのIKは0.60，固定wrist −0.1は0.57（OBJECT_PHYSICS.md §11.4）．**学習中のスコアは誤差の範囲で横ばいだった．**

行動の分析（`runs/so101-grasp/lowdim-action-stats.txt`，選択用5条件×4 seed，決定的方策）:
- 飽和はほぼない（|a| > 0.95 は0〜4%のstep）．wrist残差の標準偏差は0.025〜0.046 rad．
- 低摩擦の条件ではwristをやや負に（平均 −0.02〜−0.03 rad 追加），3-Dでは0.07/0.07で持ち上げを遅く（速度×0.87）していた．状態（観測した接触後の動き）に応じた行動の変化はあるが，小さい．

## 6. 報酬設計（Phase 6）

v2の毎stepの傾き罰は使わず，v3の報酬（持ち上げ進捗のpotential，Hold終了までの実行，安定した受け渡し＋後段IK Transport/Placeの成功）をそのまま使った．
v3の効果とRL補正の効果を分けるため，**固定オフセットもv3と同じ受け渡し（Hold後）で評価した**（`wrist_roll_policy(end_at="base")`）．これにより§3.3で，以前の固定補正の優位が受け渡しのタイミングから来ていたことが分かった．

## 7. 最終比較（Phase 7）

`scripts/run_model_eval.sh runs/wrist-sweep 6 lowdim-wrist-seed-42 lowdim-wrist-lift-seed-42 v3-h0-seed-42`，表は`scripts/compare_final.py runs/wrist-sweep`（`runs/wrist-sweep/final_comparison.md`）．
最終評価seed 3,000,000〜029，各条件30回，19条件．右端の列は，同じ（条件，seed）での全工程成功の勝ち/負け数と符号検定（McNemar）のp値（基準: IK＋wrist −0.15）．v3はA seed 42（legacyの比較でv3の中で最良だったモデル）．


### legacy, seeds 3,000,000+ (Grasp / held / overall success)

| controller | learned | easy /150 | hard /300 | out of range /120 | overall /570 | vs IK + wrist −0.15 (hand-off after Hold): +/− (p) |
|---|---|---|---|---|---|---|
| IK | no | 150/150/149 | 174/120/68 | 55/33/23 | 240 | +21/−22 (1.00, n=570) |
| IK + wrist −0.15 (hand-off after Hold) | no | 150/150/150 | 178/136/73 | 50/28/18 | 241 | — |
| IK + wrist −0.05 (hand-off after Hold) | no | 150/150/150 | 168/126/74 | 54/32/23 | 247 | +26/−20 (0.46, n=570) |
| IK + wrist −0.10 (hand-off at 0.3 s lift, original) | no | 150/150/150 | 219/136/80 | 64/32/23 | 253 | +26/−14 (0.08, n=570) |
| IK + wrist 0 (hand-off at 0.3 s lift) | no | — | 205/136/78 | 77/31/25 | — | +27/−15 (0.09, n=420) |
| v3 A seed 42 (SAC, 6-D) | yes | 149/149/147 | 159/129/72 | 34/27/16 | 235 | +20/−26 (0.46, n=570) |
| low-dim SAC: wrist (1-D) | yes | 150/150/150 | 186/137/75 | 50/32/20 | 245 | +21/−17 (0.63, n=570) |
| low-dim SAC: wrist+lift (3-D) | yes | 150/150/150 | 170/130/72 | 40/30/20 | 242 | +21/−20 (1.00, n=570) |

### pairwise-same, seeds 3,000,000+ (Grasp / held / overall success)

| controller | learned | easy /150 | hard /300 | out of range /120 | overall /570 | vs IK + wrist −0.15 (hand-off after Hold): +/− (p) |
|---|---|---|---|---|---|---|
| IK | no | 150/150/150 | 209/169/121 | 64/32/31 | 302 | +20/−29 (0.25, n=570) |
| IK + wrist −0.15 (hand-off after Hold) | no | 150/150/150 | 219/171/129 | 66/32/32 | 311 | — |
| IK + wrist −0.05 (hand-off after Hold) | no | 150/150/150 | 221/182/133 | 68/38/32 | 315 | +27/−23 (0.67, n=570) |
| v3 A seed 42 (SAC, 6-D) | yes | 150/150/148 | 204/173/119 | 41/32/31 | 298 | +20/−33 (0.10, n=570) |
| low-dim SAC: wrist (1-D) | yes | 150/150/150 | 228/187/134 | 71/38/31 | 315 | +26/−22 (0.67, n=570) |
| low-dim SAC: wrist+lift (3-D) | yes | 150/150/150 | 225/181/132 | 57/34/32 | 314 | +31/−28 (0.79, n=570) |

### pairwise-nominal, seeds 3,000,000+ (Grasp / held / overall success)

| controller | learned | easy /150 | hard /300 | out of range /120 | overall /570 | vs IK + wrist −0.15 (hand-off after Hold): +/− (p) |
|---|---|---|---|---|---|---|
| IK | no | 150/150/150 | 248/225/145 | 80/41/35 | 330 | +40/−50 (0.34, n=570) |
| IK + wrist −0.15 (hand-off after Hold) | no | 150/150/150 | 257/225/154 | 67/40/36 | 340 | — |
| IK + wrist −0.05 (hand-off after Hold) | no | 150/150/150 | 249/215/148 | 74/41/31 | 329 | +32/−43 (0.25, n=570) |


- **低次元Residual RLは固定補正を上回らなかった．** legacyで1-D 245，3-D 242（基準241，IK 240，v3 235，元の固定補正253）．pairwiseで1-D 315，3-D 314（基準311，wrist −0.05 315）．いずれも基準との差は符号検定で有意でない（p = 0.63〜1.00）．
- v3（6-D残差）はlegacy 235，pairwise 298で，IKと同程度かやや下．範囲外で弱い（legacy 16/120）．
- 学習時と異なる物理（pairwise，legacyで学習）でも，低次元方策はIK・固定補正と同程度を保った（6-Dのv3は範囲外で低下）．
- 最も大きな差を生んだのは学習ではなく，(1) 接触モデル（legacy → pairwise で同じ摩擦値でも全工程240 → 302），(2) 受け渡しのタイミング（+12/420），(3) Placeの開き方（§4，+10/55）だった．

## 8. 結論

1. **物理モデルに問題はあったか**: あった．(a) グリッパはSTS3215のストールトルク2.94 N·mで常に飽和し（実機の定格は約1 N·m），片側約39 Nで握る．(b) 把持は指先の半径0.75 mmの球と小さなboxだけで行われ，めり込みは1.3〜2.7 mmと球の直径を超え，しかも物体が軽いほど深い（MuJoCoのsoft contactの性質）．(c) legacyの摩擦DRは，ジョーとテーブルの摩擦を連動させるうえ，condim/solref/priorityも変えていた．この副作用だけで難条件のIK全工程は124 → 66/300に下がった．
2. **摩擦と質量で最適な手首補正角は変わったか**: 変わったとは言えない．7つのオフセットの差は，同じseedでの並べ替え検定でノイズと区別できず（p = 0.19〜0.96），条件ごとの最良は選択用seedと最終seedで一致しなかった．
3. **Place失敗の条件**: 容易な条件では，離した後にゴール上で静止した立方体を，0.6 radまで開くジョーと退避する手が倒す失敗が主だった（学習Graspの失敗16件中11件）．学習Graspが立方体を高い位置・奥で渡すほど起きやすい．開きを0.25 radにすると39 → 49/55（IK・ジョー低摩擦でも40 → 52/56）．静止待ちは効果なし，接地してから離すと悪化．テーブル摩擦が低いと離した後に滑る失敗が残る．
4. **低次元Residual RLは固定補正を上回ったか**: 上回らなかった（legacy 245/242 vs 241，pairwise 315/314 vs 311，いずれも有意差なし）．状態に応じた行動の変化は小さく，成功率に結びつかなかった．
5. **改善に寄与した変更**: 学習以外の3点（接触モデルの副作用の除去，早い受け渡し，Placeの開きを小さくすること）．手首角の固定補正・学習補正はどちらも寄与が確認できなかった．
6. **比較上の制約**: legacyとpairwiseは別の実験である．v1〜v3と今回の低次元方策はlegacyで学習しており，pairwiseでの評価は学習時と異なる物理での評価になる．グリッパの力とsolref/solimpは既存結果との比較のため変更しておらず，実機より強い把持とめり込みの質量依存は全結果に含まれている．Placeの変種（gentleなど）は既定のPlaceを変えておらず，§7の全工程成功はすべて従来のIK Placeによる．学習は各1 seed・3万ステップの短い実行である．

### 再現方法（テスト）

```bash
cargo fmt --all --check && cargo clippy --workspace --all-targets
cargo test --workspace            # 今回58件成功（新規 contact_model 5件を含む）
maturin develop && pytest -q      # 今回166件成功（新規 test_physics_audit 6件を含む）
```
