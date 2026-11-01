# AGENTS.md — SO-101 Pick & Place（Physical AI講座 最終課題）

このファイルは作業方針と引継ぎ情報。作業開始時に必ず読み、状況が変わったら更新してコミットする。
詳細な計測値・経緯は [docs/PICK_PLACE.md](docs/PICK_PLACE.md)。

## 1. ゴールと締切

- MuJoCo上のSO-101で、**立方体の初期位置と目標位置が変化しても** Pick & Place を成功させる。
- **締切: 2026-11-02 10:00**。提出物は 動作動画1〜3分 / public GitHubリポジトリ / PDFレポート。
- **AI利用が必須**: 少なくとも一部の動作で学習済みモデルの推論を実際に使う。IK＋固定状態機械だけではAI利用とみなされない。
- 間に合わない場合は機能を縮小してでも「実際に動く成果物」を優先する。
- 実機なし・シミュレーションのみ。物体/目標位置はMuJoCoの真値を使ってよい（画像認識は対象外）。
- 対象外: 未知の障害物・未知形状・照明変化。

## 2. 制御アーキテクチャ

Grasp → Transport → Place の3 Skill。成功判定に基づいて順に遷移し、前Skillの終了状態が次Skillの初期状態になる。

- `crates/mujoco-runtime/src/skills.rs`: `Skill` trait、成功判定 `predicates`（lifted / transported / placed / dropped）、
  IK版 `ScriptedGrasp/Transport/Place`、`run_episode`、`EpisodeReport`。
- **成功判定はRust側の `predicates` を唯一の定義とし、学習環境・評価・デモで共用する。** Python側で別定義を作らない。
- 学習済みSkillとIK版Skillは `Skill::learned()` / `SkillReport.learned` で必ず区別して記録・表示する。
- Skill間の状態分布の不一致に注意: 学習時の初期状態は実行時と同じ前段Skillの終了状態から作る（＋小ノイズ）。

## 3. 技術上の制約・方針

- ロボット制御・シミュレータ内部処理・タスク管理は原則Rust。RLはPythonのStable-Baselines3（既存SAC+HER基盤 `python/robot_sim/goal_learning.py`）。
- 既存の関節範囲・アクチュエータ制限（`Simulator::action_bounds`）・制御周期（30 Hz、physics 0.005 s）を守る。
- 立方体はMuJoCoの接触物理だけで動かす。座標を書き込むのはreset時の `PickPlaceSim::set_cube` のみ。
- **既存のReachタスク・テストを壊さない。** `scenes/robot.xml` と `Simulator::from_spec` の挙動は変えない（Pick & Placeは `from_spec_with_scene` + `scenes/so101_pick_place.xml`）。
- 変更は小さく分離し、原因を特定しやすくする。
- 既存実装は必要なら再利用する。CodexなどのAgentも積極的に使ってよい。
- 予備マシン: NixOS / Core Ultra 9 / 32GB を一時利用可（長い学習向け）。外部GPUはなし。

## 4. 検証と報告のルール

- **コンパイル成功と物理的な成功（把持・運搬・配置）は別の検証として扱う。**
- **実行していないテスト・学習・評価の結果を成功として報告しない。** 数値は実行したコマンドの出力からのみ書く。
- 最低限記録するもの: 成功/失敗条件、試行回数、seed、最終位置誤差、Skillごとの成功率、どのSkillが学習済みか。
- 変更後は関連するRust/Pythonテストを実行して結果を報告する（下のコマンド）。

## 5. コミット方針

- **Conventional Commits** で区切りごとにどんどんコミットしてよい（ユーザー了承済み）。例: `feat(mujoco-runtime): ...`, `test(...)`, `docs: ...`, `fix(...)`, `refactor(...)`。
- 1コミット1目的、各コミット単独でビルドが通る状態を保つ。
- 末尾に `Co-Authored-By:` 行を付ける（使用モデルのもの）。
- push・リモート作成・publicリポジトリ化はユーザーに確認してから行う。
- `runs/`、`data/`、`.venv*`、`*.so` はコミットしない（`.gitignore` 済み）。

## 6. 環境とコマンド

Mac mini M1（aarch64-darwin）。すべて `nix develop` 内で実行する。

```bash
nix develop                     # .venv があれば自動activate
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace          # 2026-11-01 時点: 66件成功
maturin develop                 # Pythonテスト前に native extension を再ビルド
pytest -q                       # 2026-11-01 時点: 178 passed

# Pick & Place（IK版）評価: 引数 = エピソード数, 先頭seed。JSON lines を stdout、要約を stderr
cargo run --release -p mujoco-runtime --example so101_pick_place -- 200 0
PICK_TRACE=1 cargo run --release -p mujoco-runtime --example so101_pick_place -- 1 25   # ステップ追跡
cargo run -p mujoco-runtime --example pick_place_probe   # ジョー形状・IKの計測

# 既存Reach checkpoint の確認（2026-10-30: 100 episode・5mmで到達率98%）
python examples/python/eval_reach.py --robot so101 \
  --model runs/so101-reach-5mm/base-policy/best.zip --episodes 100 --threshold 0.005

# 動画: ffmpeg / ffprobeをdevShellへ追加済み（§7，docs/PICK_PLACE.md §10）
python examples/python/record_pick_place.py \
  --model runs/so101-transport/seed-42/best.zip --seeds 0 1 7 25 42 99 \
  --width 1280 --height 720 --hold-frames 90 --output runs/video/pick_place_demo.mp4
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size -of json runs/video/pick_place_demo.mp4
```

Python依存を入れ直す場合: `uv venv --python "$NIX_PYTHON" --seed && uv sync --locked --extra learning --no-install-project && maturin develop`。

## 7. 現状（2026-10-30，Transportとパイプライン統合を含む）

完了:
- シーン `scenes/so101_pick_place.xml`: テーブル(z=0)、25mm・30g立方体（freejoint、condim 4、摩擦1.2）、目標マーカー（mocap・非接触）。
- `pick_place.rs` の `PickPlaceSim`: seed付き配置（半径0.15–0.25m、方位±0.8rad、相互距離≥0.08m、yaw=0）、接触判定、挟持中心FK、DLS IK（位置＋指先方向＋閉じ方向）。
- IK版 Grasp/Transport/Place: **200/200 seed成功、最終XY誤差 平均2.4mm / P95 4.1mm / 最大6.9mm**。
- `tests/pick_place.rs`: 静止ドリフト、seed再現性、IK精度、両ジョー接触で30mm以上持ち上げ、**グリッパを閉じなければ持ち上がらない対照**、テレポートなし。
- この全IK版では学習済みモデルを使わない．学習Transportの追加結果は下記を参照．

追加実装（2026-10-30）:
- `robot_sim._native.PickPlaceSimulator`（`robot_sim`からも公開）: 専用ワーカー所有、GIL解放、panic時のRuntimeError化と再利用拒否。
- seed付きreset、状態・接触・Rustのpredicates/default Criteria取得、関節指令、IK Skillの一括・逐次実行、全episode実行。
- APIの仕様と追加検証結果は [docs/PICK_PLACE.md](docs/PICK_PLACE.md) §6。
- 追加検証: Rust 45件成功、pytest 122件成功（既存96件＋追加26件）、警告9件。fmt / clippy / maturin developも成功。
- Python経由の全IK版: seeds 0〜4で5/5成功、各Skillも5/5、最終XY誤差 平均2.313098mm / 最大2.476371mm、すべて`learned=False`。

追加実装（2026-10-30，Transport）:
- `GoalTransportEnv`とSAC+HERの学習・評価スクリプトを追加．実際のGrasp終端から開始し，成功・落下はRustの判定を使う．訓練は150ステップ固定．
- Rust 45件成功，`maturin develop`成功，pytest 136件成功（警告9件）．
- seed 42で3,008ステップの短い学習を実行し，最適化504回，`best.zip`と`final.zip`の保存を確認．学習165.64ステップ/秒（評価除外），環境206.82ステップ/秒（把持reset込み）．
- finalモデルをseeds 2,000,000〜2,000,009で評価し，Grasp 10/10，学習Transport 0/10，落下0/10．最終XY誤差は平均142.920mm / P95 207.880mm / 最大219.714mm．運搬性能の達成は未確認．
- 別出力先への`--resume`で3,008→3,168ステップ，追加の最適化40回を確認．実行コマンドと仕様は[docs/PICK_PLACE.md](docs/PICK_PLACE.md) §7を参照．
- 今回の指示に従い，変更はコミット・pushしていない．

追加実装（2026-10-30，パイプライン統合）:
- `pick_place_pipeline.run_pick_place_episode`でGrasp(IK) → Transport(学習) → Place(IK)を統合．Transport失敗時はPlaceを実行せず，フォールバックも行わない．最終成功にはRustの`placed`も要求する．
- `transport="scripted"`で全IK版を同じ関数から評価できる．観測と行動変換は`GoalTransportEnv`との共通関数へ抽出し，環境の意味は変更していない．
- 全Skillの各制御ステップ後に`on_step(state, skill_name, learned)`を呼ぶ．逐次実行の失敗理由取得用に読み取り専用の`skill_report()`をbindingへ追加した．
- `examples/python/eval_pick_place.py`はepisodeごとのJSON linesをstdout，集計をstderrへ出力する．モデルの`config.json`から行動制限とhorizonを復元する．仕様とコマンドは[docs/PICK_PLACE.md](docs/PICK_PLACE.md) §8を参照．
- Rust 45件成功，`maturin develop`成功，pytest 142件成功（既存警告9件，60.97秒）．fmtとclippyも成功した．
- 全IK版はseeds 0〜19で20/20成功，各Skillも20/20，落下0件．最終XY誤差は平均2.215473mm / P95 2.550412mm / 最大2.738648mm，すべて`learned=False`．
- smokeモデル`runs/so101-transport/smoke-seed-42/best.zip`（2,000ステップ時点）をseeds 0〜4で評価し，全体0/5，Grasp 5/5，学習Transport 0/5，Place未実行，落下0件．全件150ステップでtimeout．最終XY誤差は平均144.282837mm / P95 180.767333mm / 最大188.652089mm．学習TransportからPlaceへの物理的な成功は未確認である．
- `runs/so101-transport/seed-42/`の250kステップ背景学習は今回の作業対象外とし，操作していない．新しい学習実行は開始せず，変更はコミット・pushしていない．

本学習と全工程評価（2026-10-31）:
- `runs/so101-transport/seed-42/best.zip`（30kステップ時点。学習は50kで停止）。
- **Grasp(IK) → Transport(SAC学習済み) → Place(IK) を seeds 0〜199 で 200/200 成功、落下0、最終XY誤差 平均2.42 / P95 3.43 / 最大6.42 mm**。同seedの全IK版は 200/200、2.43 / 3.94 / 6.95 mm。詳細は docs/PICK_PLACE.md §9。

動画描画の実装と事前検証（2026-10-31）:
- `PickPlaceRenderer(sim, width, height).render()`はRGB bytesを返す．コンパイル済みモデルと全状態を描画用にコピーし，元の物理状態を変更しない．macOSはCGL（GPUが使えなければAppleのソフトウェア描画），他OSは非表示GLFWを使う．
- `examples/python/record_pick_place.py`を追加した．各制御ステップを30fpsで出力し，seed，Skill，`LEARNED (SAC)` / `IK (scripted)`，XY誤差，終了結果を表示する．モデルの設定を復元し，JSON lines，集計JSON，3場面のPNGも保存する．
- 指定モデルでseeds 0 / 1 / 7 / 25 / 42 / 99を再評価し，全工程・各Skillとも6/6成功，落下0，最終XY誤差は平均2.546967 / P95 3.499503 / 最大3.814673mmであった．Transportだけが学習済みである．
- `runs/video/pick_place_demo_{grasp,transport,place}.png`を1280×720で生成し，目視確認した．seed 0を描画付きで再実行した結果は描画なしと完全一致した．
- `cargo test --workspace`は45件成功，`maturin develop`，fmt，clippyも成功した．`pytest -q`は143 passed / 1 skipped / 警告9件であった．描画テストは成功し，動画テストはffmpeg / ffprobeがないためスキップした．
- **MP4生成済み**: `runs/video/pick_place_demo.mp4`（ffprobe実測 1280×720・30fps・1944フレーム・64.8秒・5.5MB）．6/6成功．ffmpeg必須の動画テストも2 passed，pytest全体144 passed．
- `runs/so101-transport/`への書き込み，学習，コミット，pushは行っていない．

未知の物理特性（質量・摩擦）とGrasp学習（2026-10-31，詳細と全数値は [docs/OBJECT_PHYSICS.md](docs/OBJECT_PHYSICS.md)）:
- 物体DR `ObjectRandomization`（reset時のみ，独立乱数，慣性スケール＋`mj_setConst`）．Python `set_object_randomization(mass=, friction=)` / `object_physics()`（ログ用の真値，`state()`には含めない）．
- **SO-101ジョーは`priority=1`のため，既存シーンの立方体摩擦1.2はジョー接触に効いていなかった．** 摩擦DR時だけ立方体をpriority 2にしている．その結果，ジョー接触のcondimは6→4，テーブル接触のsolrefは0.015→0.01に変わる（`contact_probe`で実測）．
- IK頑健性（870 episode）: 指定範囲0.02–0.05 kg・μ0.8–1.5では，Grasp・保持とも270/270．失敗するのはμ ≤ 0.1だけ（ジョー面が平行でないための押し出し．挟む力を変えても効かない）．
- `GoalGraspEnv`（IKでClose直前まで進め，その後IK指令に残差を加える．履歴K=4のモデルBも同じ環境）を，0.02–0.1 kg・μ0.05–1.5で各150kステップ学習した（A/B×seed 42/7，`runs/so101-grasp/h{0,4}-seed-{42,7}/best.zip`）．
- 最終比較（seed 3,000,000〜，各条件30回，`scripts/eval_grasp_comparison.sh`）: **学習残差は，IKにも固定のwrist_roll −0.1 radにも勝たなかった．** 難条件のGrasp成功: IK 174，固定 219，学習 150〜211（/300）．容易な条件の全工程成功は，IK 149 → 学習123〜143（/150）に低下した（傾いたまま低い位置で渡すためPlaceで失敗）．
- 適応: 接触前の行動は条件によらず一致し，接触の1 step後に分かれる（観測した反応に応じて変えている）．ただし主にグリッパの残差に現れ，方向は学習seedごとに異なり，性能向上には結びつかなかった．
- 球体（`shape="sphere"`）: 学習なしのIKでGrasp 30/30，Transport 30/30，Place 0/30（転がり）．独立した発展課題として扱う．
- 動画 `runs/video/grasp_hidden_physics.mp4`（50.7秒，学習Grasp＋IK fallback，真の物理を画面に表示）．
- tilt罰付きv2（`runs/so101-grasp/v2-tilt-h{0,4}-seed-42`）は改善しなかった（B v2は持ち上げを避ける局所解に入り，容易な条件のGraspが99/150）．受け渡しの品質は後段の成否を含めて評価する必要がある．

- 見直し（docs/OBJECT_PHYSICS.md §11）: 学習Graspは，判定（30 mm・0.3 s）の時点で0.4 m/sで動き，傾いたまま渡していた．Transportの判定も速度条件がなく，IKでも0.12 m/sで終わる．Placeの失敗はすべて離した後の転倒で，ジョー内の傾きが原因である．IKの安定化（`stabilize`）を挟んでも回復しない．グリッパはLift/Holdで常に力制限に飽和している．
- v3（`--reward v3`: 持ち上げの進捗，Hold終了まで実行，安定した受け渡しと後段IK成功を終端報酬にし，全工程成功率で選択．40kステップ）は容易な条件の低下を解消した（146〜149/150）．全工程合計（/570）は IK 240，固定wrist 253，v1 192〜237，v3 222〜235で，**学習は固定補正にもIKにも勝っていない**．

接触物理の監査・手首補正・Place・低次元RL（2026-10-31，詳細と全数値は [docs/PHYSICS_AUDIT.md](docs/PHYSICS_AUDIT.md)）:
- **接触環境が2つある**: `legacy`（従来．摩擦DRで立方体priority 2，condim/solrefも変わる）と`pairwise`（`PickPlaceSimulator(contact_model="pairwise")`，ジョー/テーブルの滑り摩擦を別々に書き，他は不変．DRなしでは従来とビット一致）．**結果を混ぜない．** v1〜v3・低次元RLはlegacyで学習．
- 物理の問題: グリッパはストールトルク2.94 N·mで常時飽和（実機定格約1 N·m，片側39 N）．把持は指先の0.75 mm球だけで，めり込み1.3〜2.7 mm，軽いほど深い（soft contactの質量依存）．legacyの副作用だけで難条件IK全工程124→66/300．Grasp失敗はジョーとテーブルの摩擦が両方低いときに起きる．
- 固定手首補正（7値×19条件，選択seed 4M/最終3M）: **オフセットの差はノイズと区別できない**（対応あり並べ替え検定 p=0.19〜0.96）．以前の「wrist −0.1が最良」は**受け渡しのタイミング**（0.3 s持ち上げ時点で渡す）の効果で，オフセット0でも同じ（難＋範囲外 103/420 vs IK 91）．
- Place失敗: 主因は離した後に開くジョー・退避する手が立方体を倒すこと．`place="place_gentle"`（0.25 radまで開く）で学習Grasp後39→49/55，IK（ジョー低摩擦）40→52/56．静止待ちは効果なし，接地後に離すと悪化．既定のPlaceは変えていない．
- 低次元Residual SAC（wrist 1-D / wrist+持ち上げ方向・速度 3-D，基準 IK+wrist −0.15，v3報酬，3万step）: legacy最終 245 / 242 vs 固定241・IK 240・v3 235（/570），pairwise 315 / 314 vs 311．**固定補正を上回らない（有意差なし）**．`runs/so101-grasp/lowdim-*`．
- テスト: Rust 58件，pytest 166件成功（2026-10-31）．

把持力制御（論文 Lach et al. IROS 2024 の再現・移植，2026-11-01，詳細と全数値は [docs/FORCE_CONTROL.md](docs/FORCE_CONTROL.md)）:
- 公開実装 `llach/learning_fc` には**ライセンス表記がない**．コードはコピーせず，論文（arXiv 2311.07245v1，CC BY 4.0）から再実装する．論文・コード・SO-101の対応表と，1自由度では再現できない要素は同文書§1．
- 既存モデルの「指令→力」（`grip_command_probe`）: 接触角から約−20 mrad以内なら飽和せず力は指令で変わる．以前の飽和は閉指令−0.17 radのせい．力/角度の傾きは物体質量で約4倍変わる（soft contactの人工的な依存）．
- 独立環境 `force_grasp.rs` + `scenes/so101_force_grasp.xml`: 4面の表皮板をばね・ダンパ付きスライド関節で芯に付けた柔軟物体（**F = k·d + c·ḋ を陽に持つ**．めり込みは別に記録）．潰れる前はジョー法線力＝ばね力（0.05 N以内）．
- 成立性（学習なし，20試行）: 力PIは平均力を目標±0.005 N以内に保った（潰れ1/20）．固定閉指令は20/20で潰す（39 N）．課題は接触直後のピーク（中央値6.6 N・最大34 N，1自由度で物体をすき間分押し渡すため）．PPO学習は未実施．
- テスト: Rust 64件（`tests/force_grasp.rs` 6件を追加），pytest 166件成功（2026-11-01）．
- PPO（2026-11-01，FORCE_CONTROL.md §6〜9）: `ForceGraspEnv`（`robot_sim.force_grasp_env`，観測に物体の真値なし，物理ステップ単位のピーク），学習 `train_force_grasp_ppo.py`（出力先が空でなければ拒否），評価 `eval_force_grasp.py`（選択seed 7M〜 / 最終seed 8M〜）．`runs/force-grasp-ppo/`に計約8.7M step・53分．
  - 最終評価200試行: 公開スクリプト相当のP1（r_objなし・φなし）は平均誤差0.208 N（力PI 0.278 N），ピーク平均13.6 N（PI 20.1 N），ただし潰れ65%・成功15%（PI 16.5%）．論文相当（r_obj＋SO-101版φ，P2）は誤差2.7 N．
  - 論文ベース報酬ではピーク荷重は下がらない．独自のピーク罰（重みを0→20で焼きなまし，P8）でピーク平均3.4 N・潰れ5%になるが，両ジョー接触の維持45%（柔らかい物体を把持しない）．
  - 押し渡しを遅くすれば衝突は避けられる（0.015 rad/sで潰れ12.5%）が5 sに収まらない．押すだけで潰れる物体は3%．
  - 失敗した設定: 全開0.6 radから開始（接触しない），SO-101版φ = 0.1（押し渡しが進まない），毎周期のピーク罰・ゼロからの強いピーク罰（全開で接触回避）．
- テスト（PPO追加後）: Rust 65件，pytest 176件成功．
- 失敗の分解（FORCE_CONTROL.md §10，`runs/force-grasp-eval/diag2-200`）: グリッパ単独の安全把持の成功は力PI 16.5%・二段階PI 18.5%・P1 15.0%・P8 17.0%（45%が非接触）・P9（接触獲得を達成条件にした報酬）19.0%．潰れの大半は目標力の前の押し渡し中（可動ジョーが物体を固定ジョーへ押す間に面が最大圧縮）．ゆっくり押しても36%で起き，グリッパ単独制御の限界と判断．
  - 腕の把持方向の微小補正（`ForceGraspSim::shift_arm`，`ArmAssistPi`，別実験）で成功44.0%・物体移動1.8 mm・潰れ20%．次のRL候補は行動に腕補正を加えたPPO（未実施）．
  - テスト: Rust 66件，pytest 178件成功．

未検証・未実装: PDFレポート，立方体yaw≠0，Placeの学習，球体のPlace（転がり），物理モデルの修正（別ブランチで行う）．

## 8. 次の作業（優先順）

1. **Python binding完了**（`crates/sim-python`）: `PickPlaceSimulator`。仕様と検証結果は上記§6を参照。
2. **Transport を SAC+HER で学習 — 完了**（30kステップでTransport単体20/20、全工程200/200。§7参照）
   - obs: 5 arm関節q/qvel，挟持中心，立方体位置・速度，グリッパ角，立方体と挟持中心の相対位置（合計23次元）．achieved_goal=立方体位置，desired_goal=目標XY＋Rustのcriteriaから得るcube_half+carry_height．
   - action: 5 arm関節の正規化delta（既存 GoalReachEnv と同じ max_delta / rate limit）。グリッパは `GRIPPER_CLOSED` 固定。
   - reset: 毎回 `ScriptedGrasp` を実行して把持状態を作る。報酬は goal のみの関数（HER再計算可）。固定horizon。
   - 目安 200–300k step。`best.zip` を随時保存し、時間切れでも途中モデルで評価できるようにする。
3. **統合・評価完了**（§7）。以下は当初の計画: Grasp(IK) → Transport(学習) → Place(IK)．次は長期学習が終わったモデルを`eval_pick_place.py`で全IK版と同じseed列で評価し，全体成功率と誤差，Placeへの接続を確認する．学習中の出力先を変更・上書きしない．
4. **動画完了**（`runs/video/pick_place_demo.mp4`，64.8秒）．次は**PDFレポート**，public GitHubリポジトリ化（ユーザー確認後）．
5. 余裕があれば Place / Grasp の学習、yawランダム化、質量・摩擦DR。

## 9. ハマりどころ（実測で判明済み）

- M1は4P+4Eコア．評価を7並列にするとepisodeあたり数倍遅くなる（1 episode ≈ 0.5 s → 3 s）．並列6までにする．
- JSONの`json!`はキーをアルファベット順に出す（`"type"`は行の最後）．行頭で種類を判定しない．
- `PickPlaceSimulator`の変種skill（`place_gentle`など）は`on_step`に自分のskill名で通知される．

- `gripperframe` は挟持中心ではない。site の **+x が指先方向、+z が固定ジョー→可動ジョー方向**。
  25mm立方体の中心は site 座標で約 (−6, 0.2, −8) mm（`PickPlaceSim::grasp_offset`）。
- グリッパ角: 0 rad で指先間隔16mm、0.3 radで40mm。開=`0.6`、閉=`-0.17`（力制限で押し付け、25mm立方体では約0.10 radで停止）。
- IKの姿勢誤差に外積を使うと指先「真上」の解に収束する → 方向ベクトルの差を使う（修正済み）。
- 固定ジョー内面と立方体側面のすき間は約1mm。下降時は固定ジョーを −z 方向（立方体から離れる向き）へ4mmずらす。符号を逆にすると成功率が半減した。
- 運搬中の立方体は1制御ステップで最大約12mm動く（テレポート判定は30mm）。
- ログインシェルは **zsh**: `$VAR` に入れた複数パスは単語分割されない。複数ファイルを扱うスクリプトは配列か `bash` を使う。
- `.venv` はリポジトリ移動前のパスを参照して壊れていたため再作成済み（旧環境は `.venv-old-moved-path/`）。
- 既存checkpointは `runs/so101-reach-5mm/base-policy/best.zip`、`runs/so101-reach-hold/best.zip`（gitには含めない）。
- `nix develop` のshellHookは `PYTHONPATH` を上書きする．別のビルドを使うときは `nix develop -c bash -c 'export PYTHONPATH=…; …'` の中で設定する．
- 学習・評価のプロセスが editable の `_native*.so` を読み込んでいる間は `maturin develop` しない（実行中プロセスの.soを置き換えてしまう）．待てない場合は `maturin build -o DIR` ＋ `pip install --target` で別の場所にwheelを入れて使う．
- 既存シーン（立方体中心12.5 mm）でグリッパを閉じると，ジョーのメッシュがテーブルに当たり腕を押し戻すことがある（力制御環境で発見，Pick & Placeへの影響は未検証）．力制御環境は物体を高さ40 mmにして回避した．
- 摩擦DRの下限は0.01（以前は0.05で，μ=0.03の評価が拒否された）．
