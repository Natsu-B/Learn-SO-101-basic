# Validation (Apple Silicon macOS, 2026-09-26)

## Baseline / phases

変更前はRust 7 integration + 2 binding unit tests、Python 15 tests成功。
初回のGit未追跡flake問題と既存回避策はBASELINE.md参照。
Phase 2 (dynamic state/controller)、3 (RobotSpec/runtime)、4 (Panda)、5 (UR5e)、
6 (Reach)、7 (control timing)、8 (joint/ctrl validation)、9 (worker ownership)、
10 (Gymnasium/Python3.12)、11 (Rerun)、12 (docs/examples)ごとにNix内で
fmt / clippy / workspace testを実行した。Python変更ではmaturin + pytestも実行。

途中のfailureと対処:

- range fixtureのMJCF既定角度がdegreeだった。compiler angle=radianへ修正。
- Nixのpytest executableがvenvを見なかった。旧環境ではpython -m pytest、
  新環境ではuvでvenvへpytestをinstallし通常のpytestが成功。
- worker初期化errorをString化するとFileNotFoundErrorが失われた。PyErrを保持して転送。
- cdylib/rlibのPyO3 doctest metadata衝突。Python extensionのdoctestを無効にし、
  実行される2件のpanic境界unit testsとPython testsを維持。
- Rerun 0.27.3はset_duration_secsとInto<PathBuf>を使用。SDK sourceで確認して修正。
- maturinのworkspace feature指定はsim-python/rerunへ限定。
- 通常wheelに旧packageが含まれなかった。python-packagesへ両packageを明示し検証。
  このwheelはNix native libsへlinkする開発用。standalone配布用repairはscope外。

## Commands

すべてnix developのshellHookを適用した環境で実行。

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo clippy -p sim-cli --all-targets --features rerun,viewer
cargo clippy -p sim-python --all-targets --features rerun
cargo test -p telemetry-rerun --features sdk
uv sync --locked --no-install-project
maturin develop
pytest
maturin develop --features sim-python/rerun
pytest
```

Rust workspaceは21 tests (runtime11 / codec1 / binding2 / legacy7)。
Rerun sdk featureではGUIなしRRD testを追加実行。
Pythonは33 tests (既存15 + Gymnasium17 + telemetry1)、GUIは一切起動しない。
Gym checkerはSI action範囲と無限観測Boxについて9件の推奨警告を出す。
Nix cc-wrapperはarm64 target spellingについてwarningを出すがbuild/link成功。

3機種のmodel load、cached actuator discovery、1000 step/reset、seed、Reach reward/
successを検証。fixtureで追加actuatorを含むID [2,0]のsubsetとunbounded limitsを検証。
SO101のwrist_rollはctrlrangeよりjoint rangeが狭いことを検証。
30Hz control x 6000 calls / 200Hz physicsで200秒、time誤差<1e-8秒。

3機種のheadless CLI、Rust Reach examples、Python Reach examples、random policy、
旧Python basic/sineも実行。旧SO101 home/sineはPythonとRust CLIの結果が完全一致。

## Visualization

Rust CLIからSO101 300 frames / 10秒のRRDを保存し、rerun rrd printで
joint position/target・time・reward・end_effector/targetのentityと各300 rowsを確認。
PythonからRust sinkを使うUR5e RRD保存も成功。
通常のRerun ViewerでRRDを開き、5秒間エラーなく動作後に検証用processを終了。

Rerunのscreenshot-to経路は画像保存後に40000px SurfaceのGPUサイズエラーと
store hub panicでexit102となった。小windowでも再現したが、通常Viewerでは再現しない。
保存画像ではtime-seriesと3D pointsを確認した。このSDK/Viewerのscreenshot専用問題は
今回のsimulation実装では修正していない。

新CLIのSO101/Panda/UR5eと旧SO101 sineのGLFW Viewerを各100 physics stepsで実行。
Viewer実描画はtest suiteから分離し、headless testsにdisplayを要求しない。

## Unsafe inventory

- mujoco-runtime/src/model_access.rs: unsafe extern C宣言と1つの呼び出しblock。
  rusty_mujoco 0.4.0のactuator_trnid getterはjoint IDとobject enumを比較するため、
  native mjModelのactuator_trntype/trnidを読んで初期化時のtransmission対応を検証。
  C側でactuator index/typeをcheckし、pointerを保持せずread-onlyで使用。
- mujoco-runtime/src/simulation.rs: site_xpos取得の1 block。
  同じSimulatorが所有する対応model/dataで、load時に検証/cacheしたsite IDを使用。
  native allocationを交換するmutable callbackは公開していない。

Pythonにunsafe block、unsafe impl Send/Syncはない。GLFW contextはmain threadで所有。
依存crate内部のFFI unsafeは上記repository-owned inventoryには含めない。

## TODO

Camera rendering/offscreen context、3D mesh/link hierarchy、到達可能なtarget sampling、
Panda tendon gripper action codec、実機calibration、LeRobot adapter。
Viewer二種類の同時利用とRerun screenshot専用問題も後続項目。
ROS/他simulation backend/driver/学習algorithm/recorderはscope外。
