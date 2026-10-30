# 変更前の基準 (2026-09-26)

- Workspace: so101-core / so101-cli / so101-python。
- MuJoCo 3.10.0、rusty_mujoco 0.4.0、PyO3 0.27.2、maturin 1.15.0。
- Nix Python 3.13.15。Menagerie c96a32d28fb5da84da38c1da4d749e7a13212855。
- JointState/JointCommandとruntimeが6軸固定。ctrlrangeだけで入力を検証。
- 名前でactuator/hinge IDを取得するがnu==6を要求。SO101 scene template + mj_loadXML。
- coreにGUIなし、CLI viewer featureにGLFW。PythonはMutex + unsafe Send。
- Rust 7 integration + 2 Python panic boundary tests成功。
- Python 15 tests成功 (home/sine CLI一致・別thread利用を含む)。

変更前の `nix develop` は全ファイルGit未追跡のためflake.nixを見つけられなかった。
`nix develop path:. --command bash -e -c 'cargo test --workspace; python -m pytest'`
で基準を実行し全成功。flake.nix/flake.lockをgit add -Nで可視化し、以降は通常の
nix developで実行。commitは作成していない。

既知の既存回避策: Darwin CMakeのar/ranlib明示とMuJoCoのlibrary-only override。
mjSpec Drop二重解放を避けるmj_loadXML経路。これらとphysics version pinは維持。
