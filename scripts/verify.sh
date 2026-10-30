#!/usr/bin/env bash
# Run inside nix develop. No GUI is started by these checks.
set -euo pipefail
bash scripts/check-phase.sh
learning_args=()
if [[ "${1:-}" == "--learning" ]]; then learning_args=(--extra learning); fi
uv sync --locked --no-install-project "${learning_args[@]}"
maturin develop
pytest
cargo clippy -p sim-cli --all-targets --features rerun,viewer
cargo clippy -p sim-python --all-targets --features rerun
cargo test -p telemetry-rerun --features sdk
maturin develop --features sim-python/rerun
pytest
