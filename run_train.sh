#!/usr/bin/env bash
# Train the PPO tracking policy (single-trajectory tracking).
#
#   - sources the venv (inherits system mujoco/torch/numpy on this Jetson)
#   - runs scripts/train.py on the Orin GPU
#
# Usage:
#   ./run_train.sh                                  # defaults
#   ./run_train.sh --total-steps 2000000 --num-envs 8 --run-name pick_v1
# Args are forwarded to scripts/train.py.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[run_train] creating venv at $VENV_DIR (system-site-packages)"
    python3 -m venv --system-site-packages "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$REPO_DIR/requirements.txt"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python - <<'PY'
import torch, mujoco
print(f"[run_train] torch {torch.__version__}  cuda={torch.cuda.is_available()}  mujoco {mujoco.__version__}")
PY

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec python scripts/train.py "$@"
