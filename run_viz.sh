#!/usr/bin/env bash
# Launch the Vega + f5d6 tracking visualizer.
#
#   - exports the X display to :1 (the running X server on this Jetson)
#   - creates/sources a venv that inherits the system MuJoCo/glfw build
#     (--system-site-packages avoids recompiling mujoco on aarch64)
#   - runs scripts/viz.py
#
# Usage:
#   ./run_viz.sh                 # play synthetic reference, looping
#   ./run_viz.sh --random        # random residual actions
#   ./run_viz.sh --ref foo.npz   # play a saved reference trajectory
# Any args are forwarded to scripts/viz.py.
set -euo pipefail

# --- repo root (this script's dir) -------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- display -----------------------------------------------------------------
export DISPLAY=:1
# MuJoCo on Linux uses GLFW/EGL; force GLFW for an on-screen window.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

# --- venv (inherits system mujoco/glfw/numpy already built for Jetson) -------
VENV_DIR="$REPO_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[run_viz] creating venv at $VENV_DIR (system-site-packages)"
    python3 -m venv --system-site-packages "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    # only installs what the system python is missing
    pip install --quiet -r "$REPO_DIR/requirements.txt"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# --- sanity check ------------------------------------------------------------
python - <<'PY'
import mujoco, glfw, numpy
print(f"[run_viz] mujoco {mujoco.__version__}  numpy {numpy.__version__}  DISPLAY={__import__('os').environ.get('DISPLAY')}")
PY

# --- run ---------------------------------------------------------------------
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec python scripts/viz.py "$@"
