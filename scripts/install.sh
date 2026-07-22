#!/usr/bin/env bash
# Create .venv in the repo root with pinned deps (CPU torch) and smoke-test it.
# Needs python 3.10-3.12; set PYTHON_BIN to force an interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-}"
if [[ -z "$PY" ]]; then
    for c in python3.10 python3.11 python3.12 python3; do
        if command -v "$c" >/dev/null 2>&1 \
           && "$c" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info < (3, 13)))'; then
            PY="$c"
            break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "error: no Python 3.10-3.12 found; set PYTHON_BIN=/path/to/python" >&2
    exit 1
fi
echo "using $("$PY" --version) at $(command -v "$PY")"

"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Smoke test: the imports the sweep needs, plus env registration.
# mpe_envs' PettingZooWrapper entry point lives under epymarl/src, so that
# needs to be on the path too (see mpe_envs/__init__.py).
PYTHONPATH=.:epymarl/src .venv/bin/python - <<'EOF'
import torch, torch_scatter, sacred
import gymnasium as gym
import matrix_envs
import mpe_envs
import lbforaging
gym.make("climbing-2p-v0")
gym.make("SimpleSpread-4ag-v0")
gym.make("lbforaging:Foraging-8x8-2p-3f-v3")
print(f"install OK: torch {torch.__version__}")
EOF
