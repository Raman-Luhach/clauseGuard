#!/usr/bin/env bash
# Train ClauseGuard's transformer models on a Linux GPU machine (Lambda, Kaggle, Colab, a
# server) and write one results archive. Run from anywhere inside the repository:
#
#   OUT=~ bash scripts/train_on_gpu.sh                      # everything (Lambda H100: ~4-5 h)
#   MODULES="contract" JOINT=0 OUT=~ bash scripts/train_on_gpu.sh
#
# MODULES  modules to train in this run (default: all three); each trains the models listed
#          under gpu_candidates in configs/<module>.yaml
# JOINT    1 (default): also train the joint model over all three modules; 0: skip it
# OUT      where the results archive goes (default: current directory)
# SCRATCH  large downloads: virtual environment, caches (default: /tmp/clauseguard-cache)
#
# Back on the Mac: clauseguard results import <archive>, then make after-import.
set -euo pipefail

MODULES="${MODULES:-tos privacy contract}"
JOINT="${JOINT:-1}"
OUT="${OUT:-$PWD}"
SCRATCH="${SCRATCH:-/tmp/clauseguard-cache}"
TAG="$(echo "$MODULES" | tr ' ' '-')"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv" UV_CACHE_DIR="$SCRATCH/uv" HF_HOME="$SCRATCH/hf"
export HF_HUB_DISABLE_PROGRESS_BARS=1 COLUMNS=120
cd "$(dirname "$0")/.."
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

step "installing (Python 3.12, dependencies, CUDA PyTorch)"
export PATH="$HOME/.local/bin:$PATH"  # where pip --user and the uv installer put uv
command -v uv >/dev/null || python3 -m pip install -q --user uv \
  || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
if ! uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  step "PyTorch from PyPI cannot use this GPU driver; installing the CUDA 12.6 build instead"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --reinstall torch \
    --index-url https://download.pytorch.org/whl/cu126
  export UV_NO_SYNC=1  # keep that build: don't let `uv run` restore the locked one
fi
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; \
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

step "datasets (downloaded and checked, splits from data/splits)"
make data
step "quick tests"
make test-fast
step "the whole session in miniature (CPU): $MODULES"
make train-gpu-smoke MODULES="$MODULES"
if [ "$JOINT" = "1" ]; then uv run clauseguard joint train --smoke; fi

step "training: $MODULES"
make train-gpu MODULES="$MODULES"
step "test results, ONNX export, robustness, latency, packaging: $MODULES"
make after-gpu MODULES="$MODULES"
if [ "$JOINT" = "1" ]; then
  step "the joint model (all three modules, one encoder)"
  uv run clauseguard joint train --final
fi

step "writing the results archive"
ARCHIVE="$OUT/clauseguard-results-${TAG}.tgz"
uv run clauseguard results pack "$ARCHIVE" $(for m in $MODULES; do echo --module "$m"; done)
step "done: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
