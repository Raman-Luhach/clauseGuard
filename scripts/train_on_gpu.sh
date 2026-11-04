#!/usr/bin/env bash
# Train ClauseGuard's transformer models on a Linux GPU machine (Kaggle, Colab, a server) and
# write one results archive. Run from anywhere inside the repository:
#
#   MODULES="tos privacy" OUT=/kaggle/working bash scripts/train_on_gpu.sh
#   MODULES="contract"    OUT=/kaggle/working bash scripts/train_on_gpu.sh
#
# MODULES  modules to train in this run (default: all three)
# OUT      where the results archive goes (default: current directory)
# SCRATCH  large downloads: virtual environment, caches (default: /tmp/clauseguard-cache)
set -euo pipefail

MODULES="${MODULES:-tos privacy contract}"
OUT="${OUT:-$PWD}"
SCRATCH="${SCRATCH:-/tmp/clauseguard-cache}"
TAG="$(echo "$MODULES" | tr ' ' '-')"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv" UV_CACHE_DIR="$SCRATCH/uv" HF_HOME="$SCRATCH/hf"
export HF_HUB_DISABLE_PROGRESS_BARS=1 COLUMNS=120
cd "$(dirname "$0")/.."
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

step "installing (Python 3.12, dependencies, CUDA PyTorch)"
command -v uv >/dev/null || pip install -q uv
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

step "training: $MODULES"
make train-gpu MODULES="$MODULES"
step "test results, ONNX export, robustness, latency, packaging: $MODULES"
make after-gpu MODULES="$MODULES"

step "writing the results archive"
for name in final_results.md final_results.json gpu_session.json; do
  cp "reports/generated/$name" "reports/generated/${name%.*}_${TAG}.${name##*.}"
done
cp experiments/log.csv "experiments/log_${TAG}.csv"
mkdir -p "$OUT"
tar czf "$OUT/clauseguard-results-${TAG}.tgz" \
  deploy/models reports/generated "experiments/log_${TAG}.csv" data/data_cards
step "done: $OUT/clauseguard-results-${TAG}.tgz ($(du -h "$OUT/clauseguard-results-${TAG}.tgz" | cut -f1))"
