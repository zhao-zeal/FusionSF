#!/usr/bin/env bash
set -euo pipefail

# Train the migrated Stage 4B CoRA-inspired adapter. This is intentionally not
# appended to run_night_zero_shot.sh: unlike the frozen baselines, it trains an
# adapter and requires an audited Stage 4A token cache.
#
# Usage:
#   bash scripts/run_night05_stage4b_correlation_adapter.sh GPU_ID CACHE_DIR [SEED]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/torch/bin/python"
MODEL_DIR="/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
GPU_ID="${1:-}"
CACHE_DIR="${2:-}"
SEED="${3:-2021}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT}/outputs/stage4b_correlation_adapter/${RUN_STAMP}_seed${SEED}"

if [[ -z "${GPU_ID}" || -z "${CACHE_DIR}" ]]; then
    echo "Usage: bash $0 GPU_ID CACHE_DIR [SEED]" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -d "${MODEL_DIR}" ]]; then
    echo "Chronos-2 Python environment or local model snapshot is unavailable" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${ROOT}"

COMMON_ARGS=(
    --cache-dir "${CACHE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL_DIR}"
    --device cuda:0
    --seed "${SEED}"
    --shuffle-seed "${SEED}"
    --train-batch-size 128
    --eval-batch-size 256
    --epochs 100
    --learning-rate 3e-4
    --weight-decay 1e-4
    --patience 20
    --local-files-only
)
if [[ -f "${CACHE_DIR}/fusion_aligned_predictions.npy" ]]; then
    COMMON_ARGS+=(--stage4a-predictions "${CACHE_DIR}/fusion_aligned_predictions.npy")
fi

"${PYTHON_BIN}" scripts/run_stage4b_correlation_adapter.py \
    "${COMMON_ARGS[@]}" --preflight-only

exec "${PYTHON_BIN}" -u scripts/run_stage4b_correlation_adapter.py \
    "${COMMON_ARGS[@]}"
