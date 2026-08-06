#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_night04_chronos2_mmsp_power_baseline.sh GPU_ID SOURCE_DIR
#
# SOURCE_DIR must be the artifact directory containing inputs.npy, targets.npy,
# site_ids.npy, input/forecast boundary timestamps, and per-horizon timestamps.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/torch/bin/python"
MODEL_DIR="/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
GPU_ID="${1:-}"
SOURCE_DIR="${2:-}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT}/outputs/chronos2_baseline/${RUN_STAMP}_chronos2_mmsp_power_only"

if [[ -z "${GPU_ID}" || -z "${SOURCE_DIR}" ]]; then
    echo "Usage: bash $0 GPU_ID SOURCE_DIR" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Local Chronos-2 model snapshot is unavailable: ${MODEL_DIR}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${ROOT}"

COMMON_ARGS=(
    --source-dir "${SOURCE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL_DIR}"
    --device-map cuda
    --batch-size 64
    --quantile-levels 0.1,0.5,0.9
    --point-quantile 0.5
    --local-files-only
)

"${PYTHON_BIN}" scripts/run_chronos2_mmsp_power_baseline_v2.py \
    "${COMMON_ARGS[@]}" --preflight-only

exec "${PYTHON_BIN}" -u scripts/run_chronos2_mmsp_power_baseline_v2.py \
    "${COMMON_ARGS[@]}"
