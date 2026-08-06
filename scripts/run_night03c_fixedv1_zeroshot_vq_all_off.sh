#!/usr/bin/env bash
set -euo pipefail

# fixed_v1 cross-site run: train sites 10-19, unseen test sites 0-9, all VQ off.
# Usage: bash scripts/run_night03c_fixedv1_zeroshot_vq_all_off.sh [GPU_ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/FusionSF/bin/python"
GPU_ID="${1:-2}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd "${ROOT}"
exec "${PYTHON_BIN}" -u main.py \
    experiment=fusionsf_pipeline_v1_legacy_matched_zeroshot_vq_all_off
