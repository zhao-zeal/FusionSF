#!/usr/bin/env bash
set -euo pipefail

# Run the leakage-safe fixed_v1 pipeline with the legacy full-modal experiment
# hyperparameters. This is the controlled counterpart of night01.
#
# Usage:
#   bash scripts/run_night03_fixedv1_legacy_matched_full.sh [PHYSICAL_GPU_ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/FusionSF/bin/python"
GPU_ID="${1:-2}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd "${ROOT}"
exec "${PYTHON_BIN}" -u main.py \
    experiment=fusionsf_pipeline_v1_legacy_matched_full
