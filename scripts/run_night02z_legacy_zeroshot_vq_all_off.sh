#!/usr/bin/env bash
set -euo pipefail

# Legacy cross-site reproduction: train sites 10-19, unseen test sites 0-9,
# with every VQ branch disabled.
# Usage: bash scripts/run_night02z_legacy_zeroshot_vq_all_off.sh [GPU_ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/FusionSF/bin/python"
GPU_ID="${1:-2}"
TASK_NAME="fusionSF_legacy_zeroshot_vq_all_off_20260806_seed42"
RUN_DIR="${ROOT}/logs/${TASK_NAME}/runs/1"

if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    exit 2
fi
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
    experiment=fusionsf_3modal_zeroshot \
    task_name="${TASK_NAME}" \
    seed=42 \
    resume=False \
    datamodule.batch_size=16 \
    +datamodule.test_batch_size=64 \
    datamodule.num_workers=4 \
    datamodule.pin_memory=True \
    pl_module.model.ctx_masking_ratio=0.99 \
    pl_module.model.ts_masking_ratio=0 \
    pl_module.model.mlp_ratio=4 \
    pl_module.model.vq_in_ts=False \
    pl_module.model.vq_in_ctx=False \
    pl_module.model.vq_in_guide=False \
    trainer.strategy=ddp \
    trainer.max_epochs=100 \
    callbacks.early_stopping.patience=20
