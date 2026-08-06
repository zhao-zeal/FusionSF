#!/usr/bin/env bash
set -euo pipefail

# No-VQ ablation on the legacy full-modal MMSP protocol. The only experimental
# change relative to run_night01_legacy_full_repro.sh is that all VQ branches
# are disabled.
#
# Usage:
#   bash scripts/run_night02_legacy_vq_all_off.sh [PHYSICAL_GPU_ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/zhaopp/miniconda3/envs/FusionSF/bin/python"
GPU_ID="${1:-2}"
TASK_NAME="fusionSF_legacy_vq_all_off_repro_20260806_seed42"
RUN_DIR="${ROOT}/logs/${TASK_NAME}/runs/1"

if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd "${ROOT}"
exec "${PYTHON_BIN}" -u main.py \
    experiment=fusionsf_3modal \
    datamodule.dataset.num_sites=10 \
    datamodule.dataset.num_ignored_sites=0 \
    datamodule.batch_size=16 \
    datamodule.test_batch_size=64 \
    datamodule.num_workers=4 \
    datamodule.pin_memory=True \
    seed=42 \
    resume=False \
    task_name="${TASK_NAME}" \
    pl_module.model.ts_masking_ratio=0 \
    pl_module.model.ctx_masking_ratio=0.99 \
    pl_module.model.vq_in_ts=False \
    pl_module.model.vq_in_ctx=False \
    pl_module.model.vq_in_guide=False \
    trainer.strategy=ddp \
    trainer.max_epochs=100 \
    callbacks.early_stopping.patience=20
