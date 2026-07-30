#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_fusionsf_modality_ablation.sh [GPU_ID]
# Example:
#   bash scripts/run_fusionsf_modality_ablation.sh 7

export CUDA_VISIBLE_DEVICES="${1:-${GPU_ID:-7}}"
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

run_ablation() {
    local experiment="$1"
    local task_name="$2"

    # Keep batch_size=16, matching the existing all-modality optimizer schedule.
    python main.py \
        experiment="${experiment}" \
        datamodule.dataset.num_sites=10 \
        datamodule.dataset.num_ignored_sites=0 \
        datamodule.batch_size=16 \
        datamodule.test_batch_size=64 \
        datamodule.num_workers=4 \
        datamodule.pin_memory=True \
        seed=42 \
        resume=False \
        task_name="${task_name}" \
        pl_module.model.ts_masking_ratio=0 \
        pl_module.model.ctx_masking_ratio=0.99 \
        pl_module.model.vq_in_ts=True \
        pl_module.model.vq_in_ctx=False \
        pl_module.model.vq_in_guide=False \
        "~trainer.strategy" \
        trainer.max_epochs=100 \
        callbacks.early_stopping.patience=20
}

echo "[1/3] Training FusionSF with power only on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
run_ablation fusionsf_power_only fusionSF_power_optimized_train10_test10

echo "[2/3] Training FusionSF with power + NWP"
run_ablation fusionsf_power_nwp fusionSF_power_nwp_optimized_train10_test10

echo "[3/3] Comparing both runs with the existing all-modality run"
python scripts/compare_fusionsf_modalities.py
