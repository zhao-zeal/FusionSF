#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <baseline|guided_ts_vq|codebook_l1|codebook_l0> <seed> <gpu>" >&2
  exit 2
fi

method="$1"
seed="$2"
gpu="$3"

case "$method" in
  baseline)
    experiment="fusionsf_pipeline_v1_chronos_prior_baseline"
    extra_overrides=()
    ;;
  guided_ts_vq)
    experiment="fusionsf_pipeline_v1_chronos_guided_ts_vq"
    extra_overrides=()
    ;;
  codebook_l1)
    experiment="fusionsf_pipeline_v1_chronos_codebook_guided_ts_vq"
    extra_overrides=("pl_module.model.chronos_vq_guidance_lambda=1.0")
    ;;
  codebook_l0)
    experiment="fusionsf_pipeline_v1_chronos_codebook_guided_ts_vq"
    extra_overrides=("pl_module.model.chronos_vq_guidance_lambda=0.0")
    ;;
  *)
    echo "unknown method: $method" >&2
    exit 2
    ;;
esac

experiment_id="20260820_chronos_vq_stability_p10_${method}_seed${seed}"

cd /home/zhaopp/workspace/FusionSF
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONNOUSERSITE=1

exec /home/zhaopp/miniconda3/envs/FusionSF/bin/python -u main.py \
  "experiment=${experiment}" \
  "experiment_id=${experiment_id}" \
  "task_name=${experiment_id}" \
  "seed=${seed}" \
  resume=False \
  trainer.devices=1 \
  trainer.strategy=null \
  datamodule.num_workers=0 \
  callbacks.early_stopping.patience=10 \
  "${extra_overrides[@]}"
