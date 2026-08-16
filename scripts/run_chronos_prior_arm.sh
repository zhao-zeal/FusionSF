#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <baseline|mlp|frozen> <gpu>" >&2
  exit 2
fi

arm="$1"
gpu="$2"
case "$arm" in
  baseline|mlp|frozen) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

cd /home/zhaopp/workspace/FusionSF
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONNOUSERSITE=1

exec /home/zhaopp/miniconda3/envs/FusionSF/bin/python -u main.py \
  "experiment=fusionsf_pipeline_v1_chronos_prior_${arm}" \
  seed=42 \
  resume=False \
  trainer.devices=1 \
  trainer.strategy=null
