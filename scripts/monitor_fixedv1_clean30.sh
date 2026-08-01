#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaopp/workspace/FusionSF
PYTHON=/home/zhaopp/miniconda3/envs/FusionSF/bin/python
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-7200}
LOG_DIR="$ROOT/outputs/pipeline_v1_fixed/clean30_monitor"
LOG_FILE="$LOG_DIR/monitor.log"

RUN_DIRS=(
  "$ROOT/outputs/pipeline_v1_fixed/20260731_223710_fusionsf_fixedv1_clean30_power_sites10_seed42"
  "$ROOT/outputs/pipeline_v1_fixed/20260731_223710_fusionsf_fixedv1_clean30_power_nwp_sites10_seed42"
  "$ROOT/outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_full_sites10_seed42"
  "$ROOT/outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42"
)
TRAINING_SESSIONS=(power_clean30 power_nwp_clean30 full_clean30 zeroshot_clean30)
FULL_DIR=${RUN_DIRS[2]}
ZERO_DIR=${RUN_DIRS[3]}
FULL_EMBEDDING_DIR="$FULL_DIR/embedding_export_clean30_test"
ZERO_EMBEDDING_DIR="$ZERO_DIR/embedding_export_clean30_test"

mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1

timestamp() {
  date '+%F %T %Z'
}

training_is_active() {
  local session
  for session in "${TRAINING_SESSIONS[@]}"; do
    if tmux -L fixedv1 has-session -t "$session" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

validate_training_outputs() {
  local run_dir
  for run_dir in "${RUN_DIRS[@]}"; do
    test -s "$run_dir/config_resolved.yaml"
    test -s "$run_dir/metrics.json"
    test -s "$run_dir/metrics_by_horizon.csv"
    test -s "$run_dir/metrics_by_site.csv"
    test -s "$run_dir/run_summary.json"
    local checkpoint
    checkpoint=$(jq -r '.best_checkpoint // empty' "$run_dir/run_summary.json")
    test -n "$checkpoint"
    test -s "$checkpoint"
  done
}

extract_one() {
  local run_dir=$1
  local output_dir=$2
  local seen_flag=$3
  local checkpoint
  checkpoint=$(jq -r '.best_checkpoint' "$run_dir/run_summary.json")
  if test -e "$output_dir"; then
    echo "[$(timestamp)] Embedding output already exists; refusing overwrite: $output_dir"
    return 1
  fi
  local command=(
    "$PYTHON" "$ROOT/scripts/extract_embeddings.py"
    --config "$run_dir/config_resolved.yaml"
    --checkpoint-path "$checkpoint"
    --output-dir "$output_dir"
    --splits test
    --embedding-type both
    --pooling none mean
  )
  if test "$seen_flag" = true; then
    command+=(--site-seen-during-training)
  fi
  "${command[@]}"
}

validate_embedding_metadata() {
  "$PYTHON" - "$FULL_EMBEDDING_DIR" "$ZERO_EMBEDDING_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

full_dir, zero_dir = map(Path, sys.argv[1:])
for directory in (full_dir, zero_dir):
    split = directory / "embeddings/test"
    metadata = pd.read_csv(split / "metadata.csv")
    arrays = [
        np.load(split / "ts_embedding.npy", mmap_mode="r"),
        np.load(split / "ts_embedding_mean.npy", mmap_mode="r"),
        np.load(split / "fusion_embedding.npy", mmap_mode="r"),
        np.load(split / "fusion_embedding_mean.npy", mmap_mode="r"),
    ]
    assert all(len(array) == len(metadata) for array in arrays)
    assert all(np.isfinite(array).all() for array in arrays)

full_manifest = json.loads((full_dir / "extraction_manifest.json").read_text())
zero_manifest = json.loads((zero_dir / "extraction_manifest.json").read_text())
assert full_manifest["training_site_ids"] == list(range(10))
assert full_manifest["extraction_site_ids"] == list(range(10))
assert full_manifest["site_seen_during_training"] is True
assert zero_manifest["training_site_ids"] == list(range(10, 20))
assert zero_manifest["extraction_site_ids"] == list(range(10))
assert zero_manifest["site_seen_during_training"] is False
print("embedding metadata validation passed")
PY
}

echo "[$(timestamp)] clean30 monitor started; interval=${CHECK_INTERVAL_SECONDS}s"
while training_is_active; do
  echo "[$(timestamp)] training still active"
  for session in "${TRAINING_SESSIONS[@]}"; do
    if tmux -L fixedv1 has-session -t "$session" 2>/dev/null; then
      tmux -L fixedv1 capture-pane -p -t "$session" -S -1 | tail -1
    else
      echo "$session: finished"
    fi
  done
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "[$(timestamp)] all training sessions ended; validating outputs"
validate_training_outputs
extract_one "$FULL_DIR" "$FULL_EMBEDDING_DIR" true
extract_one "$ZERO_DIR" "$ZERO_EMBEDDING_DIR" false
validate_embedding_metadata
echo "[$(timestamp)] embedding extraction and validation completed"
