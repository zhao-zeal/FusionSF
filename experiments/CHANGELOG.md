# Experiment pipeline changelog

## pipeline_v1_fixed — 2026-07-30

- Added target-time-based chronological window assignment so train, validation, and test target
  timestamps cannot overlap. Validation and test may use preceding history, but their targets are
  isolated.
- Added a train-only scaler interface and metadata for scaler fit ranges. The legacy global-fit
  path remains selectable and unchanged for reproduction.
- Added overall, daylight, normalized-capacity interface, and per-horizon MAE/RMSE evaluation.
- Added explicit modality availability masks and named missing-modality evaluation modes.
- Separated validation and test metric state and added aligned site/timestamp output metadata.
- Added unique, non-overwriting experiment directories and automatic command, resolved config,
  environment, Git status, and metric records.
- Added explicit TS and fused-representation extraction plus Chronos-compatible covariate export.
- Indexed, but did not copy, move, modify, or delete `baseline_v0_legacy` artifacts.

These changes require retraining for valid `pipeline_v1_fixed` results and are expected to change
reported metrics. They do not alter the stored legacy checkpoints, predictions, logs, Hydra
configs, or W&B information.

### Files and impact

- `src/datasets/split_utils.py`, `src/datasets/tscontext_3modal_dataset.py`, and
  `src/datamodules/tscontext_3modal_datamodule.py`: split ownership, timestamp assertions,
  train-only scaling, grid-local NWP interpolation, aligned metadata. These changes affect all new
  metrics and require retraining; `data_pipeline.version=legacy_v0` retains the old path.
- `src/models/fusionSF_3modal.py`: fixed-ratio TS/context masking, explicit modality masks,
  non-dead fixed_v1 output activation, and embedding API. New checkpoints are not metric-comparable
  to legacy checkpoints without a controlled rerun.
- `src/pl_modules/pl_context_3modal.py`, `src/metrics/forecast_metrics.py`, and
  `src/utils/experiment_records.py`: separate test metric state, daylight/horizon evaluation,
  aligned outputs, and provenance recording.
- `main.py`: removed the pre-training manual `training_step`; fixed_v1 writes unique structured
  outputs and appends the registry. Legacy artifact paths remain supported.
- `scripts/extract_embeddings.py` and `scripts/validate_pipeline_v1.py`: checkpoint-only embedding
  export and bounded real-data validation.
- `configs/*/*fixed_v1*.yaml`, `configs/experiment/fusionsf_pipeline_v1_smoke.yaml`, and `tests/`:
  new-only configuration and regression coverage. Existing first-batch experiment configs were not
  edited.

### Validation

- 14 unit tests passed.
- Real MMSP check passed with two sites: split sizes 15,314/5,090/5,090, finite forward output,
  full/missing-satellite/missing-NWP modes, and both embedding types.
- One-epoch smoke tests were bounded to two train, validation, and test batches; they are registered
  as `valid_or_debug`, not paper results.
- Checkpoint extraction produced 5,090 aligned test rows with TS and fusion shapes
  `(5090, 24, 16)` and pooled shapes `(5090, 16)`.
- The 3-site/3-epoch integration run completed without output collapse. Validation MAE improved
  from 0.213976 to 0.041122; test MAE/RMSE were 0.044213/0.088583 over 7,635 windows. This remains
  a debug integration result, not a paper result.
- Added explicit raw and non-negative-clipped predictions and metrics. The integration checkpoint
  has 2.40% negative raw values; clipping changes MAE only from 0.044213 to 0.044184.
- Added checkpoint-only evaluation and ran four availability modes. Missing NWP, satellite, and
  both increased raw MAE by 124.90%, 255.68%, and 304.52%. These are sensitivity diagnostics only:
  whole-modality missing tokens were not trained in the source full-modality run.
