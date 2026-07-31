# Experiment records

This directory is the versioned index for FusionSF experiments. Large artifacts remain in their
original output directories and are referenced rather than copied.

- `baseline_v0_legacy/`: immutable index of the first preliminary runs. These runs use the legacy
  window-index split and global scaler fit and are not final paper results.
- `pipeline_v1_fixed/`: configurations, notes, and outputs produced by the leakage-safe pipeline.
- `experiment_registry.csv`: one row per run or extraction job.
- `CHANGELOG.md`: pipeline changes that may affect metrics or reproducibility.

Pipeline versions must never share an output directory. `legacy_v0` remains available only for
reproduction; new experiments should use `fixed_v1`.
