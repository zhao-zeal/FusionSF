# Chronos-guided TS-VQ Task 4 acceptance

Acceptance date: 2026-08-18

## Verdict

The approved seed-42 experiment completed successfully and passes numerical and
artifact-integrity acceptance. It is the best of the four directly comparable
arms on both MAE and RMSE. The result remains a single-seed finding rather than a
claim of statistically reliable improvement.

## Protocol and configuration

- Dataset: MMSP.
- Train/validation: first 60%/next 20% chronological target slices of sites 10-19.
- Test: final 20% chronological target slice of unseen sites 0-9; sites 20-21 unused.
- Model: FusionSF3M Partial-VQ with frozen Chronos-2 representation.
- Guidance: frozen 768-to-64 projection conditions the TS VQ input (`pre_ts_vq`, scale 1.0).
- Modalities: power, future NWP, and satellite.
- Sequence/prediction length: 24/24; seed: 42; batch size: 16; learning rate: 0.0016.
- Maximum epochs: 100; early stopping selected epoch 3 after 20 non-improving checks.
- Test set: 25,450 windows; target tensor shape `(25450, 24, 1)`.
- Recorded source commit: `adcdf9ad7072355e9689dec6ab92f9f6fa763a48`; the run recorded a dirty worktree.

## Results

| Arm | MAE | MAE change vs baseline | RMSE | RMSE change vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Task 1 baseline | 0.042308796 | 0.000000000 (0.000%) | 0.090031847 | 0.000000000 (0.000%) |
| Parameter-matched history MLP | 0.045678257 | +0.003369461 (+7.964%) | 0.087989389 | -0.002042458 (-2.269%) |
| Frozen Chronos prior | 0.041150926 | -0.001157870 (-2.737%) | 0.087311562 | -0.002720286 (-3.021%) |
| Chronos-guided TS-VQ | **0.041070426** | **-0.001238370 (-2.927%)** | **0.086440667** | **-0.003591181 (-3.989%)** |

Relative to the previous best frozen-prior arm, guided TS-VQ improves MAE by
0.000080500 (0.196%) and RMSE by 0.000870895 (0.997%). Daylight MAE/RMSE are
0.074491512/0.112666729 versus baseline 0.075934473/0.118480857.

## Integrity checks and limitations

- All four arms use exactly equal test targets and tensor shapes.
- Predictions and targets contain no NaN or infinite values; no all-zero forecast degeneration occurred.
- Raw negative prediction fraction is 42.158%; clipped MAE is 0.041022970.
- NMAE/NRMSE are unavailable because station-capacity metadata was not supplied.
- MAPE is unstable around zero-power/nighttime targets and is not an acceptance metric.
- Only seed 42 was approved and run; the small gain over the frozen-prior arm has not been validated across seeds.
- Focused implementation tests passed: 8 tests passed with 6 dependency deprecation warnings.
- `git diff --check` passed at acceptance time.

## Artifacts

- Output: `outputs/pipeline_v1_fixed/20260816_161011_fusionsf_fixedv1_chronos_guided_ts_vq_seed42`
- Metrics: `outputs/pipeline_v1_fixed/20260816_161011_fusionsf_fixedv1_chronos_guided_ts_vq_seed42/metrics.json`
- Horizon/site tables: `metrics_by_horizon.csv`, `metrics_by_site.csv` in the output directory.
- Best checkpoint: `checkpoints/epoch_epoch=003.ckpt` in the output directory.
- Resolved configuration: `config_resolved.yaml` in the output directory.
- Log: `logs/task4_chronos_guided_ts_vq_seed42.log`.
- Run-level registry: `experiments/experiment_registry.csv`.
- No task-specific chart was generated; numerical tables and raw predictions are retained.

## Conclusion

Chronos-guided TS-VQ is accepted as the best seed-42 arm under the approved MMSP
protocol. The minimum follow-up, if broader evidence is later authorized, is a
predeclared multi-seed confirmation against the frozen-prior arm.
