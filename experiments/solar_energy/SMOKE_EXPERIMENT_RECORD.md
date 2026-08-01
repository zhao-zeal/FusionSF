# Stage 2 prototype smoke experiment record

> **Superseded architecture notice:** these two runs used
> `model_family=prototype_horizon_gru`, `architecture_version=prototype_horizon_gru_v1`.
> They validate only the migrated data pipeline and must not be cited as FusionSF
> model results. The aligned formal architecture is recorded in
> [`STAGE2A_PROTOCOL_RECORD.md`](STAGE2A_PROTOCOL_RECORD.md).

Date: 2026-08-01

Status: completed smoke only; awaiting review

Code commit containing the executed prototype implementation: `913a9fa`

Execution provenance: the runs were executed from the corresponding dirty working tree,
then the same implementation was committed as `913a9fa`.

## Scope and guardrails

This record covers only the first migration validation on GEFCom zone1. No full
three-dataset experiment, 30/100-epoch training, satellite branch, Chronos-2 code
change, FusionSF embedding extraction, Chronos-2 fusion, hyperparameter search, or
MMSP output/registry change was performed.

The design and data inventory are recorded separately in
[`MIGRATION_AUDIT.md`](MIGRATION_AUDIT.md).

## Shared configuration

| Item | Value |
|---|---|
| Dataset/site | GEFCom / zone1 |
| Source file | `/home/zhaopp/workspace/solar-energy/dataset/GEFCom/gefcom15_by_zone/zone1.csv` |
| Time grain | 1 hour |
| Power field/rule | `POWER`, clipped to `[0, 1]` |
| NWP fields | `VAR78,VAR79,VAR157,VAR164,VAR169,VAR178` |
| Split | chronological 60%/10%/30% |
| Input/prediction | `seq_len=336`, `pred_len=1` |
| Seed | 42 |
| Batch size | 8 |
| Budget | 1 epoch; at most 2 train, 2 validation, 2 test batches |
| Scaler | fit on training-time rows only |
| Chronos reference | saved zone1/pred1 `t_origin.npy`, Chronos context 2048 |

## Commands

Power:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python scripts/run_fusionsf_solar_smoke.py \
  --csv /home/zhaopp/workspace/solar-energy/dataset/GEFCom/gefcom15_by_zone/zone1.csv \
  --chronos-origin /home/zhaopp/workspace/solar-energy/results/solar_chronos2_gefcom/GEFCOM_zone1_2048_1_Chronos2_ctx2048_cl0_covFut_freq1h_strict1_tr0p3_skipIr1_mb64_wb256_ql0p4-saveQ0_Exp_0/t_origin.npy \
  --output-dir outputs/solar_energy_smoke/gefcom_zone1_pred1_power \
  --mode power --seq-len 336 --pred-len 1 --max-epochs 1 \
  --limit-train-batches 2 --limit-val-batches 2 --limit-test-batches 2 --seed 42
```

Power+NWP used the same command with:

```bash
--output-dir outputs/solar_energy_smoke/gefcom_zone1_pred1_power_nwp --mode power_nwp
```

## Shapes and optimization checks

| Mode | History power | Future NWP | Prediction | Backward |
|---|---|---|---|---|
| Power | `[8,336,1]` | not read | `[8,1,1]` | passed |
| Power+NWP | `[8,336,1]` | `[8,1,6]` | `[8,1,1]` | passed |

Both runs produced finite predictions. Each evaluated 16 test windows and saved 16
metadata rows, so metadata/prediction alignment passed.

## Smoke results

These values validate the pipeline only and are not formal performance estimates.

| Mode | Last train loss | Validation loss | Test loss | MAE | RMSE | Samples |
|---|---:|---:|---:|---:|---:|---:|
| Power | 1.042262 | 1.439746 | 1.443212 | 0.285521 | 0.308516 | 16 |
| Power+NWP | 1.009761 | 1.391344 | 1.169157 | 0.254767 | 0.277683 | 16 |

Relative to Power, Power+NWP changed MAE by `-0.030754` (`-10.77%`) and RMSE by
`-0.030833` (`-9.99%`). Power+NWP is the best smoke result, but the sample and training
budgets are too small for a scientific comparison.

## Shared sample manifest verification

- Each generated manifest has 16,264 data rows plus one header: 11,054 train,
  1,562 validation, and 3,648 test rows.
- The 3,648 FusionSF test forecast starts match all 3,648 saved Chronos-2 origins
  after adding one hour.
- Alignment result: `true` for every test row.
- First/last shared forecast timestamp: 2013-12-31 01:00 / 2014-06-01 00:00.
- Published manifest columns are dataset, site, split, input start/end, forecast
  start/end, `seq_len`, and `pred_len`.

## Tests

Migration-specific tests: `9 passed`.

Full FusionSF suite after implementation: `33 passed, 6 dependency deprecation warnings`.

The tests cover target-time separation, training-only scaler fitting, causal input
and forecast ordering, five native horizons, gradient propagation, Power-mode NWP
isolation, exact future-NWP slicing, and output shapes.

## Artifacts

Power: `outputs/solar_energy_smoke/gefcom_zone1_pred1_power/`

Power+NWP: `outputs/solar_energy_smoke/gefcom_zone1_pred1_power_nwp/`

Each directory contains `model.pt`, `metrics.json`, `losses.json`, `shapes.json`,
`run_config.json`, `sample_manifest.csv`, `prediction_metadata.csv`,
`predictions.npy`, and `targets.npy`. These runtime artifacts are intentionally
git-ignored and remain local. No chart or dedicated console log was generated.

## Failures and unresolved items

1. The first Power invocation stopped before training with `ModuleNotFoundError: src`;
   repository-root bootstrapping was added and both reruns passed.
2. CSG sites 6–8 have a current header/data column-count mismatch and were not used.
3. Whether CSG native future weather is forecast-available NWP remains unverified.
4. A verified SKIPPD Chronos-2 test-origin artifact was not found.
5. Chronos-2 uses a 2048-step context while FusionSF uses 336; target timestamps are
   aligned, but histories are intentionally not identical.

Conclusion: the native-horizon migration path and shared GEFCom test protocol passed
the minimum smoke gate. Next minimum action is review approval before resolving the
CSG/SKIPPD protocol gaps or running another dataset/horizon.
