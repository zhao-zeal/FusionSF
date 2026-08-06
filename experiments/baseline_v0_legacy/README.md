# baseline_v0_legacy

The first FusionSF/Chronos-2 runs are preserved in place under `logs/`; this directory is an index,
not a duplicate. Artifact paths below are relative to the repository root.

## Shared setup

- Date: 2026-07-28 through 2026-07-30
- Dataset: MMSP
- Input/prediction length: 24/24 hours; direct multi-horizon forecast; stride 1
- Split: 60%/20%/20% after constructing windows (`window_split_v0`)
- Scalers: fitted on the complete NWP/satellite arrays (`global_fit_v0`)
- Seed: 42; batch size: 16; optimizer: AdamW; learning rate: 0.0016
- Maximum epochs: 100; early stopping/checkpoint metric: validation MAE
- Git commit: not recorded at run time; the worktree contained local modifications

## Indexed FusionSF runs

| Experiment | Sites | Modalities | max observed epoch | masking (ctx/ts) | VQ (ts/ctx/guide) | MAE | RMSE | MAPE | Artifacts |
|---|---|---|---:|---|---|---:|---:|---:|---|
| `fusionSF_train10_test10` | train/test 0–9 | power+future NWP+satellite | 28 | 0.99/0 | true/true/false | 0.040520 | 0.085809 | 1392.65 | `logs/fusionSF_train10_test10/runs/1/` |
| `fusionSF_power_optimized_train10_test10` | train/test 0–9 | power | 21 | 0.99/0 | true/false/false | 0.178738 | 0.304493 | 0.697821 | `logs/fusionSF_power_optimized_train10_test10/runs/1/` |
| `fusionSF_power_nwp_optimized_train10_test10` | train/test 0–9 | power+future NWP | 21 | 0.99/0 | true/false/false | 0.178738 | 0.304493 | 0.697821 | `logs/fusionSF_power_nwp_optimized_train10_test10/runs/1/` |
| `fusionsf_zeroshot_train10_19_test0_9` | train 10–19, test 0–9 | power+future NWP+satellite | 42 | 0.85/0 | false/false/false | 0.044397 | 0.095651 | 1482.84 | `logs/fusionsf_zeroshot_train10_19_test0_9/runs/1/` |

Each FusionSF run directory contains its resolved `.hydra/config.yaml`, overrides, checkpoint(s),
CSV metrics, execution time, and `src.models.fusionSF_3modal.FusionSF3M/{inputs,outputs,targets}.npy`.
Exact paths and settings are recorded in `../experiment_registry.csv` and `artifact_index.csv`.

## Known issues and intended use

该实验为第一批预实验结果，可能存在滑动窗口切分边界重叠、
全数据归一化以及模态缺失评估不完整等问题。

该结果仅作为代码跑通、趋势观察和后续修复对照，
不能直接作为最终论文结果。

In addition, the power-only and power+NWP FusionSF predictions are entirely zero and must be
treated as failed/degenerate ablations. MAPE is distorted by zero nighttime targets. No capacity
field was available for nMAE/nRMSE, and only one seed was run. No historical embedding was saved;
`embedding_extracted=false` is recorded rather than inferred after the fact.
