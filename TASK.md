# FusionSF Project Status

Last updated: 2026-08-15

## Active

- `chronos_reverse_guidance_task1_3_20260815`: approved Task 1-3 study of
  frozen Chronos-2 temporal representations injected into the current fixed-v1
  Partial-VQ FusionSF. The fixed MMSP protocol is train/validation/test sites
  10-19/20-21/0-9, seq/pred 24/24, seed 42. The three fair arms are the unchanged
  FusionSF baseline, a 49.3k-parameter history-power MLP control, and a 49.2k-
  parameter projection of mean-pooled frozen Chronos-2 encoder states. Training
  and validation are chronological slices from sites 10-19; the held-out test
  sites are 0-9. Guided-VQ
  and all out-of-scope datasets/models remain deferred until this gate is reviewed.

## Completed / reviewed

- `night_reproduction_20260806`: completed the seed-42 MMSP legacy, fixed_v1,
  cross-site zero-shot, VQ-ablation, Chronos-2 baseline, and Stage4b seed-2021
  runs. Consolidated metrics are in `reports/weekly_experiments_20260803_20260810.csv`.
- `legacy_zeroshot_partial_vq_identity_20260810`: completed the single-variable
  identity-output rerun on MMSP train sites 10-19/test sites 0-9, seed 42,
  seq/pred 24/24. Test MAE/RMSE are 0.042264875/0.089217305 versus the degenerate
  ReLU run's 0.178765759/0.304493129; the activation change removed the
  near/all-zero prediction failure.
- `stage4b_correlation_adapter_migration_20260806`: completed the auditable
  seed-2021 run with train/validation/test sites 10-19/20-21/0-9. The aligned
  adapter reached MAE/RMSE 0.046355557/0.097611664; seeds 2022 and 2023 have not
  been run.

- `mmsp_fusion_embedding_to_chronos2`: completed frozen inference on 25,450 windows from unseen MMSP sites 0–9 after user confirmation that MMSP future NWP is available at forecast origin. Formal fusion-node sensitivity to both NWP and satellite passed, but Fusion aligned underperformed baseline and TS and exactly matched shuffled Fusion predictions in Chronos-2. The negative result is retained in the solar-energy artifacts.

- `mmsp_embedding_to_chronos2`: blocked at Stage 3B preflight because MMSP NWP
  provides valid time but no issue/publication time or forecast cycle. No model
  inference or new embedding export was run; see the solar-energy audit.
- `fixed_v1_preliminary`: MMSP fixed_v1 Power, Power+NWP, full-modal and zero-shot preliminary experiments are recorded in `experiments/experiment_registry.csv`.
- `fusionsf_solar_migration`: Solar migration framework and smoke protocols are recorded in `experiments/solar_energy/experiment_registry.csv`; native Solar leaderboard execution was moved to the Solar repository.
- `mmsp_ts_embedding_to_chronos2`: completed a frozen TS-branch-only experiment on unseen MMSP sites 0–9. The isolated read-only interface exactly matches the original forward TS node; prediction artifacts are in the solar-energy repository. The aligned representation did not outperform its shuffled control.

## Planned

- No Fusion method change is planned; the completed negative result is retained without tuning.

## Boundaries

- Do not run Solar-native leaderboard experiments here.
- Do not modify fixed_v1 historical records or `FusionSF3M` behavior.
