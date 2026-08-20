# FusionSF Project Status

Last updated: 2026-08-19

## Active

- `chronos_codebook_guided_ts_vq`: implementation and focused unit tests are
  complete. The new mode adds frozen-Chronos similarity only to TS codebook
  selection and exactly follows the original VQ path at lambda zero. No training
  or large experiment has been run.

## Completed / reviewed

- `chronos_guided_vq_task4_20260816`: completed and audited the approved seed-42
  MMSP follow-up. Conditioning the TS VQ input with the frozen Chronos-2 768-to-64
  projection reached MAE/RMSE 0.041070/0.086441 versus the Task 1 baseline's
  0.042309/0.090032 (improvements of 2.927%/3.989%). This is the best result in
  the four-arm gate, but the incremental MAE gain over the frozen-prior arm is
  only 0.196% and has not been validated across seeds. See
  `reports/chronos_guided_vq_task4_20260816_acceptance.md`.

- `chronos_reverse_guidance_task1_3_20260815`: completed the seed-42 MMSP
  three-arm gate. Training and validation are the first 60% and next 20%
  chronological slices of sites 10-19; testing is the final 20% slice of unseen
  sites 0-9, and sites 20-21 are not used. Baseline, parameter-matched history
  MLP, and frozen Chronos-2 prior reached MAE/RMSE 0.042309/0.090032,
  0.045678/0.087989, and 0.041151/0.087312, respectively.

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

- Review the codebook-guided implementation before authorizing any MMSP training run.

## Boundaries

- Do not run Solar-native leaderboard experiments here.
- Do not modify fixed_v1 historical records or `FusionSF3M` behavior.
