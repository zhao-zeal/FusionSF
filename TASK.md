# FusionSF Project Status

Last updated: 2026-08-06

## Active

- `night_reproduction_20260806`: approved sequential experiment program under user supervision.
  The ordered scope is legacy FusionSF reproduction, VQ ablation, matched-configuration
  fixed_v1 training, Chronos-2 baselines, and the Stage 4 improved Chronos-2 structure.
  The first prepared run is the legacy full-modal FusionSF reproduction on MMSP sites 0-9,
  seed 42, seq/pred 24/24, maximum 100 epochs, on physical GPU 2. Historical
  output directories must remain untouched and no two training runs may overlap. No run
  has been started by Codex; execution is waiting for the user's manual command in
  `scripts/run_night01_legacy_full_repro.sh`.
- `night_legacy_vq_comparison_20260806`: approved and prepared, awaiting manual
  execution. `run_night01_legacy_full_repro.sh` uses the paper-best setting
  (power/TS and satellite/context VQ on; NWP/guide VQ off), while
  `run_night02_legacy_vq_all_off.sh` disables all VQ branches.
- `night_fixedv1_legacy_matched_20260806`: approved and prepared, awaiting manual
  execution through `scripts/run_night03_fixedv1_legacy_matched_full.sh`. It matches
  the night01 legacy model/training hyperparameters while retaining fixed_v1
  correctness changes and structured outputs.
- `night_fixedv1_vq_comparison_20260806`: two approved fixed_v1 legacy-matched
  runs are prepared: all VQ off (`run_night03a_fixedv1_vq_all_off.sh`) and the
  paper-best setting (`run_night03b_fixedv1_vq_paper_best.sh`). They must be run
  sequentially.
- `night_fusionsf_zeroshot_comparison_20260806`: the original four same-site
  reproduction runs do not test unseen sites. Four cross-site counterparts are
  now prepared: legacy and fixed_v1, each with all VQ off and paper-best VQ.
  Every run trains on sites 10-19 and tests on unseen sites 0-9; execution is
  manual and sequential through the `run_night01z`, `02z`, `03c`, and `03d`
  scripts.
- `night_chronos2_mmsp_baselines_20260806`: code prepared for two auditable
  Chronos-2 zero-shot baselines on the same aligned MMSP windows: power-only
  and power plus future NWP. Both emit 0.1/0.5/0.9 quantiles and use the median
  for point metrics. Each runs a metadata/timestamp/hash preflight before model
  loading. Execution remains manual and sequential through
  `scripts/run_night04_chronos2_mmsp_power_baseline.sh` and
  `scripts/run_night04b_chronos2_mmsp_future_nwp.sh`.
- `stage4b_correlation_adapter_migration_20260806`: the archived Stage 4B
  CoRA-inspired dual-branch adapter has been migrated into FusionSF with an
  explicit Stage 4A cache contract, strict 10-19/20-21/0-9 site checks, frozen
  Chronos-2 verification, aligned/shuffled controls, and a manual standalone
  entrypoint. No Stage 4B training has been started while the night zero-shot
  queue is running.

## Completed / reviewed

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
