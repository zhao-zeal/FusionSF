# FusionSF Project Status

Last updated: 2026-08-02

## Active

- None. No FusionSF training or Chronos-2 embedding fusion is currently running in this repository.

## Completed / reviewed

- `mmsp_embedding_to_chronos2`: blocked at Stage 3B preflight because MMSP NWP
  provides valid time but no issue/publication time or forecast cycle. No model
  inference or new embedding export was run; see the solar-energy audit.
- `fixed_v1_preliminary`: MMSP fixed_v1 Power, Power+NWP, full-modal and zero-shot preliminary experiments are recorded in `experiments/experiment_registry.csv`.
- `fusionsf_solar_migration`: Solar migration framework and smoke protocols are recorded in `experiments/solar_energy/experiment_registry.csv`; native Solar leaderboard execution was moved to the Solar repository.
- `mmsp_ts_embedding_to_chronos2`: completed a frozen TS-branch-only experiment on unseen MMSP sites 0–9. The isolated read-only interface exactly matches the original forward TS node; prediction artifacts are in the solar-energy repository. The aligned representation did not outperform its shuffled control.

## Planned

- Complete Fusion embedding cannot resume without NWP availability provenance.

## Boundaries

- Do not run Solar-native leaderboard experiments here.
- Do not modify fixed_v1 historical records or `FusionSF3M` behavior.
