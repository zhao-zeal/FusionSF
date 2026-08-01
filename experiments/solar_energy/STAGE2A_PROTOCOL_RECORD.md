# Stage 2A: FusionSF architecture alignment and protocol closure

Date: 2026-08-01

Status: six long-horizon smoke runs passed; formal training not started.

Implementation commit: `d9d9689`.

## Architecture status

The previous GRU/MultiheadAttention model remains available as
`PrototypeHorizonGRU`, identified by `prototype_horizon_gru_v1`. It is retained only
for data-pipeline provenance and is not registered as FusionSF.

The official migration model is `FusionSFSolar`, with:

- `model_family=FusionSF`
- `model_class=FusionSFSolar`
- `architecture_version=fusionsf_solar_v1`
- no `nn.GRU`
- original `fusionSF_3modal.Transformer` as the historical TS encoder
- original `fusionSF_3modal.CrossTransformer` as horizon-to-history cross-attention
- original FusionSF `Attention`, `FeedForward`, linear embedding, LayerNorm and
  per-time-step prediction-head design

No source line in `FusionSF3M` was changed, and its forward signature/checkpoint
layout remains intact.

## Data flows and embeddings

Power:

```text
history_power [B,336,1] + history_time_features [B,336,8]
→ FusionSF linear TS embedding
→ FusionSF Transformer
→ ts_embedding/history_tokens [B,336,D]

learned horizon offsets + future_time_features [B,pred_len,8]
→ horizon queries
→ FusionSF CrossTransformer reads history tokens
→ horizon representation [B,pred_len,D]
→ FusionSF-style fusion encoder and prediction head
→ prediction [B,pred_len,1]
```

Power+Weather adds:

```text
future weather [B,pred_len,C]
→ guide embedding + FusionSF-style Transformer blocks
→ guide representation [B,pred_len,D]
→ horizon + guide fusion
→ fusion_embedding [B,pred_len,D]
```

No history repetition, horizon clipping, interpolation, or future-power input is
used. In Power mode, no weather tensor is returned by the dataset and the guide
branch receives no gradients.

## Deterministic time coordinates

Both modes use the same eight timestamp-only values: sine/cosine pairs for
hour-of-day, day-of-year, minute-of-day, and quarter-of-hour. They are generated from
history and target timestamps only and never inspect target power or weather.

## Covariate semantics

| Dataset | Weather source | Registered semantics | Restriction |
|---|---|---|---|
| GEFCom | `VAR78,VAR79,VAR157,VAR164,VAR169,VAR178` | `future_nwp` | matches current Chronos protocol |
| SKIPPD | ERA5, 8 fields | `future_reanalysis` | oracle/reanalysis; not operational NWP |
| CSG | ERA5, 8 fields | `future_reanalysis` | oracle/reanalysis; not operational NWP |
| CSG native station weather | observed station columns | prohibited | no evidence that values are issued forecasts |

Power runs register `covariate_semantics=none`.

## CSG schema conclusion

Raw dataset files were not modified. The loader reads headerless rows against an
explicit per-site schema and validates the exact field count, timestamp position and
power position. Repository headers and the existing solar-energy loader establish
that sites 6–8 contain eight fields, including air temperature before pressure,
humidity and power. Therefore site6–8 schema is confirmed and the eighth field is not
dropped. Site8 additionally contains six non-15-minute gaps (five 1-day+15-minute,
one 3-day+15-minute), so it is excluded from the current preliminary protocol pending
an explicit irregular-window policy. The smoke uses confirmed, regular site1.

The first 32 CSG 15-minute timestamps have no causal ERA5 value after local→UTC
alignment; they remain missing and affected windows are skipped. No backward fill is
used.

## SKIPPD shared protocol

Index-only export command:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python scripts/export_solar_chronos_manifest.py \
  --dataset skippd --site-id skippd \
  --csv /home/zhaopp/workspace/solar-energy/dataset/skippd.csv \
  --weather-path /home/zhaopp/workspace/solar-energy/dataset/ERA5/skippd_stanford_all.csv \
  --output-dir outputs/solar_energy_protocol/skippd_chronos_strict_ctx2048_pred288 \
  --chronos-context 2048 --fusionsf-context 336 --pred-len 288
```

- selection logic: existing `run_chronos2_zero_shot_solar_v4.py::build_windows`
- strict test-only, last 30%, irregular windows skipped
- Chronos context: 2048; FusionSF context: 336
- shared origins/complete target intervals: 5,835
- paths: `outputs/solar_energy_protocol/skippd_chronos_strict_ctx2048_pred288/`
  (`t_origin.npy`, `sample_manifest.csv`, `meta.json`)

## Long-horizon smoke results

All runs use seed 42, batch size 2, one epoch, and at most two train/validation/test
batches. Only four test windows are scored, so metrics are pipeline checks, not
performance conclusions. Predictions are clipped using each dataset's established
power bounds before metrics.

| Dataset/config | Covariates | History | Future time | Weather | Prediction | TS embedding | Fusion embedding | MAE | RMSE |
|---|---|---|---|---|---|---|---|---:|---:|
| GEFCom zone1 336→72 Power | none | `[2,336,1]` | `[2,72,8]` | — | `[2,72,1]` | `[2,336,32]` | `[2,72,32]` | 0.181455 | 0.223054 |
| GEFCom zone1 336→72 Power+Weather | future_nwp `[2,72,6]` | `[2,336,1]` | `[2,72,8]` | `[2,72,6]` | `[2,72,1]` | `[2,336,32]` | `[2,72,32]` | 0.185560 | 0.225909 |
| SKIPPD 336→288 Power | none | `[2,336,1]` | `[2,288,8]` | — | `[2,288,1]` | `[2,336,32]` | `[2,288,32]` | 9.021803 | 10.162564 |
| SKIPPD 336→288 Power+Weather | future_reanalysis `[2,288,8]` | `[2,336,1]` | `[2,288,8]` | `[2,288,8]` | `[2,288,1]` | `[2,336,32]` | `[2,288,32]` | 8.515740 | 9.673270 |
| CSG site1 336→288 Power | none | `[2,336,1]` | `[2,288,8]` | — | `[2,288,1]` | `[2,336,32]` | `[2,288,32]` | 13.699261 | 16.817307 |
| CSG site1 336→288 Power+Weather | future_reanalysis `[2,288,8]` | `[2,336,1]` | `[2,288,8]` | `[2,288,8]` | `[2,288,1]` | `[2,336,32]` | `[2,288,32]` | 16.181086 | 19.203598 |

Every run produced finite predictions, completed backward propagation, aligned all
test-manifest origins and complete target intervals, and saved model/config/shape/
metric/prediction metadata under `outputs/solar_energy_smoke_v1/`.

## Verification and boundaries

- migration tests: 16 passed
- full suite: 40 passed, 6 dependency deprecation warnings
- formal 30/100-epoch training: **not started**
- Chronos-2 model inference/retraining: **not run**
- embedding extraction or FusionSF→Chronos-2 fusion: **not started**
- satellite migration and hyperparameter search: **not started**

Conclusion: `fusionsf_solar_v1` and the GEFCom/SKIPPD/CSG shared protocols pass the
Stage-2A smoke gate. The next action is review, not formal training.
