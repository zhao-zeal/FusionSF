# pipeline_v1_fixed

New leakage-safe experiments belong here or under the unique runtime directories in
`outputs/pipeline_v1_fixed/`. Do not place legacy artifacts in this directory.

No full experiment has been run yet. Validation remains intentionally limited to 2–3 sites and
1–3 epochs. The completed 3-site/3-epoch integration run is registered as
`20260730_211537_fusionsf3m_sites3_seq24_pred24_seed42` with status `valid_or_debug`.

## Guarantees and current scope

- Windows are assigned from complete forecast-target ranges. Input history may precede a split,
  while target timestamps are asserted disjoint.
- Satellite and NWP scalers fit only timestamps before the training boundary; parameters and fit
  ranges are saved with the run.
- fixed_v1 NWP interpolation is isolated by `(lat, lon)` and coordinates are not weather channels.
  MMSP therefore uses 15 guide features instead of the legacy 17 columns.
- Prediction is direct 24-step output, not recursive. Overall, daylight, and horizon metrics are
  saved. nMAE/nRMSE remain empty until authoritative station capacities are supplied.
- Negative power is preserved because no authoritative clipping policy is available. fixed_v1
  raises on missing power instead of silently converting missing values to zero.
- fixed_v1 cross-site `dataset_test` is deliberately blocked until the training-site scaler can be
  injected into the unseen-site dataset, preventing accidental fitting on test sites.

## Validation and training

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python -m pytest -q tests
/home/zhaopp/miniconda3/envs/FusionSF/bin/python scripts/validate_pipeline_v1.py
```

One-epoch smoke run:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python main.py experiment=fusionsf_pipeline_v1_smoke
```

Reviewed 3-site/3-epoch run:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python main.py \
  experiment=fusionsf_pipeline_v1_fixed \
  datamodule.dataset.num_sites=3 trainer.max_epochs=3
```

## Embedding extraction

Extraction requires an explicit best-validation checkpoint and resolved fixed_v1 config and never
invokes training:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python scripts/extract_embeddings.py \
  --config outputs/pipeline_v1_fixed/<experiment_id>/config_resolved.yaml \
  --checkpoint-path outputs/pipeline_v1_fixed/<experiment_id>/checkpoints/<best>.ckpt \
  --output-dir outputs/pipeline_v1_fixed/<experiment_id>/embedding_export_<timestamp> \
  --splits train validation test --embedding-type both --pooling none mean \
  --site-seen-during-training
```

Each split contains raw and pooled arrays plus row-aligned metadata. Chronos covariates use
`site_id + forecast_start_timestamp` as the join key. Cross-site exports must omit
`--site-seen-during-training`.

## Checkpoint-only evaluation

Evaluate raw and non-negative-clipped predictions without training:

```bash
/home/zhaopp/miniconda3/envs/FusionSF/bin/python scripts/evaluate_checkpoint.py \
  --config outputs/pipeline_v1_fixed/<experiment_id>/config_resolved.yaml \
  --checkpoint-path outputs/pipeline_v1_fixed/<experiment_id>/checkpoints/<best>.ckpt \
  --output-dir outputs/pipeline_v1_fixed/<experiment_id>/evaluation_<timestamp> \
  --modes full_modalities missing_satellite missing_nwp missing_satellite_and_nwp
```

A missing-mode result is only a sensitivity diagnostic unless the source model was explicitly
trained with whole-modality dropout. Do not label it as robustness or retrained-model ablation.
