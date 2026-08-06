# Stage 4B FusionSF–Chronos-2 correlation adapter

This directory owns the migrated Stage 4B implementation. The method is
CoRA-inspired; it is not a complete or vendored CoRA model.

## Architecture

Both FusionSF and Chronos-2 are frozen. Unpooled FusionSF tokens with shape
`[B,24,64]` are projected to the Chronos hidden dimension and used as
cross-attention keys/values. Chronos encoder tokens are the queries. A second
branch mean-pools the projected FusionSF tokens and applies a two-layer global
MLP. The adapted hidden state is

`H_new = H + alpha * dynamic_attention(H, Z) + beta * global_mlp(mean(Z))`.

`alpha` and `beta` are initialized to zero, so the initial model is exactly the
frozen Chronos-2 baseline.

## Required cache contract

The runner consumes an audited Stage 4A cache rather than silently loading a
solar-energy path. For each of `train`, `validation`, and `test`, the cache must
contain:

- `<split>_window_manifest.csv`
- `<split>_contexts.npy` with shape `[N,24]`
- `<split>_targets.npy` with shape `[N,24]`
- `<split>_fusion_tokens.npy` with shape `[N,24,64]`

The default protocol enforces train sites 10–19, validation sites 20–21, and
unseen test sites 0–9. The three site sets must be disjoint.

## Run

Use the existing archived cache only for exact replication:

```bash
bash scripts/run_night05_stage4b_correlation_adapter.sh \
  2 \
  /home/zhaopp/workspace/solar-energy/results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021 \
  2021
```

The shell script runs cache preflight first and refuses to overwrite an output
directory. It uses the `torch` Conda environment and the local Chronos-2 model
snapshot. The formal training budget is 100 maximum epochs with early-stopping
patience 20. It must not be run concurrently with other GPU experiments.
