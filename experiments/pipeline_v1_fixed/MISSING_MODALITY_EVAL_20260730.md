# Three-site checkpoint sensitivity evaluation

Source experiment: `20260730_211537_fusionsf3m_sites3_seq24_pred24_seed42`

Checkpoint: `outputs/pipeline_v1_fixed/20260730_211537_fusionsf3m_sites3_seq24_pred24_seed42/checkpoints/epoch_epoch=002.ckpt`

Evaluation output: `outputs/pipeline_v1_fixed/20260730_211537_fusionsf3m_sites3_seq24_pred24_seed42/evaluation_20260730_212957/`

## Results

| Mode | Raw MAE | Clipped MAE | Raw RMSE | MAE change vs full | Prediction mean/std |
|---|---:|---:|---:|---:|---:|
| full modalities | 0.044213 | 0.044184 | 0.088583 | — | 0.177985 / 0.238372 |
| missing NWP | 0.099437 | 0.099437 | 0.168990 | +124.90% | 0.115363 / 0.126047 |
| missing satellite | 0.157259 | 0.157259 | 0.258575 | +255.68% | 0.041868 / 0.033155 |
| missing satellite and NWP | 0.178852 | 0.178852 | 0.285807 | +304.52% | 0.037171 / 0.006969 |

The full-mode raw prediction has 2.40% negative values. Clipping changes MAE from 0.044213 to
0.044184 and has negligible RMSE impact, so both raw and physically clipped metrics are retained.

## Interpretation boundary

This is a checkpoint sensitivity/stress test, not a trained missing-modality robustness result and
not a modality ablation. The source checkpoint was trained with all modalities available; the
whole-modality missing tokens were therefore not optimized through modality dropout. The large
degradation shows dependence on satellite and NWP inputs, but it does not measure how well a model
trained for missing inputs would recover. Formal ablation requires separately trained modality
configurations, while robustness requires modality-dropout training.
