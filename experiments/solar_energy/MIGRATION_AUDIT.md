# FusionSF → solar-energy migration audit

Audit date: 2026-08-01. Evidence comes from the current files under
`/home/zhaopp/workspace/solar-energy`; no dataset file was modified. Values below are
measured from CSVs, not inferred from MMSP.

## Protocol findings

- The three dataset families are **SKIPPD**, **StateGrid/CSG Solar**, and
  **GEFCom 2014 Solar**.
- Existing classical scripts use `seq_len=336`. Their horizons are `1/16/288`
  for 15-minute SKIPPD/CSG and `1/4/72` for hourly GEFCom.
- Existing `run_chronos2_zero_shot_solar_v4.py` runs a different context protocol:
  `seq_len=2048`, `test_ratio=0.3`, `strict_test_only=1`. A window is retained only
  when its entire history starts inside the last 30%, and irregular windows are
  skipped. Thus “Chronos-2” and “seq_len=336 baseline” were not previously the same
  sample set.
- For the migration smoke, FusionSF uses `seq_len=336` but filters its test
  forecast origins to the saved Chronos-2 `t_origin.npy`. This preserves the required
  forecast start/target timestamps while using the requested FusionSF input length.
  The exported `sample_manifest.csv` is the authority for subsequent comparisons.
- Existing verified hybrid code uses chronological 60%/10%/30% boundaries and
  fits scaling on the first 60% only. Chronos-2 only defines a 30% test boundary;
  this migration adopts the verified 60/10/30 train/validation/test boundaries and
  the Chronos saved test origins.
- Existing Chronos artifacts are `y_pred.npy`, `y_true.npy`, `t_origin.npy`, and
  `meta.json`. `t_origin` is the last history timestamp; forecast start is one data
  interval later. The verified supervised pipeline also emits JSON/CSV metrics and
  prediction tables. Core metrics are MAE, RMSE, capacity-normalized MAE/RMSE
  accuracy, and day/night MAE.

## Dataset inventory

### SKIPPD

- Data file: `solar-energy/dataset/skippd.csv`; ERA5:
  `solar-energy/dataset/ERA5/skippd_stanford_all.csv`.
- Time range/sites: 2017-01-01 00:00 through 2017-12-31 23:45; one site; 35,040 rows.
- Time grain: 15 minutes (measured); ERA5 is hourly and requires causal alignment to
  the 15-minute grid.
- Input/horizons: `seq_len=336`; `pred_len=1/16/288` (15 min, 4 h, 72 h).
- Power field: `OT`; no capacity is encoded in the CSV. Verified code associates
  SKIPPD with 30 MW, which must remain an explicit configuration, not be guessed by a loader.
- NWP: the power CSV has none. ERA5 fields are `temperature_2m`, `dew_point_2m`,
  `surface_pressure`, `shortwave_radiation`, `cloud_cover`, `wind_speed_10m`,
  `precipitation`, `relative_humidity_2m`.
- Missing values: zero in both current CSVs. DST localization is explicitly handled
  by verified code (`America/Los_Angeles`, ambiguous fall-back treated as standard time).
- Existing split: verified 60/10/30 chronological; older Chronos shell protocol for
  SKIPPD was not found in the checked root scripts, so no saved Chronos test-index
  artifact was verified for this audit.
- Chronos test indices: **missing/unverified**; must be exported before formal runs.

### StateGrid / CSG Solar

- Data files: eight `solar-energy/dataset/csg_solar/Solar_station_site_*.csv` files;
  matching hourly ERA5 files exist for all eight sites.
- Time range/sites: eight sites. Sites 1,2,4,5 cover 2019-01-01 through 2020-12-31
  (70,176 rows); site3 ends 2019-07-31 (20,352 rows). The current edited headers for
  sites 6–8 have seven names for eight values, so pandas mis-parses their timestamps;
  their ranges are not considered verified until that external dataset issue is fixed.
- Time grain: 15 minutes in valid files; ERA5 hourly.
- Input/horizons: `seq_len=336`; `pred_len=1/16/288` (15 min, 4 h, 72 h).
- Power field: current renamed header `Power`; capacity is encoded in filenames:
  sites 1–8 = 50/130/30/130/110/35/30/30 MW.
- NWP: native site weather includes irradiance (total/direct/global), pressure,
  optional temperature and humidity; matching ERA5 has the eight fields listed for
  SKIPPD. Which source is “forecast-available NWP” needs protocol confirmation before
  formal Power+NWP runs; observed future weather must not silently stand in for a forecast.
- Missing values: current valid CSVs and ERA5 files report zero cells missing; negative
  power exists and the verified loader clips it to zero. Causal fill limit is four steps.
- Existing split: verified 60/10/30 chronological. Chronos-v4 uses last 30%, entire
  context inside test, 15-minute regularity checks.
- Chronos test indices: saved per site/horizon under
  `solar-energy/results/solar_chronos2_csg/*/t_origin.npy`; context is 2048, not 336.

### GEFCom 2014 Solar

- Data files: `solar-energy/dataset/GEFCom/gefcom15_by_zone/zone1.csv` through
  `zone3.csv`.
- Time range/sites: 2012-04-01 01:00 through 2014-06-01 00:00; three zones;
  18,984 rows per zone.
- Time grain: 1 hour (measured).
- Input/horizons: `seq_len=336`; `pred_len=1/4/72` (1 h, 4 h, 72 h).
- Power field: `POWER`, already normalized in the competition-derived files; stage-two
  smoke clips to `[0, 1]` and does not apply an invented MW capacity.
- NWP fields: `VAR78`, `VAR79`, `VAR134`, `VAR157`, `VAR164`, `VAR165`, `VAR166`,
  `VAR167`, `VAR169`, `VAR175`, `VAR178`, `VAR228`. Existing Chronos-v4 Power+NWP
  selects `VAR78,VAR79,VAR157,VAR164,VAR169,VAR178`; migration smoke uses exactly these.
- Missing values: zero in the three materialized zone CSVs; no duplicate timestamps.
- Existing split: verified 60/10/30 chronological; Chronos-v4 last 30% with entire
  2048-step context inside test and regular-window filtering.
- Chronos test indices: saved `t_origin.npy` per zone/horizon under
  `solar-energy/results/solar_chronos2_gefcom/`. For zone1/pred1 there are 3,648
  origins, from 2013-12-31 00:00 through 2014-05-31 23:00; forecast timestamps are
  one hour later. This is the smoke reference.

## Shared manifest and leakage rules

`sample_manifest.csv` contains dataset, site, split, input start/end, forecast
start/end, `seq_len`, and `pred_len`. An internal `start_index` is used only while
constructing tensors and is omitted from the published manifest. Windows never cross
partition boundaries; input end precedes forecast start; scaler statistics are fit
only on rows before the 60% train boundary; no backward fill, interpolation, repeated
horizon, or random split is used. Power mode never indexes NWP. Power+NWP indexes
exactly `[forecast_start, forecast_end]`.

## Model status correction and formal contract

The first Stage-2 smoke used a GRU prototype, now explicitly named
`PrototypeHorizonGRU` with `model_family=prototype_horizon_gru`. It is not a FusionSF
result. The formal `FusionSFSolar` (`fusionsf_solar_v1`) contains no GRU and directly
reuses the original FusionSF `Transformer` for history tokens and `CrossTransformer`
for horizon-to-history attention. Its TS embedding is `[B,seq_len,D]`; deterministic
future-time coordinates condition learned horizon queries; weather is encoded per
target offset; and the fusion embedding is `[B,pred_len,D]`. No Stage-3 embedding
extraction or Chronos-2 fusion has been run.
