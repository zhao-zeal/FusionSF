#!/usr/bin/env python3
"""Export the existing Chronos-v4 strict-window indices without model inference."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fusionsf_solar_smoke import make_spec
from src.datasets.solar_energy_dataset import build_sample_manifest, load_solar_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("skippd",), required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--weather-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chronos-context", type=int, default=2048)
    parser.add_argument("--fusionsf-context", type=int, default=336)
    parser.add_argument("--pred-len", type=int, required=True)
    args = parser.parse_args()
    spec = make_spec(args)
    frame = load_solar_frame(spec)
    n, test_start = len(frame), len(frame) - int(math.ceil(len(frame) * 0.3))
    last_start = n - args.chronos_context - args.pred_len
    starts = np.arange(test_start, last_start + 1, dtype=np.int64)
    timestamps = frame[spec.timestamp_col].to_numpy(dtype="datetime64[ns]")
    delta = np.timedelta64(pd.Timedelta(spec.frequency).value, "ns")
    starts = np.asarray([
        start for start in starts
        if np.all(np.diff(timestamps[start : start + args.chronos_context + args.pred_len]) == delta)
    ], dtype=np.int64)
    origins = timestamps[starts + args.chronos_context - 1]
    manifest = build_sample_manifest(frame, spec, args.fusionsf_context, args.pred_len, origins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "t_origin.npy", origins)
    manifest.drop(columns="start_index").to_csv(args.output_dir / "sample_manifest.csv", index=False)
    metadata = {"selection_code": "run_chronos2_zero_shot_solar_v4.py::build_windows", "strict_test_only": 1, "test_ratio": 0.3, "chronos_context_length": args.chronos_context, "fusionsf_context_length": args.fusionsf_context, "pred_len": args.pred_len, "origins": len(origins), "frequency": spec.frequency}
    (args.output_dir / "meta.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
