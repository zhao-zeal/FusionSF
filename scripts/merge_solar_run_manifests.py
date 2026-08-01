#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def merge_run_manifests(run_dirs: list[Path], registry: Path):
    rows = [json.loads((directory / "run_manifest.json").read_text()) for directory in run_dirs]
    frame = pd.DataFrame(rows)
    for (_, dataset), pair in frame.groupby(["site_id", "dataset"]):
        if set(pair["mode"]) == {"power", "power_weather"} and pair["test_manifest_sha256"].nunique() != 1:
            raise ValueError(f"Power/Power+Weather test manifests differ for {dataset}")
    existing = pd.read_csv(registry) if registry.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    if "experiment_id" in combined and combined.experiment_id.duplicated().any():
        duplicates = combined.loc[combined.experiment_id.duplicated(), "experiment_id"].tolist()
        raise ValueError(f"Duplicate experiment ids: {duplicates}")
    combined.to_csv(registry, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_run_manifests(args.run_dir, args.registry)
    print(f"registry rows={len(merged)} path={args.registry}")


if __name__ == "__main__": main()
