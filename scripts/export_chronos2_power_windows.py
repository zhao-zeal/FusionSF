#!/usr/bin/env python3
"""Export target-free, row-aligned MMSP history windows for Chronos-2 embedding."""

import argparse
import hashlib
import json
import os
import site
import sys
from pathlib import Path

# Match the pinned training entrypoint: do not let a broken user-site package
# shadow this environment's dependencies.
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

# Lightning 1.8 still imports pkg_resources, which setuptools 81 removed.
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    from pip._vendor import pkg_resources

    sys.modules["pkg_resources"] = pkg_resources

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from torch.utils.data import Subset


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def export_split(dataset, split: str, output_dir: Path, batch_size: int) -> dict:
    del batch_size  # Kept in the CLI for backward-compatible invocation records.
    if not isinstance(dataset, Subset):
        raise TypeError("fixed-v1 split must be represented by torch.utils.data.Subset")
    base = dataset.dataset
    windows_per_site = len(base.window_records)
    arrays = {
        "power": np.empty((len(dataset), base.seq_len, 1), dtype=np.float32),
        "site_ids": np.empty(len(dataset), dtype=np.int64),
        "input_start": np.empty(len(dataset), dtype=np.int64),
        "input_end": np.empty(len(dataset), dtype=np.int64),
    }
    # Read only the power tensor and window metadata directly. Calling __getitem__
    # would unnecessarily materialize 64x64 satellite frames for every row.
    for row, global_index in enumerate(dataset.indices):
        site_index = int(global_index) // windows_per_site
        record = base.window_records[int(global_index) % windows_per_site]
        start, end = record.start_index, record.start_index + base.seq_len
        arrays["power"][row, :, 0] = base.data_sp[site_index]["values"][start:end].numpy()
        arrays["site_ids"][row] = int(base.data_sp[site_index]["site"])
        arrays["input_start"][row] = int(base.data_sp_time_dt.iloc[start].value)
        arrays["input_end"][row] = int(base.data_sp_time_dt.iloc[end - 1].value)
    paths = {}
    for name, value in arrays.items():
        path = output_dir / f"{split}_{name}.npy"
        np.save(path, value)
        paths[name] = path
    power = np.load(paths["power"], mmap_mode="r")
    if power.shape[1:] != (24, 1) or not np.isfinite(power).all():
        raise ValueError(f"unexpected/non-finite {split} power array: {power.shape}")
    return {
        "rows": int(len(power)),
        "power_shape": list(power.shape),
        "site_ids": [int(x) for x in np.unique(np.load(paths["site_ids"]))],
        "files": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=fusionsf_pipeline_v1_legacy_matched_zeroshot_vq_paper_best",
                "datamodule.num_workers=0",
            ],
        )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    splits = {
        "train": export_split(datamodule.data_train, "train", output_dir, args.batch_size),
        "validation": export_split(datamodule.data_val, "validation", output_dir, args.batch_size),
        "test": export_split(datamodule.data_test, "test", output_dir, args.batch_size),
    }
    manifest = {
        "protocol": "fixed_fusionsf_chronos2_history_power_v1",
        "dataset": "MMSP",
        "train_sites": "10-19",
        "validation_sites": "10-19",
        "test_sites": "0-9",
        "seq_len": 24,
        "pred_len": 24,
        "seed": 42,
        "contains_future_target": False,
        "split_version": "chronological_target_split_v1",
        "scaler_version": "train_sites_and_time_only_fit_v1",
        "splits": splits,
    }
    expected_sites = {
        "train": list(range(10, 20)),
        "validation": list(range(10, 20)),
        "test": list(range(10)),
    }
    for split, expected in expected_sites.items():
        if splits[split]["site_ids"] != expected:
            raise ValueError(
                f"unexpected {split} sites: {splits[split]['site_ids']} != {expected}"
            )
    (output_dir / "window_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
