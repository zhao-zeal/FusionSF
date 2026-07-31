#!/usr/bin/env python3
"""Fast real-data validation for fixed_v1 without fitting or writing experiment outputs."""

import json
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    from pip._vendor import pkg_resources
    sys.modules["pkg_resources"] = pkg_resources

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.metrics.forecast_metrics import compute_forecast_metrics
from src.datasets.tscontext_3modal_dataset import Ts3MDataset


def main():
    config_dir = str(ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=fusionsf_pipeline_v1_smoke"])
        cross_site_cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=fusionsf_pipeline_v1_zeroshot_smoke",
                "datamodule.dataset.num_sites=3",
                "datamodule.dataset.num_ignored_sites=1",
                "datamodule.dataset.dataset_test.num_sites=1",
            ],
        )
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    split_sizes = {
        "train": len(datamodule.data_train),
        "validation": len(datamodule.data_val),
        "test": len(datamodule.data_test),
    }
    if not all(split_sizes.values()):
        raise AssertionError(f"empty fixed_v1 split: {split_sizes}")
    batch = next(iter(datamodule.test_dataloader()))
    module = hydra.utils.instantiate(cfg.pl_module)
    model = module.model.eval()
    predictions = {}
    with torch.no_grad():
        for mode in ("full_modalities", "missing_satellite", "missing_nwp"):
            output = model(
                batch["stl_input"].float(), batch["stl_coords"].float(),
                batch["ts_input"].float(), batch["ts_coords"].float(),
                batch["ts_time"].float(), batch["ec_input"].float(), mask=False,
                modality_availability=batch["modality_availability"], evaluation_mode=mode,
            ).mean(dim=2)
            predictions[mode] = output.numpy()
        embeddings = model.extract_embeddings(batch, "both", "mean")
    target = batch["ts_target"].unsqueeze(-1).numpy()
    if predictions["full_modalities"].shape != target.shape:
        raise AssertionError("prediction/target shape mismatch")
    legacy_dataset = Ts3MDataset(
        str(ROOT / "data/MMSP/data"), seq_len=24, pred_len=24, num_sites=2,
        modality_mode="power", data_pipeline={"version": "legacy_v0"},
    )
    legacy_sample = legacy_dataset[0]
    cross_site_datamodule = hydra.utils.instantiate(cross_site_cfg.datamodule)
    cross_site_datamodule.setup()
    train_sites = {int(site["site"]) for site in cross_site_datamodule.data_all.data_sp}
    test_sites = {int(site["site"]) for site in cross_site_datamodule.data_test_all.data_sp}
    if not train_sites.isdisjoint(test_sites):
        raise AssertionError(f"cross-site validation has overlapping sites: {train_sites & test_sites}")
    cross_site_batch = next(iter(cross_site_datamodule.test_dataloader()))
    injected_state = cross_site_datamodule.data_test_all.scaler_state
    if injected_state.get("source") != "external_training_dataset":
        raise AssertionError("cross-site dataset did not mark its scaler as externally injected")
    for modality in ("satellite", "nwp"):
        if injected_state[modality].get("source") != "external_training_dataset":
            raise AssertionError(f"cross-site {modality} scaler was not injected")
    summary = {
        "split_sizes": split_sizes,
        "scaler_fit_end_exclusive": datamodule.data_all.scaler_state["fit_end_exclusive"],
        "prediction_shape": list(predictions["full_modalities"].shape),
        "ts_embedding_shape": list(embeddings["ts"].shape),
        "fusion_embedding_shape": list(embeddings["fusion"].shape),
        "finite": all(np.isfinite(value).all() for value in predictions.values()),
        "debug_metrics": compute_forecast_metrics(predictions["full_modalities"], target),
        "legacy_pipeline_selectable": (
            legacy_dataset.pipeline_version == "legacy_v0" and legacy_sample["ts_input"].shape == (24, 1)
        ),
        "cross_site": {
            "train_sites": sorted(train_sites),
            "test_sites": sorted(test_sites),
            "test_windows": len(cross_site_datamodule.data_test),
            "batch_ts_shape": list(cross_site_batch["ts_input"].shape),
            "batch_nwp_shape": list(cross_site_batch["ec_input"].shape),
            "scaler_source": injected_state["source"],
            "nwp_fit_coordinates": cross_site_datamodule.data_all.scaler_state["nwp"]["fit_coordinates"],
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
