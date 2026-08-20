from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from src.datasets.tscontext_3modal_dataset import Ts3MDataset


class CachedRepresentationDataset(Dataset):
    """Attach an immutable, row-aligned representation array to a dataset."""

    def __init__(self, dataset: Dataset, representation_path: Path):
        self.dataset = dataset
        self.representations = np.load(representation_path, mmap_mode="r")
        if self.representations.ndim != 2:
            raise ValueError("cached representations must have shape [N, D]")
        if len(self.representations) != len(dataset):
            raise ValueError(
                f"cache rows ({len(self.representations)}) do not match dataset rows ({len(dataset)})"
            )
        if not np.isfinite(self.representations).all():
            raise ValueError("cached representations contain NaN or Inf")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["chronos_representation"] = torch.from_numpy(
            np.array(self.representations[index], dtype=np.float32, copy=True)
        )
        return sample


class Ts3MDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2,
        representation_cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.test_batch_size = test_batch_size or batch_size
        self.representation_cache_dir = representation_cache_dir

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_all = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        train_ratio = self.train_ratio
        valid_ratio = self.valid_ratio
        test_ratio = self.test_ratio
        if not self.data_train and not self.data_val and not self.data_test:
            data_all = Ts3MDataset(
                **self.hparams.dataset,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
            )
            self.data_all = data_all
            if data_all.pipeline_version == "fixed_v1":
                split_ids = np.array([
                    {"train": 0, "validation": 1, "test": 2}[record.split]
                    for _ in data_all.data_sp for record in data_all.window_records
                ])
                self.data_train = Subset(data_all, np.flatnonzero(split_ids == 0))
                self.data_val = Subset(data_all, np.flatnonzero(split_ids == 1))
                self.data_test = Subset(data_all, np.flatnonzero(split_ids == 2))
                if self.hparams.dataset.get('dataset_test'):
                    test_config = OmegaConf.to_container(
                        self.hparams.dataset.dataset_test, resolve=True
                    )
                    test_config.setdefault(
                        "data_pipeline",
                        OmegaConf.to_container(self.hparams.dataset.data_pipeline, resolve=True),
                    )
                    test_config.setdefault("modality_mode", self.hparams.dataset.modality_mode)
                    data_all_test = Ts3MDataset(
                        **test_config,
                        train_ratio=train_ratio,
                        valid_ratio=valid_ratio,
                        test_ratio=test_ratio,
                        precomputed_scaler_state=data_all.scaler_state,
                    )
                    train_site_ids = {int(site["site"]) for site in data_all.data_sp}
                    test_site_ids = {int(site["site"]) for site in data_all_test.data_sp}
                    overlap = train_site_ids.intersection(test_site_ids)
                    if overlap:
                        raise ValueError(f"fixed_v1 cross-site datasets overlap at sites: {sorted(overlap)}")
                    test_split_ids = np.array([
                        {"train": 0, "validation": 1, "test": 2}[record.split]
                        for _ in data_all_test.data_sp for record in data_all_test.window_records
                    ])
                    self.data_test_all = data_all_test
                    self.data_test = Subset(data_all_test, np.flatnonzero(test_split_ids == 2))
                self._attach_representation_cache()
                return
            data_len = len(data_all)
            all_indices = np.arange(0, int(data_len))
            all_indices = all_indices.reshape([data_all.num_sites - data_all.num_ignored_sites, -1])
            N, L = all_indices.shape
            train_indices = all_indices[:, :int(L * train_ratio)].reshape(-1)
            valid_indices = all_indices[:, int(L * train_ratio): int(L * (train_ratio + valid_ratio))].reshape(-1)
            test_indices = all_indices[:, -int(L * test_ratio):].reshape(-1)
            self.data_train = Subset(data_all, train_indices)
            self.data_val = Subset(data_all, valid_indices)
            self.data_test = Subset(data_all, test_indices)
            if self.hparams.dataset.get('dataset_test'):
                print('use a different test dataset')
                data_all_test = Ts3MDataset(**self.hparams.dataset.dataset_test)
                all_indices_test = np.arange(0, int(len(data_all_test)))
                all_indices_test = all_indices_test.reshape([data_all_test.num_sites - data_all_test.num_ignored_sites, -1])
                N, L = all_indices_test.shape
                test_indices = all_indices_test[:, -int(L * test_ratio):].reshape(-1)
                self.data_test = Subset(data_all_test, test_indices)

    def _attach_representation_cache(self):
        if not self.representation_cache_dir:
            return
        cache_dir = Path(self.representation_cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"representation cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("pooling") != "mean" or manifest.get("final_shape", [None])[-1] != 768:
            raise ValueError("representation cache must contain mean-pooled 768-d Chronos-2 embeddings")
        for split, attribute in (
            ("train", "data_train"), ("validation", "data_val"), ("test", "data_test")
        ):
            dataset = getattr(self, attribute)
            expected = manifest.get("splits", {}).get(split, {}).get("rows")
            if expected != len(dataset):
                raise ValueError(
                    f"manifest {split} rows ({expected}) do not match dataset rows ({len(dataset)})"
                )
            setattr(
                self,
                attribute,
                CachedRepresentationDataset(dataset, cache_dir / f"{split}_mean.npy"),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )
