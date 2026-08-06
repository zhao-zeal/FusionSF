from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]


def test_preliminary_seed42_configs_share_one_budget_and_correct_modalities():
    expected = {
        "fusionsf_pipeline_v1_preliminary_power": ("power", 10, 0, None),
        "fusionsf_pipeline_v1_preliminary_power_nwp": ("power_nwp", 10, 0, None),
        "fusionsf_pipeline_v1_preliminary_full": ("all", 10, 0, None),
        "fusionsf_pipeline_v1_preliminary_zeroshot": ("all", 20, 10, 10),
    }
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
        for experiment, (mode, num_sites, ignored, test_sites) in expected.items():
            cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
            assert cfg.seed == 42
            assert cfg.experiment_stage == "preliminary_seed42"
            assert cfg.trainer.max_epochs == 30
            assert cfg.trainer.deterministic is True
            assert cfg.callbacks.early_stopping.monitor == "val/mae"
            assert cfg.callbacks.early_stopping.patience == 10
            assert cfg.trainer.get("limit_train_batches", 1.0) == 1.0
            assert cfg.datamodule.dataset.modality_mode == mode
            assert cfg.pl_module.model.modality_mode == mode
            assert cfg.pl_module.model.masking_policy == "fixed_ratio"
            assert cfg.pl_module.model.ctx_masking_ratio == 0
            assert cfg.pl_module.model.ts_masking_ratio == 0
            assert cfg.pl_module.model.satellite_modality_dropout == 0
            assert cfg.pl_module.model.nwp_modality_dropout == 0
            assert cfg.pl_module.model.vq_in_ts is False
            assert cfg.pl_module.model.vq_in_ctx is False
            assert cfg.pl_module.model.vq_in_guide is False
            assert cfg.pl_module.model.output_activation == "identity"
            assert cfg.datamodule.dataset.num_sites == num_sites
            assert cfg.datamodule.dataset.num_ignored_sites == ignored
            assert cfg.datamodule.dataset.data_pipeline.scaler_version == "train_sites_and_time_only_fit_v1"
            if test_sites is None:
                assert cfg.datamodule.dataset.get("dataset_test") is None
            else:
                assert cfg.datamodule.dataset.dataset_test.num_sites == test_sites
                assert cfg.datamodule.dataset.dataset_test.num_ignored_sites == 0
