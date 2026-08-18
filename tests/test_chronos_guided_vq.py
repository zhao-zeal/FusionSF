from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.fusionSF_3modal import FusionSF3M
from src.models.modules.positional_encoding import Cyclical_embedding


ROOT = Path(__file__).resolve().parents[1]


def model_kwargs():
    return dict(
        image_size=[2, 2], patch_size=[1, 1],
        time_coords_encoder=Cyclical_embedding([12, 31, 24]), dim=8, depth=1,
        heads=1, mlp_ratio=1, ctx_channels=1, ts_channels=1, guide_channels=17,
        ts_length=4, dim_head=8, decoder_dim=8, decoder_depth=1,
        decoder_heads=1, decoder_dim_head=8, dropout=0, num_mlp_heads=1,
        ctx_masking_ratio=0, ts_masking_ratio=0, vq_in_ctx=False,
        vq_in_guide=False, modality_mode="all", masking_policy="fixed_ratio",
        output_activation="identity", max_freq=16,
    )


def test_guided_vq_requires_ts_vq_and_is_mutually_exclusive_with_post_prior():
    with pytest.raises(ValueError, match="requires vq_in_ts"):
        FusionSF3M(**model_kwargs(), vq_in_ts=False, chronos_vq_guidance_mode="pre_ts_vq")
    with pytest.raises(ValueError, match="mutually exclusive"):
        FusionSF3M(
            **model_kwargs(), vq_in_ts=True, temporal_prior_mode="chronos",
            chronos_vq_guidance_mode="pre_ts_vq",
        )


def test_task4_config_moves_chronos_to_pre_ts_vq_only():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=fusionsf_pipeline_v1_chronos_guided_ts_vq"],
        )
    model = cfg.pl_module.model
    assert model.vq_in_ts is True
    assert model.vq_in_ctx is True
    assert model.vq_in_guide is False
    assert model.temporal_prior_mode == "none"
    assert model.chronos_vq_guidance_mode == "pre_ts_vq"
    assert model.chronos_vq_guidance_scale == 1.0
    assert cfg.seed == 42
