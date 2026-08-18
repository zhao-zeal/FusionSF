from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from vector_quantize_pytorch import ResidualVQ

from src.models.chronos_guided_vq import chronos_guided_residual_vq
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


def test_codebook_guided_vq_output_shape_and_lambda_zero_exactly_matches_vq():
    vq = ResidualVQ(dim=2, num_quantizers=1, codebook_size=2, kmeans_init=False).eval()
    tokens = torch.tensor([[[0.1, 0.0], [0.9, 0.0]]])
    prior = torch.tensor([[1.0, 0.0]])

    expected = vq(tokens)
    actual = chronos_guided_residual_vq(vq, tokens, prior, guidance_lambda=0.0)

    assert actual[0].shape == tokens.shape
    assert actual[1].shape == (1, 2, 1)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(actual_tensor, expected_tensor)


def test_codebook_guidance_can_change_selected_code():
    vq = ResidualVQ(dim=2, num_quantizers=1, codebook_size=2, kmeans_init=False).eval()
    codebook = vq.layers[0]._codebook
    codebook.embed.copy_(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    tokens = torch.tensor([[[0.0, 0.9]]])
    prior = torch.tensor([[1.0, 0.0]])

    _, baseline_indices, _ = vq(tokens)
    _, guided_indices, _ = chronos_guided_residual_vq(
        vq, tokens, prior, guidance_lambda=4.0
    )

    assert baseline_indices.item() == 0
    assert guided_indices.item() == 1


def test_codebook_guidance_trains_chronos_projection_input():
    vq = ResidualVQ(dim=2, num_quantizers=1, codebook_size=2, kmeans_init=False).eval()
    codebook = vq.layers[0]._codebook
    codebook.embed.copy_(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    tokens = torch.tensor([[[0.0, 0.9]]])
    prior = torch.tensor([[0.8, 0.2]], requires_grad=True)

    quantized, _, _ = chronos_guided_residual_vq(vq, tokens, prior, guidance_lambda=1.0)
    quantized[..., 0].sum().backward()

    assert prior.grad is not None
    assert torch.count_nonzero(prior.grad).item() > 0


def test_codebook_surrogate_does_not_add_ts_input_gradient():
    vq = ResidualVQ(dim=2, num_quantizers=1, codebook_size=2, kmeans_init=False).eval()
    codebook = vq.layers[0]._codebook
    codebook.embed.copy_(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    tokens = torch.tensor([[[0.0, 0.9]]], requires_grad=True)
    prior = torch.tensor([[0.8, 0.2]], requires_grad=True)

    quantized, _, _ = chronos_guided_residual_vq(vq, tokens, prior, guidance_lambda=1.0)
    token_grad, prior_grad = torch.autograd.grad(
        quantized[..., 0].sum(), (tokens, prior), allow_unused=True
    )

    assert token_grad is None
    assert prior_grad is not None
    assert torch.count_nonzero(prior_grad).item() > 0


def test_codebook_guidance_runs_full_shared_eight_quantizer_path():
    vq = ResidualVQ(
        dim=4, num_quantizers=8, codebook_size=8,
        shared_codebook=True, kmeans_init=False,
    ).eval()
    tokens = torch.randn(2, 3, 4)
    prior = torch.randn(2, 4, requires_grad=True)

    quantized, indices, losses = chronos_guided_residual_vq(
        vq, tokens, prior, guidance_lambda=0.5
    )
    quantized.sum().backward()

    assert quantized.shape == tokens.shape
    assert indices.shape == (2, 3, 8)
    assert losses.shape[-1] == 8
    assert prior.grad is not None


def test_codebook_guided_config_only_changes_ts_vq_selection():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=fusionsf_pipeline_v1_chronos_codebook_guided_ts_vq"],
        )
    model = cfg.pl_module.model
    assert model.vq_in_ts is True
    assert model.vq_in_ctx is True
    assert model.vq_in_guide is False
    assert model.temporal_prior_mode == "none"
    assert model.chronos_vq_guidance_mode == "codebook_guided"
    assert model.chronos_vq_guidance_lambda == 1.0


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
