import torch

from src.models.fusionSF_3modal import FusionSF3M
from src.models.modules.positional_encoding import Cyclical_embedding


def tiny_model():
    return FusionSF3M(
        image_size=[2, 2], patch_size=[1, 1], time_coords_encoder=Cyclical_embedding([12, 31, 24]),
        dim=8, depth=1, heads=1, mlp_ratio=1, ctx_channels=1, ts_channels=1,
        guide_channels=17, ts_length=4, dim_head=8, decoder_dim=8, decoder_depth=1,
        decoder_heads=1, decoder_dim_head=8, dropout=0, num_mlp_heads=1,
        ctx_masking_ratio=0, ts_masking_ratio=0, vq_in_ts=False, vq_in_ctx=False,
        vq_in_guide=False, modality_mode="all", masking_policy="fixed_ratio",
        output_activation="identity", max_freq=16,
    ).eval()


def batch():
    return {
        "stl_input": torch.randn(2, 4, 1, 2, 2), "stl_coords": torch.randn(2, 2, 2, 2),
        "ts_input": torch.randn(2, 4, 1), "ts_coords": torch.randn(2, 2, 1, 1),
        "ts_time": torch.randint(1, 12, (2, 4, 3, 2, 2)).float(),
        "ec_input": torch.randn(2, 4, 17), "modality_availability": torch.ones(2, 2),
    }


def test_ts_embedding_is_independent_of_missing_context_but_fusion_changes():
    torch.manual_seed(42)
    model, data = tiny_model(), batch()
    full = model.extract_embeddings(data, "both", "none", "full_modalities")
    missing = model.extract_embeddings(data, "both", "none", "missing_satellite_and_nwp")
    assert torch.equal(full["ts"], missing["ts"])
    assert not torch.equal(full["fusion"], missing["fusion"])


def test_modality_availability_distinguishes_missing_from_real_zero():
    model, data = tiny_model(), batch()
    data["stl_input"].zero_(); data["ec_input"].zero_()
    available = model.extract_embeddings(data, "fusion", "mean", "full_modalities")["fusion"]
    missing = model.extract_embeddings(data, "fusion", "mean", "missing_satellite_and_nwp")["fusion"]
    assert not torch.equal(available, missing)


def test_ts_masking_ratio_changes_ts_encoder_input_when_masking_is_enabled():
    model, data = tiny_model(), batch()
    model.ts_masking_ratio = 0.99
    unmasked = model.extract_embeddings(data, "ts", "none")["ts"]
    torch.manual_seed(42)
    _, masked = model(
        data["stl_input"], data["stl_coords"], data["ts_input"], data["ts_coords"],
        data["ts_time"], data["ec_input"], mask=True,
        modality_availability=data["modality_availability"], return_embeddings=True,
    )
    assert not torch.equal(unmasked, masked["ts"])


def test_fixed_ctx_masking_uses_configured_ratio_not_random_range():
    model, data = tiny_model(), batch()
    model.ctx_masking_ratio = 0.5
    observed = []
    original = model.random_masking

    def capture(x, ratio):
        observed.append(ratio)
        return original(x, ratio)

    model.random_masking = capture
    model(
        data["stl_input"], data["stl_coords"], data["ts_input"], data["ts_coords"],
        data["ts_time"], data["ec_input"], mask=True,
        modality_availability=data["modality_availability"],
    )
    assert observed == [0.5]


def test_training_modality_dropout_gives_missing_tokens_gradients_and_eval_disables_it():
    model, data = tiny_model(), batch()
    model.satellite_modality_dropout = 1.0
    model.nwp_modality_dropout = 1.0
    model.train()
    output, _ = model(
        data["stl_input"], data["stl_coords"], data["ts_input"], data["ts_coords"],
        data["ts_time"], data["ec_input"], mask=True,
        modality_availability=data["modality_availability"],
    )
    output.sum().backward()
    assert model.missing_ctx_token.grad is not None
    assert model.missing_guide_token.grad is not None
    assert model.missing_ctx_token.grad.abs().sum() > 0
    assert model.missing_guide_token.grad.abs().sum() > 0

    model.eval()
    first = model.extract_embeddings(data, "both", "mean")
    second = model.extract_embeddings(data, "both", "mean")
    assert torch.equal(first["ts"], second["ts"])
    assert torch.equal(first["fusion"], second["fusion"])
