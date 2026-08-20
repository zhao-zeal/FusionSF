import pytest
import torch

from src.models.temporal_prior import TemporalPriorInjector


def test_none_is_exact_identity():
    injector = TemporalPriorInjector("none", fusion_dim=64, history_length=24)
    tokens = torch.randn(2, 24, 64)
    assert injector(tokens, torch.randn(2, 24, 1)) is tokens
    assert injector.trainable_parameter_count == 0


def test_mlp_and_chronos_controls_are_parameter_matched():
    mlp = TemporalPriorInjector("mlp", fusion_dim=64, history_length=24, mlp_hidden_dim=553)
    chronos = TemporalPriorInjector("chronos", fusion_dim=64, history_length=24, chronos_dim=768)
    assert mlp.trainable_parameter_count == 49_281
    assert chronos.trainable_parameter_count == 49_216
    relative_difference = abs(mlp.trainable_parameter_count - chronos.trainable_parameter_count) / 49_216
    assert relative_difference < 0.002


@pytest.mark.parametrize("mode", ["mlp", "chronos"])
def test_prior_is_broadcast_to_every_power_token(mode):
    injector = TemporalPriorInjector(mode, fusion_dim=4, history_length=3, chronos_dim=5, mlp_hidden_dim=4)
    tokens = torch.zeros(2, 3, 4)
    history = torch.randn(2, 3, 1)
    chronos = torch.randn(2, 5) if mode == "chronos" else None
    output = injector(tokens, history, chronos)
    torch.testing.assert_close(output[:, 0], output[:, 1])
    assert output.shape == tokens.shape


def test_chronos_mode_rejects_missing_or_wrong_shape():
    injector = TemporalPriorInjector("chronos", fusion_dim=4, history_length=3, chronos_dim=5)
    tokens, history = torch.zeros(2, 3, 4), torch.zeros(2, 3, 1)
    with pytest.raises(ValueError, match="requires chronos"):
        injector(tokens, history)
    with pytest.raises(ValueError, match="must have shape"):
        injector(tokens, history, torch.zeros(2, 6))


def test_project_prior_can_condition_vq_input_without_post_encoder_residual():
    injector = TemporalPriorInjector("chronos", fusion_dim=4, history_length=3, chronos_dim=5)
    tokens, history = torch.zeros(2, 3, 4), torch.zeros(2, 3, 1)
    representation = torch.randn(2, 5)
    guidance = injector.project_prior(tokens, history, representation)
    assert guidance.shape == (2, 4)
    torch.testing.assert_close(injector(tokens, history, representation), guidance.unsqueeze(1).expand(-1, 3, -1))
