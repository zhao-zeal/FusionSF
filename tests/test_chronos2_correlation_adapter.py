import torch

from src.models.chronos2_correlation_adapter import (
    FusionChronosCorrelationAdapter,
    freeze_backbones,
    trainable_parameter_names,
)


def test_zero_initialized_adapter_is_exact_identity():
    adapter = FusionChronosCorrelationAdapter(
        fusion_dim=8, chronos_dim=24, heads=4, global_hidden=6
    )
    chronos = torch.randn(2, 5, 24)
    fusion = torch.randn(2, 7, 8)
    torch.testing.assert_close(adapter(chronos, fusion), chronos, rtol=0, atol=0)
    assert adapter.alpha.item() == 0.0
    assert adapter.beta.item() == 0.0


def test_dynamic_and_global_branches_change_output_after_gates_open():
    torch.manual_seed(7)
    adapter = FusionChronosCorrelationAdapter(
        fusion_dim=8, chronos_dim=24, heads=4, global_hidden=6
    )
    adapter.alpha.data.fill_(1.0)
    adapter.beta.data.fill_(1.0)
    chronos = torch.randn(2, 5, 24)
    fusion = torch.randn(2, 7, 8)
    assert not torch.equal(adapter(chronos, fusion), chronos)
    assert not torch.equal(adapter(chronos, fusion), adapter(chronos, fusion.flip(0)))


def test_misaligned_batches_are_rejected():
    adapter = FusionChronosCorrelationAdapter(
        fusion_dim=8, chronos_dim=24, heads=4, global_hidden=6
    )
    try:
        adapter(torch.randn(2, 5, 24), torch.randn(3, 7, 8))
    except ValueError as error:
        assert "batches must align" in str(error)
    else:
        raise AssertionError("misaligned batches must fail")


def test_only_adapter_remains_trainable():
    chronos = torch.nn.Linear(3, 4)
    fusionsf = torch.nn.Linear(5, 6)
    adapter = FusionChronosCorrelationAdapter(
        fusion_dim=8, chronos_dim=24, heads=4, global_hidden=6
    )
    freeze_backbones(chronos, fusionsf)
    assert not any(parameter.requires_grad for parameter in chronos.parameters())
    assert not any(parameter.requires_grad for parameter in fusionsf.parameters())
    names = trainable_parameter_names(adapter)
    assert "alpha" in names and "beta" in names
    assert "fusion_projection.weight" in names
