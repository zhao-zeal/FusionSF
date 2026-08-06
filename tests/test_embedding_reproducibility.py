import torch

from tests.test_missing_modalities import batch, tiny_model


def test_same_model_batch_and_seed_are_reproducible_in_eval_mode():
    torch.manual_seed(42)
    model, data = tiny_model(), batch()
    first = model.extract_embeddings(data, "both", "mean")
    second = model.extract_embeddings(data, "both", "mean")
    assert torch.equal(first["ts"], second["ts"])
    assert torch.equal(first["fusion"], second["fusion"])


def test_embedding_order_is_independent_of_batch_size():
    torch.manual_seed(42)
    model, data = tiny_model(), batch()
    full = model.extract_embeddings(data, "both", "mean")
    chunks = []
    for index in range(2):
        one = {key: value[index:index + 1] for key, value in data.items()}
        chunks.append(model.extract_embeddings(one, "both", "mean"))
    # Batched GEMM kernels may differ at the last floating-point bits; row order and values must
    # still agree to numerical precision.
    assert torch.allclose(full["ts"], torch.cat([chunk["ts"] for chunk in chunks]), atol=1e-6)
    assert torch.allclose(full["fusion"], torch.cat([chunk["fusion"] for chunk in chunks]), atol=1e-6)
