from tests.test_missing_modalities import batch, tiny_model


def test_embedding_pooling_shapes_and_finiteness():
    model, data = tiny_model(), batch()
    raw = model.extract_embeddings(data, "both", "none")
    mean = model.extract_embeddings(data, "both", "mean")
    assert raw["ts"].shape == (2, 4, 8)
    assert raw["fusion"].shape == (2, 4, 8)
    assert mean["ts"].shape == (2, 8)
    assert raw["ts"].isfinite().all() and raw["fusion"].isfinite().all()
