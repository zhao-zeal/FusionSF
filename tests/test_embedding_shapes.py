import torch

from tests.test_missing_modalities import batch, tiny_model


def test_embedding_pooling_shapes_and_finiteness():
    model, data = tiny_model(), batch()
    raw = model.extract_embeddings(data, "both", "none")
    mean = model.extract_embeddings(data, "both", "mean")
    assert raw["ts"].shape == (2, 4, 8)
    assert raw["fusion"].shape == (2, 4, 8)
    assert mean["ts"].shape == (2, 8)
    assert raw["ts"].isfinite().all() and raw["fusion"].isfinite().all()


def test_read_only_ts_interface_matches_forward_ts_node():
    model, data = tiny_model(), batch()
    model.eval()
    with torch.inference_mode():
        isolated = model.extract_ts_embeddings(
            data["ts_input"].float(), data["ts_time"].float(), pooling="mean"
        )
        full = model.extract_embeddings(data, "ts", "mean")["ts"]
    assert torch.equal(isolated, full)
