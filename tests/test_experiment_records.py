from omegaconf import OmegaConf

from src.utils.experiment_records import missing_modality_interpretation


def test_missing_modality_interpretation_tracks_training_dropout():
    standard = OmegaConf.create({
        "satellite_modality_dropout": 0.0,
        "nwp_modality_dropout": 0.0,
    })
    robust = OmegaConf.create({
        "satellite_modality_dropout": 0.2,
        "nwp_modality_dropout": 0.2,
    })
    assert "diagnostic only" in missing_modality_interpretation(standard, "missing_nwp")
    assert "robustness evaluation" in missing_modality_interpretation(robust, "missing_nwp")
    assert missing_modality_interpretation(robust, "full_modalities") == ""
