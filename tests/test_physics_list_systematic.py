"""The physics-list systematic: structure, and that it reads only the pilot's unsealed samples."""

from pathlib import Path

import pytest

from ams_ecal import physics_list_systematic as systematic
from ams_ecal.proton import ProtonShowerModel

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"
HAVE_DATA = ARTIFACT.is_dir() and (systematic.DATA / "high_energy_model" / "E10GeV").is_dir()


def test_it_names_only_pilot_samples_and_never_a_sealed_directory() -> None:
    source = (ROOT / "src" / "ams_ecal" / "physics_list_systematic.py").read_text(encoding="utf-8")

    assert set(systematic.SAMPLES.values()) == {"high_energy_model", "alternate"}
    assert "geant4_proton_pilot" in str(systematic.DATA)
    assert "open_set" not in source and "geant4_proton_sealed" not in source.split('"""', 2)[2]


@pytest.mark.skipif(not HAVE_DATA, reason="the calibration artifact or the pilot samples are absent")
def test_a_comparison_reports_the_model_and_the_ftfp_floor_for_every_cell() -> None:
    result = systematic.compare(ProtonShowerModel.from_config(), "QGSP_BERT", 10.0, 200)

    assert set(result) == {
        "crossing readout",
        "crossing deposition",
        "interacting readout",
        "interacting deposition",
    }
    for cell in result.values():
        assert set(cell["model_vs_alternative"]) == set(systematic.OBSERVABLES)
        assert set(cell["ftfp_vs_alternative"]) == set(systematic.OBSERVABLES)
        assert all(0.0 <= v <= 1.0 for v in cell["model_vs_alternative"].values())
        assert cell["n_alternative"] > 100
