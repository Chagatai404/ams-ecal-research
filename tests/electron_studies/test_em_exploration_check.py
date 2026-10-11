"""The EM-versus-Geant4 exploration comparison: structure, scale and refusal to touch sealed data."""

from pathlib import Path

import pytest

from ams_ecal.electron_studies import em_exploration_check
from ams_ecal.electron_studies.em_exploration_check import (
    DATA,
    OBSERVABLES,
    compare_energy,
)

ROOT = Path(__file__).parents[2]


def test_the_exploration_data_are_the_unsealed_sample_not_a_sealed_directory() -> None:
    assert "geant4_electron_sample" in str(em_exploration_check.DATA)
    assert "sealed" not in str(em_exploration_check.DATA)
    source = (ROOT / "src" / "ams_ecal" / "electron_studies" / "em_exploration_check.py").read_text(encoding="utf-8")
    assert "open_set" not in source
    assert "geant4_electron_sealed" not in source.split('"""', 2)[2]  # only the docstring names it


@pytest.mark.skipif(
    not (DATA / "E10GeV").is_dir(), reason="the exploration electron sample is not present"
)
def test_the_comparison_reports_every_contract_observable_and_the_harness_rows() -> None:
    result = compare_energy(10.0, n_boot=5, n_permutations=2)

    assert result["n_events"] == 1000
    assert set(result["observables"]) == set(OBSERVABLES)
    for row in result["observables"].values():
        assert 0.0 <= row["ks"] <= 1.0
        assert len(row["model_quantiles"]) == len(row["geant4_quantiles"]) == 3
    assert len(result["mean_layer_energy_ratio_model_over_geant4"]) == 18
    assert len(result["added_checks"]["sparsity"]) == 3
    assert 0.0 <= result["added_checks"]["classifier_auc"] <= 1.0
    median = result["observables"]["energy_mev"]["model_quantiles"][1]
    assert 0.9 * 10_000 < median < 1.01 * 10_000  # the deposition scale, not the fibre scale
