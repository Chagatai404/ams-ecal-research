"""Distinguishability diagnostic: the AUC of identical and separated features, the ablations and the sealed-data rule."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_studies import em_lateral_distinguishability as dist
from ams_ecal.electron_studies import em_production_lateral_check as check

needs_samples = pytest.mark.skipif(
    not (
        (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
        and (check.BASELINE_DIR / "E10GeV" / "events.npz").exists()
        and lat.DEFAULT_ARTIFACT.exists()
        and dist.FASTER_TAIL_ARTIFACT.exists()
    ),
    reason="needs the git-ignored exposed samples (regenerate them from their configs) and the lateral artifacts",
)


def test_the_auc_of_two_samples_from_one_distribution_is_near_one_half() -> None:
    rng = np.random.default_rng(0)

    auc = dist.cross_validated_auc(rng.normal(size=(300, 4)), rng.normal(size=(300, 4)))

    assert 0.4 < auc < 0.6


def test_the_auc_of_separated_samples_is_near_one() -> None:
    rng = np.random.default_rng(1)

    auc = dist.cross_validated_auc(rng.normal(size=(300, 3)), rng.normal(loc=3.0, size=(300, 3)))

    assert auc > 0.95


def test_the_ablations_keep_exactly_the_declared_feature_groups() -> None:
    names = ["width_mm", "n_hit_cells", "occupied_cells_0", "occupied_cells_1", "energy_mev"]
    rng = np.random.default_rng(2)
    model_f = rng.normal(size=(200, len(names)))
    reference_f = rng.normal(size=(200, len(names)))
    reference_f[:, 1] += 5.0  # only the hit-cell count separates

    result = dist.ablations(model_f, reference_f, names)

    assert set(result) == set(dist.GROUPS)
    assert result["only_hit_cell_count"] > 0.95
    assert result["without_hit_cells"] < 0.65
    assert result["without_hit_cells_and_width"] < 0.65
    assert result["all_features"] > 0.95


def test_the_diagnostic_never_reads_a_sealed_set_and_is_never_a_target() -> None:
    text = Path(dist.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text
    assert "never a tuning target" in text


@needs_samples
def test_the_diagnostic_runs_end_to_end_on_a_small_slice() -> None:
    results = dist.distinguishability((10.0,), events=200, permutations=3)

    row = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(row) == {"geant4_floor", "slice2", "faster_tail"}
    assert set(row["slice2"]["feature_group_auc"]) == set(dist.GROUPS)
    assert 0.0 <= row["geant4_floor"]["auc"] <= 1.0
