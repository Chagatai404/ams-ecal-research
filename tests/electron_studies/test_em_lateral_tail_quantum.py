"""Extension E2's measurement: the pooled energy per occupied cell far from the track."""

import numpy as np
import pytest

from ams_ecal.electron_studies import em_lateral_tail_quantum as quantum
from ams_ecal.proton_model.proton_validation import CELL_THRESHOLD_MEV


def test_only_occupied_cells_beyond_the_distance_are_counted() -> None:
    offsets = np.array([[0.0, 50.0, 120.0, -130.0, 200.0]])
    grid = np.array([[900.0, 40.0, 2.0, 1.0, CELL_THRESHOLD_MEV / 2]])

    energy, cells = quantum.occupied_energy_far(grid, offsets, far_mm=108.0)

    assert (energy, cells) == (3.0, 2)


def test_the_measurement_pools_energy_over_cells_and_stays_inside_its_own_range() -> None:
    measured = quantum.measure_tail_quantum()

    values = list(measured["per_energy_mev"].values())
    assert min(values) <= measured["tail_quantum_mev"] <= max(values)
    assert measured["tail_quantum_mev"] == pytest.approx(1.36, abs=0.05)
