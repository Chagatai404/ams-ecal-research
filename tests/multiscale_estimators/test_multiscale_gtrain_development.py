"""The exploratory conventional reductions of the large-sample development run."""

import numpy as np

from ams_ecal.multiscale_estimators import multiscale_gtrain_development as dev
from ams_ecal.multiscale_estimators import multiscale_test as mt


def test_the_reductions_drop_exactly_the_named_columns() -> None:
    grids = np.random.default_rng(0).exponential(1.0, size=(6, 18, 72))
    names, table = mt.conventional_features(grids)

    blocks = dev.derived_blocks(names, table)

    assert blocks["conv"].shape == table.shape
    assert blocks["conv_noE"].shape[1] == table.shape[1] - 1
    assert blocks["conv_noE_nocells"].shape[1] == table.shape[1] - 2
    kept = [n for n in names if n not in ("log_total_energy", "occupied_cells")]
    assert np.array_equal(blocks["conv_noE_nocells"], table[:, [names.index(n) for n in kept]])


def test_every_exploratory_rung_pairs_with_its_multiscale_extension() -> None:
    for base, extended in dev.GAIN_PAIRS:
        assert base in dev.RUNGS
        assert dev.RUNGS[extended] == (*dev.RUNGS[base], "ms")
