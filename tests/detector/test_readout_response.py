import numpy as np
import pytest

from ams_ecal.detector.readout_response import (
    MIP_FIBRE_MEV,
    ReadoutResponseConfig,
    apply_readout_response,
    crosstalk,
    draw_gain_map,
)
from ams_ecal.validation.added_checks import MIP_CELL_MEV


def fibre(n_events: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n_events, 18, 72)
    return rng.exponential(2.0, size=shape) * (rng.random(shape) < 0.3)


def fresh(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_mip_constant_matches_the_repository_value() -> None:
    assert MIP_FIBRE_MEV == MIP_CELL_MEV


def test_all_switches_off_is_the_identity() -> None:
    data = fibre()
    assert np.array_equal(
        apply_readout_response(data, ReadoutResponseConfig(), fresh()), data
    )


def test_photoelectrons_preserve_the_mean_and_follow_poisson_variance() -> None:
    energy = 3.0 * MIP_FIBRE_MEV
    data = np.full((20000, 18, 4), energy)
    out = apply_readout_response(data, ReadoutResponseConfig(pe_per_mip=8.0), fresh(1))
    assert out.mean() == pytest.approx(energy, rel=0.005)
    quantum = MIP_FIBRE_MEV / 8.0
    assert out.var() == pytest.approx(
        quantum * energy, rel=0.03
    )  # Poisson: var(N) = mu


def test_photoelectron_noise_is_35_percent_at_one_mip() -> None:
    data = np.full((40000, 18, 2), MIP_FIBRE_MEV)
    out = apply_readout_response(data, ReadoutResponseConfig(pe_per_mip=8.0), fresh(2))
    assert out.std() / out.mean() == pytest.approx(1 / np.sqrt(8.0), rel=0.03)


def test_zero_energy_stays_zero_under_photoelectrons() -> None:
    out = apply_readout_response(
        np.zeros((10, 18, 72)), ReadoutResponseConfig(pe_per_mip=8.0), fresh(3)
    )
    assert not out.any()


def test_gain_map_is_shared_by_all_events_and_differs_between_realisations() -> None:
    first = draw_gain_map(0.024, 72, fresh(4))
    second = draw_gain_map(0.024, 72, fresh(5))
    assert first.shape == (18, 72) and not np.array_equal(first, second)
    out = apply_readout_response(
        np.ones((50, 18, 72)), ReadoutResponseConfig(gain_sigma=0.024), fresh(), first
    )
    assert np.allclose(out, np.broadcast_to(first, out.shape))
    assert first.std() == pytest.approx(0.024, rel=0.1)


def test_gain_sigma_zero_is_identity_even_with_a_map() -> None:
    data = fibre()
    out = apply_readout_response(
        data, ReadoutResponseConfig(), fresh(), draw_gain_map(0.05, 72, fresh())
    )
    assert np.array_equal(out, data)


def test_gain_without_a_map_is_an_error() -> None:
    with pytest.raises(ValueError):
        apply_readout_response(
            fibre(5), ReadoutResponseConfig(gain_sigma=0.024), fresh()
        )


def test_crosstalk_conserves_the_total_including_edges() -> None:
    data = fibre()
    assert np.allclose(crosstalk(data, 0.1).sum(axis=2), data.sum(axis=2))
    edge = np.zeros((1, 18, 72))
    edge[..., 0] = 1.0
    moved = crosstalk(edge, 0.1)
    assert moved[0, 0, 0] == pytest.approx(0.9 + 0.05)
    assert moved[0, 0, 1] == pytest.approx(0.05)


def test_crosstalk_moves_energy_to_both_neighbours_in_the_same_layer_only() -> None:
    single = np.zeros((1, 18, 72))
    single[0, 5, 30] = 10.0
    out = crosstalk(single, 0.2)
    assert out[0, 5, 30] == pytest.approx(8.0)
    assert out[0, 5, 29] == pytest.approx(1.0) and out[0, 5, 31] == pytest.approx(1.0)
    assert out[0, 4].sum() == 0 and out[0, 6].sum() == 0


def test_threshold_zeroes_cells_below_the_cut_in_mip_units() -> None:
    data = np.array([[[0.4, 0.6, 1.0]]]).repeat(18, axis=1) * MIP_FIBRE_MEV
    out = apply_readout_response(
        data, ReadoutResponseConfig(threshold_mips=0.5), fresh()
    )
    assert out[0, 0, 0] == 0 and out[0, 0, 1] > 0 and out[0, 0, 2] > 0


def test_order_flag_matters_only_when_both_gain_and_threshold_are_active() -> None:
    data = fibre(50, 9)
    gain = draw_gain_map(0.07, 72, fresh(6))
    both = ReadoutResponseConfig(gain_sigma=0.07, threshold_mips=1.0)
    both_swapped = ReadoutResponseConfig(
        gain_sigma=0.07, threshold_mips=1.0, threshold_before_gain=True
    )
    assert not np.array_equal(
        apply_readout_response(data, both, fresh(), gain),
        apply_readout_response(data, both_swapped, fresh(), gain),
    )
    only = ReadoutResponseConfig(threshold_mips=1.0)
    only_swapped = ReadoutResponseConfig(threshold_mips=1.0, threshold_before_gain=True)
    assert np.array_equal(
        apply_readout_response(data, only, fresh()),
        apply_readout_response(data, only_swapped, fresh()),
    )


def test_same_seed_gives_the_same_output() -> None:
    data = fibre(30, 11)
    config = ReadoutResponseConfig(
        pe_per_mip=8.0, crosstalk_fraction=0.03, threshold_mips=0.5
    )
    assert np.array_equal(
        apply_readout_response(data, config, fresh(7)),
        apply_readout_response(data, config, fresh(7)),
    )


def test_every_parameter_carries_a_provenance_label() -> None:
    labelled = ReadoutResponseConfig(pe_per_mip=8.0, gain_sigma=0.024).labelled()
    assert labelled["crosstalk_fraction"]["provenance"] == "no source"
    assert labelled["threshold_mips"]["provenance"] == "no source"
    assert str(labelled["pe_per_mip"]["provenance"]).startswith("documented")
    assert all("provenance" in entry for entry in labelled.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pe_per_mip": 0.0},
        {"gain_sigma": 0.5},
        {"crosstalk_fraction": 1.5},
        {"threshold_mips": -1.0},
    ],
)
def test_invalid_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        ReadoutResponseConfig(**kwargs)


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_readout_response(np.zeros((3, 17, 72)), ReadoutResponseConfig(), fresh())
