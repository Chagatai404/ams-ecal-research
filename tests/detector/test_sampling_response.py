import numpy as np
import pytest

from ams_ecal.detector.sampling_response import N_LAYERS, SamplingResponse


def synthetic(n_events: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deposits with a known layer-dependent sampling fraction and a 30% multiplicative scatter."""

    rng = np.random.default_rng(seed)
    shape = (n_events, N_LAYERS, 12)
    deposit = rng.lognormal(mean=1.0, sigma=0.8, size=shape) * (rng.random(shape) < 0.4)
    fraction = np.linspace(0.05, 0.08, N_LAYERS)[None, :, None]
    fibre = deposit * fraction * rng.lognormal(mean=-0.045, sigma=0.3, size=shape)
    return deposit, fibre


def test_zero_deposit_gives_zero_fibre_energy() -> None:
    deposit, fibre = synthetic(400, 1)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    out = model.apply(deposit, np.random.default_rng(0))
    assert np.all(out[deposit == 0] == 0)


def test_layer_means_close_on_independent_events() -> None:
    deposit, fibre = synthetic(1500, 2)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    held_deposit, held_fibre = synthetic(1500, 3)
    out = model.apply(held_deposit, np.random.default_rng(0))
    closure = out.sum(axis=(0, 2)) / held_fibre.sum(axis=(0, 2)) - 1
    assert (
        np.abs(closure).max() < 0.05
    )  # wiring check on 1500 events; the 3% criterion is run on real events


def test_scatter_is_reproduced_not_removed() -> None:
    deposit, fibre = synthetic(1500, 4)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    out = model.apply(deposit, np.random.default_rng(0))
    mask = deposit > 0
    modelled = np.std(np.log(out[mask] / deposit[mask]))
    reference = np.std(np.log(fibre[mask] / deposit[mask]))
    assert modelled == pytest.approx(reference, rel=0.25)


def test_same_seed_gives_same_output() -> None:
    deposit, fibre = synthetic(300, 5)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    first = model.apply(deposit, np.random.default_rng(7))
    second = model.apply(deposit, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, model.apply(deposit, np.random.default_rng(8)))


def test_save_and_load_round_trip(tmp_path) -> None:
    deposit, fibre = synthetic(300, 6)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    model.save(tmp_path / "model.json")
    loaded = SamplingResponse.load(tmp_path / "model.json")
    assert np.allclose(
        model.apply(deposit, np.random.default_rng(1)),
        loaded.apply(deposit, np.random.default_rng(1)),
    )
    assert loaded.particle == "test" and loaded.fitted_from == "synthetic"


def test_wrong_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        SamplingResponse.fit(
            np.zeros((3, 17, 4)), np.zeros((3, 17, 4)), particle="x", fitted_from="x"
        )
    deposit, fibre = synthetic(100, 7)
    model = SamplingResponse.fit(
        deposit, fibre, particle="test", fitted_from="synthetic"
    )
    with pytest.raises(ValueError):
        model.apply(np.zeros((3, 4)), np.random.default_rng(0))
