import numpy as np
import pytest

from ams_ecal.detector.component_sampling_response import (
    N_COMPONENTS,
    RatioTable,
    Statistics,
    actual_layers,
    closure,
)

SHAPE = (N_COMPONENTS, 18, 72)


def synthetic(n_events: int, ratios: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    deposit = rng.exponential(2.0, size=(n_events, *SHAPE)) * (
        rng.random((n_events, *SHAPE)) < 0.4
    )
    fibre = deposit * ratios[None, :, None, None]
    return deposit.astype(np.float32), fibre.astype(np.float32), np.arange(n_events)


def stats_of(deposit, fibre, index) -> Statistics:
    s = Statistics.zeros()
    s.add_events(deposit, fibre, index)
    return s


def test_constant_component_ratios_are_recovered_and_closure_is_zero() -> None:
    ratios = np.linspace(0.05, 0.2, N_COMPONENTS)
    fit = stats_of(*synthetic(400, ratios, 1))
    table = RatioTable.fit(fit)
    test = stats_of(*synthetic(400, ratios, 2))
    assert np.abs(closure(table, test)).max() < 1e-3


def test_in_sample_closure_is_exact_when_every_class_is_dense() -> None:
    rng = np.random.default_rng(3)
    deposit = np.where(rng.random((200, *SHAPE)) < 0.4, 1.0, 0.0).astype(np.float32)
    fibre = deposit * rng.uniform(0.5, 1.5, deposit.shape).astype(np.float32)
    s = stats_of(deposit, fibre, np.arange(200))
    table = RatioTable.fit(s)
    assert (table.level[deposit.any(axis=(0, 3))[..., None] & (s.count.sum(0) > 0)] == 0).all()
    assert np.allclose(table.predict_layers(s), actual_layers(s), rtol=1e-5)


def test_parts_keeps_the_axis_and_selects_by_event_index() -> None:
    deposit, fibre, index = synthetic(40, np.full(N_COMPONENTS, 0.1), 5)
    s = stats_of(deposit, fibre, index)
    odd = s.parts((1, 3))
    assert odd.n_events.shape == (4,) and odd.n_events.tolist() == [0, 10, 0, 10]
    assert odd.sum_deposit[0].sum() == 0 and odd.sum_deposit[1].sum() > 0


def test_orphan_fibre_energy_is_counted_in_the_actual_but_not_the_prediction() -> None:
    deposit = np.zeros((4, *SHAPE), np.float32)
    fibre = np.zeros((4, *SHAPE), np.float32)
    deposit[:, 0, 0, :5] = 1.0
    fibre[:, 0, 0, :5] = 0.1
    fibre[:, 1, 0, 0] = 0.5  # fibre energy where component 1 has no deposit
    s = stats_of(deposit, fibre, np.arange(4))
    assert s.orphan_fibre.sum() == pytest.approx(4 * 0.5)
    table = RatioTable.fit(s)
    assert actual_layers(s)[0] > table.predict_layers(s)[0]


def test_sparse_classes_use_the_fallback_and_it_is_reported() -> None:
    deposit, fibre, index = synthetic(3, np.full(N_COMPONENTS, 0.1), 6)
    s = stats_of(deposit, fibre, index)
    table = RatioTable.fit(s)  # far fewer than 400 cells per class
    share = table.fallback_share(s)
    assert share[0] == 0.0 and share[1] + share[2] == pytest.approx(1.0)
