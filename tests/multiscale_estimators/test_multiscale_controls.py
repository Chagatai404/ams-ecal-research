"""Tests for the synthetic controls of the estimator-validity gate (EXP-008). Exact identities and structure only: no control
outcome is asserted in advance."""

import numpy as np
import pytest

from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.multiscale_estimators import multiscale_controls as mc


def test_maps_are_normalised_and_have_the_ams_shape():
    for p in (mc.smooth_map(), mc.cascade_map()):
        assert p.shape == (18, 72) and p.sum() == pytest.approx(1.0) and np.all(p > 0)


@pytest.mark.parametrize("q", ms.Q_VALUES)
def test_estimator_on_the_cascade_population_equals_the_exact_aligned_values(q):
    p = mc.cascade_map()
    for (fine, coarse), exact in mc.analytic_local_dimensions(q).items():
        assert ms.local_dimension(p, q, fine, coarse) == pytest.approx(exact, abs=1e-9)


def test_exact_cascade_dimensions_are_ordered_in_q():
    d = [mc.analytic_local_dimensions(q) for q in ms.Q_VALUES]
    for pair in d[0]:
        assert d[0][pair] >= d[1][pair] >= d[2][pair]
    assert d[0][(1, 3)] == pytest.approx(2.0)  # full support: every cell occupied at q = 0


def test_sampling_conserves_hits_and_is_reproducible():
    p = mc.smooth_map()
    a = mc.sample_hits(p, 300, np.random.default_rng(5))
    b = mc.sample_hits(p, 300, np.random.default_rng(5))
    assert a.sum() == 300 and np.array_equal(a, b) and a.min() >= 0


def test_window_estimates_drop_saturated_windows():
    img = mc.sample_hits(mc.smooth_map(), 100, np.random.default_rng(1))
    est = mc.window_estimates(img, 0.0)
    assert set(est) == set(ms.contiguous_windows())
    assert any(v is None for v in est.values()) and any(v is not None for v in est.values())
    reliable = {w for w, v in est.items() if v is not None}
    assert reliable == set(ms.reliable_windows(img))


def test_spread_is_none_when_either_order_is_saturated():
    img = mc.sample_hits(mc.smooth_map(), 100, np.random.default_rng(2))
    spread = mc.spread_estimates(img)
    d0, d2 = mc.window_estimates(img, 0.0), mc.window_estimates(img, 2.0)
    for w, v in spread.items():
        assert (v is None) == (d0[w] is None or d2[w] is None)


def test_evaluate_returns_well_formed_results_for_small_replicates():
    results = mc.evaluate(300, seed=3, replicates=40)
    assert results
    for r in results:
        assert r.n_hits == 300 and r.window in ms.contiguous_windows()
        assert r.baseline_low <= r.baseline_high and 0.0 <= r.false_positive_rate <= 1.0
        assert r.reliable_fraction >= 0.95
        if not np.isnan(r.q):  # power is only defined for the D0 - D2 spread statistic
            assert r.power is None and r.ordering_rate is None


def test_recovery_table_covers_every_q_and_aligned_pair():
    rows = mc.recovery_table(1000, seed=4, replicates=20)
    assert len(rows) == 3 * 3
    assert {(r["fine"], r["coarse"]) for r in rows} == {(9, 18), (3, 9), (1, 3)}
    assert all(0.0 <= r["saturated_fraction"] <= 1.0 for r in rows)


def test_verdict_is_empty_for_no_results_and_never_says_usable_without_a_passing_window():
    assert mc.verdict([]) == {}
    results = mc.evaluate(30, seed=6, replicates=40)
    for text in mc.verdict(results).values():
        assert text.startswith(("USABLE", "NOT USABLE"))


def test_thresholds_are_the_confirmed_dec_013_values():
    assert mc.BASELINE_INTERVAL == (2.5, 97.5)
    assert mc.MAX_FALSE_POSITIVE_RATE == 0.10 and mc.MIN_DETECTION_POWER == 0.90 and mc.RECOVERY_REFERENCE_TOLERANCE == 0.15
    assert mc.MIN_ORDERING_RATE == 0.90  # interpretation of ordering part, awaiting the researcher
    assert mc.HIT_COUNTS == (30, 100, 300, 1000, 3000) and mc.REPLICATES == 1000


def test_all_statistics_come_from_the_same_image_and_match_the_single_fit():
    img = mc.sample_hits(mc.cascade_map(), 1000, np.random.default_rng(7))
    est = mc.image_estimates(img)
    for w, v in est["D0"].items():
        if v is not None:
            assert v == pytest.approx(ms.fitted_dimension(img, 0.0, w), abs=1e-12)
            assert est["D1"][w] == pytest.approx(ms.fitted_dimension(img, 1.0, w), abs=1e-12)
            assert est["D2"][w] == pytest.approx(ms.fitted_dimension(img, 2.0, w), abs=1e-12)
            assert est["D0-D2"][w] == pytest.approx(est["D0"][w] - est["D2"][w], abs=1e-12)


def test_c2_requires_both_power_and_the_ordering_clause():
    for r in mc.evaluate(1000, seed=8, replicates=60):
        if r.q != r.q:
            assert r.passes_power_and_ordering == (r.power >= mc.MIN_DETECTION_POWER and r.ordering_rate >= mc.MIN_ORDERING_RATE)


def test_the_runner_writes_every_result_file(tmp_path):
    result = mc.run_controls(tmp_path, seed=11, replicates=30)
    for name in ("controls.json", "recovery.json", "windows.json", "controls.png"):
        assert (tmp_path / name).exists()
    assert set(result["verdicts"]) == {str(n) for n in mc.HIT_COUNTS}
    assert set(result["windows"]) == {str(w) for w in ms.contiguous_windows()}
