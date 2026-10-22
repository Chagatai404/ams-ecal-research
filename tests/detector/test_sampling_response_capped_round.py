import numpy as np

from ams_ecal.detector.sampling_response import SamplingResponse
from ams_ecal.detector.sampling_response_capped_round import (
    classes_v1,
    classes_v2,
    classes_v3,
    eligible,
    interpolate_in_log_energy,
)


def test_v1_puts_crossing_protons_in_their_own_class():
    share = np.array([0.05, 0.15, 0.25, 0.4, 0.95, 0.05])
    interacting = np.array([True, True, True, True, True, False])
    assert classes_v1(share, interacting).tolist() == [1, 2, 3, 4, 8, 0]


def test_v2_and_v3_bin_edges():
    assert classes_v2(np.array([0.1, 0.3, 0.5, 0.95])).tolist() == [0, 1, 2, 5]
    assert classes_v3(np.array([0.0, 0.05, 0.2, 0.9])).tolist() == [0, 1, 3, 4]


def test_eligibility_needs_no_regression_and_two_improvements():
    baseline = {10.0: 0.08, 20.0: 0.05, 50.0: 0.05, 100.0: 0.05}
    better = {10.0: 0.06, 20.0: 0.05, 50.0: 0.039, 100.0: 0.05}
    one_gain = {10.0: 0.06, 20.0: 0.05, 50.0: 0.05, 100.0: 0.05}
    regress = {10.0: 0.06, 20.0: 0.051, 50.0: 0.03, 100.0: 0.05}
    assert eligible(better, baseline)
    assert not eligible(one_gain, baseline)
    assert not eligible(regress, baseline)


def _model(level: float) -> SamplingResponse:
    return SamplingResponse(
        particle="t",
        tables=np.full((2, 2, 2, 3), level),
        mean_ratio=np.full((2, 2, 2), level),
        n_fitted_cells=np.full((2, 2, 2), 500),
        fitted_from="t",
    )


def test_log_energy_interpolation_is_linear_in_log_energy():
    mid = interpolate_in_log_energy(
        _model(1.0), _model(3.0), 10.0, 100.0, np.sqrt(10.0 * 100.0), "m"
    )
    assert np.allclose(mid.mean_ratio, 2.0)
    assert np.allclose(mid.tables, 2.0)
