"""Extended-depth tail study: the machinery on planted profiles, and the new sample where it is present."""

import numpy as np
import pytest

from ams_ecal.electron_studies import em_depth_origin_calibration as calibration
from ams_ecal.electron_studies import em_extended_tail_study as study
from ams_ecal.electron_studies import em_profile_shape_study as shape

CRITICAL_ENERGY_MEV = 7.6
ENERGIES = (10.0, 50.0)
needs_sample = pytest.mark.skipif(
    not (study.EXTENDED_DIR / "E10GeV" / "events.npz").exists(),
    reason="the extended-depth sample is git-ignored: regenerate it from configs/geant4_electron_extended.yaml",
)


def planted_events(spec: calibration.ModelSpec, params: dict[str, float], n: int = 150, seed: int = 2):
    """Extended events scattered around a known 270-layer profile."""

    rng = np.random.default_rng(seed)
    model = study.predicted_layer_fractions(spec, params, np.array(ENERGIES), CRITICAL_ENERGY_MEV, 270)
    events = {}
    for energy, row in zip(ENERGIES, model, strict=True):
        deposit = np.clip(row * (1.0 + 0.05 * rng.standard_normal((n, 270))), 0.0, None)
        events[energy] = study.Extended(energy, deposit, 0.065 * deposit, rng.uniform(0, 9, n), rng.uniform(0, 9, n))
    return events


def test_the_layer_bounds_are_contiguous_and_17_over_18_x0_thick() -> None:
    bounds = study.layer_bounds(270)

    assert bounds.shape == (270, 2)
    assert bounds[0, 0] == 0.0
    assert bounds[:-1, 1] == pytest.approx(bounds[1:, 0])
    assert bounds[0, 1] == pytest.approx(17.0 / 18.0)


def test_the_exponential_tail_has_nothing_in_front_of_its_onset_and_decays_at_its_rate() -> None:
    bounds = study.layer_bounds(80)

    tail = calibration.exponential_tail_fractions(bounds, 10.0, 0.3)

    assert tail[bounds[:, 1] <= 10.0].sum() == 0.0
    assert tail.sum() == pytest.approx(1.0, abs=2e-3)
    assert tail[40] / tail[41] == pytest.approx(np.exp(0.3 * study.LAYER_X0), rel=1e-6)


def test_the_decay_length_of_a_planted_exponential_is_recovered() -> None:
    depth = (np.arange(270) + 0.5) * study.LAYER_X0
    profile = 0.05 * np.exp(-0.28 * depth)
    rng = np.random.default_rng(3)
    deposit = profile * (1.0 + 0.05 * rng.standard_normal((300, 270)))
    events = {10.0: study.Extended(10.0, deposit, deposit, np.zeros(300), np.zeros(300))}

    result = study.decay_length(events, n_boot=40)

    assert result["10"]["rate_per_x0"] == pytest.approx(0.28, abs=0.01)
    assert result["10"]["attenuation_length_x0"] == pytest.approx(1.0 / 0.28, rel=0.05)


def test_local_decay_rates_of_a_planted_exponential_are_the_same_in_every_window() -> None:
    depth = (np.arange(270) + 0.5) * study.LAYER_X0
    deposit = np.tile(0.05 * np.exp(-0.27 * depth), (20, 1))
    events = {10.0: study.Extended(10.0, deposit, deposit, np.zeros(20), np.zeros(20))}

    rates = study.local_decay_rates(events)["10"]

    assert rates == pytest.approx([0.27] * len(rates), abs=1e-6)


def test_the_prefix_consistency_of_a_sample_with_itself_is_zero() -> None:
    rng = np.random.default_rng(4)
    deposit = rng.dirichlet(np.ones(270) * 2.0, size=100)
    events = {10.0: study.Extended(10.0, deposit, deposit, np.zeros(100), np.zeros(100))}
    baseline = {10.0: type("Baseline", (), {"fractions": {"deposition": deposit[:, :18]}})()}

    result = study.prefix_consistency(events, baseline)["10"]

    assert result["max_abs_z"] == pytest.approx(0.0, abs=1e-9)
    assert result["ks_contained_prefix"] == 0.0


def test_the_prefix_only_prediction_finds_a_planted_tail_that_a_gamma_cannot() -> None:
    spec = shape.extension_specs()[shape.CHOSEN_TAIL_FAMILY]
    truth = {"z0": -1.0, "beta": 0.6, "tail_w": 0.15, "tail_alpha": 4.3}
    events = planted_events(spec, truth)

    result = study.prefix_prediction(events, CRITICAL_ENERGY_MEV, ("primary", shape.CHOSEN_TAIL_FAMILY))

    assert result[shape.CHOSEN_TAIL_FAMILY]["ratio_predicted_over_measured"] == pytest.approx([1.0, 1.0], abs=0.05)
    assert max(result["primary"]["ratio_predicted_over_measured"]) < 0.95
    assert result["measured_behind_prefix_mean"][0] > 0.0


def test_the_full_depth_fit_returns_finite_group_errors_with_empty_layers() -> None:
    spec = shape.extension_specs()[shape.CHOSEN_TAIL_FAMILY]
    events = planted_events(spec, {"z0": -1.0, "beta": 0.6, "tail_w": 0.15, "tail_alpha": 4.3})
    for ext in events.values():
        ext.deposit[:, 70:] = 0.0  # layers with no energy at all must not produce infinities

    fits = study.full_depth_fits(events, CRITICAL_ENERGY_MEV)

    assert np.all(np.isfinite(fits["gamma_only"]["rms_relative_layers_4_17"]))
    assert fits["tail_literature_rate"]["chi2"] < fits["gamma_only"]["chi2"]
    assert fits["tail_literature_rate"]["tail_attenuation_length_x0"] == pytest.approx(3.6)


def test_the_event_level_statistics_have_the_expected_structure() -> None:
    spec = shape.extension_specs()[shape.CHOSEN_TAIL_FAMILY]
    events = planted_events(spec, {"z0": -1.0, "beta": 0.6, "tail_w": 0.15, "tail_alpha": 4.3}, n=80)

    result = study.event_level(events, calibration_origin=-1.0)

    block = result["10"]
    assert 0.0 <= block["r2_of_prefix_gamma_for_behind"] <= 1.0
    assert block["sd_behind_prefix_measured"] >= 0.0
    assert {"spearman_residual_ln_t_prefix", "spearman_residual_entry_x", "ln_t_prefix_skewness"} <= set(block)


@needs_sample
def test_the_extended_sample_has_270_layers_and_contains_almost_all_the_energy() -> None:
    ext = study.load_extended(10.0, max_events=50)

    assert ext.deposit.shape == (50, 270)
    assert 0.98 < ext.deposit.sum(axis=1).mean() < 1.001
    assert np.all(ext.behind_prefix >= 0.0)


@needs_sample
def test_the_study_runs_end_to_end_and_writes_tables_and_plots(tmp_path) -> None:
    families = ("primary", shape.CHOSEN_TAIL_FAMILY)

    results, artefacts = study.analyse((10.0, 100.0), max_events=120, families=families)

    assert set(results["B_prefix_only_prediction"]) >= {"primary", shape.CHOSEN_TAIL_FAMILY}
    assert results["A_prefix_consistency"]["10"]["rms_z"] < 3.0
    assert "D_event_level" in results

    study.write_tables(results, tmp_path)
    from ams_ecal.electron_studies.em_extended_tail_study_plots import make_plots

    make_plots(results, artefacts, tmp_path / "plots")

    assert (tmp_path / "extended_tail_study_prediction.csv").exists()
    assert (tmp_path / "plots" / "tail_decay_rate.png").exists()
    assert (tmp_path / "plots" / "prefix_only_prediction.png").exists()
