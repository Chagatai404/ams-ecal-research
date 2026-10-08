"""Lateral structure study: the closed-form strip law, planted recovery, the spot-number estimator and the sealed-data rule."""

from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.electron_studies import em_lateral_structure_study as lat

needs_samples = pytest.mark.skipif(
    not (lat.DATA / "E10GeV" / "events.npz").exists(),
    reason="the exposed baseline electron sample is git-ignored: regenerate it from configs/geant4_electron_sample.yaml",
)


def strip() -> lat.StripGeometry:
    return lat.StripGeometry(load_geometry(lat.GEOMETRY_CONFIG))


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")  # the 1/x^3 tail of an infinite range
def test_the_strip_cdf_matches_numerical_integration_of_the_two_dimensional_density() -> None:
    radius = 7.0

    def marginal(x: float) -> float:
        """The density 2 r R^2/(r^2+R^2)^2 / (2 pi r) integrated over the fibre direction."""

        return integrate.quad(
            lambda y: radius**2 / (np.pi * (x * x + y * y + radius**2) ** 2),
            -np.inf,
            np.inf,
            limit=400,
        )[0]

    numeric = integrate.quad(marginal, -np.inf, 5.0, limit=400)[0]

    assert lat.strip_cdf(np.array([5.0]), radius)[0] == pytest.approx(numeric, abs=1e-5)
    assert lat.strip_cdf(np.array([0.0]), radius)[0] == pytest.approx(0.5)


def test_the_fractions_are_normalised_to_the_central_mass_and_symmetric_in_the_mean() -> None:
    geometry = strip()

    fractions = geometry.fractions(2.0, 15.0, 0.7)

    assert fractions.sum() < 1.0
    assert fractions.sum() > 0.95
    assert (fractions >= 0).all()
    assert fractions[lat.AXIS_CELL - 4] == pytest.approx(fractions[lat.AXIS_CELL + 4], rel=0.25)


def test_the_two_component_fit_recovers_planted_parameters() -> None:
    geometry = strip()
    truth = (2.5, 16.0, 0.8)

    core, tail, weight = lat.fit_radial_law(geometry, geometry.fractions(*truth))

    assert core == pytest.approx(truth[0], rel=0.1)
    assert tail == pytest.approx(truth[1], rel=0.1)
    assert weight == pytest.approx(truth[2], abs=0.05)


def test_the_one_component_fit_cannot_describe_a_two_component_profile() -> None:
    geometry = strip()
    observed = geometry.fractions(2.0, 20.0, 0.8)
    two = geometry.fractions(*lat.fit_radial_law(geometry, observed))
    one = geometry.fractions(*lat.fit_radial_law(geometry, observed, single=True))

    assert lat.total_variation(two, observed) < 0.2 * lat.total_variation(one, observed)


def test_the_spot_number_is_recovered_after_removing_the_depth_dependence() -> None:
    rng = np.random.default_rng(3)
    spots, events = 400, 6000
    depth = rng.normal(9.0, 0.8, events)
    probability = np.clip(0.9 - 0.04 * (depth - 9.0), 0.05, 0.95)  # the radial law moves with the event's depth
    fraction = rng.binomial(spots, probability) / spots

    result = lat.spot_number(fraction, depth)

    assert result["spot_number_conditioned"] == pytest.approx(spots, rel=0.15)
    assert result["spot_number_raw"] < 0.8 * result["spot_number_conditioned"]
    assert 0.0 < result["explained_variance_fraction"] < 1.0


def test_the_radial_containment_is_the_complement_of_the_enclosed_fraction() -> None:
    assert lat.radial_fraction_outside(0.0, 2.0, 10.0, 0.6) == pytest.approx(1.0)
    assert lat.radial_fraction_outside(1e6, 2.0, 10.0, 0.6) == pytest.approx(0.0, abs=1e-9)
    assert lat.radial_fraction_outside(2.0, 2.0, 10.0, 1.0) == pytest.approx(0.5)


def test_the_published_constants_are_returned_in_millimetres_with_the_stated_assumptions() -> None:
    published = lat.grindhammer_peters_radial(50.0, 5.0, 18.0)

    assert 0.0 < published["core_weight"] <= 1.0
    assert published["core_mm"] < published["tail_mm"]
    assert "NOT verified" in lat.grindhammer_peters_radial.__doc__


def test_the_study_never_reads_a_sealed_set() -> None:
    text = Path(lat.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_study_runs_end_to_end_on_a_small_slice(tmp_path) -> None:
    results = lat.study((10.0,), max_events=150)

    block = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert len(block["layers"]) == 18
    mid = block["layers"][8]["tv_points"]
    assert mid["two_component_fit"] < mid["one_component_fit"]
    lat.write_outputs(results, tmp_path)
    assert (tmp_path / "lateral_structure_study.json").exists()
    assert (tmp_path / "lateral_structure_study_layers.csv").exists()
