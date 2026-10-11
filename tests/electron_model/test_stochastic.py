from math import exp, isfinite
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaincc

from ams_ecal.detector.event import ECALEvent
from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model.fastmc_config import StochasticEMConfig, load_fastmc_config
from ams_ecal.electron_model.lateral import AMSLateralShowerModel
from ams_ecal.electron_model.longitudinal import AMSLongitudinalGammaModel
from ams_ecal.electron_model.stochastic import StochasticEMShowerModel

PROJECT_ROOT = Path(__file__).parents[2]
GEOMETRY_CONFIG_PATH = PROJECT_ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG_PATH = PROJECT_ROOT / "configs" / "fastmc.yaml"

VALID_SHA256 = "a" * 64
SIMULATION_VERSION = "test"


@pytest.fixture
def model() -> StochasticEMShowerModel:
    geometry = load_geometry(GEOMETRY_CONFIG_PATH)
    fastmc_config = load_fastmc_config(FASTMC_CONFIG_PATH)
    return StochasticEMShowerModel(
        config=fastmc_config.stochastic_em,
        longitudinal=AMSLongitudinalGammaModel(
            config=fastmc_config.longitudinal_em,
            geometry=geometry,
            regime=fastmc_config.regime,
        ),
        lateral=AMSLateralShowerModel(
            config=fastmc_config.lateral_em,
            geometry=geometry,
        ),
    )


@pytest.fixture
def track() -> TrackState:
    return TrackState(
        x0_mm=0.0,
        y0_mm=0.0,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=0.0,
    )


def make_event(
    model: StochasticEMShowerModel,
    track: TrackState,
    random_seed: int,
    energy_mev: float = 100_000.0,
) -> ECALEvent:
    return model.generate_event(
        event_id=f"test-{random_seed:06d}",
        primary_energy_mev=energy_mev,
        track=track,
        random_seed=random_seed,
        simulation_version=SIMULATION_VERSION,
        configuration_sha256=VALID_SHA256,
    )


# --- the width law -----------------------------------------------------


def test_fluctuation_width_narrows_with_increasing_energy(
    model: StochasticEMShowerModel,
) -> None:
    widths = tuple(
        model.fluctuation_width(energy_mev)
        for energy_mev in (1.0e3, 1.0e4, 1.0e5, 1.0e6)
    )

    assert all(width > 0 for width in widths)
    assert widths == tuple(sorted(widths, reverse=True))


def test_fluctuation_width_matches_the_published_form(
    model: StochasticEMShowerModel,
) -> None:
    # s(E) = 1 / (-2.5 + 1.25 ln(E / E_c)) with E_c = 7.6 MeV.
    assert model.fluctuation_width(100_000.0) == pytest.approx(0.1069, abs=1e-4)


def test_rejects_energies_below_the_width_law_pole(
    model: StochasticEMShowerModel,
) -> None:
    # The denominator vanishes at E / E_c = exp(2), i.e. near 56 MeV. This
    # binds above the mean model's own ln(E / E_c) > 0.5 floor near 12.5 MeV,
    # so it is the stochastic block that must reject these energies.
    pole_energy_mev = model.critical_energy_mev * exp(2.0)

    assert model.fluctuation_width(1.01 * pole_energy_mev) > 0

    with pytest.raises(ValueError, match="width law"):
        model.fluctuation_width(0.99 * pole_energy_mev)


# --- the centring identity ---------------------------------------------


@pytest.mark.parametrize("energy_mev", [1.0e4, 1.0e5, 1.0e6])
def test_sampled_shower_maximum_is_centred_on_the_deterministic_mean(
    model: StochasticEMShowerModel,
    energy_mev: float,
) -> None:
    # This is the invariant the -0.5 s^2 term in mu(E) exists to guarantee:
    # the arithmetic mean of the lognormal T0 equals the deterministic mean
    # depth, so the stochastic model does not bias the accepted mean model.
    rng = np.random.default_rng(20260921)
    draws = np.array(
        [
            model.sample_shower_max_depth_x0(energy_mev, rng)
            for _ in range(40_000)
        ]
    )
    expected_mean_x0 = model.longitudinal.shower_max_depth_x0(energy_mev)
    standard_error = draws.std(ddof=1) / len(draws) ** 0.5

    assert abs(draws.mean() - expected_mean_x0) < 4.0 * standard_error


def test_relative_spread_of_shower_maximum_matches_the_width(
    model: StochasticEMShowerModel,
) -> None:
    energy_mev = 100_000.0
    rng = np.random.default_rng(1)
    draws = np.array(
        [
            model.sample_shower_max_depth_x0(energy_mev, rng)
            for _ in range(40_000)
        ]
    )
    # For a lognormal, sd(ln T0) is the configured width by construction.
    assert np.log(draws).std(ddof=1) == pytest.approx(
        model.fluctuation_width(energy_mev),
        rel=0.05,
    )


def test_an_uncentred_lognormal_would_have_been_biased_deep(
    model: StochasticEMShowerModel,
) -> None:
    # Documents why the centring term is not cosmetic: drawing with
    # mu = ln(T_bar) would place the mean at T_bar * exp(s^2 / 2).
    energy_mev = 100_000.0
    width = model.fluctuation_width(energy_mev)
    mean_depth_x0 = model.longitudinal.shower_max_depth_x0(energy_mev)

    assert model.lognormal_location(energy_mev) < np.log(mean_depth_x0)
    assert exp(0.5 * width**2) > 1.0


# --- one event ---------------------------------------------------------


def test_generates_a_valid_eighteen_by_seventytwo_event(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    event = make_event(model, track, random_seed=11)
    geometry = model.geometry

    assert isinstance(event, ECALEvent)
    assert len(event.cell_energies_mev) == geometry.number_of_layers

    for layer_energies_mev in event.cell_energies_mev:
        assert len(layer_energies_mev) == geometry.cells_per_layer
        assert all(
            isfinite(energy_mev) and energy_mev >= 0.0
            for energy_mev in layer_energies_mev
        )


def test_event_energy_never_exceeds_the_primary_energy(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # Fractions are never renormalized, so the deficit is physical leakage.
    energy_mev = 100_000.0
    events = tuple(
        make_event(model, track, random_seed=seed, energy_mev=energy_mev)
        for seed in range(30)
    )

    assert all(event.total_ecal_energy_mev <= energy_mev for event in events)
    assert all(event.total_ecal_energy_mev > 0.0 for event in events)


def test_records_its_own_seed_and_configuration_in_provenance(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    event = make_event(model, track, random_seed=4242)

    assert event.provenance.simulation_backend == "fastmc"
    assert event.provenance.random_seed == 4242
    assert event.provenance.configuration_sha256 == VALID_SHA256


# --- energy accounting -------------------------------------------------


def accounting(
    model: StochasticEMShowerModel,
    track: TrackState,
    random_seed: int,
    energy_mev: float = 100_000.0,
) -> tuple[float, float, float, float]:
    """Split one event's primary energy into its three destinations.

    Replays the event's own draw from its seed, so the split is computed from
    the same sampled shape parameter the event used, independently of the
    event's cell grid. Returns (alpha, contained, longitudinal, lateral).
    """

    rng = np.random.default_rng(random_seed)
    alpha = model.sample_shape_parameter(energy_mev, rng)
    layer_fractions = model.longitudinal.layer_energy_fractions_for_shape(alpha)
    cell_fractions = model.lateral.track_centered_cell_fractions(
        track, energy_mev
    )

    longitudinal_leak = energy_mev * (1.0 - sum(layer_fractions))
    lateral_leak = sum(
        energy_mev * layer_fraction * (1.0 - sum(row))
        for layer_fraction, row in zip(
            layer_fractions, cell_fractions, strict=True
        )
    )
    contained = sum(
        energy_mev * layer_fraction * cell_fraction
        for layer_fraction, row in zip(
            layer_fractions, cell_fractions, strict=True
        )
        for cell_fraction in row
    )
    return alpha, contained, longitudinal_leak, lateral_leak


def test_primary_energy_is_contained_or_leaks_and_nothing_else(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    energy_mev = 100_000.0

    for seed in range(20):
        event = make_event(model, track, random_seed=seed, energy_mev=energy_mev)
        _, contained, longitudinal, lateral = accounting(
            model, track, seed, energy_mev
        )

        assert event.total_ecal_energy_mev == pytest.approx(contained, rel=1e-12)
        assert longitudinal > 0.0
        assert lateral > 0.0
        assert contained + longitudinal + lateral == pytest.approx(
            energy_mev, rel=1e-12
        )


def test_longitudinal_leakage_is_the_gamma_tail_beyond_the_detector(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # The leak is the event's own gamma profile integrated from the back face
    # to infinity, at the event's sampled shape, not the mean model's tail.
    energy_mev = 100_000.0
    back_face = model.longitudinal.config.gamma_rate * (
        model.geometry.total_depth_x0
    )

    for seed in range(20):
        alpha, _, longitudinal, _ = accounting(model, track, seed, energy_mev)

        assert longitudinal / energy_mev == pytest.approx(
            float(gammaincc(alpha, back_face)), rel=1e-9, abs=1e-15
        )


def test_lateral_leakage_grows_as_the_track_nears_the_side(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # Same seed, so the same longitudinal draw: only the entry point differs.
    near_side = TrackState(
        x0_mm=310.0, y0_mm=310.0, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0
    )
    _, central_contained, central_long, central_lat = accounting(
        model, track, 7
    )
    _, side_contained, side_long, side_lat = accounting(model, near_side, 7)

    assert side_long == pytest.approx(central_long, rel=1e-12)
    assert side_lat > 10.0 * central_lat
    assert side_contained < central_contained


# --- reproducibility ---------------------------------------------------


def test_identical_seeds_reproduce_identical_events(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    first = make_event(model, track, random_seed=99)
    second = make_event(model, track, random_seed=99)

    assert first.cell_energies_mev == second.cell_energies_mev


def test_different_seeds_produce_different_events(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    first = make_event(model, track, random_seed=1)
    second = make_event(model, track, random_seed=2)

    assert first.cell_energies_mev != second.cell_energies_mev


def test_event_reproducibility_is_independent_of_generation_order(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # Each event seeds its own generator, so an event produced third in one
    # run is identical to the same event produced first in another.
    in_sequence = [
        make_event(model, track, random_seed=seed) for seed in (5, 6, 7)
    ]
    alone = make_event(model, track, random_seed=7)

    assert in_sequence[2].cell_energies_mev == alone.cell_energies_mev


def test_spawned_seeds_are_distinct_and_reproducible(
    model: StochasticEMShowerModel,
) -> None:
    seeds = model.spawn_event_seeds(base_seed=2026, count=64)

    assert len(set(seeds)) == 64
    assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    assert seeds == model.spawn_event_seeds(base_seed=2026, count=64)
    assert seeds != model.spawn_event_seeds(base_seed=2027, count=64)


def test_generates_an_ensemble_of_independently_seeded_events(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    count = 12
    events = tuple(
        model.generate_events(
            primary_energies_mev=(100_000.0,) * count,
            tracks=(track,) * count,
            base_seed=7,
            simulation_version=SIMULATION_VERSION,
            configuration_sha256=VALID_SHA256,
        )
    )

    assert len(events) == count
    assert len({event.event_id for event in events}) == count
    assert len({event.provenance.random_seed for event in events}) == count
    assert len({event.total_ecal_energy_mev for event in events}) == count


def test_rejects_mismatched_ensemble_inputs(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    with pytest.raises(ValueError, match="same length"):
        tuple(
            model.generate_events(
                primary_energies_mev=(1.0e5, 1.0e5),
                tracks=(track,),
                base_seed=0,
                simulation_version=SIMULATION_VERSION,
                configuration_sha256=VALID_SHA256,
            )
        )


# --- the fluctuation is real, and only longitudinal --------------------


def test_shower_depth_and_containment_genuinely_fluctuate(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    events = tuple(
        make_event(model, track, random_seed=seed) for seed in range(60)
    )
    layer_profiles = np.array([event.layer_energies_mev for event in events])
    deepest_layer = layer_profiles.argmax(axis=1)
    contained = layer_profiles.sum(axis=1)

    assert len(set(deepest_layer.tolist())) > 1
    assert contained.std(ddof=1) > 0.0


def test_lateral_shape_stays_deterministic_within_every_layer(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # EM event generator deliberately fluctuates only the longitudinal weights, so two
    # events differ layer-by-layer only by one scalar factor per layer.
    energy_mev = 100_000.0
    first = make_event(model, track, random_seed=3, energy_mev=energy_mev)
    second = make_event(model, track, random_seed=4, energy_mev=energy_mev)
    expected = model.lateral.track_centered_cell_fractions(track, energy_mev)

    for layer_index, layer_fractions in enumerate(expected):
        layer_total = sum(first.cell_energies_mev[layer_index])
        assert layer_total > 0.0
        recovered = tuple(
            cell / layer_total for cell in first.cell_energies_mev[layer_index]
        )
        normalized_reference = tuple(
            fraction / sum(layer_fractions) for fraction in layer_fractions
        )
        assert recovered == pytest.approx(normalized_reference, rel=1e-9)

    ratios = [
        sum(first.cell_energies_mev[index])
        / sum(second.cell_energies_mev[index])
        for index in range(model.geometry.number_of_layers)
    ]
    assert len({round(ratio, 9) for ratio in ratios}) > 1


# --- the deterministic limit -------------------------------------------


def test_deterministic_limit_is_approached_in_proportion_to_the_width(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # Driving s(E) towards zero must recover the mean model. The convergence
    # is a limit, not an identity, so the honest check is that the residual
    # falls in proportion to the forced width: that proves the residual is the
    # width propagating through the profile rather than a numerical defect.
    energy_mev = 100_000.0
    deterministic_layer_energies = tuple(
        mean_energy_mev * sum(layer_fractions)
        for mean_energy_mev, layer_fractions in zip(
            model.longitudinal.mean_layer_energies_mev(energy_mev),
            model.lateral.track_centered_cell_fractions(track, energy_mev),
            strict=True,
        )
    )

    def residual(forced_width: float) -> float:
        degenerate = StochasticEMShowerModel(
            config=StochasticEMConfig(
                deposition_width_intercept=1.0 / forced_width,
                deposition_width_log_slope=0.0,
                sampling_width_intercept=1.0 / forced_width,
                sampling_width_log_slope=0.0,
            ),
            longitudinal=model.longitudinal,
            lateral=model.lateral,
        )
        worst = 0.0
        for seed in range(5):
            event = degenerate.generate_event(
                event_id=f"limit-{seed}",
                primary_energy_mev=energy_mev,
                track=track,
                random_seed=seed,
                simulation_version=SIMULATION_VERSION,
                configuration_sha256=VALID_SHA256,
            )
            worst = max(
                worst,
                max(
                    abs(actual - expected) / expected
                    for actual, expected in zip(
                        event.layer_energies_mev,
                        deterministic_layer_energies,
                        strict=True,
                    )
                ),
            )
        return worst

    wide, narrow = residual(1.0e-9), residual(1.0e-12)

    assert narrow < 1.0e-9
    # A thousand-fold narrower width must give a thousand-fold smaller residual.
    assert 0.5 < (wide / narrow) / 1.0e3 < 2.0


# --- construction guards -----------------------------------------------


def test_accepts_an_equivalent_geometry_loaded_twice(
    model: StochasticEMShowerModel,
) -> None:
    other_geometry = load_geometry(GEOMETRY_CONFIG_PATH)
    lateral_config = load_fastmc_config(FASTMC_CONFIG_PATH).lateral_em
    equivalent = AMSLateralShowerModel(
        config=lateral_config,
        geometry=other_geometry,
    )

    # An identical geometry loaded twice is still the same detector, so this
    # must be accepted; only a genuinely different detector is an error.
    StochasticEMShowerModel(
        config=model.config,
        longitudinal=model.longitudinal,
        lateral=equivalent,
    )


@pytest.mark.parametrize(
    ("random_seed", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
    ],
)
def test_rejects_invalid_event_seeds(
    model: StochasticEMShowerModel,
    track: TrackState,
    random_seed: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="random_seed"):
        model.generate_event(
            event_id="invalid-seed",
            primary_energy_mev=100_000.0,
            track=track,
            random_seed=random_seed,
            simulation_version=SIMULATION_VERSION,
            configuration_sha256=VALID_SHA256,
        )


def test_rejects_a_non_generator_random_source(
    model: StochasticEMShowerModel,
) -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        model.sample_shower_max_depth_x0(100_000.0, np.random.RandomState(0))


# --- the perfect event -------------------------------------------------


def test_true_deposition_variant_switches_both_mean_and_width(
    model: StochasticEMShowerModel,
) -> None:
    # The two regimes are a matched pair: a deposition mean must not be used
    # with a sampling width, so switching regime has to move both.
    perfect = model.true_deposition()
    energy_mev = 100_000.0

    assert model.regime == "sampling"
    assert perfect.regime == "deposition"

    # The perfect event peaks deeper, because no sampling shift is applied.
    assert perfect.longitudinal.shower_max_depth_x0(
        energy_mev
    ) > model.longitudinal.shower_max_depth_x0(energy_mev)

    # And fluctuates less, because sampling adds shape fluctuation.
    assert perfect.fluctuation_width(energy_mev) < model.fluctuation_width(
        energy_mev
    )

    assert perfect.longitudinal.sampling_shower_max_correction_x0 == 0.0
    assert perfect.longitudinal.describes_true_deposition


def test_sampling_correction_is_energy_independent_and_negative(
    model: StochasticEMShowerModel,
) -> None:
    correction = model.longitudinal.sampling_shower_max_correction_x0

    assert correction < 0.0
    assert model.true_deposition().longitudinal.sampling_shower_max_correction_x0 == 0.0

    for energy_mev in (1.0e3, 1.0e5, 1.0e6):
        gap = model.true_deposition().longitudinal.shower_max_depth_x0(
            energy_mev
        ) - model.longitudinal.shower_max_depth_x0(energy_mev)
        assert gap == pytest.approx(0.665, abs=5.0e-3)


def test_same_seed_recovers_the_perfect_event_behind_a_sampled_one(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    # The whole point of the regime switch: the seed carries the randomness,
    # the regime carries the physics, so one seed names a pair of events.
    perfect = model.true_deposition()
    energy_mev = 100_000.0

    for seed in (0, 7, 20260928):
        sampled_x0 = model.sample_shower_max_depth_x0(
            energy_mev, np.random.default_rng(seed)
        )
        perfect_x0 = perfect.sample_shower_max_depth_x0(
            energy_mev, np.random.default_rng(seed)
        )

        # Same underlying standard normal, so the same quantile in both.
        sampled_z = (
            np.log(sampled_x0) - model.lognormal_location(energy_mev)
        ) / model.fluctuation_width(energy_mev)
        perfect_z = (
            np.log(perfect_x0) - perfect.lognormal_location(energy_mev)
        ) / perfect.fluctuation_width(energy_mev)

        assert sampled_z == pytest.approx(perfect_z, abs=1.0e-12)


def test_perfect_event_is_reproducible_and_differs_from_the_sampled_one(
    model: StochasticEMShowerModel,
    track: TrackState,
) -> None:
    perfect = model.true_deposition()

    def build(m: StochasticEMShowerModel) -> ECALEvent:
        return m.generate_event(
            event_id="perfect-vs-sampled",
            primary_energy_mev=100_000.0,
            track=track,
            random_seed=31337,
            simulation_version=SIMULATION_VERSION,
            configuration_sha256=VALID_SHA256,
        )

    first, second = build(perfect), build(perfect)

    assert first.cell_energies_mev == second.cell_energies_mev
    assert first.cell_energies_mev != build(model).cell_energies_mev
    # A shallower sampled shower leaks less out of the back, so the perfect
    # event should contain no more than the sampled one at the same seed.
    assert first.total_ecal_energy_mev <= build(model).total_ecal_energy_mev
