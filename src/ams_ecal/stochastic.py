"""Stochastic electromagnetic event generation for the AMS ECAL FastMC.

Blocks 4 and 5 produce the *mean* electromagnetic shower: at a given primary
energy and track, every event is identical. This module adds the first source
of event-to-event variation in the project.

The accepted Block 6A model gives the whole longitudinal fluctuation to a
single random variable, the depth of shower maximum ``T0``:

    T_bar(E) = ln(E / E_c) + offset(regime)
    s(E)     = 1 / (intercept(regime) + slope(regime) * ln(E / E_c))
    mu(E)    = ln(T_bar(E)) - 0.5 * s(E)**2
    ln(T0)   ~ Normal(mu(E), s(E)**2)
    alpha    = 1 + beta * T0

``beta`` stays fixed at the AMS-reported value 0.65. This matches AMS: the
published AMS form is ``alpha = 1 + b*T0`` with ``b`` fixed for all showers and
all energies and ``T0`` fitted per shower (Kounine et al., NIM A 869 (2017)
110-117, p. 113; Aguilar et al., Physics Reports 894 (2021), section 1.7.1).
The origin is detector-entry referenced: no shower-start depth is sampled.

The ``-0.5 s**2`` term in ``mu`` is the centring that makes
``E[T0] = T_bar(E)`` exactly, so the stochastic model reduces on average to the
mean model rather than biasing it deep by ``exp(s**2 / 2)``.

REGIMES. ``offset`` and the width coefficients are a MATCHED PAIR selected by
the configured regime; mixing one regime's mean with another's width is
incoherent.

    deposition : true energy deposition in the lead/fibre composite, the
                 "perfect event". No detector effect of any kind is applied.
    sampling   : signal-level longitudinal shape, already distorted by the
                 fact that only the fibres are read out. Peaks shallower
                 because e/mip falls with depth, and fluctuates more because
                 sampling adds longitudinal shape fluctuation.

SEED CORRESPONDENCE. ``sample_shower_max_depth_x0`` draws one standard normal
variate from the generator and only then applies the regime transformation, so
one seed names a CORRESPONDING PAIR of events. ``true_deposition()`` returns
the perfect event lying behind any sampled event of the same seed. That is what
keeps the Block 6/7 boundary usable.

PROVENANCE. ``beta`` is AMS-specific evidence. Both width laws and the sampling
depth shift are TRANSFERRED APPROXIMATIONS from Grindhammer and Peters,
arXiv:hep-ex/0001020, appendices A.1.2, A.2.2 and A.2.3; no AMS-specific
fitted fluctuation width is known to this project, and AMS publishes no
mean-depth formula at all. Whether the transfer is adequate is to be tested
later against detailed Geant4 transport.

SCOPE. The lateral profile is deliberately left deterministic here, so every
event-to-event difference enters through the longitudinal shape. A stochastic
lateral model, a fluctuating ``beta``, an explicit shower-start variable, and
independent per-layer noise are all explicitly excluded from Block 6A.

BLOCK 7 BOUNDARY. Detector impurities - photostatistics, light attenuation,
noise, thresholds, gains, saturation and dead channels - are Block 7 and are
applied nowhere in this module, in either regime. Under ``regime: sampling``
the longitudinal SHAPE has already been moved to signal level, so a Block 7
model must not re-apply the depth shift or the extra shape fluctuation that
Grindhammer and Peters section 3.2 attributes to sampling. Under
``regime: deposition`` nothing detector-related has been applied and Block 7
owns all of it; that is the cleaner base to build Block 7 against.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite, log

import numpy as np

from ams_ecal.event import ECALEvent, EnergyGrid, EventProvenance
from ams_ecal.fastmc_config import ShowerRegime, StochasticEMConfig
from ams_ecal.geometry import ECALGeometry
from ams_ecal.lateral import AMSLateralShowerModel
from ams_ecal.longitudinal import (
    AMSLongitudinalGammaModel,
    ElectromagneticParticleType,
)
from ams_ecal.tracking import TrackState


def _validate_seed(random_seed: object, name: str = "random_seed") -> int:
    """Return a nonnegative integer seed, matching EventProvenance's rule."""

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError(f"{name} must be an integer")

    if random_seed < 0:
        raise ValueError(f"{name} must be nonnegative")

    return random_seed


@dataclass(frozen=True, slots=True)
class StochasticEMShowerModel:
    """Event-by-event electromagnetic shower generator for the AMS ECAL.

    Composes the two deterministic Block 4/5 models rather than reimplementing
    them: the gamma profile is integrated at a *sampled* shape parameter, and
    the mean lateral grid is applied unchanged to the resulting layer energies.
    """

    config: StochasticEMConfig
    longitudinal: AMSLongitudinalGammaModel
    lateral: AMSLateralShowerModel

    def __post_init__(self) -> None:
        if not isinstance(self.config, StochasticEMConfig):
            raise TypeError("config must be a StochasticEMConfig")

        if not isinstance(self.longitudinal, AMSLongitudinalGammaModel):
            raise TypeError(
                "longitudinal must be an AMSLongitudinalGammaModel"
            )

        if not isinstance(self.lateral, AMSLateralShowerModel):
            raise TypeError("lateral must be an AMSLateralShowerModel")

        # One event describes one detector. Two models built from different
        # geometries would silently produce an incoherent event.
        if self.longitudinal.geometry != self.lateral.geometry:
            raise ValueError(
                "longitudinal and lateral models must share one ECALGeometry"
            )

    @property
    def geometry(self) -> ECALGeometry:
        """Return the detector geometry shared by both component models."""

        return self.longitudinal.geometry

    @property
    def regime(self) -> ShowerRegime:
        """Return the regime of the underlying longitudinal model."""

        return self.longitudinal.regime

    def as_regime(self, regime: ShowerRegime) -> StochasticEMShowerModel:
        """Return the same model reading the other regime of constants.

        Generating an event from this model and from ``as_regime(...)`` with
        the SAME seed yields corresponding events: both consume one standard
        normal variate from that seed, so they describe the same underlying
        shower fluctuation expressed in the two regimes. In particular

            model.as_regime("deposition").generate_event(..., random_seed=s)

        recovers the perfect, true-deposition event behind the sampled event
        that the same seed produced. Detector impurities - noise, thresholds,
        gains, dead channels - remain Block 7 and are applied to neither.
        """

        return replace(
            self,
            longitudinal=replace(self.longitudinal, regime=regime),
        )

    def true_deposition(self) -> StochasticEMShowerModel:
        """Return the perfect-event variant of this model."""

        return self.as_regime("deposition")

    @property
    def critical_energy_mev(self) -> float:
        """Return the effective ECAL critical energy from detector geometry."""

        return self.longitudinal.critical_energy_mev

    def fluctuation_width(self, primary_energy_mev: float) -> float:
        """Return the standard deviation of ``ln(T0)`` at one primary energy.

        The width law has a pole: its denominator vanishes, and then changes
        sign, at ``E / E_c = exp(-intercept / slope)``. Below that energy the
        parametrization does not describe a fluctuation at all, so this raises
        rather than returning a meaningless or negative width.
        """

        # Reuse the mean model's own energy validation and domain check. A
        # positive mean depth is also the precondition for taking its
        # logarithm in lognormal_location below.
        self.longitudinal.shower_max_depth_x0(primary_energy_mev)

        scaled_energy_log = log(
            float(primary_energy_mev) / self.critical_energy_mev
        )
        if self.longitudinal.describes_true_deposition:
            intercept = self.config.deposition_width_intercept
            slope = self.config.deposition_width_log_slope
        else:
            intercept = self.config.sampling_width_intercept
            slope = self.config.sampling_width_log_slope

        denominator = intercept + slope * scaled_energy_log

        if not isfinite(denominator) or denominator <= 0:
            raise ValueError(
                "primary_energy_mev is below the valid domain of the "
                "stochastic shower-maximum width law"
            )

        width = 1.0 / denominator

        if not isfinite(width) or width <= 0:
            raise ArithmeticError(
                "computed shower-maximum fluctuation width must be positive"
            )

        return width

    def lognormal_location(self, primary_energy_mev: float) -> float:
        """Return ``mu(E)``, the mean of ``ln(T0)``.

        Subtracting ``s**2 / 2`` is the centring that makes the *arithmetic*
        mean of the lognormal variable equal the deterministic mean depth. A
        lognormal drawn with ``mu = ln(T_bar)`` instead would be biased deep by
        a factor ``exp(s**2 / 2)``.
        """

        width = self.fluctuation_width(primary_energy_mev)
        mean_shower_max_x0 = self.longitudinal.shower_max_depth_x0(
            primary_energy_mev
        )
        return log(mean_shower_max_x0) - 0.5 * width**2

    def sample_shower_max_depth_x0(
        self,
        primary_energy_mev: float,
        rng: np.random.Generator,
    ) -> float:
        """Draw one event's shower-maximum depth ``T0`` in radiation lengths."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        # Draw the standard normal variate FIRST, then apply the
        # regime-dependent transformation. Doing it in this order is what
        # makes a seed reproduce corresponding events in both regimes: the
        # randomness belongs to the seed, the physics to the regime. Calling
        # rng.lognormal directly would work only by relying on an internal
        # detail of numpy, which is not something to depend on silently.
        standard_normal = float(rng.standard_normal())
        shower_max_x0 = exp(
            self.lognormal_location(primary_energy_mev)
            + self.fluctuation_width(primary_energy_mev) * standard_normal
        )

        if not isfinite(shower_max_x0) or shower_max_x0 <= 0:
            raise ArithmeticError(
                "sampled shower-maximum depth must be finite and positive"
            )

        return shower_max_x0

    def sample_shape_parameter(
        self,
        primary_energy_mev: float,
        rng: np.random.Generator,
    ) -> float:
        """Draw one event's gamma shape parameter ``alpha = 1 + beta * T0``."""

        return 1.0 + self.longitudinal.config.gamma_rate * (
            self.sample_shower_max_depth_x0(primary_energy_mev, rng)
        )

    def sample_layer_energy_fractions(
        self,
        primary_energy_mev: float,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        """Draw one event's 18 longitudinal energy fractions.

        The fractions are not renormalized to the finite detector depth, so
        their deficit from unity remains physical longitudinal leakage.
        """

        return self.longitudinal.layer_energy_fractions_for_shape(
            self.sample_shape_parameter(primary_energy_mev, rng)
        )

    def sample_layer_energies_mev(
        self,
        primary_energy_mev: float,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        """Draw one event's 18 longitudinal layer energies in MeV."""

        energy_mev = float(primary_energy_mev)
        return tuple(
            energy_mev * fraction
            for fraction in self.sample_layer_energy_fractions(
                primary_energy_mev,
                rng,
            )
        )

    def sample_cell_energies_mev(
        self,
        primary_energy_mev: float,
        track: TrackState,
        rng: np.random.Generator,
    ) -> EnergyGrid:
        """Draw one event's full 18 x 72 ideal energy deposition.

        The lateral grid is the deterministic Block 5 result: only the
        longitudinal weights fluctuate. Lateral leakage is preserved because
        the mean cell fractions are themselves unnormalized.
        """

        layer_energies_mev = self.sample_layer_energies_mev(
            primary_energy_mev,
            rng,
        )
        lateral_fractions = self.lateral.track_centered_cell_fractions(
            track,
            primary_energy_mev,
        )

        return tuple(
            tuple(
                layer_energy_mev * cell_fraction
                for cell_fraction in layer_fractions
            )
            for layer_energy_mev, layer_fractions in zip(
                layer_energies_mev,
                lateral_fractions,
                strict=True,
            )
        )

    def generate_event(
        self,
        *,
        event_id: str,
        primary_energy_mev: float,
        track: TrackState,
        random_seed: int,
        simulation_version: str,
        configuration_sha256: str,
        particle_type: ElectromagneticParticleType = "electron",
    ) -> ECALEvent:
        """Generate one reproducible stochastic electromagnetic event.

        The event builds its own generator from ``random_seed``, so it is
        reproducible from the seed recorded in its own provenance, independent
        of how many events were generated before it.
        """

        seed = _validate_seed(random_seed)
        rng = np.random.default_rng(seed)

        return ECALEvent(
            event_id=event_id,
            particle_type=particle_type,
            primary_energy_mev=float(primary_energy_mev),
            track=track,
            geometry=self.geometry,
            cell_energies_mev=self.sample_cell_energies_mev(
                primary_energy_mev,
                track,
                rng,
            ),
            provenance=EventProvenance(
                simulation_backend="fastmc",
                simulation_version=simulation_version,
                configuration_sha256=configuration_sha256,
                random_seed=seed,
            ),
        )

    def spawn_event_seeds(
        self,
        base_seed: int,
        count: int,
    ) -> tuple[int, ...]:
        """Derive ``count`` independent per-event seeds from one base seed.

        SeedSequence spawning gives streams that are statistically independent
        by construction, which simple seed arithmetic such as ``base + index``
        does not guarantee. Recording each event's own seed keeps every event
        individually reproducible.
        """

        seed = _validate_seed(base_seed, "base_seed")

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")

        if count < 0:
            raise ValueError("count must be nonnegative")

        children = np.random.SeedSequence(seed).spawn(count)
        # A uint32 state word is already a nonnegative integer below 2**32,
        # which is what both default_rng and EventProvenance accept. It is
        # converted to a Python int so the seed stays platform independent.
        return tuple(
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in children
        )

    def generate_events(
        self,
        *,
        primary_energies_mev: Sequence[float],
        tracks: Sequence[TrackState],
        base_seed: int,
        simulation_version: str,
        configuration_sha256: str,
        particle_type: ElectromagneticParticleType = "electron",
        event_id_prefix: str = "fastmc-em",
    ) -> Iterator[ECALEvent]:
        """Generate an ensemble of independently seeded events.

        ``primary_energies_mev`` and ``tracks`` are zipped strictly, so an
        energy list and a track list of different lengths is an error rather
        than a silently truncated dataset.
        """

        energies = tuple(primary_energies_mev)
        track_states = tuple(tracks)

        if len(energies) != len(track_states):
            raise ValueError(
                "primary_energies_mev and tracks must have the same length"
            )

        seeds = self.spawn_event_seeds(base_seed, len(energies))

        for index, (energy_mev, track, seed) in enumerate(
            zip(energies, track_states, seeds, strict=True)
        ):
            yield self.generate_event(
                event_id=f"{event_id_prefix}-{index:06d}",
                primary_energy_mev=energy_mev,
                track=track,
                random_seed=seed,
                simulation_version=simulation_version,
                configuration_sha256=configuration_sha256,
                particle_type=particle_type,
            )
