"""Geant4-calibrated proton phenomenology for the AMS ECAL FastMC (Block 6B).

WORDING. This is a Geant4-DERIVED PHENOMENOLOGY of a proton in the thin AMS-like
ECAL (17 X0, ~0.6 lambda_I): a calibration of Geant4 11.4.1 in this project's
material model over 10-100 GeV at normal incidence. It is not a true proton
shower model, not the exact AMS proton response, and its effective interaction
length is not a physical interaction length of AMS.

WHAT IS IMPLEMENTED (Slices 1-2 of the plan)

* THE INTERACTION DRAW. One uniform variate gives ``S_int ~ Exponential(lambda)``
  by inversion. ``S_int >= L`` (the path length through the ECAL) means the
  proton crosses without an inelastic interaction; otherwise the first
  interaction is at depth ``S_int``. Status and depth come from one draw, so
  they cannot disagree with the survival law ``exp(-z / lambda)``.
* THE CROSSING PROTON. Exact fibre geometry (``ams_ecal.crossing``) gives the
  chord the straight track cuts in each layer; each layer's energy is then a
  draw from the calibrated quantile function conditioned on that chord and on
  the primary energy, and is spread over cells in proportion to chord.

WHAT IS NOT YET IMPLEMENTED: interacting events (Slices 3-5). They wait for the
researcher's decision on the interacting-event factorization
(``research/plans/2026-09-29_block6b_slice0_dependency_analysis.md``, D1).
``generate_event`` therefore raises ``NotImplementedError`` for an interacting
draw rather than returning something wrong; ``generate_crossing_event`` forces
the crossing branch and is what the crossing validation uses.

KNOWN LIMITATIONS OF THE CROSSING BRANCH (Slice 0 / Slice 2 findings)

* Layers are drawn INDEPENDENTLY. The real event total has 1.5x (10 GeV) to
  5.5x (100 GeV) the summed layer variance, because rare bursts (a hard
  delta-ray, a small cascade) span several layers. The event-total tail is
  therefore under-represented; the minimal additional variable is an
  event-level burst.
* The two representations (``readout`` and ``deposition``) draw their layer
  fluctuations independently from the same seed: the real per-layer rank
  correlation of a crossing proton is only 0.15-0.24. What one seed shares
  between them is the interaction draw.

REPRODUCIBILITY. Each event builds its own generator from its seed. Draw order:
one variate for the interaction, then 18 for the ``readout`` layers, then 18
for the ``deposition`` layers, whichever representation is generated.
"""

from dataclasses import dataclass, replace
from math import cos, log1p
from pathlib import Path

import numpy as np

from ams_ecal.crossing import CrossedFibres, FibreCrossingGeometry
from ams_ecal.event import ECALEvent, EventProvenance
from ams_ecal.fastmc_config import config_digest
from ams_ecal.geant4_backend import PROJECT_ROOT
from ams_ecal.geometry import ECALGeometry, load_geometry
from ams_ecal.proton_calibration import ProtonCalibration
from ams_ecal.proton_config import (
    ProtonConfig,
    ProtonRepresentation,
    load_proton_config,
)
from ams_ecal.tracking import TrackState

MODEL_NAME = "block6b-proton"
MODEL_VERSION = "1-slice2"  # crossing branch only

PROTON_CONFIG = PROJECT_ROOT / "configs" / "fastmc_proton.yaml"
GEOMETRY_CONFIG = PROJECT_ROOT / "configs" / "geometry.yaml"


@dataclass(frozen=True, slots=True)
class InteractionDraw:
    """Outcome of the single interaction draw for one proton."""

    depth_mm: float | None  # first-inelastic depth along the path; None = crosses
    path_mm: float  # path length through the ECAL

    @property
    def interacts(self) -> bool:
        return self.depth_mm is not None


def spawn_event_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Derive ``count`` independent per-event seeds from one base seed.

    The same construction as Block 6A: ``SeedSequence`` spawning gives
    statistically independent streams, unlike ``base + index``.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    children = np.random.SeedSequence(base_seed).spawn(count)
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)


@dataclass(frozen=True, slots=True)
class ProtonShowerModel:
    """Block 6B proton generator (crossing branch)."""

    config: ProtonConfig
    calibration: ProtonCalibration
    crossing: FibreCrossingGeometry
    configuration_sha256: str

    def __post_init__(self) -> None:
        if self.config.calibration.physics_list != self.calibration.physics_list:
            raise ValueError(
                f"configuration asks for the {self.config.calibration.physics_list!r} "
                f"calibration but the artifact is {self.calibration.physics_list!r}"
            )
        low, high = self.calibration.crossing.energy_range_gev
        domain = self.config.domain
        if not (low <= domain.energy_min_gev and domain.energy_max_gev <= high):
            raise ValueError(
                f"the configured domain [{domain.energy_min_gev:g}, {domain.energy_max_gev:g}] "
                f"GeV exceeds the calibrated range [{low:g}, {high:g}] GeV"
            )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = PROTON_CONFIG,
        geometry_path: str | Path = GEOMETRY_CONFIG,
        root: Path = PROJECT_ROOT,
    ) -> ProtonShowerModel:
        """Load the configuration, its calibration artifact and the geometry."""

        config = load_proton_config(config_path)
        artifact = root / config.calibration.artifact
        calibration = ProtonCalibration.load(artifact)
        geometry = load_geometry(geometry_path)
        digest = config_digest([geometry_path, config_path, artifact / "manifest.json"])
        return cls(config, calibration, FibreCrossingGeometry(geometry), digest)

    @property
    def geometry(self) -> ECALGeometry:
        return self.crossing.geometry

    @property
    def representation(self) -> ProtonRepresentation:
        return self.config.representation

    def as_representation(self, representation: ProtonRepresentation) -> ProtonShowerModel:
        """Return the same model generating the other representation.

        One seed gives the same interaction draw in both; the layer
        fluctuations are independent (see the module docstring).
        """

        return replace(self, config=replace(self.config, representation=representation))

    @property
    def effective_length_mm(self) -> float:
        override = self.config.interaction.effective_length_mm
        return self.calibration.effective_length_mm if override is None else override

    @property
    def effective_length_source(self) -> str:
        return "calibration" if self.config.interaction.effective_length_mm is None else "configuration"

    # ------------------------------------------------------------------

    def _check_request(self, primary_energy_mev: float, track: TrackState) -> None:
        if not self.config.domain.contains_energy_mev(primary_energy_mev):
            domain = self.config.domain
            raise ValueError(
                f"primary energy {primary_energy_mev / 1000.0:g} GeV is outside the "
                f"calibrated domain [{domain.energy_min_gev:g}, {domain.energy_max_gev:g}] GeV"
            )
        if track.theta_rad > self.config.domain.max_theta_rad:
            raise ValueError(
                f"track angle {track.theta_rad:g} rad exceeds the calibrated "
                f"maximum {self.config.domain.max_theta_rad:g} rad"
            )

    def path_length_mm(self, track: TrackState) -> float:
        """Return the path the track takes through the ECAL depth."""

        return self.calibration.depth_mm / cos(track.theta_rad)

    def _draw_interaction(self, rng: np.random.Generator, track: TrackState) -> InteractionDraw:
        # inversion: S = -lambda ln(1 - U), U in [0, 1) so the argument is > 0
        distance = -self.effective_length_mm * log1p(-float(rng.random()))
        path = self.path_length_mm(track)
        return InteractionDraw(depth_mm=distance if distance < path else None, path_mm=path)

    def interaction_for_seed(
        self, primary_energy_mev: float, track: TrackState, random_seed: int
    ) -> InteractionDraw:
        """Return the interaction draw that ``random_seed`` names."""

        self._check_request(primary_energy_mev, track)
        return self._draw_interaction(np.random.default_rng(random_seed), track)

    # ------------------------------------------------------------------

    def _crossing_layer_energies(
        self, primary_energy_mev: float, track: TrackState, rng: np.random.Generator
    ) -> tuple[np.ndarray, CrossedFibres]:
        crossed = self.crossing.cross(track)
        chords = crossed.layer_path_mm(self.geometry.number_of_layers)
        uniforms = {
            name: rng.random(self.geometry.number_of_layers) for name in ("readout", "deposition")
        }
        energies = self.calibration.crossing.sample_layer_energies(
            self.representation, primary_energy_mev / 1000.0, chords, uniforms[self.representation]
        )
        return energies, crossed

    def _provenance(self, random_seed: int, status: str) -> EventProvenance:
        details = {
            "calibration_content_sha256": self.calibration.content_sha256,
            "calibration_schema_version": str(self.calibration.manifest["schema_version"]),
            "effective_length_mm": f"{self.effective_length_mm:.6g}",
            "effective_length_source": self.effective_length_source,
            "geant4_version": str(self.calibration.manifest["source"]["geant4_version"]),
            "interaction_status": status,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "physics_list": self.calibration.physics_list,
            "representation": self.representation,
        }
        return EventProvenance(
            simulation_backend="fastmc",
            simulation_version=f"{MODEL_NAME}-{MODEL_VERSION}",
            configuration_sha256=self.configuration_sha256,
            random_seed=random_seed,
            model_details=tuple(sorted(details.items())),
        )

    def generate_crossing_event(
        self,
        *,
        event_id: str,
        primary_energy_mev: float,
        track: TrackState,
        random_seed: int,
    ) -> ECALEvent:
        """Generate the event of a proton that crosses without interacting.

        The interaction variate is still consumed, so that the layer draws of
        an event are the same whether or not it is reached through
        ``generate_event``.
        """

        self._check_request(primary_energy_mev, track)
        rng = np.random.default_rng(random_seed)
        self._draw_interaction(rng, track)  # keeps the draw order fixed
        layer_energy, crossed = self._crossing_layer_energies(primary_energy_mev, track, rng)
        grid = self.crossing.spread_layer_energies(track, crossed, layer_energy)
        return ECALEvent(
            event_id=event_id,
            particle_type="proton",
            primary_energy_mev=float(primary_energy_mev),
            track=track,
            geometry=self.geometry,
            cell_energies_mev=tuple(tuple(float(v) for v in row) for row in grid),
            provenance=self._provenance(random_seed, "crossing"),
        )

    def generate_event(
        self,
        *,
        event_id: str,
        primary_energy_mev: float,
        track: TrackState,
        random_seed: int,
    ) -> ECALEvent:
        """Generate one proton event from its seed (crossing branch only)."""

        draw = self.interaction_for_seed(primary_energy_mev, track, random_seed)
        if draw.interacts:
            raise NotImplementedError(
                f"seed {random_seed} draws an interaction at {draw.depth_mm:.1f} mm; interacting "
                "events (Block 6B Slices 3-5) wait for the researcher's decision D1 in "
                "research/plans/2026-09-29_block6b_slice0_dependency_analysis.md"
            )
        return self.generate_crossing_event(
            event_id=event_id,
            primary_energy_mev=primary_energy_mev,
            track=track,
            random_seed=random_seed,
        )
