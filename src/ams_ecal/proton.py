"""Geant4-calibrated proton phenomenology for the AMS ECAL FastMC.

WORDING. This is a Geant4-DERIVED PHENOMENOLOGY of a proton in the thin AMS-like
ECAL (17 X0, ~0.6 lambda_I): a calibration of Geant4 11.4.1 in this project's
material model over 10-100 GeV at normal incidence. It is not a true proton
shower model, not the exact AMS proton response, and its effective interaction
length is not a physical interaction length of AMS.

WHAT IS IMPLEMENTED (interaction draw and crossing branch (steps 1-2) of the plan)

* THE INTERACTION DRAW. One uniform variate gives ``S_int ~ Exponential(lambda)``
  by inversion. ``S_int >= L`` (the path length through the ECAL) means the
  proton crosses without an inelastic interaction; otherwise the first
  interaction is at depth ``S_int``. Status and depth come from one draw, so
  they cannot disagree with the survival law ``exp(-z / lambda)``.
* THE CROSSING PROTON. Exact fibre geometry (``ams_ecal.crossing``) gives the
  chord the straight track cuts in each layer; each layer's energy is then a
  draw from the calibrated quantile function conditioned on that chord and on
  the primary energy, and is spread over cells in proportion to chord.

WHAT IS NOT YET IMPLEMENTED: interacting events (interacting-event model (steps 3-5)). They wait for the
researcher's decision on the interacting-event factorization
(``research/plans/2026-09-29_proton_dependency_analysis.md``, the interacting-proton factorization).
``generate_event`` therefore raises ``NotImplementedError`` for an interacting
draw rather than returning something wrong; ``generate_crossing_event`` forces
the crossing branch and is what the crossing validation uses.

TWO CROSSING PATHS, chosen by the calibration artifact.

* Schema-2 artifact with a crossing STRUCTURE (``ams_ecal.proton_structure``): a burst
  latent (onset, amplitude, downstream extent, fibre share), a lateral spill that splits a
  layer's energy over cells off the track, and a coupling of the bulk layer draws. One seed
  gives the SAME burst in both representations. Model version ``2-crossing-structure``.
  Calibrated and tested; NOT yet validated (the repaired branch is validated once, on the
  sealed Geant4 set).
* Schema-1 artifact (the per-layer table alone): layers are drawn INDEPENDENTLY and every
  layer's energy goes to the crossed cells. This is the branch whose first validation
  failed (event total, hit cells, maximum cell, containment): the real event total has 1.5x
  (10 GeV) to 5.5x (100 GeV) the summed layer variance because rare bursts span several
  layers. Kept so that the failed record stays reproducible. Model version ``1-slice2``.

REPRODUCIBILITY. Each event builds its own generator from its seed. Draw order after the
interaction variate: schema 1 draws 18 uniforms for the ``readout`` layers, then 18 for the
``deposition`` layers; schema 2 draws the burst uniforms and jitter normals, the bulk normals
(``readout`` then ``deposition``) and the spill uniforms (``readout`` then ``deposition``).
Both representations' draws are made whichever representation is generated.
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
from ams_ecal.proton_structure import (
    BURST_UNIFORMS,
    N_LAYERS,
    REPRESENTATIONS,
    SPILL_DRAWS,
    coupled_uniforms,
    reference_for,
    sample_bulk,
    sample_bursts,
    sample_spill,
)
from ams_ecal.tracking import TrackState

MODEL_NAME = "block6b-proton"
MODEL_VERSION = "1-slice2"  # crossing branch, layers drawn independently (schema 1 artifact)
MODEL_VERSION_STRUCTURED = "2-crossing-structure"  # crossing branch with burst, spill, bulk coupling

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

    The same construction as the EM event generator: ``SeedSequence`` spawning gives
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
    """Proton model generator (crossing branch)."""

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

    @property
    def model_version(self) -> str:
        return MODEL_VERSION if self.calibration.structure is None else MODEL_VERSION_STRUCTURED

    def _structured_crossing_grid(
        self, primary_energy_mev: float, track: TrackState, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, str]]:
        """Crossing event with the burst latent, the bulk coupling and the lateral spill.

        Draw order, fixed whichever representation is generated (after the interaction
        variate): burst uniforms, burst jitter normals, bulk normals for ``readout`` then
        ``deposition``, spill uniforms for ``readout`` then ``deposition``.
        """

        structure = self.calibration.structure
        if structure is None:
            raise ValueError("this calibration carries no crossing structure")
        table = self.calibration.crossing
        crossed = self.crossing.cross(track)
        chords = crossed.layer_path_mm(N_LAYERS)[None, :]
        bins = table.chord_bin(chords)
        energy_gev = np.array([primary_energy_mev / 1000.0])

        burst_uniforms = rng.random((1, BURST_UNIFORMS))
        burst_normals = rng.standard_normal((1, N_LAYERS))
        bulk_normals = {name: rng.standard_normal((1, N_LAYERS)) for name in REPRESENTATIONS}
        spill_uniforms = {name: rng.random((1, N_LAYERS, SPILL_DRAWS)) for name in REPRESENTATIONS}

        name = self.representation
        burst = sample_bursts(structure.burst, energy_gev, burst_uniforms, burst_normals)
        bulk = sample_bulk(
            table.quantiles_mev[name],
            table.levels,
            table.energies_gev,
            bins,
            energy_gev,
            coupled_uniforms(structure.coupling_factor(name), bulk_normals[name]),
        )
        total = bulk + burst.excess_mev[name]
        ratio = total / reference_for(
            structure.reference_median_mev[name], table.energies_gev, energy_gev, bins
        )
        spill = sample_spill(structure.spill[name], ratio, spill_uniforms[name])
        grid = self.crossing.place_layer_energies(
            track, crossed, total[0], spill.fraction[0], spill.offset[0], spill.weight[0], spill.cells[0]
        )
        latent = {
            "burst_onset_layer": str(int(burst.onset[0])) if burst.has_burst[0] else "none",
            "spill_layers": str(int((spill.cells[0] > 0).sum())),
        }
        return grid, latent

    def _provenance(
        self, random_seed: int, status: str, latent: dict[str, str] | None = None
    ) -> EventProvenance:
        details = {
            "calibration_content_sha256": self.calibration.content_sha256,
            "calibration_schema_version": str(self.calibration.manifest["schema_version"]),
            "effective_length_mm": f"{self.effective_length_mm:.6g}",
            "effective_length_source": self.effective_length_source,
            "geant4_version": str(self.calibration.manifest["source"]["geant4_version"]),
            "interaction_status": status,
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "physics_list": self.calibration.physics_list,
            "representation": self.representation,
        }
        details.update(latent or {})
        return EventProvenance(
            simulation_backend="fastmc",
            simulation_version=f"{MODEL_NAME}-{self.model_version}",
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
        latent: dict[str, str] | None = None
        if self.calibration.structure is None:
            layer_energy, crossed = self._crossing_layer_energies(primary_energy_mev, track, rng)
            grid = self.crossing.spread_layer_energies(track, crossed, layer_energy)
        else:
            grid, latent = self._structured_crossing_grid(primary_energy_mev, track, rng)
        return ECALEvent(
            event_id=event_id,
            particle_type="proton",
            primary_energy_mev=float(primary_energy_mev),
            track=track,
            geometry=self.geometry,
            cell_energies_mev=tuple(tuple(float(v) for v in row) for row in grid),
            provenance=self._provenance(random_seed, "crossing", latent),
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
                "events (proton model steps 3-5) wait for the researcher's decision on the interacting-proton factorization in "
                "research/plans/2026-09-29_proton_dependency_analysis.md"
            )
        return self.generate_crossing_event(
            event_id=event_id,
            primary_energy_mev=primary_energy_mev,
            track=track,
            random_seed=random_seed,
        )
