"""Geant4 detailed-transport backend for the AMS-like ECAL.

Scope: the thin vertical slice the Block 6B proton calibration pilot needs,
built so that the full RQ-001 study can grow from it rather than replace it.
Geant4 performs all particle transport in C++ through ``geant4_pybind``; this
module only builds the geometry, configures scoring, and reads each finished
event. Geant4 is a model of the detector, not detector truth.

WHAT ONE EVENT PRODUCES

* ``fibre_grid``   - energy deposited in the scintillating fibres, projected to
                     the canonical 18 x 72 readout. Sensitive-material
                     deposition: NOT a calibrated or digitized AMS signal.
* ``prefix_grid``  - energy deposited in ALL material of the AMS-like volume
                     (lead, glue, fibre), same 18 x 72 projection. Comparable
                     to FastMC's ``deposition`` regime.
* ``extended_grid``- only for the extended geometry: the prefix plus the
                     research-instrument extension, in the same alternating
                     readout continued to 270 layers.
* fine deposits    - per-fibre deposits and 3 x 3 x 4.625 mm voxel deposits of
                     the prefix, stored sparsely for later RQ-001 reuse.
* truth            - the primary's first inelastic interaction, by the
                     definition in ``ams_ecal.geant4_truth``.

Scoring never runs Python per step. Fibre deposits use a primitive scorer on
the fibre volume; all-material deposits use voxel meshes in a parallel world.
Python is called once per new track (to catch the primary's secondaries) and
once per event.

REPRODUCIBILITY. Every event gets its own seed, spawned from the batch seed and
the primary energy. The Geant4 engine and the entry-point draw are both seeded
from it at the start of the event, so an event is reproducible from its seed
alone, independently of which worker ran it or in which order. Every batch
writes ``metadata.json`` with the git commit, Geant4 and dataset versions,
physics list, configuration digest and material depths.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from math import cos, isfinite, sin
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ams_ecal.event import ECALEvent, EventProvenance
from ams_ecal.fastmc_config import config_digest
from ams_ecal.geant4_truth import SecondaryRecord, first_inelastic_interaction
from ams_ecal.geometry import ECALGeometry, load_geometry
from ams_ecal.projection import project_deposits
from ams_ecal.tracking import TrackState
from ams_ecal.transport_geometry import (
    FibreLayout,
    MeshGrid,
    TransportConfig,
    extended_layer_count,
    extension_mesh,
    load_transport_config,
    prefix_mesh,
    superlayer_composition,
)

OUTPUT_SCHEMA_VERSION = 1
EXPECTED_PILOT_SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_CONFIG = PROJECT_ROOT / "configs" / "geometry.yaml"
TRANSPORT_CONFIG = PROJECT_ROOT / "configs" / "geant4_transport.yaml"
PILOT_CONFIG = PROJECT_ROOT / "configs" / "geant4_proton_pilot.yaml"

GeometryVariant = Literal["ams", "extended"]

SEED_POLICY = (
    "per-event seeds = numpy SeedSequence([base_seed, round(energy_gev * 1000)])"
    ".spawn(n_events), first uint32 state word of each child shifted right by one"
    " bit (31-bit seeds fit Geant4's C long on Windows); the Geant4 engine"
    " (G4Random.setTheSeed) and the entry-point draw (numpy default_rng) are"
    " both seeded from the event seed at the start of the event"
)

# Geant4 dataset environment variables and the directory-name prefixes of the
# datasets they point to (the layout geant4_pybind downloads to).
DATASET_ENVIRONMENT = {
    "G4LEDATA": "G4EMLOW",
    "G4LEVELGAMMADATA": "PhotonEvaporation",
    "G4RADIOACTIVEDATA": "RadioactiveDecay",
    "G4PARTICLEXSDATA": "G4PARTICLEXS",
    "G4PIIDATA": "G4PII",
    "G4REALSURFACEDATA": "RealSurface",
    "G4SAIDXSDATA": "G4SAIDDATA",
    "G4ABLADATA": "G4ABLA",
    "G4INCLDATA": "G4INCL",
    "G4ENSDFSTATEDATA": "G4ENSDFSTATE",
    "G4CHANNELINGDATA": "G4CHANNELING",
    "G4NEUTRONHPDATA": "G4NDL",
}


def configure_geant4_data(data_dir: str | Path | None = None) -> dict[str, str]:
    """Point Geant4 at its datasets; call BEFORE importing geant4_pybind.

    Returns ``{environment variable: dataset directory name}`` for provenance.
    Setting ``GEANT4_DATA_DIR`` also stops geant4_pybind from prompting to
    download datasets interactively.
    """

    root = Path(
        data_dir or os.environ.get("GEANT4_DATA_DIR") or Path.home() / ".geant4_pybind"
    )
    found: dict[str, str] = {}
    for variable, prefix in DATASET_ENVIRONMENT.items():
        if variable in os.environ:
            found[variable] = Path(os.environ[variable]).name
            continue
        matches = sorted(p for p in root.glob(prefix + "[0-9]*") if p.is_dir())
        if matches:
            os.environ[variable] = str(matches[-1])
            found[variable] = matches[-1].name
    os.environ.setdefault("GEANT4_DATA_DIR", str(root))
    return found


def geant4_available() -> bool:
    """Return whether geant4_pybind is importable (without importing it)."""

    from importlib.util import find_spec

    return find_spec("geant4_pybind") is not None


# ----------------------------------------------------------------------
# Pilot configuration
# ----------------------------------------------------------------------


class PilotConfigError(ValueError):
    """Raised when the pilot configuration is malformed."""


@dataclass(frozen=True, slots=True)
class EntrySpot:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float


@dataclass(frozen=True, slots=True)
class SampleSpec:
    name: str
    geometry: GeometryVariant
    physics_list: str
    events_per_energy: int
    base_seed: int


@dataclass(frozen=True, slots=True)
class PilotConfig:
    particle: str
    energies_gev: tuple[float, ...]
    theta_rad: float
    phi_rad: float
    entry_spot: EntrySpot
    gun_z_mm: float
    physics_lists: tuple[str, ...]
    production_cut_mm: float
    samples: dict[str, SampleSpec]
    store_fine_prefix_deposits: bool
    output_dir: Path


def load_pilot_config(config_path: str | Path = PILOT_CONFIG) -> PilotConfig:
    """Load and validate ``configs/geant4_proton_pilot.yaml``."""

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "particle",
        "energies_gev",
        "incidence",
        "entry_spot",
        "gun_z",
        "physics",
        "samples",
        "store_fine_prefix_deposits",
        "output_dir",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise PilotConfigError(f"pilot configuration keys must be {sorted(expected)}")
    if raw["schema_version"] != EXPECTED_PILOT_SCHEMA_VERSION:
        raise PilotConfigError("unsupported pilot schema_version")

    lists = tuple(raw["physics"]["lists"])
    samples: dict[str, SampleSpec] = {}
    for name, spec in raw["samples"].items():
        if spec["geometry"] not in ("ams", "extended"):
            raise PilotConfigError(f"sample {name}: geometry must be ams|extended")
        if spec["physics_list"] not in lists:
            raise PilotConfigError(f"sample {name}: physics list not in physics.lists")
        samples[name] = SampleSpec(
            name=name,
            geometry=spec["geometry"],
            physics_list=spec["physics_list"],
            events_per_energy=int(spec["events_per_energy"]),
            base_seed=int(spec["base_seed"]),
        )

    spot = raw["entry_spot"]
    energies = tuple(float(e) for e in raw["energies_gev"])
    if not energies or any(not isfinite(e) or e <= 0 for e in energies):
        raise PilotConfigError("energies_gev must be positive")
    return PilotConfig(
        particle=str(raw["particle"]),
        energies_gev=energies,
        theta_rad=float(raw["incidence"]["theta"]),
        phi_rad=float(raw["incidence"]["phi"]),
        entry_spot=EntrySpot(
            float(spot["x_min"]),
            float(spot["x_max"]),
            float(spot["y_min"]),
            float(spot["y_max"]),
        ),
        gun_z_mm=float(raw["gun_z"]),
        physics_lists=lists,
        production_cut_mm=float(raw["physics"]["production_cut"]),
        samples=samples,
        store_fine_prefix_deposits=bool(raw["store_fine_prefix_deposits"]),
        output_dir=PROJECT_ROOT / str(raw["output_dir"]),
    )


# ----------------------------------------------------------------------
# Seeds and entry points
# ----------------------------------------------------------------------


def event_seeds(base_seed: int, energy_gev: float, count: int) -> tuple[int, ...]:
    """Return ``count`` independent per-event seeds for one batch.

    Mixing the energy into the SeedSequence entropy keeps batches at different
    energies statistically independent without seed arithmetic.
    """

    if base_seed < 0 or count < 0:
        raise ValueError("base_seed and count must be nonnegative")
    children = np.random.SeedSequence([base_seed, round(energy_gev * 1000)]).spawn(
        count
    )
    # 31 bits, because Geant4's engine seed is a C long, 32-bit signed on
    # Windows; one bit of a 32-bit state word is dropped.
    return tuple(int(c.generate_state(1, dtype=np.uint32)[0] >> 1) for c in children)


def entry_point_mm(seed: int, spot: EntrySpot) -> tuple[float, float]:
    """Return the front-face entry point of an event, drawn from its seed."""

    rng = np.random.default_rng(seed)
    return (
        float(rng.uniform(spot.x_min_mm, spot.x_max_mm)),
        float(rng.uniform(spot.y_min_mm, spot.y_max_mm)),
    )


# ----------------------------------------------------------------------
# One event's result
# ----------------------------------------------------------------------


@dataclass(slots=True)
class EventResult:
    event_index: int
    seed: int
    entry_x_mm: float
    entry_y_mm: float
    truth: dict[str, Any]
    fibre_ids: np.ndarray
    fibre_edep_mev: np.ndarray
    prefix_voxels: np.ndarray
    prefix_voxel_edep_mev: np.ndarray
    extension_voxels: np.ndarray
    extension_voxel_edep_mev: np.ndarray
    cpu_seconds: float


def _loud(method):
    """Print a Python traceback before an exception crosses into Geant4.

    An exception raised inside a Geant4 callback otherwise surfaces only as an
    access violation in C++, with no Python frame to show where it came from.
    """

    import functools
    import traceback

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            raise

    return wrapper


# ----------------------------------------------------------------------
# The Geant4 application
# ----------------------------------------------------------------------


class ECALTransport:
    """A configured Geant4 run manager for one geometry and physics list.

    Geant4 allows one run manager per process, so build one of these per
    worker process and reuse it for every event that worker runs.
    """

    def __init__(
        self,
        *,
        geometry: ECALGeometry,
        transport: TransportConfig,
        variant: GeometryVariant,
        physics_list: str,
        production_cut_mm: float,
        particle: str,
        theta_rad: float,
        phi_rad: float,
        gun_z_mm: float,
        spot: EntrySpot,
    ) -> None:
        self.datasets = configure_geant4_data()
        import geant4_pybind as g4

        self.g4 = g4
        self.geometry = geometry
        self.transport = transport
        self.variant = variant
        self.layout = FibreLayout(geometry)
        self.prefix_grid_spec = prefix_mesh(geometry, transport)
        self.extension_grid_spec = (
            extension_mesh(geometry, transport) if variant == "extended" else None
        )
        self._keep: list[object] = []  # Geant4 objects must outlive the run
        self._records: list[SecondaryRecord] = []
        self._results: list[EventResult] = []
        self._seeds: list[int] = []
        self._indices: list[int] = []
        self.material_summary: dict[str, float] = {}

        self.run_manager = g4.G4RunManagerFactory.CreateRunManager(
            g4.G4RunManagerType.Serial
        )
        detector = self._detector_construction()
        detector.RegisterParallelWorld(self._scoring_world())
        self.run_manager.SetUserInitialization(detector)

        physics = getattr(g4, physics_list)()
        parallel = g4.G4ParallelWorldPhysics("scoring")
        physics.RegisterPhysics(parallel)
        self._keep += [detector, physics, parallel]
        self.run_manager.SetUserInitialization(physics)

        self.run_manager.SetUserInitialization(
            self._actions(particle, theta_rad, phi_rad, gun_z_mm, spot)
        )
        ui = g4.G4UImanager.GetUIpointer()
        for command in (
            "/control/verbose 0",
            "/run/verbose 0",
            "/event/verbose 0",
            "/tracking/verbose 0",
            # Setting the cut through SetDefaultCutValue crashes
            # geant4_pybind 0.1.3 on Windows; the UI command is equivalent.
            f"/run/setCut {production_cut_mm} mm",
        ):
            ui.ApplyCommand(command)
        self.run_manager.Initialize()
        self.physics_list = physics_list

    # --- geometry -------------------------------------------------------

    def _materials(self):
        g4 = self.g4
        nist = g4.G4NistManager.Instance()
        lead = nist.FindOrBuildMaterial(self.transport.absorber_nist)
        fibre = nist.FindOrBuildMaterial(self.transport.fibre_nist)
        glue = g4.G4Material(
            "ECAL_glue",
            self.transport.glue_density_g_cm3 * g4.g / g4.cm3,
            len(self.transport.glue_atoms),
        )
        for symbol, count in self.transport.glue_atoms:
            glue.AddElement(nist.FindOrBuildElement(symbol), count)

        rho_lead = lead.GetDensity() / (g4.g / g4.cm3)
        rho_fibre = fibre.GetDensity() / (g4.g / g4.cm3)
        rho_glue = self.transport.glue_density_g_cm3
        composition = superlayer_composition(
            self.layout,
            self.transport.matrix_constraint,
            lead_density=rho_lead,
            fibre_density=rho_fibre,
            glue_density=rho_glue,
        )

        share = composition.lead_share_of_matrix
        rho_matrix = share * rho_lead + (1 - share) * rho_glue
        matrix = g4.G4Material("ECAL_matrix", rho_matrix * g4.g / g4.cm3, 2)
        matrix.AddMaterial(lead, share * rho_lead / rho_matrix)
        matrix.AddMaterial(glue, (1 - share) * rho_glue / rho_matrix)

        w_lead, w_fibre, w_glue = composition.mass_fractions(
            rho_lead, rho_fibre, rho_glue
        )
        composite = g4.G4Material(
            "ECAL_composite", composition.density_g_cm3 * g4.g / g4.cm3, 3
        )
        composite.AddMaterial(lead, w_lead)
        composite.AddMaterial(fibre, w_fibre)
        composite.AddMaterial(glue, w_glue)

        depth = self.geometry.depth_z_mm
        self.material_summary = {
            "matrix_constraint": self.transport.matrix_constraint,
            "lead_volume_fraction": composition.lead_volume_fraction,
            "fibre_volume_fraction": composition.fibre_volume_fraction,
            "glue_volume_fraction": composition.glue_volume_fraction,
            "composite_density_g_cm3": composition.density_g_cm3,
            "composite_radiation_length_mm": composite.GetRadlen() / g4.mm,
            "composite_nuclear_interaction_length_mm": (
                composite.GetNuclearInterLength() / g4.mm
            ),
            "prefix_depth_x0": depth / (composite.GetRadlen() / g4.mm),
            "prefix_depth_lambda_i": depth
            / (composite.GetNuclearInterLength() / g4.mm),
            "configured_depth_x0": self.geometry.total_depth_x0,
            "configured_depth_lambda_i": self.geometry.total_depth_lambda_i,
        }
        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        self._keep += [glue, matrix, composite]
        return vacuum, fibre, matrix, composite

    def _detector_construction(self):
        g4 = self.g4
        app = self
        mm = g4.mm
        geometry, layout = self.geometry, self.layout

        class Detector(g4.G4VUserDetectorConstruction):
            @_loud
            def Construct(self):
                vacuum, fibre, matrix, composite = app._materials()
                keep = app._keep
                half_w = geometry.width_x_mm / 2
                depth = geometry.depth_z_mm
                extension = (
                    app.transport.extension_depth_mm if app.variant == "extended" else 0.0
                )
                world_half_z = depth + extension + app.transport.upstream_gap_mm
                world_half_xy = app.transport.world_half_width_mm

                world_s = g4.G4Box("World", world_half_xy * mm, world_half_xy * mm, world_half_z * mm)
                world_l = g4.G4LogicalVolume(world_s, vacuum, "World")
                world_p = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_l, "World", None, False, 0)

                ecal_s = g4.G4Box("ECAL", half_w * mm, half_w * mm, depth / 2 * mm)
                ecal_l = g4.G4LogicalVolume(ecal_s, vacuum, "ECAL")
                ecal_p = g4.G4PVPlacement(
                    None, g4.G4ThreeVector(0, 0, depth / 2 * mm), ecal_l, "ECAL", world_l, False, 0
                )

                thickness = geometry.sampling_structure.superlayer_thickness_mm
                fibre_s = g4.G4Tubs(
                    "Fibre", 0.0, layout.fibre_radius_mm * mm, half_w * mm, 0.0, 360 * g4.deg
                )
                fibre_l = g4.G4LogicalVolume(fibre_s, fibre, "Fibre")
                keep += [world_s, world_l, world_p, ecal_s, ecal_l, ecal_p, fibre_s, fibre_l]

                superlayer_s = g4.G4Box("Superlayer", half_w * mm, half_w * mm, thickness / 2 * mm)
                superlayer_logical = {}
                for axis in ("x", "y"):
                    logical = g4.G4LogicalVolume(superlayer_s, matrix, f"Superlayer_{axis}")
                    rotation = g4.G4RotationMatrix()
                    if axis == "x":  # fibres run along x, measure y
                        rotation.rotateY(90 * g4.deg)
                    else:  # fibres run along y, measure x
                        rotation.rotateX(90 * g4.deg)
                    keep += [logical, rotation]
                    for row in range(layout.rows_per_superlayer):
                        z = (layout.row_offset_mm(row) - thickness / 2) * mm
                        for index, centre in enumerate(layout.fibre_centres_mm(row)):
                            position = (
                                g4.G4ThreeVector(0, centre * mm, z)
                                if axis == "x"
                                else g4.G4ThreeVector(centre * mm, 0, z)
                            )
                            copy = row * layout.max_fibres_per_row + index
                            keep.append(
                                g4.G4PVPlacement(
                                    rotation, position, fibre_l, "Fibre", logical, False, copy
                                )
                            )
                    superlayer_logical[axis] = logical
                keep.append(superlayer_s)

                for superlayer, axis in enumerate(geometry.superlayer_fiber_axes):
                    z = (-depth / 2 + (superlayer + 0.5) * thickness) * mm
                    keep.append(
                        g4.G4PVPlacement(
                            None,
                            g4.G4ThreeVector(0, 0, z),
                            superlayer_logical[axis],
                            "Superlayer",
                            ecal_l,
                            False,
                            superlayer,
                        )
                    )

                if extension > 0:
                    ext_s = g4.G4Box("Extension", half_w * mm, half_w * mm, extension / 2 * mm)
                    ext_l = g4.G4LogicalVolume(ext_s, composite, "Extension")
                    ext_p = g4.G4PVPlacement(
                        None,
                        g4.G4ThreeVector(0, 0, (depth + extension / 2) * mm),
                        ext_l,
                        "Extension",
                        world_l,
                        False,
                        0,
                    )
                    keep += [ext_s, ext_l, ext_p]

                app._fibre_logical = fibre_l
                return world_p

            @_loud

            def ConstructSDandField(self):
                # index = superlayer * (rows * fibres_per_row) + copy number,
                # which is exactly FibreLayout.encode(superlayer, row, index).
                detector = g4.G4MultiFunctionalDetector("fibreSD")
                g4.G4SDManager.GetSDMpointer().AddNewDetector(detector)
                scorer = g4.G4PSEnergyDeposit3D(
                    "edep",
                    geometry.number_of_superlayers,
                    1,
                    layout.rows_per_superlayer * layout.max_fibres_per_row,
                    1,
                    2,
                    0,
                )
                detector.RegisterPrimitive(scorer)
                self.SetSensitiveDetector("Fibre", detector)
                app._keep += [detector, scorer]

        return Detector()

    def _scoring_world(self):
        g4 = self.g4
        app = self
        mm = g4.mm

        def build_mesh(parent_logical, name: str, grid: MeshGrid, keep: list):
            nx, ny, nz = grid.shape
            vx, vy, vz = grid.voxel.x_mm, grid.voxel.y_mm, grid.voxel.z_mm
            half = (nx * vx / 2, ny * vy / 2, nz * vz / 2)
            box_s = g4.G4Box(name, half[0] * mm, half[1] * mm, half[2] * mm)
            box_l = g4.G4LogicalVolume(box_s, None, name)
            centre_z = grid.origin_mm[2] + half[2]
            box_p = g4.G4PVPlacement(
                None, g4.G4ThreeVector(0, 0, centre_z * mm), box_l, name, parent_logical, False, 0
            )
            z_s = g4.G4Box(name + "Z", half[0] * mm, half[1] * mm, vz / 2 * mm)
            z_l = g4.G4LogicalVolume(z_s, None, name + "Z")
            z_p = g4.G4PVReplica(name + "Z", z_l, box_l, g4.kZAxis, nz, vz * mm)
            y_s = g4.G4Box(name + "Y", half[0] * mm, vy / 2 * mm, vz / 2 * mm)
            y_l = g4.G4LogicalVolume(y_s, None, name + "Y")
            y_p = g4.G4PVReplica(name + "Y", y_l, z_l, g4.kYAxis, ny, vy * mm)
            x_s = g4.G4Box(name + "X", vx / 2 * mm, vy / 2 * mm, vz / 2 * mm)
            x_l = g4.G4LogicalVolume(x_s, None, name + "X")
            x_p = g4.G4PVReplica(name + "X", x_l, y_l, g4.kXAxis, nx, vx * mm)
            keep += [box_s, box_l, box_p, z_s, z_l, z_p, y_s, y_l, y_p, x_s, x_l, x_p]
            return x_l

        class ScoringWorld(g4.G4VUserParallelWorld):
            @_loud
            def Construct(self):
                ghost = self.GetWorld().GetLogicalVolume()
                self.cells = {"prefixMesh": build_mesh(ghost, "prefixMesh", app.prefix_grid_spec, app._keep)}
                if app.extension_grid_spec is not None:
                    self.cells["extensionMesh"] = build_mesh(
                        ghost, "extensionMesh", app.extension_grid_spec, app._keep
                    )

            @_loud

            def ConstructSD(self):
                for name, logical in self.cells.items():
                    grid = (
                        app.prefix_grid_spec if name == "prefixMesh" else app.extension_grid_spec
                    )
                    nx, ny, nz = grid.shape
                    detector = g4.G4MultiFunctionalDetector(name)
                    g4.G4SDManager.GetSDMpointer().AddNewDetector(detector)
                    scorer = g4.G4PSEnergyDeposit3D("edep", nz, ny, nx, 2, 1, 0)
                    detector.RegisterPrimitive(scorer)
                    self.SetSensitiveDetector(logical, detector)
                    app._keep += [detector, scorer]

        world = ScoringWorld("scoring")
        self._keep.append(world)
        return world

    # --- actions --------------------------------------------------------

    def _actions(self, particle, theta, phi, gun_z, spot):
        g4 = self.g4
        app = self
        direction = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))

        class Generator(g4.G4VUserPrimaryGeneratorAction):
            def __init__(self):
                super().__init__()
                self.gun = g4.G4ParticleGun(1)
                self.gun.SetParticleDefinition(
                    g4.G4ParticleTable.GetParticleTable().FindParticle(particle)
                )
                self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(*direction))

            @_loud

            def GeneratePrimaries(self, event):
                local = event.GetEventID()
                seed = app._seeds[local]
                g4.G4Random.setTheSeed(seed)
                x, y = entry_point_mm(seed, spot)
                # Place the gun upstream so the track crosses z = 0 at (x, y).
                step = gun_z / direction[2]
                self.gun.SetParticlePosition(
                    g4.G4ThreeVector(
                        (x + step * direction[0]) * g4.mm,
                        (y + step * direction[1]) * g4.mm,
                        gun_z * g4.mm,
                    )
                )
                self.gun.SetParticleEnergy(app._energy_gev * g4.GeV)
                app._entry = (x, y)
                self.gun.GeneratePrimaryVertex(event)

        hadronic = g4.G4ProcessType.fHadronic

        class Stacking(g4.G4UserStackingAction):
            @_loud
            def ClassifyNewTrack(self, track):
                if track.GetParentID() == 1:
                    process = track.GetCreatorProcess()
                    position = track.GetPosition()
                    definition = track.GetDefinition()
                    app._records.append(
                        SecondaryRecord(
                            parent_id=1,
                            creator_process=process.GetProcessName(),
                            creator_is_hadronic=process.GetProcessType() == hadronic,
                            global_time_ns=track.GetGlobalTime() / g4.ns,
                            x_mm=position.x / g4.mm,
                            y_mm=position.y / g4.mm,
                            z_mm=position.z / g4.mm,
                            kinetic_energy_mev=track.GetKineticEnergy() / g4.MeV,
                            total_energy_mev=track.GetTotalEnergy() / g4.MeV,
                            pdg=definition.GetPDGEncoding(),
                            charge=definition.GetPDGCharge(),
                        )
                    )
                return g4.G4ClassificationOfNewTrack.fUrgent

            @_loud

            def PrepareNewEvent(self):
                app._records = []

        class Events(g4.G4UserEventAction):
            @_loud
            def BeginOfEventAction(self, event):
                app._t0 = time.perf_counter()

            @_loud

            def EndOfEventAction(self, event):
                app._collect(event)

        class Actions(g4.G4VUserActionInitialization):
            @_loud
            def Build(self):
                self.SetUserAction(Generator())
                self.SetUserAction(Stacking())
                self.SetUserAction(Events())

        actions = Actions()
        self._keep.append(actions)
        return actions

    def _hits(self, hce, name: str) -> tuple[np.ndarray, np.ndarray]:
        sdm = self.g4.G4SDManager.GetSDMpointer()
        collection_id = sdm.GetCollectionID(name)
        if collection_id < 0:
            return np.zeros(0, np.int64), np.zeros(0)
        hits = hce.GetHC(collection_id)
        items = list(hits) if hits is not None else []
        if not items:
            return np.zeros(0, np.int64), np.zeros(0)
        keys, values = zip(*items, strict=True)
        return np.asarray(keys, np.int64), np.asarray(values, float) / self.g4.MeV

    def _collect(self, event) -> None:
        hce = event.GetHCofThisEvent()
        fibre_ids, fibre_edep = self._hits(hce, "fibreSD/edep")
        prefix_ids, prefix_edep = self._hits(hce, "prefixMesh/edep")
        if self.extension_grid_spec is not None:
            ext_ids, ext_edep = self._hits(hce, "extensionMesh/edep")
        else:
            ext_ids, ext_edep = np.zeros(0, np.int64), np.zeros(0)
        truth = first_inelastic_interaction(self._records)
        local = event.GetEventID()
        self._results.append(
            EventResult(
                event_index=self._indices[local],
                seed=self._seeds[local],
                entry_x_mm=self._entry[0],
                entry_y_mm=self._entry[1],
                truth=asdict(truth),
                fibre_ids=fibre_ids,
                fibre_edep_mev=fibre_edep,
                prefix_voxels=prefix_ids,
                prefix_voxel_edep_mev=prefix_edep,
                extension_voxels=ext_ids,
                extension_voxel_edep_mev=ext_edep,
                cpu_seconds=time.perf_counter() - self._t0,
            )
        )

    # --- running --------------------------------------------------------

    def run(
        self, energy_gev: float, indices: list[int], seeds: list[int]
    ) -> list[EventResult]:
        """Simulate one event per seed; ``indices`` label them in the batch."""

        if len(indices) != len(seeds):
            raise ValueError("indices and seeds must have the same length")
        self._energy_gev = float(energy_gev)
        self._indices, self._seeds, self._results = list(indices), list(seeds), []
        if seeds:
            self.run_manager.BeamOn(len(seeds))
        return self._results


# ----------------------------------------------------------------------
# Batches: parallel execution, assembly, persistence
# ----------------------------------------------------------------------

_WORKER: ECALTransport | None = None


def _worker(
    task: dict[str, Any],
) -> tuple[list[EventResult], dict[str, Any], dict[str, Any]]:
    global _WORKER
    if _WORKER is None:
        _WORKER = ECALTransport(**task["setup"])
    results = _WORKER.run(task["energy_gev"], task["indices"], task["seeds"])
    return results, _WORKER.material_summary, _versions(_WORKER)


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _assemble(
    results: list[EventResult],
    geometry: ECALGeometry,
    layout: FibreLayout,
    prefix_grid: MeshGrid,
    extension_grid: MeshGrid | None,
    n_extended_layers: int,
    store_fine: bool,
) -> dict[str, np.ndarray]:
    results = sorted(results, key=lambda r: r.event_index)
    n = len(results)
    layers, cells = geometry.number_of_layers, geometry.cells_per_layer

    readout = layout.fibre_readout_map()
    size = max(readout) + 1
    fibre_layer = np.full(size, -1, np.int64)
    fibre_cell = np.full(size, -1, np.int64)
    for fibre_id, (layer, cell) in readout.items():
        fibre_layer[fibre_id] = layer
        fibre_cell[fibre_id] = -1 if cell is None else cell

    fibre_grid = np.zeros((n, layers, cells), np.float32)
    prefix = np.zeros((n, layers, cells), np.float32)
    extended = (
        np.zeros((n, n_extended_layers, cells), np.float32) if extension_grid else None
    )
    arrays: dict[str, list] = {key: [] for key in (
        "fibre_offsets", "fibre_ids", "fibre_edep", "fine_offsets", "fine_index", "fine_edep"
    )}
    fibre_offset = fine_offset = 0

    for i, result in enumerate(results):
        ids = result.fibre_ids
        mapped = fibre_cell[ids] >= 0
        np.add.at(
            fibre_grid[i],
            (fibre_layer[ids[mapped]], fibre_cell[ids[mapped]]),
            result.fibre_edep_mev[mapped],
        )

        x, y, z = prefix_grid.centres_mm(result.prefix_voxels)
        prefix[i] = project_deposits(x, y, z, result.prefix_voxel_edep_mev, geometry).grid_mev
        if extended is not None:
            ex, ey, ez = extension_grid.centres_mm(result.extension_voxels)
            tail = project_deposits(
                ex, ey, ez, result.extension_voxel_edep_mev, geometry, n_extended_layers
            ).grid_mev
            extended[i] = tail
            extended[i, :layers] += prefix[i]

        arrays["fibre_offsets"].append(fibre_offset)
        arrays["fibre_ids"].append(ids.astype(np.int32))
        arrays["fibre_edep"].append(result.fibre_edep_mev.astype(np.float32))
        fibre_offset += len(ids)
        if store_fine:
            arrays["fine_offsets"].append(fine_offset)
            arrays["fine_index"].append(result.prefix_voxels.astype(np.int32))
            arrays["fine_edep"].append(result.prefix_voxel_edep_mev.astype(np.float32))
            fine_offset += len(result.prefix_voxels)

    truth_keys = results[0].truth.keys() if results else ()
    out: dict[str, np.ndarray] = {
        "event_index": np.array([r.event_index for r in results], np.int64),
        "seed": np.array([r.seed for r in results], np.int64),
        "entry_x_mm": np.array([r.entry_x_mm for r in results]),
        "entry_y_mm": np.array([r.entry_y_mm for r in results]),
        "cpu_seconds": np.array([r.cpu_seconds for r in results]),
        "fibre_grid_mev": fibre_grid,
        "prefix_grid_mev": prefix,
        "fibre_total_mev": fibre_grid.sum(axis=(1, 2), dtype=float),
        "prefix_total_mev": np.array([r.prefix_voxel_edep_mev.sum() for r in results]),
        "extension_total_mev": np.array(
            [r.extension_voxel_edep_mev.sum() for r in results]
        ),
    }
    for key in truth_keys:
        out[f"truth_{key}"] = np.array([r.truth[key] for r in results])
    if extended is not None:
        out["extended_grid_mev"] = extended
    out["fibre_offsets"] = np.array(arrays["fibre_offsets"] + [fibre_offset], np.int64)
    out["fibre_ids"] = np.concatenate(arrays["fibre_ids"] or [np.zeros(0, np.int32)])
    out["fibre_edep_mev"] = np.concatenate(arrays["fibre_edep"] or [np.zeros(0, np.float32)])
    if store_fine:
        out["fine_offsets"] = np.array(arrays["fine_offsets"] + [fine_offset], np.int64)
        out["fine_index"] = np.concatenate(arrays["fine_index"] or [np.zeros(0, np.int32)])
        out["fine_edep_mev"] = np.concatenate(arrays["fine_edep"] or [np.zeros(0, np.float32)])
    return out


def run_batch(
    *,
    sample: str,
    energy_gev: float,
    n_events: int | None = None,
    workers: int = 1,
    out_dir: str | Path | None = None,
    seeds: list[int] | None = None,
    pilot_config: str | Path = PILOT_CONFIG,
    transport_config: str | Path = TRANSPORT_CONFIG,
    geometry_config: str | Path = GEOMETRY_CONFIG,
) -> Path:
    """Simulate one (sample, energy) batch and write it; return its directory.

    ``seeds`` replaces the configured seed stream, which reproduces chosen
    events exactly: pass the recorded seed of any stored event.
    """

    pilot = load_pilot_config(pilot_config)
    spec = pilot.samples[sample]
    geometry = load_geometry(geometry_config)
    transport = load_transport_config(transport_config)
    count = spec.events_per_energy if n_events is None else n_events
    batch_seeds = list(seeds) if seeds is not None else list(
        event_seeds(spec.base_seed, energy_gev, count)
    )
    indices = list(range(len(batch_seeds)))

    setup = {
        "geometry": geometry,
        "transport": transport,
        "variant": spec.geometry,
        "physics_list": spec.physics_list,
        "production_cut_mm": pilot.production_cut_mm,
        "particle": pilot.particle,
        "theta_rad": pilot.theta_rad,
        "phi_rad": pilot.phi_rad,
        "gun_z_mm": pilot.gun_z_mm,
        "spot": pilot.entry_spot,
    }
    workers = max(1, min(workers, len(batch_seeds) or 1))
    chunks = [indices[w::workers] for w in range(workers)]
    tasks = [
        {
            "setup": setup,
            "energy_gev": energy_gev,
            "indices": chunk,
            "seeds": [batch_seeds[i] for i in chunk],
        }
        for chunk in chunks
        if chunk
    ]

    started = time.perf_counter()
    if workers == 1:
        outputs = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            outputs = list(pool.map(_worker, tasks))
    results = [result for chunk, _, _ in outputs for result in chunk]
    material = outputs[0][1] if outputs else {}
    versions = outputs[0][2] if outputs else {}
    wall = time.perf_counter() - started

    layout = FibreLayout(geometry)
    extension_grid = extension_mesh(geometry, transport) if spec.geometry == "extended" else None
    arrays = _assemble(
        results,
        geometry,
        layout,
        prefix_mesh(geometry, transport),
        extension_grid,
        extended_layer_count(geometry, transport),
        pilot.store_fine_prefix_deposits,
    )

    directory = Path(out_dir) if out_dir else (
        pilot.output_dir / sample / f"E{energy_gev:g}GeV"
    )
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "events.npz", **arrays)

    metadata = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_state(),
        **versions,
        "sample": sample,
        "geometry_variant": spec.geometry,
        "physics_list": spec.physics_list,
        "production_cut_mm": pilot.production_cut_mm,
        "particle": pilot.particle,
        "energy_gev": energy_gev,
        "theta_rad": pilot.theta_rad,
        "phi_rad": pilot.phi_rad,
        "entry_spot_mm": asdict(pilot.entry_spot),
        "gun_z_mm": pilot.gun_z_mm,
        "n_events": len(batch_seeds),
        "base_seed": spec.base_seed if seeds is None else None,
        "explicit_seeds": seeds is not None,
        "seed_policy": SEED_POLICY,
        "configuration_sha256": config_digest(
            [geometry_config, transport_config, pilot_config]
        ),
        "configuration_files": [
            str(Path(p).resolve().relative_to(PROJECT_ROOT))
            if Path(p).resolve().is_relative_to(PROJECT_ROOT)
            else str(p)
            for p in (geometry_config, transport_config, pilot_config)
        ],
        "materials": material,
        "fibres": layout.fibres_per_superlayer * geometry.number_of_superlayers,
        "prefix_mesh_shape_xyz": list(prefix_mesh(geometry, transport).shape),
        "extension_mesh_shape_xyz": list(extension_grid.shape) if extension_grid else None,
        "extended_layers": extended_layer_count(geometry, transport)
        if extension_grid
        else None,
        "energy_representations": {
            "fibre_grid_mev": "energy deposited in scintillating fibres (sensitive material); not calibrated, not digitized",
            "prefix_grid_mev": "energy deposited in all material of the AMS-like prefix",
            "extended_grid_mev": "all-material deposit, prefix + research-instrument extension",
        },
        "workers": workers,
        "wall_seconds": wall,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def _versions(app: ECALTransport | None) -> dict[str, Any]:
    if app is None:
        return {}
    version = app.g4.G4Version
    return {
        "geant4_version": version.replace("$Name:", "").replace("$", "").strip(),
        "geant4_pybind_version": package_version("geant4-pybind"),
        "geant4_datasets": app.datasets,
    }


# ----------------------------------------------------------------------
# Reading batches back
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Batch:
    """One stored simulation batch: event arrays plus metadata."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return len(self.arrays["event_index"])

    def fine_deposits(self, event: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (flat voxel index, MeV) of one event's fine prefix deposits."""

        start, stop = self.arrays["fine_offsets"][event : event + 2]
        return self.arrays["fine_index"][start:stop], self.arrays["fine_edep_mev"][start:stop]

    def fibre_deposits(self, event: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (fibre id, MeV) of one event's per-fibre deposits."""

        start, stop = self.arrays["fibre_offsets"][event : event + 2]
        return self.arrays["fibre_ids"][start:stop], self.arrays["fibre_edep_mev"][start:stop]


def load_batch(directory: str | Path) -> Batch:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["output_schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise ValueError("unsupported Geant4 output schema version")
    with np.load(directory / "events.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return Batch(arrays=arrays, metadata=metadata)


def to_ecal_events(
    batch: Batch,
    geometry: ECALGeometry,
    representation: Literal["fibre", "prefix"] = "prefix",
) -> list[ECALEvent]:
    """Express a Geant4 batch as canonical ``ECALEvent`` objects.

    The same class FastMC emits, with ``simulation_backend = "geant4"``. The
    ``prefix`` representation (all-material deposit) is the one comparable to
    FastMC's ``deposition`` regime; ``fibre`` is sensitive-material deposit.
    """

    key = {"fibre": "fibre_grid_mev", "prefix": "prefix_grid_mev"}[representation]
    meta = batch.metadata
    version = f"{meta['geant4_version']}/{meta['physics_list']}/{representation}"
    events = []
    for i in range(len(batch)):
        grid = batch.arrays[key][i].astype(float)
        events.append(
            ECALEvent(
                event_id=f"g4-{meta['sample']}-{meta['energy_gev']:g}GeV-{int(batch.arrays['event_index'][i]):06d}",
                particle_type=meta["particle"],
                primary_energy_mev=meta["energy_gev"] * 1000.0,
                track=TrackState(
                    x0_mm=float(batch.arrays["entry_x_mm"][i]),
                    y0_mm=float(batch.arrays["entry_y_mm"][i]),
                    z0_mm=0.0,
                    theta_rad=meta["theta_rad"],
                    phi_rad=meta["phi_rad"],
                ),
                geometry=geometry,
                cell_energies_mev=tuple(tuple(float(v) for v in row) for row in grid),
                provenance=EventProvenance(
                    simulation_backend="geant4",
                    simulation_version=version,
                    configuration_sha256=meta["configuration_sha256"],
                    random_seed=int(batch.arrays["seed"][i]),
                ),
            )
        )
    return events


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one (sample, energy) batch")
    run.add_argument("--sample", required=True)
    run.add_argument("--energy", type=float, required=True, help="GeV")
    run.add_argument("--events", type=int)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--out")
    run.add_argument("--seeds", help="comma-separated explicit event seeds")
    run.add_argument("--pilot-config", default=str(PILOT_CONFIG))
    run.add_argument("--transport-config", default=str(TRANSPORT_CONFIG))

    everything = sub.add_parser("all", help="simulate every configured batch")
    everything.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    everything.add_argument("--samples", nargs="*")

    args = parser.parse_args(argv)
    if args.command == "run":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        path = run_batch(
            sample=args.sample,
            energy_gev=args.energy,
            n_events=args.events,
            workers=args.workers,
            out_dir=args.out,
            seeds=seeds,
            pilot_config=args.pilot_config,
            transport_config=args.transport_config,
        )
        print(path)
    else:
        pilot = load_pilot_config()
        for sample in args.samples or list(pilot.samples):
            for energy in pilot.energies_gev:
                started = time.perf_counter()
                path = run_batch(sample=sample, energy_gev=energy, workers=args.workers)
                print(f"{path}  ({time.perf_counter() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
