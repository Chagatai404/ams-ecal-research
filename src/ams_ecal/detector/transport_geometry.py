"""Detailed-transport geometry derived from the canonical AMS ECAL geometry.

Geant4 needs more than the readout geometry: where every scintillating fibre
sits, what fills the space between fibres, and what - if anything - continues
behind the ECAL. This module derives all of it from ``ECALGeometry`` plus the
labelled assumptions of ``configs/geant4_transport.yaml``, so the detailed
transport geometry and the 18 x 72 readout geometry cannot silently describe two
different detectors. Dimensions come only from ``geometry.yaml``.

It is pure Python and never imports Geant4, so it is tested without it.

THE FIBRE LATTICE. Each superlayer holds ``fiber_planes_per_superlayer`` rows
of round fibres at ``fiber_horizontal_pitch`` along the measured coordinate and
``fiber_row_spacing`` in depth, alternate rows shifted by
``adjacent_row_stagger_fraction`` of a pitch. The row stack is centred in the
superlayer. A fibre is placed only if it lies entirely inside the active width,
so staggered rows hold one fibre fewer. The first half of the rows belongs to
the superlayer's first readout sampling and the second half to its second.

THE MATRIX. Everything that is not fibre is a homogeneous lead + glue matrix.
The grooved lead foils and the single aluminium terminal foil are not drawn
individually. How much lead the matrix holds is a configured choice, because the
published average density and the published volume ratio disagree (see
``superlayer_composition``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import floor, isclose, isfinite, pi
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from ams_ecal.detector.geometry import ECALGeometry
from ams_ecal.detector.readout import coordinate_to_cell_index, measured_axis_for_fiber

EXPECTED_TRANSPORT_SCHEMA_VERSION = 2

MatrixConstraint = Literal["average_density", "relative_volume"]
ExtensionStructure = Literal["sampling", "homogeneous"]

_MATRIX_CONSTRAINTS = {"average_density", "relative_volume"}
_EXTENSION_STRUCTURES = {"sampling", "homogeneous"}


class TransportConfigError(ValueError):
    """Raised when a transport configuration is malformed or inconsistent."""


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TransportConfigError(f"{name} must be a real number")
    if not isfinite(value) or value <= 0:
        raise TransportConfigError(f"{name} must be finite and positive")
    return float(value)


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TransportConfigError(f"{context} must be a YAML mapping")
    return value


def _exact_keys(
    mapping: Mapping[object, object], keys: set[str], context: str
) -> None:
    missing = keys - set(mapping)
    unexpected = set(mapping) - keys
    if missing or unexpected:
        raise TransportConfigError(
            f"{context}: missing {sorted(missing)}, "
            f"unexpected {sorted(map(str, unexpected))}"
        )


def _whole_multiple(value: float, unit: float) -> bool:
    """Return whether ``value`` is a whole number of ``unit`` (float-safe)."""

    ratio = value / unit
    return isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)


@dataclass(frozen=True, slots=True)
class VoxelSize:
    """Edge lengths of one scoring-mesh voxel in millimetres."""

    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Labelled detailed-transport assumptions; see geant4_transport.yaml."""

    absorber_nist: str
    fibre_nist: str
    glue_density_g_cm3: float
    glue_atoms: tuple[tuple[str, int], ...]
    matrix_constraint: MatrixConstraint
    extension_depth_mm: float
    extension_structure: ExtensionStructure
    prefix_voxel: VoxelSize
    extension_voxel: VoxelSize
    world_half_width_mm: float
    upstream_gap_mm: float

    def __post_init__(self) -> None:
        if self.matrix_constraint not in _MATRIX_CONSTRAINTS:
            raise TransportConfigError(
                f"matrix_constraint must be one of {sorted(_MATRIX_CONSTRAINTS)}"
            )
        if self.extension_structure not in _EXTENSION_STRUCTURES:
            raise TransportConfigError(
                f"extension.structure must be one of {sorted(_EXTENSION_STRUCTURES)}"
            )


def load_transport_config(config_path: str | Path) -> TransportConfig:
    """Load and validate ``configs/geant4_transport.yaml``."""

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transport configuration not found: {path}")

    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "root")
    _exact_keys(
        raw,
        {"schema_version", "units", "materials", "extension", "mesh", "world"},
        "root",
    )
    if raw["schema_version"] != EXPECTED_TRANSPORT_SCHEMA_VERSION:
        raise TransportConfigError(
            f"unsupported transport schema_version {raw['schema_version']!r}; "
            f"expected {EXPECTED_TRANSPORT_SCHEMA_VERSION}"
        )
    if dict(_mapping(raw["units"], "units")) != {
        "length": "mm",
        "density": "g/cm^3",
    }:
        raise TransportConfigError("units must be length: mm, density: g/cm^3")

    materials = _mapping(raw["materials"], "materials")
    _exact_keys(
        materials,
        {"absorber_nist", "fibre_nist", "glue", "matrix_constraint"},
        "materials",
    )
    glue = _mapping(materials["glue"], "materials.glue")
    _exact_keys(glue, {"density", "atoms"}, "materials.glue")
    atoms = _mapping(glue["atoms"], "materials.glue.atoms")
    for symbol, count in atoms.items():
        if not isinstance(symbol, str) or isinstance(count, bool) or not (
            isinstance(count, int) and count > 0
        ):
            raise TransportConfigError(
                "materials.glue.atoms must map element symbols to positive "
                "integer counts"
            )

    extension = _mapping(raw["extension"], "extension")
    _exact_keys(extension, {"depth", "structure"}, "extension")

    mesh = _mapping(raw["mesh"], "mesh")
    _exact_keys(mesh, {"prefix_voxel", "extension_voxel"}, "mesh")

    def voxel(name: str) -> VoxelSize:
        spec = _mapping(mesh[name], f"mesh.{name}")
        _exact_keys(spec, {"x", "y", "z"}, f"mesh.{name}")
        return VoxelSize(
            _positive(spec["x"], f"mesh.{name}.x"),
            _positive(spec["y"], f"mesh.{name}.y"),
            _positive(spec["z"], f"mesh.{name}.z"),
        )

    world = _mapping(raw["world"], "world")
    _exact_keys(world, {"half_width", "upstream_gap"}, "world")

    for key in ("absorber_nist", "fibre_nist"):
        if not isinstance(materials[key], str) or not materials[key]:
            raise TransportConfigError(f"materials.{key} must be a string")

    return TransportConfig(
        absorber_nist=materials["absorber_nist"],
        fibre_nist=materials["fibre_nist"],
        glue_density_g_cm3=_positive(glue["density"], "materials.glue.density"),
        glue_atoms=tuple((str(s), int(n)) for s, n in atoms.items()),
        matrix_constraint=materials["matrix_constraint"],
        extension_depth_mm=_positive(extension["depth"], "extension.depth"),
        extension_structure=extension["structure"],
        prefix_voxel=voxel("prefix_voxel"),
        extension_voxel=voxel("extension_voxel"),
        world_half_width_mm=_positive(world["half_width"], "world.half_width"),
        upstream_gap_mm=_positive(world["upstream_gap"], "world.upstream_gap"),
    )


# ----------------------------------------------------------------------
# Fibre lattice
# ----------------------------------------------------------------------


@lru_cache(maxsize=16)
def _row_centres(geometry: ECALGeometry, staggered: bool) -> tuple[float, ...]:
    """Fibre centres of a full or staggered row (cached; geometry is frozen)."""

    structure = geometry.sampling_structure
    pitch = structure.fiber_horizontal_pitch_mm
    radius = structure.fiber_diameter_mm / 2
    lower, upper = -geometry.width_x_mm / 2, geometry.width_x_mm / 2
    first = lower + pitch / 2
    if staggered:
        first += structure.adjacent_row_stagger_fraction * pitch

    count = floor((upper - radius - first) / pitch + 1e-9) + 1
    return tuple(first + index * pitch for index in range(count))


@dataclass(frozen=True, slots=True)
class FibreLayout:
    """Positions of every scintillating fibre, derived from ``ECALGeometry``.

    Coordinates are in the package frame: the measured coordinate runs across
    the active width ``[-W/2, W/2)``, and depth is measured from the front face
    of the fibre's own superlayer.
    """

    geometry: ECALGeometry

    def __post_init__(self) -> None:
        structure = self.geometry.sampling_structure
        radius = structure.fiber_diameter_mm / 2

        if self.geometry.number_of_layers % self.geometry.number_of_superlayers:
            raise ValueError("layers must divide evenly into superlayers")
        if self.rows_per_superlayer % self.samplings_per_superlayer:
            raise ValueError("fibre rows must divide evenly into samplings")

        # Neighbouring fibres in one row, and in adjacent staggered rows, must
        # not overlap; the stack must fit inside its superlayer.
        if structure.fiber_horizontal_pitch_mm < structure.fiber_diameter_mm:
            raise ValueError("fibres in one row would overlap")
        stagger_mm = (
            structure.adjacent_row_stagger_fraction
            * structure.fiber_horizontal_pitch_mm
        )
        nearest_mm = min(
            stagger_mm, structure.fiber_horizontal_pitch_mm - stagger_mm
        )
        if (
            structure.fiber_row_spacing_mm**2 + nearest_mm**2
            < structure.fiber_diameter_mm**2
        ):
            raise ValueError("fibres in adjacent rows would overlap")
        if self.row_offset_mm(0) < radius:
            raise ValueError("fibre stack does not fit inside the superlayer")

        # Every fibre must lie entirely inside one readout sampling, so a
        # fibre's deposit belongs to exactly one of the 18 layers.
        sampling_mm = self.sampling_thickness_mm
        for row in range(self.rows_per_superlayer):
            sampling = row // self.rows_per_sampling
            centre = self.row_offset_mm(row)
            if not (
                sampling * sampling_mm <= centre - radius
                and centre + radius <= (sampling + 1) * sampling_mm
            ):
                raise ValueError(
                    f"fibre row {row} straddles a readout-sampling boundary"
                )

    @property
    def rows_per_superlayer(self) -> int:
        return self.geometry.sampling_structure.fiber_planes_per_superlayer

    @property
    def samplings_per_superlayer(self) -> int:
        return (
            self.geometry.number_of_layers // self.geometry.number_of_superlayers
        )

    @property
    def rows_per_sampling(self) -> int:
        return self.rows_per_superlayer // self.samplings_per_superlayer

    @property
    def sampling_thickness_mm(self) -> float:
        return self.geometry.mean_readout_slice_thickness_mm

    @property
    def fibre_radius_mm(self) -> float:
        return self.geometry.sampling_structure.fiber_diameter_mm / 2

    def row_offset_mm(self, row: int) -> float:
        """Return the depth of a row's fibre axes behind the superlayer front."""

        structure = self.geometry.sampling_structure
        stack_mm = (self.rows_per_superlayer - 1) * structure.fiber_row_spacing_mm
        return (
            (structure.superlayer_thickness_mm - stack_mm) / 2
            + row * structure.fiber_row_spacing_mm
        )

    def is_staggered(self, row: int) -> bool:
        return row % 2 == 1

    def fibre_centres_mm(self, row: int) -> tuple[float, ...]:
        """Return the measured-coordinate centres of every fibre in one row."""

        return _row_centres(self.geometry, self.is_staggered(row))

    @property
    def max_fibres_per_row(self) -> int:
        return max(
            len(self.fibre_centres_mm(row))
            for row in range(self.rows_per_superlayer)
        )

    @property
    def fibres_per_superlayer(self) -> int:
        return sum(
            len(self.fibre_centres_mm(row))
            for row in range(self.rows_per_superlayer)
        )

    def layer_index(self, superlayer: int, row: int) -> int:
        """Return the readout sampling (0..17) that a fibre row belongs to."""

        return (
            superlayer * self.samplings_per_superlayer
            + row // self.rows_per_sampling
        )

    def encode(self, superlayer: int, row: int, index: int) -> int:
        """Return a unique non-negative integer naming one fibre."""

        return (
            superlayer * self.rows_per_superlayer + row
        ) * self.max_fibres_per_row + index

    def decode(self, fibre_id: int) -> tuple[int, int, int]:
        """Invert ``encode``: return ``(superlayer, row, index)``."""

        stride = self.max_fibres_per_row
        superlayer_row, index = divmod(fibre_id, stride)
        superlayer, row = divmod(superlayer_row, self.rows_per_superlayer)
        return superlayer, row, index

    @property
    def fibre_volume_fraction(self) -> float:
        """Return the fibre share of a superlayer's volume.

        Derived from the explicit lattice, not from the published volume ratio;
        the two differ slightly because the ten rows do not fill the full
        superlayer depth.
        """

        structure = self.geometry.sampling_structure
        fibre_area = self.fibres_per_superlayer * pi * self.fibre_radius_mm**2
        return fibre_area / (
            self.geometry.width_x_mm * structure.superlayer_thickness_mm
        )

    def fibre_readout_map(
        self, n_superlayers: int | None = None
    ) -> dict[int, tuple[int, int | None]]:
        """Map every fibre id to its ``(layer, cell)`` in the alternating readout.

        ``n_superlayers`` beyond the nine physical ones covers the sampling
        extension, whose superlayers continue the same lattice and alternation.

        The cell comes from the existing readout function applied to the fibre
        centre, so fibres use exactly the half-open cell convention of the rest
        of the package. A fibre whose centre lies on a cell boundary goes to
        the upper cell. PROJECT APPROXIMATION: in the real light guides such a
        fibre shares its light between two anodes.
        """

        layers, cells = self.readout_lookup(n_superlayers)
        return {
            int(fibre_id): (
                int(layers[fibre_id]),
                None if cells[fibre_id] < 0 else int(cells[fibre_id]),
            )
            for fibre_id in np.flatnonzero(layers >= 0)
        }

    def readout_lookup(
        self, n_superlayers: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(layer, cell)`` arrays indexed by fibre id; -1 = none.

        A fibre's cell depends only on its row and index, so the readout
        function is evaluated once per row position and reused for every
        superlayer.
        """

        count = (
            self.geometry.number_of_superlayers
            if n_superlayers is None
            else n_superlayers
        )
        stride = self.max_fibres_per_row
        rows = self.rows_per_superlayer
        layers = np.full(count * rows * stride, -1, np.int64)
        cells = np.full(count * rows * stride, -1, np.int64)
        for row in range(rows):
            row_cells = np.array(
                [
                    -1 if cell is None else cell
                    for cell in (
                        coordinate_to_cell_index(centre, self.geometry)
                        for centre in self.fibre_centres_mm(row)
                    )
                ],
                np.int64,
            )
            for superlayer in range(count):
                start = (superlayer * rows + row) * stride
                layers[start : start + len(row_cells)] = self.layer_index(
                    superlayer, row
                )
                cells[start : start + len(row_cells)] = row_cells
        return layers, cells

    def fibre_axis(self, superlayer: int) -> str:
        """Return a superlayer's fibre direction, alternating past the ECAL."""

        axes = self.geometry.superlayer_fiber_axes
        return axes[superlayer] if superlayer < len(axes) else axes[superlayer % 2]

    def measured_axis(self, superlayer: int) -> str:
        """Return the coordinate a superlayer's fibres measure."""

        return measured_axis_for_fiber(self.fibre_axis(superlayer))


# ----------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaterialComposition:
    """Volume fractions and density of the superlayer composite."""

    lead_volume_fraction: float
    fibre_volume_fraction: float
    glue_volume_fraction: float
    lead_share_of_matrix: float
    density_g_cm3: float

    def mass_fractions(
        self, lead_density: float, fibre_density: float, glue_density: float
    ) -> tuple[float, float, float]:
        """Return lead, fibre and glue mass fractions summing to one."""

        masses = (
            self.lead_volume_fraction * lead_density,
            self.fibre_volume_fraction * fibre_density,
            self.glue_volume_fraction * glue_density,
        )
        total = sum(masses)
        return (masses[0] / total, masses[1] / total, masses[2] / total)


def superlayer_composition(
    layout: FibreLayout,
    constraint: MatrixConstraint,
    *,
    lead_density: float,
    fibre_density: float,
    glue_density: float,
) -> MaterialComposition:
    """Return the composition of one superlayer under a matrix constraint.

    The fibre fraction is fixed by the explicit lattice. What remains is the
    lead + glue matrix, whose lead share ``phi`` is set either

    * ``average_density`` - so that the superlayer reproduces the published
      average density ``rho``:
      ``rho = f_F rho_F + (1 - f_F) (phi rho_Pb + (1 - phi) rho_glue)``;
    * ``relative_volume`` - from the published lead:glue volume ratio.

    With the published AMS numbers the two disagree (about 6.8 against 7.2
    g/cm^3); which one is right is not settled by the sources the project holds.
    """

    properties = layout.geometry.material_properties
    fibre_fraction = layout.fibre_volume_fraction
    matrix_fraction = 1.0 - fibre_fraction

    if constraint == "average_density":
        matrix_density = (
            properties.average_density_g_cm3 - fibre_fraction * fibre_density
        ) / matrix_fraction
        lead_share = (matrix_density - glue_density) / (lead_density - glue_density)
    elif constraint == "relative_volume":
        lead_share = properties.relative_volume_lead / (
            properties.relative_volume_lead
            + properties.relative_volume_optical_glue
        )
    else:
        raise ValueError(f"unknown matrix constraint {constraint!r}")

    if not 0.0 < lead_share < 1.0:
        raise ValueError(
            f"matrix lead share {lead_share:.4f} is unphysical for these densities"
        )

    lead_fraction = matrix_fraction * lead_share
    glue_fraction = matrix_fraction * (1.0 - lead_share)
    density = (
        lead_fraction * lead_density
        + fibre_fraction * fibre_density
        + glue_fraction * glue_density
    )
    return MaterialComposition(
        lead_volume_fraction=lead_fraction,
        fibre_volume_fraction=fibre_fraction,
        glue_volume_fraction=glue_fraction,
        lead_share_of_matrix=lead_share,
        density_g_cm3=density,
    )


# ----------------------------------------------------------------------
# Scoring meshes
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeshGrid:
    """A regular voxel grid. Flat index = ``(iz * ny + iy) * nx + ix``.

    The flat order matches Geant4's ``G4PSEnergyDeposit3D`` with z replicated
    outermost and x innermost.
    """

    origin_mm: tuple[float, float, float]
    voxel: VoxelSize
    shape: tuple[int, int, int]  # (nx, ny, nz)

    @property
    def size(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def unravel(
        self, flat_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(ix, iy, iz)`` arrays for flat voxel indices."""

        nx, ny, _ = self.shape
        flat = np.asarray(flat_index, dtype=np.int64)
        iz, rest = np.divmod(flat, nx * ny)
        iy, ix = np.divmod(rest, nx)
        return ix, iy, iz

    def centres_mm(
        self, flat_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return voxel-centre ``(x, y, z)`` arrays in millimetres."""

        ix, iy, iz = self.unravel(flat_index)
        x0, y0, z0 = self.origin_mm
        return (
            x0 + (ix + 0.5) * self.voxel.x_mm,
            y0 + (iy + 0.5) * self.voxel.y_mm,
            z0 + (iz + 0.5) * self.voxel.z_mm,
        )


def _mesh(
    geometry: ECALGeometry, voxel: VoxelSize, z_start_mm: float, depth_mm: float
) -> MeshGrid:
    pitch = geometry.cell_pitch_mm
    sampling = geometry.mean_readout_slice_thickness_mm

    # Nesting: voxels must tile cells and samplings exactly, and the grid must
    # start on a cell edge, so that voxel centres project without ambiguity.
    for name, value, unit in (
        ("x", voxel.x_mm, pitch),
        ("y", voxel.y_mm, pitch),
        ("z", voxel.z_mm, sampling),
    ):
        if not _whole_multiple(unit, value):
            raise TransportConfigError(
                f"voxel {name} = {value} mm does not tile the readout unit {unit} mm"
            )
    if not _whole_multiple(depth_mm, sampling):
        raise TransportConfigError(
            f"mesh depth {depth_mm} mm is not a whole number of samplings"
        )

    nx = round(geometry.width_x_mm / voxel.x_mm)
    ny = round(geometry.width_y_mm / voxel.y_mm)
    nz = round(depth_mm / voxel.z_mm)
    return MeshGrid(
        origin_mm=(-geometry.width_x_mm / 2, -geometry.width_y_mm / 2, z_start_mm),
        voxel=voxel,
        shape=(nx, ny, nz),
    )


def prefix_mesh(geometry: ECALGeometry, config: TransportConfig) -> MeshGrid:
    """Return the fine voxel grid covering the AMS-like active volume."""

    return _mesh(geometry, config.prefix_voxel, 0.0, geometry.depth_z_mm)


def extension_mesh(geometry: ECALGeometry, config: TransportConfig) -> MeshGrid:
    """Return the coarse voxel grid covering the downstream extension."""

    return _mesh(
        geometry,
        config.extension_voxel,
        geometry.depth_z_mm,
        config.extension_depth_mm,
    )


def extension_superlayer_count(geometry: ECALGeometry, config: TransportConfig) -> int:
    """Return how many whole superlayers the sampling extension holds."""

    thickness = geometry.sampling_structure.superlayer_thickness_mm
    if not _whole_multiple(config.extension_depth_mm, thickness):
        raise TransportConfigError(
            "extension depth must be a whole number of superlayers"
        )
    return round(config.extension_depth_mm / thickness)


def extended_layer_count(geometry: ECALGeometry, config: TransportConfig) -> int:
    """Return the number of readout-thickness samplings, prefix + extension."""

    return round(
        (geometry.depth_z_mm + config.extension_depth_mm)
        / geometry.mean_readout_slice_thickness_mm
    )
