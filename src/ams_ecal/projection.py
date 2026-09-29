"""Projection of fine energy deposits into the canonical alternating readout.

Any backend that knows *where* energy was deposited - Geant4 steps, voxel
centres, fibre centres - reaches the canonical ``layers x 72`` representation
through ``project_deposits``. It applies exactly the conventions of
``ams_ecal.readout``:

* layer ``l`` covers depth ``[l*t, (l+1)*t)`` with ``t`` the uniform readout
  slice thickness, lower edge included;
* each layer measures the coordinate across its parent superlayer's fibres,
  alternating superlayer by superlayer;
* cells are half-open, lower edge included, and a deposit outside the active
  grid is *not* clamped to an edge cell: its energy is returned separately.

``n_layers`` may exceed the 18 physical samplings. Layers beyond them continue
the same thickness and the same superlayer alternation, which is how the
research-instrument extension behind the ECAL is read in the same
representation as the ECAL itself.

The cell rule is vectorized here for speed; ``tests/test_projection.py`` checks
it against the scalar ``coordinate_to_cell_index`` on boundaries and random
points, so the two cannot drift apart.
"""

from dataclasses import dataclass

import numpy as np

from ams_ecal.geometry import ECALGeometry
from ams_ecal.readout import measured_axis_for_fiber


@dataclass(frozen=True, slots=True)
class ProjectedDeposits:
    """Energy in the alternating readout, plus what fell outside it."""

    grid_mev: np.ndarray  # (n_layers, cells_per_layer)
    outside_mev: float


def measured_axis_for_layer(layer_index: int, geometry: ECALGeometry) -> str:
    """Return the measured coordinate of a (possibly extended) readout layer.

    Inside the ECAL this equals the readout module's answer; beyond it the
    superlayer alternation simply continues.
    """

    per_superlayer = geometry.number_of_layers // geometry.number_of_superlayers
    superlayer = layer_index // per_superlayer
    axes = geometry.superlayer_fiber_axes
    fibre_axis = axes[superlayer % 2] if superlayer >= len(axes) else axes[superlayer]
    return measured_axis_for_fiber(fibre_axis)


def layer_measures_x(geometry: ECALGeometry, n_layers: int) -> np.ndarray:
    """Return a boolean array: does layer ``l`` measure x (else y)?"""

    return np.array(
        [measured_axis_for_layer(layer, geometry) == "x" for layer in range(n_layers)]
    )


def cell_indices(coordinate_mm: np.ndarray, geometry: ECALGeometry) -> np.ndarray:
    """Vectorized ``coordinate_to_cell_index``; -1 marks outside the grid."""

    coordinate = np.asarray(coordinate_mm, dtype=float)
    grid_width_mm = geometry.cells_per_layer * geometry.cell_pitch_mm
    lower = -grid_width_mm / 2
    upper = grid_width_mm / 2
    index = np.floor((coordinate - lower) / geometry.cell_pitch_mm).astype(np.int64)
    inside = (coordinate >= lower) & (coordinate < upper)
    return np.where(inside, index, -1)


def layer_indices(
    z_mm: np.ndarray, geometry: ECALGeometry, n_layers: int
) -> np.ndarray:
    """Return the readout layer of each depth; -1 marks outside ``[0, n*t)``."""

    z = np.asarray(z_mm, dtype=float)
    thickness = geometry.mean_readout_slice_thickness_mm
    index = np.floor(z / thickness).astype(np.int64)
    inside = (z >= 0.0) & (index < n_layers)
    return np.where(inside, index, -1)


def project_deposits(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    energy_mev: np.ndarray,
    geometry: ECALGeometry,
    n_layers: int | None = None,
) -> ProjectedDeposits:
    """Sum point deposits into the ``n_layers x cells_per_layer`` readout."""

    layers = geometry.number_of_layers if n_layers is None else n_layers
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    energy = np.asarray(energy_mev, dtype=float)

    if not (x.shape == y.shape == np.shape(z_mm) == energy.shape):
        raise ValueError("x, y, z and energy must have the same shape")
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("deposited energies must be finite and nonnegative")

    layer = layer_indices(z_mm, geometry, layers)
    measures_x = layer_measures_x(geometry, layers)
    measured = np.where(measures_x[np.clip(layer, 0, None)], x, y)
    cell = cell_indices(measured, geometry)

    inside = (layer >= 0) & (cell >= 0)
    grid = np.zeros((layers, geometry.cells_per_layer))
    np.add.at(grid, (layer[inside], cell[inside]), energy[inside])
    return ProjectedDeposits(grid_mev=grid, outside_mev=float(energy[~inside].sum()))
