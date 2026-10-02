"""Exact fibre geometry of a straight track through the AMS ECAL.

Proton model's crossing-proton response is conditioned on WHERE the track lies
relative to the fibre lattice. This module answers only the geometric part,
with the lattice and the fibre-to-cell map of ``ams_ecal.transport_geometry``
- the same ones the Geant4 backend used - so there is no second, approximate
detector mapping:

* which fibres a normally incident track crosses, in every readout layer;
* the CHORD it cuts through each (``2 sqrt(r^2 - d^2)``, ``d`` the distance
  from the fibre axis);
* the readout cell each crossed fibre belongs to, using the half-open cell rule
  applied to the fibre CENTRE (a fibre on a boundary goes to the upper cell);
* the depth of each fibre row, so the fibres upstream of an interaction point
  can be selected.

VALIDATION (dependency analysis (step 0), ``ams_ecal.proton_dependency.crossing_section``): the
Geant4 energy in the fibres crossed by the straight track is 0.195-0.204
MeV/mm of chord against 0.205 for a polystyrene MIP, so this is the geometry
Geant4 saw. It explains only 3-5% of the per-event variance of a crossing
proton (delta rays and small cascades supply the rest), which is why the
response is a TABLE conditioned on this geometry, not a deterministic term.

SCOPE. Normal incidence only: no angular calibration exists, and a tilted track
would need the fibre-axis-projected chord. A tilted or out-of-grid track raises
instead of returning a wrong answer.
"""

from dataclasses import dataclass

import numpy as np

from ams_ecal.geometry import ECALGeometry
from ams_ecal.readout import coordinate_to_cell_index
from ams_ecal.tracking import TrackState
from ams_ecal.transport_geometry import FibreLayout


@dataclass(frozen=True, slots=True)
class CrossedFibres:
    """The fibres one straight track crosses, with chord, cell and depth."""

    layer: np.ndarray  # (n,) readout layer 0..17
    cell: np.ndarray  # (n,) readout cell of the fibre, -1 outside the grid
    chord_mm: np.ndarray  # (n,) path through the fibre
    depth_mm: np.ndarray  # (n,) depth of the fibre axis behind the front face

    def __len__(self) -> int:
        return len(self.layer)

    def layer_path_mm(self, n_layers: int) -> np.ndarray:
        """Return the total chord in each readout layer."""

        return np.bincount(self.layer, weights=self.chord_mm, minlength=n_layers)

    def upstream_of(self, depth_mm: float) -> CrossedFibres:
        """Return the fibres whose axis lies in front of ``depth_mm``."""

        keep = self.depth_mm < depth_mm
        return CrossedFibres(
            self.layer[keep], self.cell[keep], self.chord_mm[keep], self.depth_mm[keep]
        )


class FibreCrossingGeometry:
    """Precomputed lattice lookups for fast per-track geometry."""

    def __init__(self, geometry: ECALGeometry) -> None:
        self.geometry = geometry
        self.layout = FibreLayout(geometry)
        layout = self.layout
        self._radius = layout.fibre_radius_mm
        self._rows = layout.rows_per_superlayer
        self._superlayers = geometry.number_of_superlayers
        self._centres = [np.asarray(layout.fibre_centres_mm(row)) for row in range(self._rows)]
        self._lookup_layer, self._lookup_cell = layout.readout_lookup()
        thickness = geometry.sampling_structure.superlayer_thickness_mm
        self._row_depth = np.array(
            [
                [s * thickness + layout.row_offset_mm(row) for row in range(self._rows)]
                for s in range(self._superlayers)
            ]
        )
        self._measures_x = [layout.measured_axis(s) == "x" for s in range(self._superlayers)]

    @property
    def n_layers(self) -> int:
        return self.geometry.number_of_layers

    def _check(self, track: TrackState) -> None:
        if track.theta_rad != 0.0:
            raise ValueError(
                "only normally incident tracks are supported: no angular calibration exists"
            )
        if (
            abs(track.x0_mm) >= self.geometry.width_x_mm / 2
            or abs(track.y0_mm) >= self.geometry.width_y_mm / 2
        ):
            raise ValueError("the track lies outside the active area of the ECAL")

    def cross(self, track: TrackState) -> CrossedFibres:
        """Return every fibre a normally incident track crosses."""

        self._check(track)
        layers, cells, chords, depths = [], [], [], []
        for superlayer in range(self._superlayers):
            coordinate = track.x0_mm if self._measures_x[superlayer] else track.y0_mm
            for row in range(self._rows):
                centres = self._centres[row]
                upper = int(np.clip(np.searchsorted(centres, coordinate), 1, len(centres) - 1))
                # the two lattice neighbours of the coordinate; the pitch
                # exceeds the diameter, so at most one of them is crossed
                index = (
                    upper
                    if abs(coordinate - centres[upper]) < abs(coordinate - centres[upper - 1])
                    else upper - 1
                )
                distance = abs(coordinate - centres[index])
                if distance >= self._radius:
                    continue
                fibre_id = self.layout.encode(superlayer, row, index)
                layers.append(int(self._lookup_layer[fibre_id]))
                cells.append(int(self._lookup_cell[fibre_id]))
                chords.append(2.0 * np.sqrt(self._radius**2 - distance**2))
                depths.append(self._row_depth[superlayer, row])
        return CrossedFibres(
            np.asarray(layers, dtype=np.int64),
            np.asarray(cells, dtype=np.int64),
            np.asarray(chords, dtype=float),
            np.asarray(depths, dtype=float),
        )

    def track_cell(self, track: TrackState, layer: int) -> int:
        """Return the cell that contains the track coordinate in one layer."""

        superlayer = layer // (self.n_layers // self._superlayers)
        coordinate = track.x0_mm if self._measures_x[superlayer] else track.y0_mm
        cell = coordinate_to_cell_index(coordinate, self.geometry)
        if cell is None:
            raise ValueError("the track lies outside the active area of the ECAL")
        return cell

    def spread_layer_energies(
        self, track: TrackState, crossed: CrossedFibres, layer_energy_mev: np.ndarray
    ) -> np.ndarray:
        """Distribute each layer's energy over cells in proportion to chord.

        A layer with no crossed fibre puts its (necessarily secondary) energy
        in the cell containing the track. Energy is conserved layer by layer.
        """

        energy = np.asarray(layer_energy_mev, dtype=float)
        if energy.shape != (self.n_layers,):
            raise ValueError(f"layer_energy_mev must have shape ({self.n_layers},)")
        if np.any(energy < 0) or not np.all(np.isfinite(energy)):
            raise ValueError("layer energies must be finite and nonnegative")
        grid = np.zeros((self.n_layers, self.geometry.cells_per_layer))
        for layer in range(self.n_layers):
            if energy[layer] == 0.0:
                continue
            here = (crossed.layer == layer) & (crossed.cell >= 0)
            if not here.any():
                grid[layer, self.track_cell(track, layer)] += energy[layer]
                continue
            weights = crossed.chord_mm[here]
            np.add.at(grid[layer], crossed.cell[here], energy[layer] * weights / weights.sum())
        return grid
