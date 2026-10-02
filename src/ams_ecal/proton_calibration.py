"""Proton model calibration: the data split, the tables and the artifact.

Proton model turns the Geant4 proton pilot into a fast phenomenological generator.
Every distribution the generator draws from is built from PILOT EVENTS, so the
events used to build it and the events used to judge it must be separated
before either is looked at.

THE SPLIT. Within every stored batch (one sample at one energy) an event is
VALIDATION when its recorded ``event_index`` leaves remainder 3 on division by
4; otherwise it is CALIBRATION. The rule is fixed here, before any proton model model
choice was made, and depends on nothing but the event index:

* deterministic - no random number is drawn, so it cannot change between runs;
* interleaved - both subsets span the whole seed stream, so neither is a
  different "period" of the run;
* the same for every sample - a ``fixed_entry`` event reuses the seed of the
  baseline event with the same index (same base seed), so it inherits that
  event's status; a fixed-entry control event is held out exactly when its
  baseline twin is, which keeps the two from leaking into one another.

HONEST CAVEAT. The PILOT REPORT (``ams_ecal.pilot_report``) was run on every
event before this split existed, and its summary statistics informed the plan
for proton model. The split therefore protects the *fitted tables and every choice made
from now on*; it does not make the held-out events unseen in aggregate. What it
guarantees is that no table, threshold, binning or dependency decision in proton model is
computed from a validation event.

THE ARTIFACT. The runtime model never reads raw Geant4 events. It loads one
compact, versioned calibration built by ``build_calibration``:

    calibration.npz   the arrays (crossing quantile tables, interaction length)
    manifest.json     provenance: source run, physics list, energy anchors,
                      geometry / configuration hash, sample counts, split rule,
                      creation date, schema version, and a CONTENT HASH of the
                      arrays that ``ProtonCalibration.load`` re-checks

The build is deterministic (no random numbers): the same pilot data give the
same content hash; only ``created_utc`` differs.

    uv run python -m ams_ecal.proton_calibration build

CROSSING TABLE. For a proton that does not interact, the fibre (or all-material)
energy of ONE readout layer is a draw from a quantile function conditioned on
the primary energy and on the EXACT chord path the track cuts through that
layer's fibres (``ams_ecal.crossing``). Quantile functions are interpolated
linearly in ``ln E`` between the anchor energies, value by value, so the
interpolation is deterministic and monotone; nothing is extrapolated. Layers
are drawn independently: the dependency analysis (step 0) analysis shows this under-disperses the
event total (see ``research/plans/2026-09-29_proton_dependency_analysis.md``).
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.crossing import FibreCrossingGeometry
from ams_ecal.geant4_backend import PROJECT_ROOT, _git_state, load_batch
from ams_ecal.geometry import ECALGeometry, load_geometry
from ams_ecal.pilot_analysis import censored_exponential_rate
from ams_ecal.proton_structure import (
    K_MAX,
    SPILL_CELL_MEV,
    CrossingCalibrationInputs,
    CrossingStructure,
    build_structure,
)
from ams_ecal.proton_structure import describe as describe_structure
from ams_ecal.tracking import TrackState

VALIDATION_MODULUS = 4
VALIDATION_RESIDUE = 3

# Schema 2 may carry the crossing STRUCTURE (burst latent, lateral spill, bulk coupling) next to
# the per-layer table; schema 1 artifacts (the table alone) still load.
CALIBRATION_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "calibration" / "proton_6b"
ARTIFACT_V2 = PROJECT_ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"

REPRESENTATIONS = ("readout", "deposition")
_GRID_KEY = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}

DEFAULT_ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
DEFAULT_CHORD_BINS = 6
# Dense in the tail, where the distribution is heavy, but never beyond the
# largest observed value: 1.0 is the sample maximum.
DEFAULT_LEVELS = np.unique(
    np.concatenate(
        [
            [0.0, 0.001, 0.005, 0.01, 0.02],
            np.arange(0.05, 0.951, 0.05),
            [0.97, 0.98, 0.99, 0.995, 0.999, 1.0],
        ]
    )
)


# ----------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------


def validation_mask(event_index: np.ndarray) -> np.ndarray:
    """Return ``True`` for held-out validation events."""

    index = np.asarray(event_index)
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError("event_index must be an integer array")
    if np.any(index < 0):
        raise ValueError("event_index must be nonnegative")
    return (index % VALIDATION_MODULUS) == VALIDATION_RESIDUE


def calibration_mask(event_index: np.ndarray) -> np.ndarray:
    """Return ``True`` for events that may be used to build the model."""

    return ~validation_mask(event_index)


# ----------------------------------------------------------------------
# The crossing table
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossingTable:
    """Per-layer energy quantile functions of a proton that does not interact.

    ``quantiles_mev[representation]`` has shape
    ``(n_energies, n_chord_bins, n_levels)`` and is non-decreasing along the
    last axis. ``chord_edges_mm`` holds the INTERIOR bin edges.
    """

    energies_gev: np.ndarray
    chord_edges_mm: np.ndarray
    levels: np.ndarray
    quantiles_mev: dict[str, np.ndarray]
    counts: np.ndarray

    def __post_init__(self) -> None:
        n_energies, n_bins = len(self.energies_gev), len(self.chord_edges_mm) + 1
        if n_energies < 2 or np.any(np.diff(self.energies_gev) <= 0):
            raise ValueError("need at least two strictly increasing anchor energies")
        if np.any(np.diff(self.chord_edges_mm) <= 0):
            raise ValueError("chord edges must be strictly increasing")
        if self.levels[0] != 0.0 or self.levels[-1] != 1.0 or np.any(np.diff(self.levels) <= 0):
            raise ValueError("levels must increase strictly from 0 to 1")
        if set(self.quantiles_mev) != set(REPRESENTATIONS):
            raise ValueError(f"quantile tables are required for {list(REPRESENTATIONS)}")
        expected = (n_energies, n_bins, len(self.levels))
        for name, table in self.quantiles_mev.items():
            if table.shape != expected:
                raise ValueError(f"{name} table must have shape {expected}, got {table.shape}")
            if not np.all(np.isfinite(table)) or np.any(table < 0):
                raise ValueError(f"{name} table must be finite and nonnegative")
            if np.any(np.diff(table, axis=-1) < 0):
                raise ValueError(f"{name} quantiles must be non-decreasing")
        if self.counts.shape != (n_energies, n_bins):
            raise ValueError("counts must have shape (n_energies, n_chord_bins)")

    @property
    def energy_range_gev(self) -> tuple[float, float]:
        return float(self.energies_gev[0]), float(self.energies_gev[-1])

    def energy_weights(self, energy_gev: float) -> tuple[int, int, float]:
        """Return ``(lower, upper, w)`` so that ``ln E`` interpolates the anchors.

        Raises outside the anchor range: nothing is extrapolated.
        """

        low, high = self.energy_range_gev
        if not low <= energy_gev <= high:
            raise ValueError(
                f"energy {energy_gev:g} GeV is outside the calibrated range "
                f"[{low:g}, {high:g}] GeV; nothing is extrapolated"
            )
        upper = int(np.clip(np.searchsorted(self.energies_gev, energy_gev), 1, len(self.energies_gev) - 1))
        lower = upper - 1
        span = np.log(self.energies_gev[upper]) - np.log(self.energies_gev[lower])
        weight = (np.log(energy_gev) - np.log(self.energies_gev[lower])) / span
        return lower, upper, float(weight)

    def chord_bin(self, chord_mm: np.ndarray) -> np.ndarray:
        """Return the chord bin of each layer path."""

        return np.searchsorted(self.chord_edges_mm, np.asarray(chord_mm, dtype=float), side="right")

    def quantile_vector(self, representation: str, energy_gev: float, chord_mm: float) -> np.ndarray:
        """Return the interpolated quantile function for one layer."""

        lower, upper, weight = self.energy_weights(energy_gev)
        table = self.quantiles_mev[representation]
        index = int(self.chord_bin(np.asarray(chord_mm)))
        return (1.0 - weight) * table[lower, index] + weight * table[upper, index]

    def sample_layer_energies(
        self,
        representation: str,
        energy_gev: float,
        chord_mm: np.ndarray,
        uniform: np.ndarray,
    ) -> np.ndarray:
        """Return one energy per layer from uniform variates ``uniform``.

        A larger variate always gives a larger energy (a comonotone map), so
        the draws are deterministic given the variates.
        """

        chord = np.asarray(chord_mm, dtype=float)
        u = np.asarray(uniform, dtype=float)
        if chord.shape != u.shape:
            raise ValueError("chord_mm and uniform must have the same shape")
        if np.any((u < 0) | (u > 1)):
            raise ValueError("uniform variates must lie in [0, 1]")
        lower, upper, weight = self.energy_weights(energy_gev)
        table = self.quantiles_mev[representation]
        bins = self.chord_bin(chord)
        out = np.empty_like(u)
        for position, (bin_index, variate) in enumerate(zip(bins.ravel(), u.ravel(), strict=True)):
            vector = (1.0 - weight) * table[lower, bin_index] + weight * table[upper, bin_index]
            out.flat[position] = np.interp(variate, self.levels, vector)
        return out


# ----------------------------------------------------------------------
# Extraction of the crossing events' layer structure
# ----------------------------------------------------------------------


def crossed_cell_mask(
    crossing_geometry: FibreCrossingGeometry, track: TrackState, n_layers: int, n_cells: int
) -> np.ndarray:
    """Return the ``(n_layers, n_cells)`` mask of the cells a straight track crosses.

    A layer in which the track crosses no fibre counts the track's own cell as crossed, as
    ``FibreCrossingGeometry.spread_layer_energies`` does.
    """

    crossed = crossing_geometry.cross(track)
    mask = np.zeros((n_layers, n_cells), dtype=bool)
    inside = crossed.cell >= 0
    mask[crossed.layer[inside], crossed.cell[inside]] = True
    for layer in np.flatnonzero(~mask.any(axis=1)):
        mask[layer, crossing_geometry.track_cell(track, int(layer))] = True
    return mask


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Distance in cells from each cell of a layer row to its nearest crossed cell."""

    positions = np.flatnonzero(mask)
    cells = np.arange(mask.shape[0])
    return np.abs(cells[:, None] - positions[None, :]).min(axis=1)


def extract_crossing_inputs(
    energy_gev: float,
    arrays: dict[str, np.ndarray],
    crossing_geometry: FibreCrossingGeometry,
    n_layers: int,
    n_cells: int,
) -> CrossingCalibrationInputs:
    """Layer energies, chords and off-track structure of the calibration crossing events.

    ``arrays`` are CALIBRATION events of one batch (see ``_calibration_batch``); only the
    events whose truth says no inelastic interaction occurred are used.
    """

    rows = np.flatnonzero(~arrays["truth_occurred"])
    n = len(rows)
    chord = np.empty((n, n_layers))
    masks = np.empty((n, n_layers, n_cells), dtype=bool)
    for i, event in enumerate(rows):
        track = TrackState(
            x0_mm=float(arrays["entry_x_mm"][event]),
            y0_mm=float(arrays["entry_y_mm"][event]),
            z0_mm=0.0,
            theta_rad=0.0,
            phi_rad=0.0,
        )
        chord[i] = crossing_geometry.cross(track).layer_path_mm(n_layers)
        masks[i] = crossed_cell_mask(crossing_geometry, track, n_layers, n_cells)

    layer_energy: dict[str, np.ndarray] = {}
    spill_count: dict[str, np.ndarray] = {}
    spill_fraction: dict[str, np.ndarray] = {}
    spill_distance: dict[str, np.ndarray] = {}
    for name in REPRESENTATIONS:
        grid = arrays[_GRID_KEY[name]][rows].astype(float)
        total = grid.sum(axis=2)
        layer_energy[name] = total
        outside = np.where(masks, 0.0, grid)
        lit = outside > SPILL_CELL_MEV
        count = lit.sum(axis=2)
        spill_count[name] = np.minimum(count, K_MAX)
        spill_fraction[name] = np.divide(
            np.where(lit, outside, 0.0).sum(axis=2), total, out=np.zeros_like(total), where=total > 0
        )
        distances = np.zeros((n, n_layers, K_MAX), dtype=np.int64)
        for i, layer in zip(*np.nonzero(count), strict=True):
            cells = np.flatnonzero(lit[i, layer])
            strongest = cells[np.argsort(outside[i, layer, cells])[::-1][:K_MAX]]
            distances[i, layer, : len(strongest)] = _distance_to_mask(masks[i, layer])[strongest]
        spill_distance[name] = distances
    return CrossingCalibrationInputs(
        energy_gev=float(energy_gev),
        chord_mm=chord,
        layer_energy_mev=layer_energy,
        spill_count=spill_count,
        spill_fraction=spill_fraction,
        spill_distance=spill_distance,
    )


# ----------------------------------------------------------------------
# The artifact
# ----------------------------------------------------------------------


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes and bytes in a canonical order."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProtonCalibration:
    """A finalized the proton model calibration: tables plus provenance."""

    manifest: dict[str, Any]
    crossing: CrossingTable
    effective_length_mm: float
    effective_length_error_mm: float
    depth_mm: float
    structure: CrossingStructure | None = None

    @property
    def physics_list(self) -> str:
        return str(self.manifest["physics_list"])

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])

    def arrays(self) -> dict[str, np.ndarray]:
        return _arrays(
            self.crossing,
            self.effective_length_mm,
            self.effective_length_error_mm,
            self.depth_mm,
            self.structure,
        )

    def save(self, directory: str | Path) -> Path:
        """Write ``calibration.npz`` and ``manifest.json``."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        if content_sha256(arrays) != self.content_sha256:
            raise ValueError("manifest content_sha256 does not match the arrays")
        np.savez_compressed(directory / "calibration.npz", **arrays)
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ProtonCalibration:
        """Load and VERIFY an artifact; a tampered array raises."""

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported calibration schema_version {manifest.get('schema_version')!r}; "
                f"expected one of {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        with np.load(directory / "calibration.npz") as data:
            arrays = {name: data[name] for name in data.files}
        if content_sha256(arrays) != manifest["content_sha256"]:
            raise ValueError("calibration arrays do not match the manifest content hash")
        crossing = CrossingTable(
            energies_gev=arrays["energies_gev"],
            chord_edges_mm=arrays["chord_edges_mm"],
            levels=arrays["levels"],
            quantiles_mev={
                name: arrays[f"crossing_quantiles_{name}"] for name in REPRESENTATIONS
            },
            counts=arrays["crossing_counts"],
        )
        structure = (
            CrossingStructure.from_arrays(arrays, crossing.energies_gev)
            if "burst_probability" in arrays
            else None
        )
        return cls(
            manifest=manifest,
            crossing=crossing,
            effective_length_mm=float(arrays["effective_length_mm"]),
            effective_length_error_mm=float(arrays["effective_length_error_mm"]),
            depth_mm=float(arrays["depth_mm"]),
            structure=structure,
        )


def _arrays(
    crossing: CrossingTable,
    length_mm: float,
    length_error_mm: float,
    depth_mm: float,
    structure: CrossingStructure | None = None,
) -> dict[str, np.ndarray]:
    arrays = {
        "energies_gev": np.asarray(crossing.energies_gev, dtype=float),
        "chord_edges_mm": np.asarray(crossing.chord_edges_mm, dtype=float),
        "levels": np.asarray(crossing.levels, dtype=float),
        "crossing_counts": np.asarray(crossing.counts, dtype=np.int64),
        "effective_length_mm": np.asarray(length_mm, dtype=float),
        "effective_length_error_mm": np.asarray(length_error_mm, dtype=float),
        "depth_mm": np.asarray(depth_mm, dtype=float),
    }
    for name in REPRESENTATIONS:
        arrays[f"crossing_quantiles_{name}"] = np.asarray(crossing.quantiles_mev[name], dtype=float)
    if structure is not None:
        arrays.update(structure.arrays())
    return arrays


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


def _calibration_batch(sample: str, energy_gev: float, data: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    batch = load_batch(data / sample / f"E{energy_gev:g}GeV")
    keep = calibration_mask(batch.arrays["event_index"])
    n = len(keep)
    arrays = {
        name: value[keep]
        for name, value in batch.arrays.items()
        if getattr(value, "shape", None) and value.shape[:1] == (n,)
    }
    # the guarantee the whole split exists for
    if validation_mask(arrays["event_index"]).any():
        raise AssertionError("a validation event reached the calibration builder")
    return arrays, batch.metadata


def build_calibration(
    data: Path = DATA,
    *,
    sample: str = "baseline",
    physics_list: str = "ftfp_bert",
    energies_gev: Sequence[float] = DEFAULT_ENERGIES_GEV,
    n_chord_bins: int = DEFAULT_CHORD_BINS,
    levels: np.ndarray = DEFAULT_LEVELS,
    min_layers_per_bin: int = 20,
    geometry: ECALGeometry | None = None,
    structure: bool = True,
) -> ProtonCalibration:
    """Build the calibration from CALIBRATION events of a stored pilot sample.

    A quantile table needs enough layers in every (energy, chord bin) cell to
    mean something; below ``min_layers_per_bin`` the build refuses.

    With ``structure=True`` (the default) the per-layer table is rebuilt from burst-free
    events and the burst latent, lateral spill and bulk coupling are calibrated next to it
    (``ams_ecal.proton_structure``). ``structure=False`` builds the original table of all
    crossing layers alone, whose layers are drawn independently.
    """

    geometry = geometry or load_geometry(PROJECT_ROOT / "configs" / "geometry.yaml")
    crossing_geometry = FibreCrossingGeometry(geometry)
    n_layers = geometry.number_of_layers
    depth_mm = n_layers * geometry.mean_readout_slice_thickness_mm

    per_energy: dict[float, dict[str, Any]] = {}
    crossing_inputs: list[CrossingCalibrationInputs] = []
    for energy in energies_gev:
        arrays, metadata = _calibration_batch(sample, float(energy), data)
        if metadata["physics_list"].lower() != physics_list.lower():
            raise ValueError(
                f"batch physics list {metadata['physics_list']!r} is not {physics_list!r}"
            )
        occurred = arrays["truth_occurred"]
        crossing_rows = np.flatnonzero(~occurred)
        if structure:
            inputs = extract_crossing_inputs(
                float(energy), arrays, crossing_geometry, n_layers, geometry.cells_per_layer
            )
            crossing_inputs.append(inputs)
            paths, layer_energy = inputs.chord_mm, inputs.layer_energy_mev
        else:
            paths = np.empty((len(crossing_rows), n_layers))
            for row, event in enumerate(crossing_rows):
                track = TrackState(
                    x0_mm=float(arrays["entry_x_mm"][event]),
                    y0_mm=float(arrays["entry_y_mm"][event]),
                    z0_mm=0.0,
                    theta_rad=0.0,
                    phi_rad=0.0,
                )
                paths[row] = crossing_geometry.cross(track).layer_path_mm(n_layers)
            layer_energy = {
                name: arrays[_GRID_KEY[name]][crossing_rows].sum(axis=2).astype(float)
                for name in REPRESENTATIONS
            }
        per_energy[float(energy)] = {
            "arrays": arrays,
            "metadata": metadata,
            "crossing_rows": crossing_rows,
            "paths": paths,
            "layer_energy": layer_energy,
        }

    # chord bins: interior edges of equal-count bins over ALL calibration layers
    pooled = np.concatenate([entry["paths"].ravel() for entry in per_energy.values()])
    chord_edges = np.quantile(pooled, np.linspace(0.0, 1.0, n_chord_bins + 1)[1:-1])

    anchors = np.array(sorted(per_energy), dtype=float)
    crossing_structure: CrossingStructure | None = None
    if structure:
        crossing_structure, quantiles, structure_edges, counts = build_structure(
            crossing_inputs,
            n_chord_bins=n_chord_bins,
            bulk_levels=np.asarray(levels, dtype=float),
            min_layers_per_bin=min_layers_per_bin,
        )
        if not np.allclose(structure_edges, chord_edges):
            raise AssertionError("the structure and the table disagree on the chord bins")
    else:
        quantiles = {
            name: np.empty((len(anchors), n_chord_bins, len(levels))) for name in REPRESENTATIONS
        }
        counts = np.zeros((len(anchors), n_chord_bins), dtype=np.int64)
        for a, energy in enumerate(anchors):
            entry = per_energy[float(energy)]
            bins = np.searchsorted(chord_edges, entry["paths"], side="right")
            for b in range(n_chord_bins):
                selected = bins == b
                counts[a, b] = int(selected.sum())
                if counts[a, b] < min_layers_per_bin:
                    raise ValueError(
                        f"only {counts[a, b]} crossing layers at {energy:g} GeV in chord bin {b}"
                    )
                for name in REPRESENTATIONS:
                    quantiles[name][a, b] = np.quantile(entry["layer_energy"][name][selected], levels)

    crossing_table = CrossingTable(
        energies_gev=anchors,
        chord_edges_mm=chord_edges,
        levels=np.asarray(levels, dtype=float),
        quantiles_mev=quantiles,
        counts=counts,
    )

    # one pooled exponential rate: no energy trend was resolved in the pilot
    depths = np.concatenate(
        [e["arrays"]["truth_z_mm"][e["arrays"]["truth_occurred"]] for e in per_energy.values()]
    )
    n_crossing = int(sum((~e["arrays"]["truth_occurred"]).sum() for e in per_energy.values()))
    pooled_rate = censored_exponential_rate(depths, n_crossing, depth_mm)
    length_mm = 1.0 / pooled_rate.rate_per_mm
    length_error_mm = length_mm / np.sqrt(pooled_rate.n_interacting)

    per_energy_length = {}
    for energy, entry in per_energy.items():
        occurred = entry["arrays"]["truth_occurred"]
        rate = censored_exponential_rate(
            entry["arrays"]["truth_z_mm"][occurred], int((~occurred).sum()), depth_mm
        )
        per_energy_length[f"{energy:g}"] = {
            "effective_length_mm": 1.0 / rate.rate_per_mm,
            "standard_error_mm": (1.0 / rate.rate_per_mm) / np.sqrt(rate.n_interacting),
            "n_interacting": rate.n_interacting,
            "n_crossing": rate.n_crossing,
        }

    arrays = _arrays(crossing_table, length_mm, length_error_mm, depth_mm, crossing_structure)
    first_metadata = next(iter(per_energy.values()))["metadata"]
    git = _git_state()
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "proton_calibration",
        "physics_list": physics_list,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content_sha256": content_sha256(arrays),
        "wording": "Geant4-calibrated proton phenomenology; not a true proton shower model",
        "split_rule": {
            "validation_modulus": VALIDATION_MODULUS,
            "validation_residue": VALIDATION_RESIDUE,
            "used": "calibration events only",
        },
        "source": {
            "sample": sample,
            "particle": first_metadata["particle"],
            "geant4_version": first_metadata["geant4_version"],
            "geant4_pybind_version": first_metadata["geant4_pybind_version"],
            "geant4_datasets": first_metadata["geant4_datasets"],
            "physics_list_as_run": first_metadata["physics_list"],
            "production_cut_mm": first_metadata["production_cut_mm"],
            "geometry_variant": first_metadata["geometry_variant"],
            "incidence": {"theta_rad": first_metadata["theta_rad"], "phi_rad": first_metadata["phi_rad"]},
            "entry_spot_mm": first_metadata["entry_spot_mm"],
            "materials": {
                key: first_metadata["materials"][key]
                for key in ("matrix_constraint", "composite_density_g_cm3", "prefix_depth_lambda_i", "prefix_depth_x0")
            },
            "pilot_batches": {
                f"{energy:g}": {
                    "configuration_sha256": entry["metadata"]["configuration_sha256"],
                    "base_seed": entry["metadata"]["base_seed"],
                    "git_commit": entry["metadata"]["git"]["commit"],
                    "created_utc": entry["metadata"]["created_utc"],
                }
                for energy, entry in per_energy.items()
            },
            "approximations": first_metadata["geometry_approximations"],
        },
        "energies_gev": [float(e) for e in anchors],
        "counts": {
            f"{energy:g}": {
                "calibration_events": len(entry["arrays"]["event_index"]),
                "crossing": len(entry["crossing_rows"]),
                "interacting": int(entry["arrays"]["truth_occurred"].sum()),
            }
            for energy, entry in per_energy.items()
        },
        "interaction": {
            "effective_length_mm": float(length_mm),
            "standard_error_mm": float(length_error_mm),
            "ecal_depth_mm": float(depth_mm),
            "definition": (
                "pooled maximum-likelihood exponential rate of the first inelastic "
                "interaction depth, censored by the ECAL depth, over all anchor "
                "energies (no energy trend was resolved); an EFFECTIVE length of "
                "this geometry and material model, not a physical interaction "
                "length of AMS"
            ),
            "per_energy": per_energy_length,
        },
        "crossing": {
            "n_chord_bins": int(n_chord_bins),
            "chord_edges_mm": [float(x) for x in chord_edges],
            "levels": [float(x) for x in levels],
            "layers_per_bin": counts.tolist(),
            "representations": list(REPRESENTATIONS),
            "layers_independent": crossing_structure is None,
            "table_built_from": (
                "all crossing layers"
                if crossing_structure is None
                else "layers of crossing events without a burst"
            ),
        },
        "structure": None if crossing_structure is None else describe_structure(crossing_structure),
        "builder": {
            "module": "ams_ecal.proton_calibration",
            "git_commit": git["commit"],
            "tracked_changes": git["tracked_changes"],
        },
    }
    return ProtonCalibration(
        manifest=manifest,
        crossing=crossing_table,
        effective_length_mm=float(length_mm),
        effective_length_error_mm=float(length_error_mm),
        depth_mm=float(depth_mm),
        structure=crossing_structure,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the FTFP_BERT calibration from the pilot")
    build.add_argument("--data", type=Path, default=DATA)
    build.add_argument("--sample", default="baseline")
    build.add_argument("--physics-list", default="ftfp_bert")
    build.add_argument("--out", type=Path, default=ARTIFACT_V2)
    build.add_argument(
        "--no-structure",
        action="store_true",
        help="build the per-layer table alone (layers drawn independently), as artifact schema 1 did",
    )
    args = parser.parse_args(argv)
    calibration = build_calibration(
        args.data,
        sample=args.sample,
        physics_list=args.physics_list,
        structure=not args.no_structure,
    )
    print(calibration.save(args.out))
    print(f"content_sha256 {calibration.content_sha256}")
    print(f"effective interaction length {calibration.effective_length_mm:.1f} +- {calibration.effective_length_error_mm:.1f} mm")


if __name__ == "__main__":
    main()
