"""Block 6B Slice 0: which dependencies must the proton generator keep explicit?

The plan (section 9) asks for one small dependency analysis on the EXISTING
Geant4 pilot events before the generator's factorization is frozen: after
conditioning on the remaining depth ``R`` and / or the visible energy, what
dependence is left between lateral width, hit multiplicity, longitudinal
morphology and energy?

Everything here uses CALIBRATION events only (``ams_ecal.proton_calibration``);
no held-out event enters any number below. Definitions are simple and
interpretable - binned conditional means, rank correlations, principal
components of log layer energies - not a fitted dependency model.

NOTATION (normal incidence, AMS-only geometry)

* ``D``     first-inelastic-interaction depth from the front face, mm
            (``truth_z_mm``); ``S_int`` in the plan.
* ``R``     remaining depth ``L - D``, mm, with ``L = 166.5``.
* ``l_D``   readout layer containing the interaction, ``floor(D / t)``.
* ``k``     layer OFFSET from the interaction layer: layer ``l_D + k``.
* ``A3``    AMPLITUDE: fibre energy in layers ``l_D + 1 .. l_D + 3``. A crude
            but model-free measure of how large the post-interaction cascade
            is, defined only where those layers exist (``l_D <= 14``).
* ``j``     UPSTREAM distance: layer ``l_D - j``.

CONVENTIONS

* Rank correlations are Spearman. A PARTIAL rank correlation removes, from the
  ranked variables, their mean inside equal-count cells of the conditioning
  variable(s) before correlating. It is a diagnostic of what dependence is
  left, not a causal statement.
* Intervals are Fisher-z approximations, ``se = 1.06 / sqrt(n - 3)``, ignoring
  the estimation error of the cell means.

    uv run python -m ams_ecal.proton_dependency

writes ``results/block6b/slice0/summary.json``.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.geant4_backend import PROJECT_ROOT, Batch, load_batch
from ams_ecal.geometry import ECALGeometry, load_geometry
from ams_ecal.pilot_analysis import event_observables, track_cells
from ams_ecal.proton_calibration import calibration_mask
from ams_ecal.transport_geometry import FibreLayout

DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
RESULTS = PROJECT_ROOT / "results" / "block6b" / "slice0"

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
# Half a MIP, in the pilot's measured MIP units (results/geant4_proton_pilot/
# summary.json). Fixed constants, not re-measured, so hit counts here are the
# same observable as in the pilot report.
PILOT_CELL_THRESHOLD_MEV = 0.2798003852367401
PILOT_LAYER_THRESHOLD_MEV = 0.2843949943780899

R_BINS = 10
GRID_BINS = 5  # for the (R, visible energy) grid: 5 x 5 cells
AMPLITUDE_OFFSETS = (1, 2, 3)
L_D_GROUPS = ((0, 3), (4, 7), (8, 11))
UNIVERSALITY_OFFSETS = tuple(range(8))
UPSTREAM_DISTANCES = tuple(range(1, 7))
LATERAL_OFFSETS = (1, 2, 3, 5, 8)
# Convention adopted 2026-09-29, AFTER the first table was read: a dependence
# is "explicit" if |partial rho| >= 0.20 at two or more energies with one sign.
EXPLICIT_RHO = 0.20
WEAK_RHO = 0.10


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def rank_uniform(x: np.ndarray) -> np.ndarray:
    """Return the ranks of ``x`` scaled to (0, 1]."""

    values = np.asarray(x, dtype=float)
    return stats.rankdata(values) / len(values)


def equal_count_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to one of ``n_bins`` equal-count bins."""

    values = np.asarray(x, dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, n_bins - 1)


def residual_within_cells(y: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Subtract from ``y`` its mean inside each cell."""

    out = np.asarray(y, dtype=float).copy()
    for cell in np.unique(cells):
        selected = cells == cell
        out[selected] -= out[selected].mean()
    return out


def joint_cells(given: Sequence[tuple[np.ndarray, int]]) -> np.ndarray:
    """Combine several equal-count binnings into one cell index."""

    cells = np.zeros(len(given[0][0]), dtype=np.int64)
    for values, n_bins in given:
        cells = cells * n_bins + equal_count_bins(values, n_bins)
    return cells


def partial_spearman(
    u: np.ndarray,
    v: np.ndarray,
    given: Sequence[tuple[np.ndarray, int]] = (),
) -> float:
    """Rank correlation of ``u`` and ``v`` after removing cell means of ``given``.

    ``given`` is a sequence of ``(values, n_bins)`` pairs; with none it is the
    plain Spearman correlation.
    """

    ranked_u, ranked_v = rank_uniform(u), rank_uniform(v)
    if given:
        cells = joint_cells(given)
        ranked_u = residual_within_cells(ranked_u, cells)
        ranked_v = residual_within_cells(ranked_v, cells)
    return float(np.corrcoef(ranked_u, ranked_v)[0, 1])


def correlation_interval(rho: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% interval for a rank correlation (Fisher z)."""

    if n <= 3:
        raise ValueError("need more than three events")
    centre = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    half = z * 1.06 / np.sqrt(n - 3)
    return float(np.tanh(centre - half)), float(np.tanh(centre + half))


def interaction_layer(depth_mm: np.ndarray, layer_thickness_mm: float) -> np.ndarray:
    """Return the readout layer that contains each interaction depth."""

    return np.floor(np.asarray(depth_mm, dtype=float) / layer_thickness_mm).astype(np.int64)


def layer_offset_matrix(
    layer_energy: np.ndarray, first_layer: np.ndarray, offsets: Sequence[int]
) -> np.ndarray:
    """Return ``e[i, first_layer[i] + k]`` for each offset ``k``; NaN if outside.

    Offsets may be negative, which addresses layers upstream of ``first_layer``.
    """

    energy = np.asarray(layer_energy, dtype=float)
    n_events, n_layers = energy.shape
    out = np.full((n_events, len(offsets)), np.nan)
    rows = np.arange(n_events)
    for column, offset in enumerate(offsets):
        layer = np.asarray(first_layer) + offset
        inside = (layer >= 0) & (layer < n_layers)
        out[rows[inside], column] = energy[rows[inside], layer[inside]]
    return out


def amplitude(layer_energy: np.ndarray, first_layer: np.ndarray) -> np.ndarray:
    """Return A3, NaN where the three layers after the interaction do not exist."""

    offsets = layer_offset_matrix(layer_energy, first_layer, AMPLITUDE_OFFSETS)
    return offsets.sum(axis=1)


def first_component_share(matrix: np.ndarray) -> float:
    """Share of the total variance carried by the first principal component."""

    centred = np.asarray(matrix, dtype=float)
    centred = centred - centred.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred.T))
    return float(eigenvalues[-1] / eigenvalues.sum())


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------


def load_calibration(sample: str, energy_gev: float, data: Path = DATA) -> dict[str, Any]:
    """Return one batch's arrays restricted to CALIBRATION events."""

    batch = load_batch(data / sample / f"E{energy_gev:g}GeV")
    keep = calibration_mask(batch.arrays["event_index"])
    n = len(keep)
    arrays: dict[str, Any] = {
        name: value[keep]
        for name, value in batch.arrays.items()
        if getattr(value, "shape", None) and value.shape[:1] == (n,)
    }
    arrays["_batch"] = batch  # sparse per-fibre arrays are indexed through this
    arrays["_kept_index"] = np.flatnonzero(keep)
    return arrays


# ----------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------

PAIRS = (
    ("log_visible", "width"),
    ("log_visible", "log_hits"),
    ("log_visible", "cog"),
    ("log_visible", "rms"),
    ("log_hits", "cog"),
    ("log_hits", "rms"),
    ("log_hits", "width"),
    ("cog", "width"),
    ("rms", "width"),
    ("log_visible", "log_deposit"),
)


def dependency_table(
    arrays: dict[str, Any], geometry: ECALGeometry, depth_mm: float
) -> dict[str, Any]:
    """Raw and partial rank correlations among the pilot observables.

    Interacting events only. Each pair is reported raw, given ``R`` (ten
    equal-count bins), and - when neither member is the visible energy - given
    the ``R x visible energy`` grid (5 x 5 equal-count cells).
    """

    occurred = arrays["truth_occurred"]
    remaining = depth_mm - arrays["truth_z_mm"][occurred]

    def observables(grid_name: str) -> dict[str, np.ndarray]:
        return event_observables(
            arrays[grid_name],
            geometry,
            arrays["entry_x_mm"],
            arrays["entry_y_mm"],
            cell_threshold_mev=PILOT_CELL_THRESHOLD_MEV,
            layer_threshold_mev=PILOT_LAYER_THRESHOLD_MEV,
        )

    readout, deposit = observables("readout_grid_mev"), observables("deposit_grid_mev")
    variables = {
        "log_visible": np.log(readout["energy_mev"][occurred]),
        "log_deposit": np.log(deposit["energy_mev"][occurred]),
        "cog": readout["long_cog_mm"][occurred],
        "rms": readout["long_rms_mm"][occurred],
        "width": readout["width_mm"][occurred],
        "log_hits": np.log(np.maximum(readout["n_hit_cells"][occurred], 1)),
    }
    n = len(remaining)
    table: dict[str, Any] = {"n_events": n}
    for first, second in PAIRS:
        u, v = variables[first], variables[second]
        row: dict[str, Any] = {
            "raw": partial_spearman(u, v),
            "given_R": partial_spearman(u, v, [(remaining, R_BINS)]),
        }
        if "log_visible" not in (first, second):
            row["given_R_and_visible"] = partial_spearman(
                u, v, [(remaining, GRID_BINS), (variables["log_visible"], GRID_BINS)]
            )
        low, high = correlation_interval(row["given_R"], n)
        row["given_R_interval"] = [low, high]
        table[f"{first}~{second}"] = row
    return table


def amplitude_section(arrays: dict[str, Any], layer_mm: float) -> dict[str, Any]:
    """Is the post-interaction amplitude independent of depth, and how is it shaped?"""

    occurred = arrays["truth_occurred"]
    depth = arrays["truth_z_mm"][occurred]
    layers = arrays["readout_grid_mev"][occurred].sum(axis=2).astype(float)
    first = interaction_layer(depth, layer_mm)
    a3 = amplitude(layers, first)
    usable = np.isfinite(a3)
    a3, depth = a3[usable], depth[usable]
    tercile_edges = np.quantile(depth, [0, 1 / 3, 2 / 3, 1])
    by_tercile = []
    for index in range(3):
        selected = (depth >= tercile_edges[index]) & (depth <= tercile_edges[index + 1])
        by_tercile.append(
            {
                "depth_range_mm": [float(tercile_edges[index]), float(tercile_edges[index + 1])],
                "quantiles_5_16_50_84_95": np.quantile(
                    a3[selected], [0.05, 0.16, 0.5, 0.84, 0.95]
                ).tolist(),
            }
        )
    early = a3[depth < np.median(depth)]
    late = a3[depth >= np.median(depth)]
    ks = stats.ks_2samp(early, late)
    rho = float(stats.spearmanr(a3, depth).statistic)
    return {
        "n": len(a3),
        "median_mev": float(np.median(a3)),
        "quantiles_5_16_50_84_95": np.quantile(a3, [0.05, 0.16, 0.5, 0.84, 0.95]).tolist(),
        "normalized_quantiles_5_16_50_84_95": np.quantile(
            a3 / np.median(a3), [0.05, 0.16, 0.5, 0.84, 0.95]
        ).tolist(),
        "spearman_with_depth": rho,
        "spearman_with_depth_interval": list(correlation_interval(rho, len(a3))),
        "by_depth_tercile": by_tercile,
        "ks_early_vs_late": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
        "n_exactly_zero": int((a3 == 0).sum()),
        "_values": a3,
    }


def universality_section(arrays: dict[str, Any], layer_mm: float) -> dict[str, Any]:
    """Is layer energy a universal function of the OFFSET from the interaction?"""

    occurred = arrays["truth_occurred"]
    depth = arrays["truth_z_mm"][occurred]
    layers = arrays["readout_grid_mev"][occurred].sum(axis=2).astype(float)
    first = interaction_layer(depth, layer_mm)
    offsets = layer_offset_matrix(layers, first, UNIVERSALITY_OFFSETS)
    medians: dict[str, list[float | None]] = {}
    for low, high in L_D_GROUPS:
        group = (first >= low) & (first <= high)
        medians[f"{low}-{high}"] = [
            (
                float(np.nanmedian(offsets[group, column]))
                if np.isfinite(offsets[group, column]).sum() >= 15
                else None
            )
            for column in range(len(UNIVERSALITY_OFFSETS))
        ]
    spreads = []
    for column in range(len(UNIVERSALITY_OFFSETS)):
        values = [m[column] for m in medians.values() if m[column] is not None]
        if len(values) >= 2:
            spreads.append(float((max(values) - min(values)) / np.mean(values)))
    # rank-1 structure of the log profile, events with at least eight layers after D
    deep = first <= 8
    profile = layer_offset_matrix(layers[deep], first[deep], range(1, 9))
    log_profile = np.log(np.maximum(profile, 1e-3))
    log_profile = log_profile - log_profile.mean(axis=0)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(log_profile.T)))[::-1]
    return {
        "layer_offsets": list(UNIVERSALITY_OFFSETS),
        "median_layer_energy_by_l_D_group": medians,
        "relative_median_spread_across_groups": spreads,
        "max_relative_median_spread": float(max(spreads)) if spreads else None,
        "log_profile_pca_shares": (eigenvalues / eigenvalues.sum())[:4].tolist(),
        "n_events_for_pca": int(deep.sum()),
    }


def upstream_section(
    arrays: dict[str, Any], layer_mm: float, mip_layer_mev: float
) -> dict[str, Any]:
    """Energy in the layers BEFORE the interaction layer, and its link to the amplitude."""

    occurred = arrays["truth_occurred"]
    depth = arrays["truth_z_mm"][occurred]
    layers = arrays["readout_grid_mev"][occurred].sum(axis=2).astype(float)
    first = interaction_layer(depth, layer_mm)
    a3 = amplitude(layers, first)
    rows = []
    for distance in UPSTREAM_DISTANCES:
        upstream = layer_offset_matrix(layers, first, [-distance])[:, 0]
        usable = np.isfinite(upstream) & np.isfinite(a3)
        rho = float(stats.spearmanr(upstream[usable], a3[usable]).statistic)
        rows.append(
            {
                "distance_layers": distance,
                "n": int(usable.sum()),
                "median_mev": float(np.median(upstream[usable])),
                "median_excess_over_mip_mev": float(np.median(upstream[usable]) - mip_layer_mev),
                "quantile_84_mev": float(np.quantile(upstream[usable], 0.84)),
                "spearman_with_amplitude": rho,
            }
        )
    return {"mip_layer_mev": mip_layer_mev, "by_distance": rows}


def lateral_section(arrays: dict[str, Any], geometry: ECALGeometry) -> dict[str, Any]:
    """Lateral kernel and event-level width of the post-interaction layers."""

    occurred = arrays["truth_occurred"]
    layer_mm = geometry.mean_readout_slice_thickness_mm
    depth = arrays["truth_z_mm"][occurred]
    grid = arrays["readout_grid_mev"][occurred].astype(float)
    cells = track_cells(
        geometry, grid.shape[1], arrays["entry_x_mm"][occurred], arrays["entry_y_mm"][occurred]
    )
    first = interaction_layer(depth, layer_mm)
    offset = np.arange(grid.shape[2])[None, None, :] - cells[:, :, None]
    kernel = {}
    for k in LATERAL_OFFSETS:
        rows = np.flatnonzero(first + k <= grid.shape[1] - 1)
        layer = first[rows] + k
        energy = grid[rows, layer, :]
        distance = np.abs(offset[rows, layer, :])
        total = energy.sum(axis=1)
        good = total > 0
        kernel[str(k)] = {
            "fraction_at_abs_cell_offset_0_1_2_3to4_5plus": [
                float(((energy * mask).sum(axis=1)[good] / total[good]).mean())
                for mask in (
                    distance == 0,
                    distance == 1,
                    distance == 2,
                    (distance == 3) | (distance == 4),
                    distance >= 5,
                )
            ],
            "mean_hit_cells": float((energy > PILOT_CELL_THRESHOLD_MEV).sum(axis=1).mean()),
            "median_layer_energy_mev": float(np.median(total)),
        }
    rows = np.flatnonzero(first <= 10)
    rms = np.empty((len(rows), 6))
    for column, k in enumerate(range(1, 7)):
        layer = first[rows] + k
        energy = grid[rows, layer, :]
        distance_mm = offset[rows, layer, :] * geometry.cell_pitch_mm
        total = np.maximum(energy.sum(axis=1), 1e-9)
        rms[:, column] = np.sqrt((energy * distance_mm**2).sum(axis=1) / total)
    return {
        "kernel_by_offset": kernel,
        "layer_rms_median_mm_k1_to_k6": np.median(rms, axis=0).tolist(),
        "log_rms_first_component_share": first_component_share(np.log(np.maximum(rms, 1e-3))),
    }


def crossing_section(
    arrays: dict[str, Any], layout: FibreLayout, n_events: int = 600
) -> dict[str, Any]:
    """Validate the chord model on crossing protons, fibre by fibre.

    For each stored fibre deposit: is the straight track inside the fibre, and
    if so what energy per mm of chord did it receive? A polystyrene MIP loses
    about 0.205 MeV/mm, so a value near it says the exact-geometry mapping is
    the one Geant4 saw. Reported with the share of fibre energy that is NOT
    in a crossed fibre (delta rays, showers, scattering).
    """

    batch: Batch = arrays["_batch"]
    kept: np.ndarray = arrays["_kept_index"]
    crossing = np.flatnonzero(~arrays["truth_occurred"])[:n_events]
    radius = layout.fibre_radius_mm
    stride, rows_per = layout.max_fibres_per_row, layout.rows_per_superlayer
    centres = [np.array(layout.fibre_centres_mm(row)) for row in range(rows_per)]
    measures_x = np.array(
        [layout.measured_axis(s) == "x" for s in range(layout.geometry.number_of_superlayers)]
    )
    crossed_e, other_e, chord_sum, crossed_count = [], [], [], []
    for local in crossing:
        ids, edep = batch.fibre_deposits(int(kept[local]))
        superlayer_row, index = np.divmod(ids.astype(np.int64), stride)
        superlayer, row = np.divmod(superlayer_row, rows_per)
        coordinate = np.where(
            measures_x[superlayer], arrays["entry_x_mm"][local], arrays["entry_y_mm"][local]
        )
        centre = np.array([centres[r][i] for r, i in zip(row, index, strict=True)])
        distance = coordinate - centre
        inside = np.abs(distance) < radius
        chord = 2 * np.sqrt(np.clip(radius**2 - distance**2, 0, None))
        crossed_e.append(edep[inside].sum())
        other_e.append(edep[~inside].sum())
        chord_sum.append(chord[inside].sum())
        crossed_count.append(int(inside.sum()))
    crossed, other, chord = map(np.array, (crossed_e, other_e, chord_sum))

    # The geometry alone: how many fibres a straight track crosses, and the
    # chord it cuts, from the entry point and the lattice - no Geant4 output.
    geometric_count = np.zeros(len(crossing))
    geometric_chord = np.zeros(len(crossing))
    for superlayer in range(layout.geometry.number_of_superlayers):
        coordinate = np.where(
            measures_x[superlayer],
            arrays["entry_x_mm"][crossing],
            arrays["entry_y_mm"][crossing],
        )
        for row in range(rows_per):
            distance = np.abs(coordinate[:, None] - centres[row][None, :]).min(axis=1)
            hit = distance < radius
            geometric_count += hit
            geometric_chord += np.where(hit, 2 * np.sqrt(np.clip(radius**2 - distance**2, 0, None)), 0.0)
    return {
        "n_events": len(crossing),
        "fibres_crossed_geometry_mean": float(geometric_count.mean()),
        "fibres_crossed_with_a_deposit_mean": float(np.mean(crossed_count)),
        "chord_path_geometry_mean_mm": float(geometric_chord.mean()),
        "mev_per_mm_of_chord_in_crossed_fibres": float(crossed.sum() / chord.sum()),
        "polystyrene_mip_mev_per_mm": 0.205,
        "share_of_fibre_energy_outside_crossed_fibres": float(
            other.sum() / (crossed.sum() + other.sum())
        ),
        "corr_crossed_energy_with_chord": float(np.corrcoef(crossed, chord)[0, 1]),
        "corr_total_energy_with_chord": float(np.corrcoef(crossed + other, chord)[0, 1]),
        "sd_mev_crossed_other_total": [
            float(crossed.std()),
            float(other.std()),
            float((crossed + other).std()),
        ],
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_summary(data: Path = DATA) -> dict[str, Any]:
    geometry = load_geometry(PROJECT_ROOT / "configs" / "geometry.yaml")
    layout = FibreLayout(geometry)
    layer_mm = geometry.mean_readout_slice_thickness_mm
    depth_mm = geometry.number_of_layers * layer_mm
    summary: dict[str, Any] = {
        "status": (
            "Block 6B Slice 0. CALIBRATION events only (event_index % 4 != 3). "
            "Exploratory: the first partial-correlation table was read before the "
            "0.20 / 0.10 decision convention below was adopted."
        ),
        "decision_convention": {
            "explicit_if_abs_partial_rho_at_least": EXPLICIT_RHO,
            "weak_if_abs_partial_rho_at_least": WEAK_RHO,
            "requires": "two or more energies with one sign; held-out validation is the final arbiter",
        },
        "layer_thickness_mm": layer_mm,
        "depth_mm": depth_mm,
    }
    for energy in ENERGIES_GEV:
        arrays = load_calibration("baseline", energy, data)
        crossing = ~arrays["truth_occurred"]
        mip_layer = float(arrays["readout_grid_mev"][crossing].sum(axis=2).mean())
        summary[f"{energy:g}"] = {
            "n_calibration_events": len(arrays["truth_occurred"]),
            "dependency": dependency_table(arrays, geometry, depth_mm),
            "amplitude": amplitude_section(arrays, layer_mm),
            "universality": universality_section(arrays, layer_mm),
            "upstream": upstream_section(arrays, layer_mm, mip_layer),
            "lateral": lateral_section(arrays, geometry),
            "crossing": crossing_section(arrays, layout),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    summary = build_summary(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
