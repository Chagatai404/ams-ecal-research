"""EXP-015 realistic-regime pilot: the approved, pre-specified analysis (electrons against protons, ECAL only).

EXPLORATORY - EXP-015 realistic-regime pilot, Geant4 FTFP_BERT, ECAL-only, scintillator level.

The analysis was fixed before any data existed and is implemented literally. Where the specification was ambiguous the code says
``# SPEC AMBIGUITY:`` and the choice made is listed in the report of this module (``ambiguities()``).

    python -m ams_ecal.multiscale_estimators.realistic_regime_analysis --data-root data --out results/realistic_regime/rung0

Stages (numbers refer to the approved plan):

1. fiducial selection from the projected axis (``fiducial_mask``);
2. classes: electrons, crossing protons (``truth_occurred`` False), interacting protons (True);
3. physical-spectrum weights on log-uniform generation, with Kish effective sample size (``physical_spectrum_weights``);
4. 2D readout D0, D1, D2 with grid-offset averaging of Z_0, Z_2 and the entropy before logs and slopes (``dimension_row``);
5. 1:1 caliper matching on (occupied readout cells, edep_scintillator_mev) within a band and across proton bands 1 and 2 steps above;
6. pair-swap permutation KS of the matched D_q, the cell-shuffle surrogate excess as reference, Holm over the reported family;
7. photoelectron scan on the readout grid with the matching redone after every smearing;
8. figures F1-F5b (F6 / F7 are hooks only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.detector.readout_response import photoelectron_smearing
from ams_ecal.inference import multiplicity as mp
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.multiscale_estimators import multiscale_matched_pairs as mpair
from ams_ecal.multiscale_estimators import multiscale_test as mt

EXPLORATORY_LABEL = "EXPLORATORY - EXP-015 realistic-regime pilot, Geant4 FTFP_BERT, ECAL-only, scintillator level"

# --- fixed constants of the approved plan -------------------------------------------------------------------------------------------

ANALYSIS_SEED = 20261109
OFFSETS_PER_EVENT = 8
OFFSET_RANGE_CELLS = (
    18  # offsets are uniform integers in [0, 18) on the cell axis and on the layer axis
)
HALF_WIDTH_MM = 324.0
PREFIX_DEPTH_MM = 166.5
FIDUCIAL_MARGIN_MM = 27.0
ELECTRON_SPECTRAL_INDEX = 3.180  # AMS Phys. Rep. 894 (2021) Sec. 3, PDF p.35
PROTON_MASS_GEV = 0.938272
PROTON_RIGIDITY_REFERENCE_GV = 45.0
PROTON_SLOPE = 2.853
PROTON_BREAK_GV = 331.0
PROTON_DELTA_GAMMA = 0.22
PROTON_SMOOTHING = 0.09  # AMS Phys. Rep. 894 Eq. 12 (Sec. 4, PDF p.44)
CALIPER_SD = 0.25
PERMUTATIONS = 10_000
MIN_PAIRS = mpair.MIN_PAIRS  # 100: fewer pairs are "not testable", never "no effect"
SHUFFLES_PER_EVENT = 10
PHOTOELECTRON_SCAN = (None, 10.0, 8.0, 5.0)
SCAN_DRAWS = 3
STEPS_ABOVE = (0, 1, 2)
THETA_MAX_RAD = math.radians(25.0)
# Pre-stated SENSITIVITY stratum (applied after the fiducial cut, never to the full-sample primary or the photoelectron scan):
# equivalent uniform-face maximum angle for the AMS geometric acceptance of about 550 cm2 sr (Aguilar et al., PRL 113, 221102
# (2014)); project approximation of the AMS acceptance, not an AMS cut; researcher decision 2026-11-09.
SENSITIVITY_THETA_MAX_DEG = 12.0
SENSITIVITY_THETA_MAX_RAD = math.radians(SENSITIVITY_THETA_MAX_DEG)
SENSITIVITY_LABEL = "sensitivity, theta <= 12 deg"
STATISTIC_Q = (0.0, 1.0, 2.0)
WINDOWS = tuple(mt.REGISTERED_WINDOWS)
EXP011_DIRECTORIES = {
    "electron": "geant4_highenergy_pilot_electron",
    "proton": "geant4_highenergy_pilot_proton",
}


def ambiguities() -> list[str]:
    """Every point where the written specification left a choice to the implementer (see the ``# SPEC AMBIGUITY`` comments)."""

    return list(SPEC_AMBIGUITIES)


SPEC_AMBIGUITIES = [
    (
        "Weights: the proton density in kinetic energy includes the Jacobian dR/dE_k = (E_k+m)/R of Eq. 12 (a flux per unit "
        "rigidity); the factor is 1.0002 at 50 GeV, so the choice is immaterial. Weights are normalised so that the sum over "
        "all generated events of a class is 1 (band totals follow the integral of the physical spectrum)."
    ),
    (
        "Weights are defined from the GENERATED count of each band (before the fiducial cut); ESS is reported both on generated and "
        "on fiducial-accepted events."
    ),
    (
        "primary_energy_gev of protons is taken as the kinetic energy; electron energy as total = kinetic."
    ),
    (
        "Holm family: all testable raw-D_q pair-swap p-values of one analysis point (all pairings x 6 windows x D0,D1,D2); the "
        "within-pairing Holm (18 tests) is reported next to it. The cell-shuffle excess is a reference and is not in the family."
    ),
    (
        "Eligibility: the per-window minimum occupied cells of mt.REGISTERED_WINDOWS (336 / 572) is kept, so an event is NaN for a "
        "window below it (reported, not dropped silently)."
    ),
    (
        "Photoelectron scan: at p.e. = None the matching keys are the file values (edep_scintillator_mev, occupied readout cells); "
        "after smearing the energy key is the sum of the smeared readout grid (the file value does not describe the smeared image). "
        "The excess reference is not recomputed in the scan."
    ),
    (
        "Scan permutations default to the same B as the main analysis (10,000); the CLI allows a lower value, which is then recorded."
    ),
    (
        "Cross-energy matching pairs the electron band i with the proton band i + k, k = 0, 1, 2 (band order by lower edge)."
    ),
    (
        "Offsets: the same 8 (layer, cell) offsets are used for an event, its shuffled copies and its smeared copies; the generator "
        "of an event is default_rng([20261109, event_key])."
    ),
    (
        "Edge distance in F3 is measured at the entry point on the front face (min over x and y of 324 - |coordinate|)."
    ),
]


# --- data container -----------------------------------------------------------------------------------------------------------------


@dataclass
class ClassEvents:
    """All events of one particle class (all bands), as plain arrays. Built by ``load_class_events`` or by a test."""

    name: str
    readout_grid_mev: np.ndarray  # (n, 18, 72)
    edep_scintillator_mev: np.ndarray
    edep_total_mev: np.ndarray
    entry_x_mm: np.ndarray
    entry_y_mm: np.ndarray
    primary_energy_gev: np.ndarray
    primary_theta_rad: np.ndarray
    primary_phi_rad: np.ndarray
    truth_occurred: np.ndarray
    band_lower_gev: np.ndarray
    band_upper_gev: np.ndarray
    fine_voxel_count: np.ndarray
    generated_per_band: dict[float, int] = field(default_factory=dict)
    band_batches: list[dict[str, np.ndarray]] = field(
        default_factory=list
    )  # fine-deposit arrays per band (hook for F6)
    event_band_position: np.ndarray | None = None  # (n, 2): band batch, row in batch

    def __len__(self) -> int:
        return len(self.edep_scintillator_mev)

    @property
    def band_edges(self) -> list[tuple[float, float]]:
        return sorted(
            {
                (float(lo), float(hi))
                for lo, hi in zip(self.band_lower_gev, self.band_upper_gev, strict=True)
            }
        )

    def fine_deposits(self, event: int) -> tuple[np.ndarray, np.ndarray]:
        """(flat voxel index, MeV) of one event, from the stored per-band fine arrays."""

        if self.event_band_position is None or not self.band_batches:
            raise ValueError("this ClassEvents carries no fine deposits")
        b, row = (int(v) for v in self.event_band_position[event])
        arrays = self.band_batches[b]
        start, stop = arrays["fine_offsets"][row : row + 2]
        return arrays["fine_index"][start:stop], arrays["fine_edep_mev"][start:stop]


def load_class_events(class_dir: str | Path, name: str) -> ClassEvents:
    """Read every ``events.npz`` below ``class_dir`` (the layouts ``pilot/E*GeV`` and ``E*GeV`` both work)."""

    from ams_ecal.geant4_simulation.geant4_backend import load_batch

    directories = sorted({p.parent for p in Path(class_dir).rglob("events.npz")})
    if not directories:
        raise FileNotFoundError(f"no events.npz below {class_dir}")
    parts: dict[str, list[np.ndarray]] = {}
    generated: dict[float, int] = {}
    batches: list[dict[str, np.ndarray]] = []
    positions: list[np.ndarray] = []
    for b, directory in enumerate(directories):
        batch = load_batch(directory)
        meta, arrays = batch.metadata, batch.arrays
        lower, upper = (
            float(v) for v in meta.get("energy_band_gev", [meta["energy_gev"], np.nan])
        )
        n = len(batch)
        generated[lower] = generated.get(lower, 0) + n
        for key in (
            "readout_grid_mev",
            "edep_scintillator_mev",
            "edep_total_mev",
            "entry_x_mm",
            "entry_y_mm",
            "primary_energy_gev",
            "primary_theta_rad",
            "primary_phi_rad",
            "truth_occurred",
        ):
            parts.setdefault(key, []).append(np.asarray(arrays[key]))
        parts.setdefault("band_lower_gev", []).append(np.full(n, lower))
        parts.setdefault("band_upper_gev", []).append(np.full(n, upper))
        parts.setdefault("fine_voxel_count", []).append(np.diff(arrays["fine_offsets"]))
        batches.append(
            {k: arrays[k] for k in ("fine_offsets", "fine_index", "fine_edep_mev")}
        )
        positions.append(np.column_stack([np.full(n, b), np.arange(n)]))
    stacked = {k: np.concatenate(v) for k, v in parts.items()}
    return ClassEvents(
        name=name,
        generated_per_band=generated,
        band_batches=batches,
        event_band_position=np.vstack(positions),
        **stacked,
    )


# --- 1. fiducial selection ----------------------------------------------------------------------------------------------------------


def projected_axis(
    entry_x_mm: np.ndarray,
    entry_y_mm: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    z_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Straight-line axis (x, y) at depth ``z_mm`` below the front face (z = 0 is the entry plane, +z into the ECAL)."""

    slope = np.tan(theta_rad)
    return entry_x_mm + z_mm * slope * np.cos(
        phi_rad
    ), entry_y_mm + z_mm * slope * np.sin(phi_rad)


def fiducial_mask(
    entry_x_mm,
    entry_y_mm,
    theta_rad,
    phi_rad,
    margin_mm: float = FIDUCIAL_MARGIN_MM,
    depth_mm: float = PREFIX_DEPTH_MM,
    half_width_mm: float = HALF_WIDTH_MM,
) -> np.ndarray:
    """True where the projected axis at the prefix entry (z = 0) and exit (z = depth) is at least ``margin`` from every face edge."""

    limit = half_width_mm - margin_mm
    ok = np.ones(np.shape(entry_x_mm), dtype=bool)
    for z in (0.0, depth_mm):
        x, y = projected_axis(
            np.asarray(entry_x_mm, float),
            np.asarray(entry_y_mm, float),
            np.asarray(theta_rad, float),
            np.asarray(phi_rad, float),
            z,
        )
        ok &= (np.abs(x) <= limit) & (np.abs(y) <= limit)
    return ok


def distance_to_nearest_edge(
    entry_x_mm, entry_y_mm, half_width_mm: float = HALF_WIDTH_MM
) -> np.ndarray:
    """Distance of the entry point to the nearest edge of the square front face."""

    return half_width_mm - np.maximum(np.abs(entry_x_mm), np.abs(entry_y_mm))


# --- 2. classes ---------------------------------------------------------------------------------------------------------------------


def class_masks(
    electrons: ClassEvents | None, protons: ClassEvents | None
) -> dict[str, tuple[ClassEvents, np.ndarray]]:
    """name -> (events, boolean selection before the fiducial cut) for electrons, crossing protons, interacting protons."""

    out: dict[str, tuple[ClassEvents, np.ndarray]] = {}
    if electrons is not None:
        out["electron"] = (electrons, np.ones(len(electrons), dtype=bool))
    if protons is not None:
        occurred = protons.truth_occurred.astype(bool)
        out["proton_crossing"] = (protons, ~occurred)
        out["proton_interacting"] = (protons, occurred)
    return out


def fiducial_of(events: ClassEvents) -> np.ndarray:
    return fiducial_mask(
        events.entry_x_mm,
        events.entry_y_mm,
        events.primary_theta_rad,
        events.primary_phi_rad,
    )


def theta_stratum(events: ClassEvents, theta_max_rad: float | None) -> np.ndarray:
    """True for events with primary_theta <= theta_max_rad (all True when ``theta_max_rad`` is None)."""

    if theta_max_rad is None:
        return np.ones(len(events), dtype=bool)
    return events.primary_theta_rad <= theta_max_rad


def acceptance_table(
    classes: dict[str, tuple[ClassEvents, np.ndarray]],
    theta_max_rad: float | None = None,
) -> list[dict[str, Any]]:
    """Accepted fraction per class and band; with ``theta_max_rad`` only events inside that sensitivity stratum are counted."""

    rows = []
    for name, (events, selected) in classes.items():
        fid = fiducial_of(events)
        selected = selected & theta_stratum(events, theta_max_rad)
        for lower, upper in events.band_edges:
            in_band = selected & (events.band_lower_gev == lower)
            n = int(in_band.sum())
            rows.append(
                {
                    **({} if theta_max_rad is None else {"view": SENSITIVITY_LABEL}),
                    "class": name,
                    "band_lower_gev": lower,
                    "band_upper_gev": upper,
                    "events": n,
                    "accepted": int((in_band & fid).sum()),
                    "accepted_fraction": float((in_band & fid).sum() / n)
                    if n
                    else float("nan"),
                }
            )
    return rows


# --- 3. weights ---------------------------------------------------------------------------------------------------------------------


def electron_density(energy_gev):
    """dN/dE of the physical electron spectrum (relative): E^-3.180."""

    return np.asarray(energy_gev, float) ** -ELECTRON_SPECTRAL_INDEX


def proton_rigidity_gv(kinetic_gev):
    """R = sqrt(E_k^2 + 2 E_k m_p) for a unit-charge proton."""

    k = np.asarray(kinetic_gev, float)
    return np.sqrt(k * k + 2.0 * k * PROTON_MASS_GEV)


def proton_flux_in_rigidity(rigidity_gv):
    """AMS Phys. Rep. 894 Eq. 12 (relative): (R/45)^-2.853 (1 + (R/331)^(0.22/0.09))^0.09."""

    r = np.asarray(rigidity_gv, float)
    return (r / PROTON_RIGIDITY_REFERENCE_GV) ** -PROTON_SLOPE * (
        1.0 + (r / PROTON_BREAK_GV) ** (PROTON_DELTA_GAMMA / PROTON_SMOOTHING)
    ) ** PROTON_SMOOTHING


def proton_density(kinetic_gev):
    """dN/dE_k of the physical proton spectrum (relative).

    # SPEC AMBIGUITY: Eq. 12 is a flux per unit rigidity; the Jacobian dR/dE_k = (E_k + m)/R is included (1.0002 at 50 GeV).
    """

    k = np.asarray(kinetic_gev, float)
    r = proton_rigidity_gv(k)
    return proton_flux_in_rigidity(r) * (k + PROTON_MASS_GEV) / r


def _band_integral(
    density: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    points: int = 4001,
) -> float:
    """Integral of ``density`` over [lower, upper] on a logarithmic grid (trapezoid in ln E)."""

    grid = np.geomspace(lower, upper, points)
    return float(np.trapezoid(density(grid) * grid, np.log(grid)))


def physical_spectrum_weights(
    energy_gev: np.ndarray,
    band_lower_gev: np.ndarray,
    band_upper_gev: np.ndarray,
    generated_per_band: dict[float, int],
    density: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-event weights turning log-uniform samples (fixed count per band) into the physical spectrum ``density``.

    ``w_i = density(E_i) E_i ln(u/l) / (G_total N_band)``: inside a band the weight is the ratio of the physical to the generation
    density (generation is E^-1), and the weights of a band sum (in expectation) to its share ``G_band / G_total`` of the physical
    integral. Normalised so that the weights of ALL generated events sum to 1.
    """

    bands = sorted(
        {
            (float(lo), float(hi))
            for lo, hi in zip(band_lower_gev, band_upper_gev, strict=True)
        }
    )
    total = sum(_band_integral(density, lo, hi) for lo, hi in bands)
    energy = np.asarray(energy_gev, float)
    span = np.log(np.asarray(band_upper_gev, float) / np.asarray(band_lower_gev, float))
    counts = np.array(
        [generated_per_band[float(lo)] for lo in band_lower_gev], dtype=float
    )
    return density(energy) * energy * span / (total * counts)


def kish_effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / np.sum(w * w)) if len(w) else 0.0


def class_weights(events: ClassEvents, kind: str) -> np.ndarray:
    density = electron_density if kind == "electron" else proton_density
    return physical_spectrum_weights(
        events.primary_energy_gev,
        events.band_lower_gev,
        events.band_upper_gev,
        events.generated_per_band,
        density,
    )


def ess_table(
    classes: dict[str, tuple[ClassEvents, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for name, (events, selected) in classes.items():
        weights = class_weights(events, "electron" if name == "electron" else "proton")
        fid = fiducial_of(events)
        for lower, upper in events.band_edges:
            in_band = selected & (events.band_lower_gev == lower)
            rows.append(
                {
                    "class": name,
                    "band_lower_gev": lower,
                    "band_upper_gev": upper,
                    "generated_in_class": int(in_band.sum()),
                    "ess_kish_generated": kish_effective_sample_size(weights[in_band]),
                    "accepted": int((in_band & fid).sum()),
                    "ess_kish_accepted": kish_effective_sample_size(
                        weights[in_band & fid]
                    ),
                    "weight_sum_band": float(weights[in_band].sum()),
                }
            )
    return rows


# --- 4. 2D estimators with grid-offset averaging ------------------------------------------------------------------------------------


def _offset_box_sums(
    image: np.ndarray, side: int, layer_offset: int, cell_offset: int
) -> np.ndarray:
    """Sums over ``side x side`` boxes whose edges lie at ``offset + k side`` on both axes; partial edge boxes are kept."""

    h, w = image.shape
    pad_l, pad_c = (-layer_offset) % side, (-cell_offset) % side
    height, width = -(-(h + pad_l) // side) * side, -(-(w + pad_c) // side) * side
    padded = np.zeros((height, width))
    padded[pad_l : pad_l + h, pad_c : pad_c + w] = image
    return padded.reshape(height // side, side, width // side, side).sum(axis=(1, 3))


def offset_partition_profile(
    image: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    sides: Sequence[int] = ms.AMS_BOX_SIDES,
) -> dict[float, np.ndarray]:
    """``y_q(s)`` with Z_0, Z_2 and the entropy S averaged over the offsets BEFORE the logs (mirrors the 3D rule).

    Returns {0: ln mean Z_0, 1: mean S, 2: ln mean Z_2}, each over ``sides`` (the layout ``multiscale.window_dimension`` expects).
    """

    a = ms._checked(image)
    total = a.sum()
    z0 = np.zeros(len(sides))
    z2 = np.zeros(len(sides))
    entropy = np.zeros(len(sides))
    for k, side in enumerate(sides):
        used = (
            [(0, 0)] if side == 1 else list(offsets)
        )  # the finest scale does not depend on the offset
        for lo, co in used:
            p = _offset_box_sums(a, side, int(lo), int(co)).ravel() / total
            p = p[p > 0]
            z0[k] += len(p)
            z2[k] += np.sum(p * p)
            entropy[k] -= np.sum(p * np.log(p))
        z0[k] /= len(used)
        z2[k] /= len(used)
        entropy[k] /= len(used)
    return {0.0: np.log(z0), 1.0: entropy, 2.0: np.log(z2)}


def draw_offsets(event_key: int) -> np.ndarray:
    """The 8 (layer, cell) offsets of one event: uniform integers in [0, 18), generator ``default_rng([20261109, event_key])``."""

    return np.random.default_rng([ANALYSIS_SEED, int(event_key)]).integers(
        0, OFFSET_RANGE_CELLS, size=(OFFSETS_PER_EVENT, 2)
    )


def dimension_names() -> list[str]:
    return [f"D{int(q)}@{'-'.join(map(str, w))}" for w in WINDOWS for q in STATISTIC_Q]


SECONDARY_LABEL = "secondary, descriptive, outside DEC-013 controls"


def dimension_row(
    image: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    apply_eligibility: bool = True,
) -> np.ndarray:
    """D0, D1, D2 over every registered window, in ``dimension_names`` order; NaN where the event has too few occupied cells.

    ``apply_eligibility=False`` drops the DEC-013 per-window minimum-occupied-cells cut (the SECONDARY view only; an empty image is
    still NaN).
    """

    if not np.any(image > 0):
        return np.full(len(WINDOWS) * len(STATISTIC_Q), np.nan)
    cells = int(np.count_nonzero(image))
    profile = offset_partition_profile(image, offsets)
    out = []
    for window, minimum in mt.REGISTERED_WINDOWS.items():
        for q in STATISTIC_Q:
            out.append(
                ms.window_dimension(profile[q], ms.AMS_BOX_SIDES, q, window)
                if (cells >= minimum or not apply_eligibility)
                else np.nan
            )
    return np.array(out)


def dimension_table(grids: np.ndarray, event_keys: Sequence[int]) -> np.ndarray:
    """``(events, 18)`` offset-averaged D_q; ``event_keys`` seed the per-event offsets."""

    return np.array(
        [
            dimension_row(g, draw_offsets(k))
            for g, k in zip(grids, event_keys, strict=True)
        ]
    ).reshape(len(grids), len(WINDOWS) * len(STATISTIC_Q))


def shuffle_surrogate_mean(
    grid: np.ndarray,
    event_key: int,
    shuffles: int = SHUFFLES_PER_EVENT,
    apply_eligibility: bool = True,
) -> np.ndarray:
    """Mean over ``shuffles`` cell-shuffled copies (``permutation_nulls.shuffle_cells_within_layers``, every layer's cells permuted)
    of the offset-averaged D_q of one image, with the offsets of the real event."""

    rng = np.random.default_rng([ANALYSIS_SEED, 99, int(event_key)])
    copies = pn.shuffle_cells_within_layers(
        np.repeat(np.asarray(grid, float)[None], shuffles, axis=0), rng
    )
    offsets = draw_offsets(event_key)
    rows = np.array([dimension_row(c, offsets, apply_eligibility) for c in copies])
    return (
        np.nanmean(rows, axis=0)
        if np.isfinite(rows).any()
        else np.full(rows.shape[1], np.nan)
    )


# --- 5. matching --------------------------------------------------------------------------------------------------------------------


def occupied_cells(grids: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.asarray(grids) > 0, axis=(1, 2))


def match_on_occupancy_and_energy(
    cells_e, energy_e, cells_p, energy_p, caliper: float = CALIPER_SD
) -> tuple[np.ndarray, np.ndarray]:
    """1:1 nearest-neighbour matching without replacement on the standardized logs of (occupied cells, energy) within ``caliper``
    SD; the S3 stage-3 code (``multiscale_matched_pairs.match_pairs``) is reused."""

    cells_e, energy_e = np.asarray(cells_e, float), np.asarray(energy_e, float)
    cells_p, energy_p = np.asarray(cells_p, float), np.asarray(energy_p, float)
    good_e, good_p = (
        np.flatnonzero((cells_e > 0) & (energy_e > 0)),
        np.flatnonzero((cells_p > 0) & (energy_p > 0)),
    )
    if len(good_e) < 2 or len(good_p) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    ie, ip = mpair.match_pairs(
        np.column_stack([cells_e[good_e], energy_e[good_e]]),
        np.column_stack([cells_p[good_p], energy_p[good_p]]),
        caliper,
    )
    return good_e[ie], good_p[ip]


# --- 6. statistics ------------------------------------------------------------------------------------------------------------------


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries (untestable) are left out of the family and stay NaN."""

    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = mp.holm(p[finite])
    return out


def permutation_p_values(
    observed: np.ndarray, null: np.ndarray, valid_pairs: np.ndarray
) -> np.ndarray:
    """``(1 + #{null >= observed}) / (1 + B)``; NaN for a column with fewer than ``MIN_PAIRS`` valid pairs."""

    p = (1.0 + (null >= observed).sum(axis=0)) / (1.0 + len(null))
    return np.where((valid_pairs >= MIN_PAIRS) & np.isfinite(observed), p, np.nan)


def compare_matched_pairs(
    dims_e: np.ndarray,
    dims_p: np.ndarray,
    excess_e: np.ndarray | None,
    excess_p: np.ndarray | None,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """KS of the matched D_q (and of the excess, as a reference) with the pair-swap permutation p-value.

    A pairing with fewer than ``MIN_PAIRS`` valid pairs in a column is NOT TESTABLE for it (p = NaN), never "no effect".
    """

    n_stat = dims_e.shape[1]
    values_e, values_p = dims_e, dims_p
    if excess_e is not None and excess_p is not None:
        values_e, values_p = (
            np.hstack([dims_e, excess_e]),
            np.hstack([dims_p, excess_p]),
        )
    if len(values_e) < 2:
        nan = np.full(values_e.shape[1], np.nan)
        return {
            "observed": nan,
            "p": nan.copy(),
            "valid_pairs": np.zeros(values_e.shape[1], dtype=int),
            "n_stat": n_stat,
        }
    result = mpair.pair_swap_test(values_e, values_p, permutations, rng)
    p = permutation_p_values(result["observed"], result["null"], result["valid_pairs"])
    return {
        "observed": result["observed"],
        "p": p,
        "valid_pairs": result["valid_pairs"],
        "n_stat": n_stat,
    }


@dataclass
class PairingResult:
    electron_band_gev: float
    proton_band_gev: float
    steps_above: int
    caliper_sd: float
    pairs: int
    testable: bool
    reason: str
    standardized_mean_difference: list[float]
    rows: list[dict[str, Any]]


def run_pairing(
    e_band: float,
    p_band: float,
    steps: int,
    ie: np.ndarray,
    ip: np.ndarray,
    key_e: np.ndarray,
    key_p: np.ndarray,
    dims_of_e: Callable[[np.ndarray], np.ndarray],
    dims_of_p: Callable[[np.ndarray], np.ndarray],
    excess_of_e: Callable[[np.ndarray], np.ndarray] | None,
    excess_of_p: Callable[[np.ndarray], np.ndarray] | None,
    permutations: int,
    rng: np.random.Generator,
) -> PairingResult:
    """One pairing: balance, per-statistic KS, p, within-pairing Holm. ``ie``/``ip`` are the matched indices (into the arrays the
    ``*_of_*`` callables take); the keys are used only for the balance."""

    names = dimension_names()
    smd = mpair.balance(key_e, key_p, ie, ip) if len(ie) >= 2 else [float("nan")] * 2
    base = {
        "electron_band_gev": e_band,
        "proton_band_gev": p_band,
        "steps_above": steps,
        "caliper_sd": CALIPER_SD,
        "pairs": len(ie),
    }
    if len(ie) < MIN_PAIRS:
        return PairingResult(
            **base,
            testable=False,
            reason=f"fewer than {MIN_PAIRS} matched pairs (not testable)",
            standardized_mean_difference=smd,
            rows=[],
        )
    de, dp = dims_of_e(ie), dims_of_p(ip)
    ee = excess_of_e(ie) if excess_of_e else None
    ep = excess_of_p(ip) if excess_of_p else None
    test = compare_matched_pairs(de, dp, ee, ep, permutations, rng)
    n = len(names)
    raw_p = test["p"][:n]
    holm_within = holm_adjust(raw_p)
    rows = []
    for j, name in enumerate(names):
        row: dict[str, Any] = {
            "statistic": name,
            "valid_pairs": int(test["valid_pairs"][j]),
            "testable": bool(np.isfinite(raw_p[j])),
            "ks": float(test["observed"][j]),
            "p_raw": float(raw_p[j]),
            "p_holm_within_pairing": float(holm_within[j]),
            "median_electron": float(np.nanmedian(de[:, j]))
            if np.isfinite(de[:, j]).any()
            else float("nan"),
            "median_proton": float(np.nanmedian(dp[:, j]))
            if np.isfinite(dp[:, j]).any()
            else float("nan"),
        }
        if ee is not None and ep is not None:
            row.update(
                reference_cell_shuffle_ks_of_excess=float(test["observed"][n + j]),
                reference_cell_shuffle_p_raw=float(test["p"][n + j]),
                median_excess_electron=float(np.nanmedian(ee[:, j]))
                if np.isfinite(ee[:, j]).any()
                else float("nan"),
                median_excess_proton=float(np.nanmedian(ep[:, j]))
                if np.isfinite(ep[:, j]).any()
                else float("nan"),
                median_surrogate_electron=float(np.nanmedian(de[:, j] - ee[:, j]))
                if np.isfinite(ee[:, j]).any()
                else float("nan"),
                median_surrogate_proton=float(np.nanmedian(dp[:, j] - ep[:, j]))
                if np.isfinite(ep[:, j]).any()
                else float("nan"),
            )
        rows.append(row)
    return PairingResult(
        **base,
        testable=bool(np.isfinite(raw_p).any()),
        reason="" if np.isfinite(raw_p).any() else "no statistic has 100 valid pairs",
        standardized_mean_difference=smd,
        rows=rows,
    )


def holm_over_family(results: Sequence[PairingResult]) -> None:
    """Add ``p_holm_family`` to every testable row: Holm over all raw p-values of the analysis point.

    # SPEC AMBIGUITY: the reported family is every testable (pairing, window, q) of one analysis point.
    """

    cells = [(r, row) for r in results for row in r.rows if row["testable"]]
    adjusted = holm_adjust(np.array([row["p_raw"] for _, row in cells]))
    for (_, row), value in zip(cells, adjusted, strict=True):
        row["p_holm_family"] = float(value)
    for r in results:
        for row in r.rows:
            row.setdefault("p_holm_family", float("nan"))


# --- the analysis at one point (a photoelectron setting) ---------------------------------------------------------------------------


@dataclass
class ClassImages:
    """Fiducial electrons or interacting protons at one analysis point: images, keys, band and cache of D_q."""

    name: str
    grids: np.ndarray
    cells: np.ndarray
    energy_mev: np.ndarray
    band_lower_gev: np.ndarray
    event_keys: np.ndarray
    _dims: dict[int, np.ndarray] = field(default_factory=dict)
    _surrogate: dict[int, np.ndarray] = field(default_factory=dict)
    _dims_uncut: dict[int, np.ndarray] = field(default_factory=dict)
    _surrogate_uncut: dict[int, np.ndarray] = field(default_factory=dict)

    def dims(self, index: np.ndarray, eligibility: bool = True) -> np.ndarray:
        cache = self._dims if eligibility else self._dims_uncut
        for i in np.unique(index):
            if int(i) not in cache:
                cache[int(i)] = dimension_row(
                    self.grids[i], draw_offsets(self.event_keys[i]), eligibility
                )
        return np.array([cache[int(i)] for i in index]).reshape(len(index), -1)

    def surrogate(self, index: np.ndarray, eligibility: bool = True) -> np.ndarray:
        cache = self._surrogate if eligibility else self._surrogate_uncut
        for i in np.unique(index):
            if int(i) not in cache:
                cache[int(i)] = shuffle_surrogate_mean(
                    self.grids[i],
                    self.event_keys[i],
                    apply_eligibility=eligibility,
                )
        return np.array([cache[int(i)] for i in index]).reshape(len(index), -1)

    def excess(self, index: np.ndarray, eligibility: bool = True) -> np.ndarray:
        return self.dims(index, eligibility) - self.surrogate(index, eligibility)


def build_class_images(
    events: ClassEvents,
    selected: np.ndarray,
    name: str,
    class_code: int,
    pe_per_mip: float | None = None,
    draw: int = 0,
    theta_max_rad: float | None = None,
) -> ClassImages:
    """Fiducial events of a class; with ``pe_per_mip`` the readout grid is smeared with fixed-seed Poisson statistics (B1 only).
    ``theta_max_rad`` (sensitivity stratum only) additionally keeps events with primary_theta <= theta_max_rad after the fiducial cut."""

    keep = np.flatnonzero(
        selected & fiducial_of(events) & theta_stratum(events, theta_max_rad)
    )
    grids = events.readout_grid_mev[keep].astype(float)
    if pe_per_mip is None:
        cells, energy = (
            occupied_cells(grids),
            events.edep_scintillator_mev[keep].astype(float),
        )
    else:
        rng = np.random.default_rng(
            [ANALYSIS_SEED, int(pe_per_mip * 10), draw, class_code]
        )
        grids = photoelectron_smearing(grids, pe_per_mip, rng)
        # SPEC AMBIGUITY: after smearing the energy key is the sum of the smeared readout grid.
        cells, energy = occupied_cells(grids), grids.sum(axis=(1, 2))
    return ClassImages(
        name=name,
        grids=grids,
        cells=cells,
        energy_mev=energy,
        band_lower_gev=events.band_lower_gev[keep],
        event_keys=keep + class_code * 10_000_000,
    )


def band_pairings(
    electron_bands: Sequence[float], proton_bands: Sequence[float]
) -> list[tuple[float, float, int]]:
    """(electron band, proton band, steps above) for steps 0, 1 and 2 in the order of the lower edges."""

    all_bands = sorted(set(electron_bands) | set(proton_bands))
    out = []
    for e in sorted(electron_bands):
        i = all_bands.index(e)
        for k in STEPS_ABOVE:
            if i + k < len(all_bands) and all_bands[i + k] in proton_bands:
                out.append((e, all_bands[i + k], k))
    return out


def analyse_point(
    electrons: ClassImages,
    protons: ClassImages,
    permutations: int,
    rng: np.random.Generator,
    with_excess: bool,
    eligibility: bool = True,
) -> list[PairingResult]:
    """``eligibility=True`` is the PRIMARY analysis (DEC-013 cut, Holm over the family). ``eligibility=False`` is the SECONDARY view
    (same matched pairs, no cut, raw p-values only: ``p_holm_family`` stays NaN)."""

    results = []
    for e_band, p_band, steps in band_pairings(
        sorted(set(electrons.band_lower_gev)), sorted(set(protons.band_lower_gev))
    ):
        ie_all = np.flatnonzero(electrons.band_lower_gev == e_band)
        ip_all = np.flatnonzero(protons.band_lower_gev == p_band)
        me, mpp = match_on_occupancy_and_energy(
            electrons.cells[ie_all],
            electrons.energy_mev[ie_all],
            protons.cells[ip_all],
            protons.energy_mev[ip_all],
        )
        key_e = np.column_stack([electrons.cells, electrons.energy_mev])[ie_all]
        key_p = np.column_stack([protons.cells, protons.energy_mev])[ip_all]
        results.append(
            run_pairing(
                e_band,
                p_band,
                steps,
                me,
                mpp,
                np.maximum(key_e, 1e-300),
                np.maximum(key_p, 1e-300),
                lambda idx, ie=ie_all: electrons.dims(ie[idx], eligibility),
                lambda idx, ip=ip_all: protons.dims(ip[idx], eligibility),
                (lambda idx, ie=ie_all: electrons.excess(ie[idx], eligibility))
                if with_excess
                else None,
                (lambda idx, ip=ip_all: protons.excess(ip[idx], eligibility))
                if with_excess
                else None,
                permutations,
                rng,
            )
        )
    if eligibility:
        holm_over_family(results)
    else:
        for r in results:
            for row in r.rows:
                row["p_holm_family"] = float("nan")
    return results


def photoelectron_scan(
    electron_events: ClassEvents,
    proton_events: ClassEvents,
    proton_selected: np.ndarray,
    permutations: int,
) -> list[dict[str, Any]]:
    """p.e. per MIP in {None, 10, 8, 5}; 3 fixed-seed draws per smeared point (1 for None, which is deterministic); the matching is
    redone after every smearing; both classes are treated identically."""

    out = []
    for pe in PHOTOELECTRON_SCAN:
        for draw in range(1 if pe is None else SCAN_DRAWS):
            e_img = build_class_images(
                electron_events,
                np.ones(len(electron_events), bool),
                "electron",
                1,
                pe,
                draw,
            )
            p_img = build_class_images(
                proton_events, proton_selected, "proton_interacting", 2, pe, draw
            )
            rng = np.random.default_rng([ANALYSIS_SEED, 5, int((pe or 0) * 10), draw])
            results = analyse_point(e_img, p_img, permutations, rng, with_excess=False)
            out.append(
                {
                    "pe_per_mip": pe,
                    "draw": draw,
                    "permutations": permutations,
                    "pairings": [pairing_to_dict(r) for r in results],
                }
            )
    return out


def pairing_to_dict(r: PairingResult) -> dict[str, Any]:
    return {
        "electron_band_gev": r.electron_band_gev,
        "proton_band_gev": r.proton_band_gev,
        "steps_above": r.steps_above,
        "caliper_sd": r.caliper_sd,
        "pairs": r.pairs,
        "testable": r.testable,
        "reason": r.reason,
        "standardized_mean_difference_log_cells_log_energy": r.standardized_mean_difference,
        "rows": r.rows,
    }


# --- hook for the 3D analysis (F6) and the EXP-016 extension (F7): not part of this module's analysis -------------------------------


def three_dimensional_hook(
    electrons: ClassEvents,
    protons: ClassEvents,
    statistic_function: Callable[
        [np.ndarray, np.ndarray, Sequence[int]], float | np.ndarray
    ],
    window: Sequence[int],
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """Run a 3D statistic on the fiducial electrons and interacting protons with the SAME matching as the 2D analysis.

    ``statistic_function(fine_voxel_index, fine_edep_mev, window)`` returns the statistic(s) of one event (a float or a 1D array).
    Keys for matching are (fine voxel count, edep_scintillator_mev), as the 3D images are the fine prefix deposits. Pairs below
    ``MIN_PAIRS`` are not testable. F6 and F7 belong to other modules; this only supplies the wiring.
    """

    sel_e = np.flatnonzero(fiducial_of(electrons))
    sel_p = np.flatnonzero(fiducial_of(protons) & protons.truth_occurred.astype(bool))
    me, mpp = match_on_occupancy_and_energy(
        electrons.fine_voxel_count[sel_e],
        electrons.edep_scintillator_mev[sel_e],
        protons.fine_voxel_count[sel_p],
        protons.edep_scintillator_mev[sel_p],
    )
    if len(me) < MIN_PAIRS:
        return {
            "pairs": len(me),
            "testable": False,
            "reason": f"fewer than {MIN_PAIRS} matched pairs (not testable)",
        }

    def values(events: ClassEvents, index: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.atleast_1d(statistic_function(*events.fine_deposits(int(i)), window))
                for i in index
            ],
            float,
        )

    ve, vp = values(electrons, sel_e[me]), values(protons, sel_p[mpp])
    test = compare_matched_pairs(
        ve, vp, None, None, permutations, np.random.default_rng(ANALYSIS_SEED)
    )
    return {
        "pairs": len(me),
        "testable": True,
        "ks": test["observed"].tolist(),
        "p_raw": test["p"].tolist(),
        "p_holm": holm_adjust(test["p"]).tolist(),
        "window": list(window),
    }


# --- output helpers -----------------------------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=1), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_jsonable(rows))


def _flatten_rows(
    results: Sequence[PairingResult], view: str = "primary"
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        head = {
            "view": view,
            "electron_band_gev": r.electron_band_gev,
            "proton_band_gev": r.proton_band_gev,
            "steps_above": r.steps_above,
            "pairs": r.pairs,
        }
        if r.rows:
            out.extend({**head, **row} for row in r.rows)
        else:
            # A pairing without statistic rows is still reported, as "not testable", never silently dropped.
            out.append({**head, "testable": False})
    return out


# --- figures ------------------------------------------------------------------------------------------------------------------------

CLASS_COLOURS = {
    "electron": "#1f77b4",
    "proton_crossing": "#7f7f7f",
    "proton_interacting": "#d62728",
    "proton": "#d62728",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, title: str, name: str, numbers: dict[str, Any], out: Path) -> None:
    fig.suptitle(f"{title}\n{EXPLORATORY_LABEL}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"{name}.png", dpi=130)
    _pyplot().close(fig)
    write_json(
        out / f"{name}.json",
        {
            "status": EXPLORATORY_LABEL,
            "figure": name,
            "title": title,
            "numbers": numbers,
        },
    )


def figure_f1_generation(classes, out: Path) -> None:
    """Generated theta, phi and energy against the declared distributions, and the weighted spectrum against the target."""

    plt = _pyplot()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    numbers: dict[str, Any] = {}
    cos2_low = math.cos(THETA_MAX_RAD) ** 2
    for name, (events, selected) in classes.items():
        c = CLASS_COLOURS[name]
        theta, phi = (
            events.primary_theta_rad[selected],
            events.primary_phi_rad[selected],
        )
        u = (np.cos(theta) ** 2 - cos2_low) / (
            1 - cos2_low
        )  # uniform on [0, 1] if cos^2(theta) is uniform up to 25 degrees
        ax[0, 0].hist(
            u,
            bins=20,
            range=(0, 1),
            density=True,
            histtype="step",
            color=c,
            label=f"{name} (n={len(theta)})",
        )
        ax[0, 1].hist(
            np.mod(phi, 2 * np.pi),
            bins=20,
            range=(0, 2 * np.pi),
            density=True,
            histtype="step",
            color=c,
        )
        frac = np.log(
            events.primary_energy_gev[selected] / events.band_lower_gev[selected]
        ) / np.log(events.band_upper_gev[selected] / events.band_lower_gev[selected])
        ax[1, 0].hist(
            frac, bins=20, range=(0, 1), density=True, histtype="step", color=c
        )
        numbers[name] = {
            "n": int(selected.sum()),
            "max_theta_deg": float(np.degrees(theta.max())) if len(theta) else None,
            "cos2_theta_scaled_hist": np.histogram(
                u, bins=20, range=(0, 1), density=True
            )[0].tolist(),
            "phi_hist": np.histogram(
                np.mod(phi, 2 * np.pi), bins=20, range=(0, 2 * np.pi), density=True
            )[0].tolist(),
            "log_energy_fraction_hist": np.histogram(
                frac, bins=20, range=(0, 1), density=True
            )[0].tolist(),
        }
        weights = class_weights(events, "electron" if name == "electron" else "proton")[
            selected
        ]
        energy = events.primary_energy_gev[selected]
        edges = (
            np.geomspace(energy.min(), energy.max(), 13)
            if len(energy)
            else np.array([1.0, 2.0])
        )
        sums = np.histogram(energy, bins=edges, weights=weights)[0] / np.diff(edges)
        centres = np.sqrt(edges[1:] * edges[:-1])
        ok = sums > 0
        ax[1, 1].loglog(centres[ok], sums[ok], "o", color=c, label=f"{name} weighted")
        density = electron_density if name == "electron" else proton_density
        if ok.any():
            scale = sums[ok][0] / density(centres[ok][0])
            ax[1, 1].loglog(centres, scale * density(centres), "-", color=c, alpha=0.5)
        numbers[name]["weighted_spectrum"] = {
            "energy_gev": centres[ok].tolist(),
            "weight_per_gev": sums[ok].tolist(),
        }
    ax[0, 0].axhline(1, color="k", ls=":")
    ax[0, 0].set_xlabel(
        "(cos^2 theta - cos^2 25 deg) / (1 - cos^2 25 deg); declared: uniform"
    )
    ax[0, 1].axhline(1 / (2 * np.pi), color="k", ls=":")
    ax[0, 1].set_xlabel("phi [rad]; declared: uniform")
    ax[1, 0].axhline(1, color="k", ls=":")
    ax[1, 0].set_xlabel(
        "ln(E/E_low)/ln(E_high/E_low) in the band; declared: uniform (E^-1)"
    )
    ax[1, 1].set_xlabel("E [GeV]")
    ax[1, 1].set_ylabel("sum of weights per GeV (line: declared physical spectrum)")
    ax[0, 0].legend(fontsize=7)
    ax[1, 1].legend(fontsize=7)
    _finish(
        fig,
        "F1 generated theta, phi, E against the declared distributions",
        "F1_generation",
        numbers,
        out,
    )


def _binned_quantiles(x, y, edges):
    rows = []
    for lo, hi in pairwise(edges):
        sel = (x >= lo) & (x < hi) & np.isfinite(y)
        rows.append(
            [np.median(y[sel]), *np.percentile(y[sel], [16, 84])]
            if sel.sum()
            else [np.nan] * 3
        )
    return np.array(rows)


def figure_f2_occupancy(classes, out: Path, data_root: Path | None) -> None:
    """Occupied readout cells and fine voxels against energy, per class, with EXP-011 normal-incidence medians as reference."""

    plt = _pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    numbers: dict[str, Any] = {}
    for name, (events, selected) in classes.items():
        c = CLASS_COLOURS[name]
        energy = events.primary_energy_gev[selected]
        cells = occupied_cells(events.readout_grid_mev[selected])
        voxels = events.fine_voxel_count[selected]
        ax[0].loglog(
            energy, np.maximum(cells, 0.5), ".", color=c, alpha=0.5, label=name
        )
        ax[1].loglog(
            energy, np.maximum(voxels, 0.5), ".", color=c, alpha=0.5, label=name
        )
        numbers[name] = {
            "energy_gev": energy.tolist(),
            "occupied_readout_cells": cells.tolist(),
            "fine_voxels": voxels.tolist(),
        }
    reference = _exp011_reference(data_root)
    for label, points in reference.items():
        key = (
            "electron"
            if label == "electron"
            else (
                "proton_interacting"
                if label.endswith("interacting")
                else "proton_crossing"
            )
        )
        for panel, field_name in ((0, "median_cells"), (1, "median_voxels")):
            ax[panel].plot(
                points["energy_gev"],
                points[field_name],
                "s",
                mfc="none",
                mec=CLASS_COLOURS[key],
                ms=9,
                label=f"EXP-011 normal incidence {label}",
            )
    numbers["exp011_reference_medians"] = reference
    ax[0].set_ylabel("occupied readout cells (of 1296)")
    ax[1].set_ylabel("occupied fine voxels")
    for a in ax:
        a.set_xlabel("primary energy [GeV]")
        a.legend(fontsize=7)
    _finish(
        fig,
        "F2 occupancy against energy (all events, before the fiducial cut)",
        "F2_occupancy",
        numbers,
        out,
    )


def _exp011_reference(data_root: Path | None) -> dict[str, dict[str, list[float]]]:
    """Median occupied readout cells / fine voxels per energy of the EXP-011 normal-incidence pilots, when present."""

    out: dict[str, dict[str, list[float]]] = {}
    if data_root is None:
        return out
    for cls, directory in EXP011_DIRECTORIES.items():
        groups: dict[str, dict[float, list[tuple[float, float]]]] = {}
        for path in sorted((data_root / directory).rglob("events.npz")):
            try:
                meta = json.loads(
                    (path.parent / "metadata.json").read_text(encoding="utf-8")
                )
                with np.load(path) as data:
                    cells = (
                        (data["readout_grid_mev"] > 0)
                        .reshape(len(data["readout_grid_mev"]), -1)
                        .sum(axis=1)
                    )
                    voxels = np.diff(data["fine_offsets"])
                    occurred = data["truth_occurred"].astype(bool)
            except OSError, KeyError, ValueError:
                continue
            energy = float(meta["energy_gev"])
            for label, sel in (
                [("electron", np.ones(len(cells), bool))]
                if cls == "electron"
                else [("proton_interacting", occurred), ("proton_crossing", ~occurred)]
            ):
                if sel.any():
                    groups.setdefault(label, {}).setdefault(energy, []).append(
                        (float(np.median(cells[sel])), float(np.median(voxels[sel])))
                    )
        for label, by_energy in groups.items():
            energies = sorted(by_energy)
            out[label] = {
                "energy_gev": energies,
                "median_cells": [
                    float(np.mean([c for c, _ in by_energy[e]])) for e in energies
                ],
                "median_voxels": [
                    float(np.mean([v for _, v in by_energy[e]])) for e in energies
                ],
            }
    return out


def figure_f3_acceptance(classes, out: Path) -> None:
    """Fiducial acceptance and missing energy (E - edep_total) against distance of the entry point to the nearest edge and theta."""

    plt = _pyplot()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    numbers: dict[str, Any] = {}
    distance_edges = np.linspace(0, HALF_WIDTH_MM, 13)
    theta_edges = np.linspace(0, np.degrees(THETA_MAX_RAD) + 1e-9, 9)
    for name, (events, selected) in classes.items():
        c = CLASS_COLOURS[name]
        fid = fiducial_of(events)[selected]
        distance = distance_to_nearest_edge(
            events.entry_x_mm[selected], events.entry_y_mm[selected]
        )
        theta = np.degrees(events.primary_theta_rad[selected])
        energy_mev = events.primary_energy_gev[selected] * 1000.0
        missing = (energy_mev - events.edep_total_mev[selected]) / energy_mev
        numbers[name] = {}
        for column, (x, edges, xlabel) in enumerate(
            (
                (distance, distance_edges, "entry distance to nearest edge [mm]"),
                (theta, theta_edges, "theta [deg]"),
            )
        ):
            centres = 0.5 * (edges[1:] + edges[:-1])
            accepted = np.array(
                [
                    fid[(x >= lo) & (x < hi)].mean()
                    if ((x >= lo) & (x < hi)).any()
                    else np.nan
                    for lo, hi in pairwise(edges)
                ]
            )
            q = _binned_quantiles(x, missing, edges)
            ax[0, column].plot(centres, accepted, "o-", color=c, label=name)
            ax[1, column].errorbar(
                centres,
                q[:, 0],
                yerr=[q[:, 0] - q[:, 1], q[:, 2] - q[:, 0]],
                fmt="o-",
                color=c,
                capsize=2,
            )
            ax[1, column].set_xlabel(xlabel)
            numbers[name][("vs_distance" if column == 0 else "vs_theta")] = {
                "bin_centres": centres.tolist(),
                "fiducial_accepted_fraction": accepted.tolist(),
                "missing_energy_fraction_median_16_84": q.tolist(),
            }
    ax[0, 0].set_ylabel("fiducial accepted fraction")
    ax[1, 0].set_ylabel("(E - edep_total) / E, median and 16-84%")
    ax[0, 0].legend(fontsize=7)
    _finish(
        fig,
        "F3 fiducial acceptance and missing energy",
        "F3_acceptance_missing_energy",
        numbers,
        out,
    )


CONVENTIONAL_PANELS = (
    "log_total_energy",
    "occupied_cells",
    "layer_cog",
    "width_cells",
    "core_fraction",
    "max_cell_fraction",
)


def figure_f4_conventional(classes, out: Path) -> dict[str, Any]:
    """Conventional features (``multiscale_test.conventional_features``) of fiducial events per band and class."""

    plt = _pyplot()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    numbers: dict[str, Any] = {}
    for name, (events, selected) in classes.items():
        fid = selected & fiducial_of(events)
        if not fid.any():
            numbers[name] = {"note": "no fiducial event"}
            continue
        names, table = mt.conventional_features(
            events.readout_grid_mev[fid].astype(float)
        )
        bands = events.band_lower_gev[fid]
        numbers[name] = {}
        for k, feature in enumerate(CONVENTIONAL_PANELS):
            col = names.index(feature)
            for lower in sorted(set(bands)):
                v = table[bands == lower, col]
                q = np.percentile(v, [16, 50, 84])
                shift = {
                    "electron": -0.06,
                    "proton_crossing": 0.0,
                    "proton_interacting": 0.06,
                }[name]
                a = ax.ravel()[k]
                a.errorbar(
                    [np.log10(lower) + shift],
                    [q[1]],
                    yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                    fmt="o",
                    color=CLASS_COLOURS[name],
                    capsize=2,
                    label=name if lower == min(bands) else None,
                )
                numbers[name].setdefault(feature, {})[f"band_{lower:g}"] = {
                    "n": len(v),
                    "p16_p50_p84": q.tolist(),
                }
    for k, feature in enumerate(CONVENTIONAL_PANELS):
        ax.ravel()[k].set_title(feature, fontsize=9)
        ax.ravel()[k].set_xlabel("log10(band lower edge / GeV)")
    ax.ravel()[0].legend(fontsize=7)
    _finish(
        fig,
        "F4 conventional features, electrons against protons, per band (fiducial, unweighted)",
        "F4_conventional",
        numbers,
        out,
    )
    return numbers


def _pairing_label(r: PairingResult) -> str:
    return f"e{r.electron_band_gev:g}|p{r.proton_band_gev:g}" + (
        "" if r.testable else "*"
    )


def figure_f5_dimensions(
    results: Sequence[PairingResult],
    out: Path,
    secondary: bool = False,
    sensitivity: bool = False,
) -> None:
    """Matched-pair D0, D1, D2 per window and pairing; medians of electrons, protons and of their cell-shuffle surrogates."""

    plt = _pyplot()
    fig, ax = plt.subplots(
        len(STATISTIC_Q), len(WINDOWS), figsize=(3.2 * len(WINDOWS), 8), squeeze=False
    )
    names = dimension_names()
    numbers: dict[str, Any] = {
        "pairings": [_pairing_label(r) for r in results],
        "statistics": {},
    }
    shown = [r for r in results if r.rows]
    x = np.arange(len(shown))
    for j, name in enumerate(names):
        a = ax[j % len(STATISTIC_Q), j // len(STATISTIC_Q)]
        rows = [r.rows[j] for r in shown]
        me = np.array([row["median_electron"] for row in rows])
        mpr = np.array([row["median_proton"] for row in rows])
        a.plot(x, me, "o", color=CLASS_COLOURS["electron"], label="electron")
        a.plot(
            x,
            mpr,
            "s",
            color=CLASS_COLOURS["proton_interacting"],
            label="interacting proton",
        )
        entry = {"median_electron": me.tolist(), "median_proton": mpr.tolist()}
        if rows and "median_excess_electron" in rows[0]:
            se = np.array([row["median_surrogate_electron"] for row in rows])
            sp = np.array([row["median_surrogate_proton"] for row in rows])
            a.plot(
                x,
                se,
                "o",
                mfc="none",
                color=CLASS_COLOURS["electron"],
                label="cell-shuffle surrogate (e)",
            )
            a.plot(
                x,
                sp,
                "s",
                mfc="none",
                color=CLASS_COLOURS["proton_interacting"],
                label="cell-shuffle surrogate (p)",
            )
            entry.update(
                surrogate_median_electron=se.tolist(),
                surrogate_median_proton=sp.tolist(),
            )
        entry.update(
            ks=[row["ks"] for row in rows],
            p_raw=[row["p_raw"] for row in rows],
            p_holm_family=[row["p_holm_family"] for row in rows],
        )
        numbers["statistics"][name] = entry
        a.set_title(name, fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels([_pairing_label(r) for r in shown], rotation=90, fontsize=6)
    if not shown:
        for a in ax.ravel():
            a.text(
                0.5,
                0.5,
                "no pairing testable\n(< 100 matched pairs)",
                ha="center",
                va="center",
                transform=a.transAxes,
            )
    ax[0, 0].legend(fontsize=6)
    numbers["untestable_pairings"] = [_pairing_label(r) for r in results if not r.rows]
    if secondary:
        numbers["view"] = SECONDARY_LABEL
        numbers["multiplicity"] = "raw p-values only (no Holm)"
    if sensitivity:
        numbers["view"] = SENSITIVITY_LABEL
        numbers["theta_max_deg"] = SENSITIVITY_THETA_MAX_DEG
        numbers["multiplicity"] = "Holm within this stratum's own family"
        title = f"F5 ({SENSITIVITY_LABEL}): D0 D1 D2 of matched pairs (* = pairing not testable, < 100 pairs)"
        stem = "F5_matched_dimensions_theta12"
    elif secondary:
        title = f"F5 secondary: D0 D1 D2 of matched pairs, no eligibility cut ({SECONDARY_LABEL}; raw p only)"
        stem = "F5_matched_dimensions_secondary"
    else:
        title = "F5 2D readout D0 D1 D2 of matched pairs (* = pairing not testable, < 100 pairs)"
        stem = "F5_matched_dimensions"
    _finish(fig, title, stem, numbers, out)


def figure_f5b_scan(scan: Sequence[dict[str, Any]], out: Path) -> None:
    """Matched-pair KS against photoelectrons per MIP for the full window, within-band pairings."""

    plt = _pyplot()
    fig, ax = plt.subplots(1, len(STATISTIC_Q), figsize=(14, 4.5), squeeze=False)
    positions = {5.0: 0, 8.0: 1, 10.0: 2, None: 3}
    full = WINDOWS[0]
    numbers: dict[str, Any] = {"window": list(full), "series": {}}
    bands = sorted(
        {
            p["electron_band_gev"]
            for point in scan
            for p in point["pairings"]
            if p["steps_above"] == 0
        }
    )
    for q_index, q in enumerate(STATISTIC_Q):
        name = f"D{int(q)}@{'-'.join(map(str, full))}"
        for band in bands:
            xs, ys, untestable = [], [], []
            for pe, pos in positions.items():
                values = []
                for point in scan:
                    if point["pe_per_mip"] != pe:
                        continue
                    for p in point["pairings"]:
                        if p["electron_band_gev"] == band and p["steps_above"] == 0:
                            row = next(
                                (
                                    r
                                    for r in p["rows"]
                                    if r["statistic"] == name and r["testable"]
                                ),
                                None,
                            )
                            if row:
                                values.append(row["ks"])
                            else:
                                untestable.append((pe, p["pairs"]))
                if values:
                    xs.append(pos)
                    ys.append(values)
            if xs:
                ax[0, q_index].plot(
                    xs, [np.mean(v) for v in ys], "o-", label=f"band {band:g} GeV"
                )
                for xv, v in zip(xs, ys, strict=True):
                    ax[0, q_index].plot([xv] * len(v), v, "_", color="k", alpha=0.4)
            numbers["series"][f"{name}/band{band:g}"] = {
                "pe_positions": xs,
                "ks_per_draw": ys,
                "not_testable_points": [[str(a), b] for a, b in untestable],
            }
        ax[0, q_index].set_xticks(list(positions.values()))
        ax[0, q_index].set_xticklabels(["5", "8", "10", "no smearing"])
        ax[0, q_index].set_xlabel("photoelectrons per MIP")
        ax[0, q_index].set_title(name, fontsize=9)
    ax[0, 0].set_ylabel("matched-pair KS (marks: draws)")
    ax[0, 0].legend(fontsize=7)
    _finish(
        fig,
        "F5b matched-pair KS against photoelectrons per MIP (within-band pairings)",
        "F5b_photoelectron_scan",
        numbers,
        out,
    )


# --- orchestration ------------------------------------------------------------------------------------------------------------------


def run_analysis(
    electrons: ClassEvents | None,
    protons: ClassEvents | None,
    out: Path,
    data_root: Path | None = None,
    permutations: int = PERMUTATIONS,
    scan_permutations: int | None = None,
    run_scan: bool = True,
) -> dict[str, Any]:
    """Run every stage that the available classes allow and write figures, JSON and CSV to ``out``."""

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    classes = class_masks(electrons, protons)
    summary: dict[str, Any] = {
        "status": EXPLORATORY_LABEL,
        "ambiguities": ambiguities(),
        "permutations": permutations,
        "classes_available": {k: int(v[1].sum()) for k, v in classes.items()},
        "stages_run": [],
    }
    if not classes:
        raise ValueError("no events supplied")
    acceptance, ess = acceptance_table(classes), ess_table(classes)
    write_csv(out / "fiducial_acceptance.csv", acceptance)
    write_csv(out / "effective_sample_size.csv", ess)
    acceptance12 = acceptance_table(classes, SENSITIVITY_THETA_MAX_RAD)
    write_csv(out / "fiducial_acceptance_theta12.csv", acceptance12)
    summary.update(
        fiducial_acceptance=acceptance,
        effective_sample_size=ess,
        **{"fiducial_acceptance_" + SENSITIVITY_LABEL: acceptance12},
    )
    for name, (events, selected) in classes.items():
        kind = "electron" if name == "electron" else "proton"
        w = class_weights(events, kind)
        write_csv(
            out / f"weights_{name}.csv",
            [
                {
                    "event": int(i),
                    "band_lower_gev": float(events.band_lower_gev[i]),
                    "energy_gev": float(events.primary_energy_gev[i]),
                    "fiducial": bool(fiducial_of(events)[i]),
                    "weight": float(w[i]),
                }
                for i in np.flatnonzero(selected)
            ],
        )
    summary["stages_run"] += ["fiducial", "weights"]
    drawn = {
        k: v for k, v in classes.items() if v[1].any()
    }  # a class with no event (e.g. no crossing protons) has nothing to draw
    figure_f1_generation(drawn, out)
    figure_f2_occupancy(drawn, out, data_root)
    figure_f3_acceptance(drawn, out)
    figure_f4_conventional(drawn, out)
    summary["stages_run"] += ["F1", "F2", "F3", "F4"]
    if electrons is not None and protons is not None:
        interacting = protons.truth_occurred.astype(bool)
        e_img = build_class_images(
            electrons, np.ones(len(electrons), bool), "electron", 1
        )
        p_img = build_class_images(protons, interacting, "proton_interacting", 2)
        rng = np.random.default_rng([ANALYSIS_SEED, 1])
        results = analyse_point(e_img, p_img, permutations, rng, with_excess=True)
        rng_secondary = np.random.default_rng([ANALYSIS_SEED, 3])
        secondary = analyse_point(
            e_img,
            p_img,
            permutations,
            rng_secondary,
            with_excess=True,
            eligibility=False,
        )
        e_img12 = build_class_images(
            electrons,
            np.ones(len(electrons), bool),
            "electron",
            1,
            theta_max_rad=SENSITIVITY_THETA_MAX_RAD,
        )
        p_img12 = build_class_images(
            protons,
            interacting,
            "proton_interacting",
            2,
            theta_max_rad=SENSITIVITY_THETA_MAX_RAD,
        )
        rng_sensitivity = np.random.default_rng([ANALYSIS_SEED, 6])
        sensitivity = analyse_point(
            e_img12, p_img12, permutations, rng_sensitivity, with_excess=True
        )
        write_csv(
            out / "matched_pair_statistics.csv",
            _flatten_rows(results, "primary")
            + _flatten_rows(secondary, SECONDARY_LABEL)
            + _flatten_rows(sensitivity, SENSITIVITY_LABEL),
        )
        write_json(
            out / "matched_pair_statistics.json",
            {
                "status": EXPLORATORY_LABEL,
                "pairings": [pairing_to_dict(r) for r in results],
                SECONDARY_LABEL: {
                    "note": "same matched pairs, no per-window eligibility cut; raw p-values only (no Holm)",
                    "pairings": [pairing_to_dict(r) for r in secondary],
                },
                SENSITIVITY_LABEL: {
                    "note": (
                        "PRIMARY analysis (DEC-013 eligibility, matching, pair-swap KS, cell-shuffle reference) on the fiducial events "
                        "with primary_theta <= 12 deg; Holm within this stratum's own family; project approximation of the AMS "
                        "acceptance (about 550 cm2 sr, Aguilar et al., PRL 113, 221102 (2014)), not an AMS cut; researcher decision "
                        "2026-11-09"
                    ),
                    "theta_max_deg": SENSITIVITY_THETA_MAX_DEG,
                    "pairings": [pairing_to_dict(r) for r in sensitivity],
                },
            },
        )
        summary["matched_pairs"] = [
            {k: v for k, v in pairing_to_dict(r).items() if k != "rows"}
            for r in results
        ]
        summary["matched_pairs_" + SECONDARY_LABEL] = [
            {k: v for k, v in pairing_to_dict(r).items() if k != "rows"}
            for r in secondary
        ]
        summary["matched_pairs_" + SENSITIVITY_LABEL] = [
            {k: v for k, v in pairing_to_dict(r).items() if k != "rows"}
            for r in sensitivity
        ]
        figure_f5_dimensions(results, out)
        figure_f5_dimensions(secondary, out, secondary=True)
        figure_f5_dimensions(sensitivity, out, sensitivity=True)
        summary["stages_run"] += [
            "matching",
            "pair-swap",
            "holm (primary only)",
            "F5",
            "F5 secondary",
            "F5 theta12 (" + SENSITIVITY_LABEL + ")",
        ]
        if run_scan:
            scan = photoelectron_scan(
                electrons, protons, interacting, scan_permutations or permutations
            )
            write_json(
                out / "photoelectron_scan.json",
                {"status": EXPLORATORY_LABEL, "points": scan},
            )
            figure_f5b_scan(scan, out)
            summary["stages_run"] += ["photoelectron scan", "F5b"]
    else:
        summary["note"] = (
            "only one class available: no matching, tests or scan were run"
        )
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    write_json(out / "summary.json", summary)
    return summary


def find_class_directory(data_root: Path, name: str, rung: str) -> Path | None:
    prefix = {
        "rung0": "geant4_realistic_regime_pilot_rung0_",
        "rung1": "geant4_realistic_regime_pilot_",
    }[rung]
    path = data_root / f"{prefix}{name}"
    return path if path.is_dir() and any(path.rglob("events.npz")) else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory holding geant4_realistic_regime_pilot_{electron,proton}[ rung0 variants]",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--rung",
        choices=("auto", "rung0", "rung1"),
        default="auto",
        help="auto: rung 1 if present for a class, else rung 0",
    )
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--scan-permutations", type=int, default=None)
    parser.add_argument("--skip-scan", action="store_true")
    args = parser.parse_args(argv)
    loaded: dict[str, ClassEvents | None] = {}
    for name in ("electron", "proton"):
        rungs = ("rung1", "rung0") if args.rung == "auto" else (args.rung,)
        directory = next(
            (d for r in rungs if (d := find_class_directory(args.data_root, name, r))),
            None,
        )
        loaded[name] = load_class_events(directory, name) if directory else None
        print(f"{name}: {directory}")
    summary = run_analysis(
        loaded["electron"],
        loaded["proton"],
        args.out,
        args.data_root,
        args.permutations,
        args.scan_permutations,
        not args.skip_scan,
    )
    print(json.dumps({k: summary[k] for k in ("stages_run", "runtime_seconds")}))


if __name__ == "__main__":
    main()
