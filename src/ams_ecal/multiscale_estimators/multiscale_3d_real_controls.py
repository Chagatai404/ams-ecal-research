"""EXP-016 (version 2): 3D estimator usability tested with controls built from real Geant4 showers.

Registration: ``~/.research-os-backups/registrations/2026-11-09_exp016_3d_controls_from_real_showers.txt`` (SHA-256 prefix
55a9e9489a51580e, binding). Status of the result: EXPLORATORY, reported beside the original S4 part 1 NEGATIVE verdict (7b816ca7c1a7ccb2).
Every section number below is the registration's.

What is measured. The fine-voxel generalized-dimension descriptors ``D_0``, ``D_1``, ``D_2`` of ``multiscale_3d`` (finite-scale
slopes of ``ln Z_q`` against ``ln`` effective box side) are recomputed with grid-phase (offset) averaging (section 4.2), a per-event
saturation scale from a balls-in-boxes fit with amplitude and dimension free (4.3), a window admissibility rule (4.4) and one
window ``W*`` for all events (4.5). ``W*`` is then checked against real-shower controls (sections 5 and 6): a machinery check (C0),
the ordering ``D_0 >= D_1 >= D_2`` (C1), a matched same-class cross-energy null (C2) and power against a planted local disruption (C3).
These are descriptors of a deposit pattern, not evidence of scaling; the saturation form is a PROJECT PHENOMENOLOGICAL ASSUMPTION
(Koo and Ju 2026, arXiv:2605.27925, unweighted stochastic trajectories, single group, unreplicated).

    python -m ams_ecal.multiscale_estimators.multiscale_3d_real_controls rung0 --workers 1
    python -m ams_ecal.multiscale_estimators.multiscale_3d_real_controls rung1 --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.inference import multiplicity as mp
from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.multiscale_estimators import multiscale_3d as m3

# --- registration identity ----------------------------------------------------------------------------------------------------

REGISTRATION_FILE = (
    Path.home()
    / ".research-os-backups"
    / "registrations"
    / "2026-11-09_exp016_3d_controls_from_real_showers.txt"
)
REGISTRATION_HASH_PREFIX = "55a9e9489a51580e"
AMENDMENT_1_FILE = (
    Path.home()
    / ".research-os-backups"
    / "registrations"
    / "2026-11-09_exp016_amendment_1_lateral_partial_boxes.txt"
)
AMENDMENT_1_HASH_PREFIX = (
    "795c2d71add4c1b0"  # amends section 4.2 only: partial boxes are lateral only
)
RESULT_DIR = PROJECT_ROOT / "results" / "multiscale_controls"

# --- data (section 3) --------------------------------------------------------------------------------------------------------

ELECTRON_DIR = PROJECT_ROOT / "data" / "geant4_highenergy_pilot_electron" / "pilot"
PROTON_DIR = PROJECT_ROOT / "data" / "geant4_highenergy_pilot_proton" / "pilot"


@dataclass(frozen=True)
class EventSet:
    """One class-energy set of section 3 (electrons: all events; protons: only ``truth_occurred`` events)."""

    key: str
    particle: str
    energy_gev: float

    @property
    def directory(self) -> Path:
        root = ELECTRON_DIR if self.particle == "electron" else PROTON_DIR
        return root / f"E{self.energy_gev:g}GeV"


SETS = (
    EventSet("electron_200GeV", "electron", 200.0),
    EventSet("electron_400GeV", "electron", 400.0),
    EventSet("electron_800GeV", "electron", 800.0),
    EventSet("proton_500GeV", "proton", 500.0),
    EventSet("proton_1000GeV", "proton", 1000.0),
    EventSet("proton_2000GeV", "proton", 2000.0),
)
SET_BY_KEY = {s.key: s for s in SETS}

# --- section 4: estimator ----------------------------------------------------------------------------------------------------

GRID_OFFSET_COUNT = 8
OFFSET_SEED = 20261109  # set by the registration (4.2)
OFFSET_RANGE = (54, 54, 36)  # ox, oy uniform integers in [0, 54); oz in [0, 36)
PARTIAL_BOX_ENERGY_LIMIT = (
    0.01  # 4.2: more than 1% of the energy in partial boxes at the coarsest scale
)
PARTIAL_BOX_RULES = ("with_depth", "lateral_only")
DEFAULT_PARTIAL_BOX_RULE = "lateral_only"  # amendment 1 (795c2d71add4c1b0); "with_depth" is the original literal reading, kept for the record
FIT_FRACTAL_DIMENSION_RANGE = (
    0.5,
    3.5,
)  # 4.3: a fit with D_f outside this open interval fails
DELTA_PRIMARY = 1
DELTA_SENSITIVITY = 2
MIN_WINDOW_SCALES = 3
MIN_ADMISSIBLE_FRACTION = 0.90
LN_EFFECTIVE_SIDE = np.log(m3.EFFECTIVE_SIDE_MM)
N_WINDOWS = len(m3.WINDOWS)

# --- sections 5 and 6: controls ----------------------------------------------------------------------------------------------

VARIANT_WEIGHTS = "W"  # energies permuted among the occupied voxels of a disrupted block (occupancy and D_0 unchanged)
VARIANT_SUPPORT = "S"  # occupied voxels moved with their energy to empty voxels of the same block and layer
VARIANTS = (VARIANT_WEIGHTS, VARIANT_SUPPORT)
PHI_GRID = (0.05, 0.1, 0.2, 0.5, 1.0)
PHI_STAR = 0.2  # set by the researcher on 2026-11-09
ALPHA = 0.05
C0_SPLITS = 200
C0_PERMUTATIONS = 2000
C0_STANDARD_ERRORS = 3.0
C2_PAIRINGS = (
    ("electron_200GeV", "electron_400GeV"),
    ("electron_400GeV", "electron_800GeV"),
    ("proton_500GeV", "proton_1000GeV"),
    ("proton_1000GeV", "proton_2000GeV"),
)
C2_CALIPER_SD = 0.25
C2_MIN_PAIRS = 100
C2_PERMUTATIONS = 10_000
C3_SETS = ("electron_400GeV", "proton_1000GeV")
C3_SPLITS = 200
C3_PERMUTATIONS = 2000
C3_REALISTIC_ARM = 150
C3_MAX_ARM = 300
C3_REQUIRED_POWER = 0.90
STATISTIC_NAMES = ("D2", "D0-D2")  # primary, secondary
SWAP_CHUNK = 1000

# Seeds (fixed by this module; only OFFSET_SEED is set by the registration)
DISRUPTION_SEED = 20261110
C0_SEED = 20261111
C2_SEED = 20261112
C3_SEED = 20261113
SYNTHETIC_SEED = 20261114

VERDICT_USABLE = "usable"
VERDICT_MATCHING = "estimator usable, matching insufficient for P1"
VERDICT_NOT_USABLE = "not usable"
VERDICT_C2_NOT_TESTABLE = "no registered verdict applies: C2 not testable"
VERDICT_STOPPED = "stopped: C0 machinery check failed (the test code is wrong)"

AMBIGUITIES = (
    (
        "4.2 partial boxes: amendment 1 (SHA-256 prefix 795c2d71add4c1b0, amending registration 55a9e9489a51580e, section 4.2 only) "
        "makes the default rule lateral-only: a box is partial only if it extends beyond the mesh in x or y. The original literal "
        "reading (depth included) made every event excluded because the coarsest kz = 36 equals the mesh depth; it remains available "
        "as --partial-box-rule with_depth for the record, and both fractions are always reported."
    ),
    (
        "4.2 offsets 'drawn once': read as ONE set of 8 offsets shared by every event, drawn by "
        "rng.integers(low=(0,0,0), high=(54,54,36), size=(8,3)) from default_rng(20261109); the triples are recorded."
    ),
    (
        "4.3 units: the effective side is in mm (m3.EFFECTIVE_SIDE_MM); the amplitude A of b = A eps^-D_f depends on that unit, "
        "eps* does not. The fit is parameterised by (ln b at the finest scale, D_f), an exact reparameterisation of (A, D_f)."
    ),
    (
        "4.3 'non-convergence': scipy least_squares (method lm) status <= 0, a non-finite parameter, or an exception; one "
        "deterministic start, no retry. Capacity b < 1 is clamped to 1 in the model."
    ),
    "4.4/4.5 denominators: events excluded by 4.2 are out of every denominator; fit failures stay in and count as not admissible.",
    (
        "6 C1-C3 use every non-excluded event on W* (the registration says ONE window for all events), including events that are "
        "not admissible on W*."
    ),
    (
        "5 number of disrupted blocks: exactly max(1, floor(phi * n + 0.5)) of the n occupied (layer, lateral block) units of the event "
        "(pooled over the whole event, drawn without replacement). The minimum of one is an addition so phi = 0.05 on a small event is "
        "not empty."
    ),
    (
        "5 variant S: 'empty' means empty in the original event; the targets are drawn without replacement. When a block has fewer "
        "empty voxels than occupied ones only as many randomly chosen occupied voxels as there are empty voxels move (the rest stay), "
        "so occupancy is still exactly preserved; the number of such blocks is reported."
    ),
    (
        "6 C0 applies to both statistics (D2 and D0-D2), each with its own rate; it is 'sane' only if both rates lie in 0.05 +- 3 SE "
        "(binomial SE with 200 splits), in every set."
    ),
    (
        "6 C2 with no testable pairing is not a pass (reported as 'not testable'); the "
        "verdict is then 'no registered verdict applies: C2 not testable' (amendment 1 reporting note), with C1 and C3 as measured."
    ),
    (
        "6 C2/C3 KS: the same definition as S3 stage 3 (stratified_ks_matrix, one stratum) but computed with exact integer "
        "numerators so ties in the permutation distribution are not broken by float32 rounding."
    ),
    "6 C3: the two arms are drawn as disjoint random event subsets of the SAME set (the registration's 'independent events').",
    (
        "4.5 when no W* exists, C1-C3 are 'reported descriptively only' on a fallback window (the >= 3-scale window with the highest "
        "minimum admissibility fraction over the sets, ties: finest then widest); that fallback is not in the registration."
    ),
    "9 stop rule 'one event takes more than 2 s': evaluated as the maximum in-worker per-event time (mean and median also reported).",
    (
        "8 B1 and B3 are the smooth baselines of EXP-010, not cascades: the recovery of known dimensions is computed on the S4 cascade "
        "(aligned levels of multiscale_3d) with offset averaging, which breaks the alignment of the cascade cells with the boxes; B1 and "
        "B3 are reported descriptively (ordering, fit acceptance, saturation scale)."
    ),
    "3 prefix deposit used for the C2 matching key: arrays['edep_total_mev'] (checked against the sum of the fine prefix deposits).",
)


# --- the registration, git and provenance ---------------------------------------------------------------------------------------


def registration_hash() -> str:
    return hashlib.sha256(REGISTRATION_FILE.read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ("git", *args),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.rstrip("\n")

    status = run("status", "--porcelain").splitlines()
    return {
        "commit": run("rev-parse", "HEAD").strip(),
        "dirty": bool(status),
        "modified_or_untracked_entries": len(status),
        "modified_tracked_files": [s[3:] for s in status if s[:2].strip() == "M"],
    }


# --- section 4.2: grid-phase offsets, padded boxes, partition quantities -------------------------------------------------------


def grid_offsets() -> np.ndarray:
    """The 8 offsets ``(ox, oy, oz)`` of section 4.2, drawn once with seed 20261109 and shared by every event."""
    rng = np.random.default_rng(OFFSET_SEED)
    return rng.integers(low=(0, 0, 0), high=OFFSET_RANGE, size=(GRID_OFFSET_COUNT, 3))


def padded_box_counts(offset: Sequence[int], kx: int, kz: int) -> tuple[int, int, int]:
    """Number of boxes per axis of the padded grid for one offset (the mesh shifted by ``offset`` voxels)."""
    return (
        (m3.MESH[0] - 1 + int(offset[0])) // kx + 1,
        (m3.MESH[1] - 1 + int(offset[1])) // kx + 1,
        (m3.MESH[2] - 1 + int(offset[2])) // kz + 1,
    )


def padded_box_key(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    offset: Sequence[int],
    kx: int,
    kz: int,
) -> tuple[np.ndarray, int]:
    """Linear index of the box of every voxel at scale ``(kx, kz)``: ``((ix+ox)//kx, (iy+oy)//kx, (iz+oz)//kz)`` on the padded grid.

    Unlike ``multiscale_3d.box_shares`` the box counts are those of the padded grid, so shifted boxes never collide. Returns the key
    and the number of possible keys.
    """
    nbx, nby, nbz = padded_box_counts(offset, kx, kz)
    bx = (ix + int(offset[0])) // kx
    by = (iy + int(offset[1])) // kx
    bz = (iz + int(offset[2])) // kz
    return (bz * nby + by) * nbx + bx, nbx * nby * nbz


def occupied_box_shares(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    energy: np.ndarray,
    offset: Sequence[int],
    kx: int,
    kz: int,
) -> np.ndarray:
    """Energy fractions of the occupied boxes at one scale and offset (partial boxes at the mesh edge are kept: energy is conserved)."""
    key, size = padded_box_key(ix, iy, iz, offset, kx, kz)
    sums = np.bincount(key, weights=energy, minlength=size)
    sums = sums[sums > 0]
    return sums / sums.sum()


def partial_energy_fraction(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    energy: np.ndarray,
    offset: Sequence[int],
    kx: int,
    kz: int,
    include_depth: bool,
) -> float:
    """Fraction of the energy in boxes that are not entirely inside the mesh at scale ``(kx, kz)`` and one offset.

    A box with index ``b`` along an axis of ``n`` voxels covers ``[b k - o, (b + 1) k - o)``; it is partial if that leaves ``[0, n)``.
    """
    # Amendment 1 (SHA-256 prefix 795c2d71add4c1b0) to section 4.2: the default partial-box rule is lateral only (x, y). With
    # include_depth (the original literal reading, non-default, kept for the record) the coarsest scale (kz = 36 = mesh depth)
    # makes every box partial for oz > 0.
    partial = np.zeros(len(energy), dtype=bool)
    axes = [(ix, int(offset[0]), kx, m3.MESH[0]), (iy, int(offset[1]), kx, m3.MESH[1])]
    if include_depth:
        axes.append((iz, int(offset[2]), kz, m3.MESH[2]))
    for index, shift, k, n in axes:
        lower = ((index + shift) // k) * k - shift
        partial |= (lower < 0) | (lower + k > n)
    return float(energy[partial].sum() / energy.sum())


@dataclass(frozen=True, eq=False)
class OffsetAveragedProfile:
    """Offset-averaged partition quantities at the nine scales (section 4.2).

    ``mean_occupied_boxes`` is Z_0 and ``mean_squared_share_sum`` is Z_2, both averaged over the offsets BEFORE the logarithm;
    ``mean_entropy`` is the offset average of the Shannon entropy (q = 1). ``partial_fraction_*`` is the largest, over the offsets,
    energy fraction in partial boxes at the coarsest scale (lateral edges only, and including depth).
    """

    n_occupied_voxels: int
    mean_occupied_boxes: np.ndarray
    mean_squared_share_sum: np.ndarray
    mean_entropy: np.ndarray
    partial_fraction_lateral: float
    partial_fraction_with_depth: float

    def log_measures(self) -> dict[float, np.ndarray]:
        """``ln Z_0``, the entropy and ``ln Z_2`` at the nine scales (the ``y`` of ``multiscale_3d.window_dimension``)."""
        return {
            0.0: np.log(self.mean_occupied_boxes),
            1.0: self.mean_entropy,
            2.0: np.log(self.mean_squared_share_sum),
        }


def offset_averaged_profile(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    energy: np.ndarray,
    offsets: np.ndarray | None = None,
) -> OffsetAveragedProfile:
    """Z_0, Z_2 and the entropy at the nine scales averaged over the grid offsets (section 4.2). Only voxels with energy > 0 count."""
    offsets = grid_offsets() if offsets is None else np.asarray(offsets)
    energy = np.asarray(energy, dtype=float)
    keep = energy > 0
    ix, iy, iz = (np.asarray(a, dtype=np.int64)[keep] for a in (ix, iy, iz))
    energy = energy[keep]
    n_scales = len(m3.SCALES)
    z0 = np.empty((len(offsets), n_scales))
    z2 = np.empty_like(z0)
    entropy = np.empty_like(z0)
    p_fine = energy / energy.sum()
    fine = (
        float(len(p_fine)),
        float(np.dot(p_fine, p_fine)),
        float(-np.sum(p_fine * np.log(p_fine))),
    )
    partial_lateral = partial_depth = 0.0
    kx_max, kz_max = m3.SCALES[-1]
    for o, offset in enumerate(offsets):
        for s, (kx, kz) in enumerate(m3.SCALES):
            if s == 0:  # (1, 1) is offset-invariant: every voxel is its own box
                z0[o, s], z2[o, s], entropy[o, s] = fine
                continue
            p = occupied_box_shares(ix, iy, iz, energy, offset, kx, kz)
            z0[o, s], z2[o, s], entropy[o, s] = (
                len(p),
                float(np.dot(p, p)),
                float(-np.sum(p * np.log(p))),
            )
        partial_lateral = max(
            partial_lateral,
            partial_energy_fraction(
                ix, iy, iz, energy, offset, kx_max, kz_max, include_depth=False
            ),
        )
        partial_depth = max(
            partial_depth,
            partial_energy_fraction(
                ix, iy, iz, energy, offset, kx_max, kz_max, include_depth=True
            ),
        )
    return OffsetAveragedProfile(
        n_occupied_voxels=len(energy),
        mean_occupied_boxes=z0.mean(axis=0),
        mean_squared_share_sum=z2.mean(axis=0),
        mean_entropy=entropy.mean(axis=0),
        partial_fraction_lateral=partial_lateral,
        partial_fraction_with_depth=partial_depth,
    )


def window_dimensions(
    profile: OffsetAveragedProfile, windows: Sequence[tuple[int, ...]] = m3.WINDOWS
) -> np.ndarray:
    """``(3, n_windows)``: D_0, D_1, D_2 of every window (slopes against ln effective side, ``multiscale_3d.window_dimension``)."""
    y = profile.log_measures()
    out = np.empty((3, len(windows)))
    for k, window in enumerate(windows):
        for row, q in enumerate(ms.Q_VALUES):
            out[row, k] = m3.window_dimension(y[q], q, window)
    return out


def partial_fraction(lateral: Any, with_depth: Any, rule: str) -> Any:
    """The partial-box energy fraction that the exclusion rule uses: ``'with_depth'`` (literal) or ``'lateral_only'``."""
    if rule not in PARTIAL_BOX_RULES:
        raise ValueError(f"unknown partial-box rule {rule!r}")
    return with_depth if rule == "with_depth" else lateral


# --- section 4.3: per-event saturation scale ----------------------------------------------------------------------------------


def saturated_box_count(capacity: np.ndarray, n_effective: float) -> np.ndarray:
    """``N = b [1 - (1 - 1/b)^N_eff]``: expected occupied boxes when ``N_eff`` points fall in ``b`` equally likely boxes (b >= 1)."""
    b = np.maximum(np.asarray(capacity, dtype=float), 1.0 + 1e-9)
    return b * -np.expm1(n_effective * np.log1p(-1.0 / b))


@dataclass(frozen=True)
class SaturationFit:
    """The balls-in-boxes fit of one event: ``b(eps) = A eps^-D_f`` (eps = effective side in mm), eps* with ``b(eps*) = N_eff``."""

    accepted: bool
    converged: bool
    ln_amplitude_mm: float
    fractal_dimension: float
    saturation_side_mm: float


_FAILED_FIT = SaturationFit(False, False, float("nan"), float("nan"), float("inf"))


def fit_saturation(mean_occupied_boxes: np.ndarray, n_effective: int) -> SaturationFit:
    """Least-squares fit in ``ln N`` of the saturation law to the nine offset-averaged occupied-box counts (section 4.3).

    ``A`` and ``D_f`` are both free; the fit is parameterised by ``(ln b at the finest scale, D_f)``. Failure = non-convergence, a
    non-finite result, or ``D_f`` outside (0.5, 3.5). Statistically: ``D_f`` is the box-counting exponent of the unsaturated
    coarse scales, ``eps*`` is the side below which the ``N_eff`` occupied voxels no longer fill new boxes.
    """
    n_obs = np.asarray(mean_occupied_boxes, dtype=float)
    ln_n = np.log(n_obs)
    ln_n_eff = float(np.log(n_effective))
    ln_eps0 = LN_EFFECTIVE_SIDE[0]
    try:
        coarse = slice(4, None)  # start from the coarse (least saturated) scales
        slope, intercept = np.polyfit(
            LN_EFFECTIVE_SIDE[coarse] - ln_eps0, ln_n[coarse], 1
        )
        start = np.array(
            [max(intercept, ln_n_eff + np.log(1.05)), float(np.clip(-slope, 0.6, 3.4))]
        )

        def residual(theta: np.ndarray) -> np.ndarray:
            ln_b = theta[0] - theta[1] * (LN_EFFECTIVE_SIDE - ln_eps0)
            return ln_n - np.log(
                saturated_box_count(np.exp(np.minimum(ln_b, 700.0)), n_effective)
            )

        result = least_squares(residual, start, method="lm")
    except ValueError, FloatingPointError, np.linalg.LinAlgError:
        return _FAILED_FIT
    ln_b_fine, d_f = (float(v) for v in result.x)
    converged = bool(result.status > 0 and np.all(np.isfinite(result.x)))
    low, high = FIT_FRACTAL_DIMENSION_RANGE
    accepted = converged and low < d_f < high
    ln_eps_star = (
        ln_eps0 + (ln_b_fine - ln_n_eff) / d_f
        if converged and d_f != 0
        else float("nan")
    )
    return SaturationFit(
        accepted=accepted,
        converged=converged,
        ln_amplitude_mm=ln_b_fine + d_f * ln_eps0,
        fractal_dimension=d_f,
        saturation_side_mm=float(np.exp(ln_eps_star)) if accepted else float("inf"),
    )


# --- sections 4.4 and 4.5: admissibility and the single window W* ----------------------------------------------------------------


def admissible_windows(
    saturation_side_mm: np.ndarray, accepted: np.ndarray, delta: int
) -> np.ndarray:
    """``(n_events, n_windows)``: a window is admissible if its finest effective side is at least ``2^delta eps*`` (a failed fit never is)."""
    finest = m3.EFFECTIVE_SIDE_MM[[w[0] for w in m3.WINDOWS]]
    limit = 2.0**delta * np.asarray(saturation_side_mm, dtype=float)[:, None]
    return np.asarray(accepted, dtype=bool)[:, None] & (finest[None, :] >= limit)


def select_window(
    fractions: np.ndarray, threshold: float = MIN_ADMISSIBLE_FRACTION
) -> int | None:
    """Index into ``m3.WINDOWS`` of W* (4.5), or None.

    ``fractions`` is ``(n_sets, n_windows)``: the admissible share of events in each class-energy set. Among windows of at least
    three consecutive scales whose fraction reaches ``threshold`` in EVERY set, take the one whose finest scale is finest
    (ties: the widest).
    """
    fractions = np.asarray(fractions, dtype=float)
    candidates = [
        k
        for k, w in enumerate(m3.WINDOWS)
        if len(w) >= MIN_WINDOW_SCALES and bool(np.all(fractions[:, k] >= threshold))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda k: (m3.WINDOWS[k][0], -len(m3.WINDOWS[k])))


def fallback_window(fractions: np.ndarray) -> int:
    """Descriptive-only window when W* does not exist: the >= 3-scale window with the highest minimum fraction (ties: finest, widest)."""
    # REGISTRATION AMBIGUITY: the registration says C2-C3 are 'reported descriptively only' without naming the window.
    fractions = np.asarray(fractions, dtype=float)
    candidates = [k for k, w in enumerate(m3.WINDOWS) if len(w) >= MIN_WINDOW_SCALES]
    return min(
        candidates,
        key=lambda k: (
            -float(np.nanmin(fractions[:, k])),
            m3.WINDOWS[k][0],
            -len(m3.WINDOWS[k]),
        ),
    )


# --- per-event processing and data access -------------------------------------------------------------------------------------


def process_event(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    energy: np.ndarray,
    offsets: np.ndarray,
) -> dict[str, Any]:
    """Section 4 for one event: profile, dimensions of all 36 windows, saturation fit; with the in-process timings."""
    t0 = time.perf_counter()
    profile = offset_averaged_profile(ix, iy, iz, energy, offsets)
    t1 = time.perf_counter()
    d_values = window_dimensions(profile)
    fit = fit_saturation(profile.mean_occupied_boxes, profile.n_occupied_voxels)
    t2 = time.perf_counter()
    return {
        "n_occupied_voxels": profile.n_occupied_voxels,
        "partial_fraction_lateral": profile.partial_fraction_lateral,
        "partial_fraction_with_depth": profile.partial_fraction_with_depth,
        "d_values": d_values,
        "mean_occupied_boxes": profile.mean_occupied_boxes,
        "fit_accepted": fit.accepted,
        "fit_converged": fit.converged,
        "fractal_dimension": fit.fractal_dimension,
        "ln_amplitude_mm": fit.ln_amplitude_mm,
        "saturation_side_mm": fit.saturation_side_mm,
        "seconds_profile": t1 - t0,
        "seconds_fit_and_windows": t2 - t1,
    }


@lru_cache(maxsize=2)
def _load_set(key: str):
    from ams_ecal.geant4_simulation.geant4_backend import load_batch

    event_set = SET_BY_KEY[key]
    batch = load_batch(event_set.directory)
    if event_set.particle == "proton":
        positions = np.flatnonzero(batch.arrays["truth_occurred"])
    else:
        positions = np.arange(len(batch))
    return batch, positions


def set_positions(key: str) -> np.ndarray:
    """Row positions in the batch of the events of one set (electrons: all; protons: ``truth_occurred``)."""
    return _load_set(key)[1]


def _event_voxels(
    batch, position: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat, mev = batch.fine_deposits(int(position))
    ix, iy, iz = m3.unravel(flat)
    return ix, iy, iz, np.asarray(mev, dtype=np.float64)


def _stack(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in records]) for k in records[0]}


def _profile_chunk(args: tuple[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Worker: section 4 for the events at the given batch positions of one set."""
    key, positions = args
    batch, _ = _load_set(key)
    offsets = grid_offsets()
    records = []
    for position in positions:
        ix, iy, iz, mev = _event_voxels(batch, position)
        record = process_event(ix, iy, iz, mev, offsets)
        record["event_index"] = int(batch.arrays["event_index"][position])
        record["deposit_total_mev"] = float(batch.arrays["edep_total_mev"][position])
        record["fine_deposit_sum_mev"] = float(mev.sum())
        records.append(record)
    return _stack(records)


def _concatenate(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}


def _chunks(positions: np.ndarray, size: int) -> list[tuple[int, ...]]:
    return [
        tuple(int(p) for p in positions[i : i + size])
        for i in range(0, len(positions), size)
    ]


def _map(func: Callable, tasks: list, workers: int) -> list:
    if workers <= 1:
        return [func(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(func, tasks))


def measure_set(
    key: str, workers: int, limit: int | None = None, chunk: int = 25
) -> dict[str, np.ndarray]:
    """Section 4 for every event of one set (or the first ``limit`` by position)."""
    positions = set_positions(key)
    if limit is not None:
        positions = positions[:limit]
    return _concatenate(
        _map(_profile_chunk, [(key, c) for c in _chunks(positions, chunk)], workers)
    )


def retained(
    table: dict[str, np.ndarray], rule: str = DEFAULT_PARTIAL_BOX_RULE
) -> np.ndarray:
    """Events kept by the exclusion rule of 4.2 (more than 1% of the energy in partial boxes at the coarsest scale: excluded)."""
    fraction = partial_fraction(
        table["partial_fraction_lateral"], table["partial_fraction_with_depth"], rule
    )
    return (table["n_occupied_voxels"] > 0) & (fraction <= PARTIAL_BOX_ENERGY_LIMIT)


def admissibility_fractions(
    tables: dict[str, dict[str, np.ndarray]],
    delta: int,
    rule: str = DEFAULT_PARTIAL_BOX_RULE,
) -> tuple[list[str], np.ndarray]:
    """``(set keys, (n_sets, n_windows))``: share of the retained events of each set for which each window is admissible."""
    keys = list(tables)
    out = np.full((len(keys), N_WINDOWS), np.nan)
    for i, key in enumerate(keys):
        t = tables[key]
        keep = retained(t, rule)
        if keep.any():
            ok = admissible_windows(
                t["saturation_side_mm"][keep], t["fit_accepted"][keep], delta
            )
            out[i] = ok.mean(axis=0)
    return keys, out


# --- section 5: planted local disruption --------------------------------------------------------------------------------------


def plant_disruption(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    energy: np.ndarray,
    variant: str,
    phi: float,
    block_side: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Disrupt a random fraction ``phi`` of the occupied blocks of one event (section 5).

    Blocks are ``block_side`` x ``block_side`` voxel squares inside one z-layer, grid-aligned before the offsets. Variant W permutes the
    energies among the occupied voxels of a disrupted block (occupancy and D_0 unchanged by construction); variant S moves each occupied
    voxel with its energy to a random voxel that was empty in the same block and layer (occupancy exactly preserved).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if m3.MESH[0] % block_side or m3.MESH[1] % block_side:
        raise ValueError("block side must divide the lateral mesh")
    ix, iy, iz = (np.array(a, dtype=np.int64) for a in (ix, iy, iz))
    energy = np.array(energy, dtype=np.float64)
    n_lateral = m3.MESH[0] // block_side
    block = (iz * n_lateral + iy // block_side) * n_lateral + ix // block_side
    _, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_blocks = len(counts)
    # REGISTRATION AMBIGUITY: 'a random fraction phi of occupied blocks' (rounding; at least one block)
    n_disrupted = int(min(n_blocks, max(1, np.floor(phi * n_blocks + 0.5))))
    chosen = rng.choice(n_blocks, size=n_disrupted, replace=False)
    short_blocks = 0
    for c in chosen:
        members = order[starts[c] : starts[c] + counts[c]]
        if variant == VARIANT_WEIGHTS:
            energy[members] = energy[members][rng.permutation(len(members))]
            continue
        x0 = (ix[members[0]] // block_side) * block_side
        y0 = (iy[members[0]] // block_side) * block_side
        local = (iy[members] - y0) * block_side + (ix[members] - x0)
        empty = np.setdiff1d(
            np.arange(block_side * block_side), local, assume_unique=True
        )
        n_move = min(len(members), len(empty))
        movers = members
        if n_move < len(
            members
        ):  # REGISTRATION AMBIGUITY: more occupied than empty voxels in the block
            short_blocks += 1
            movers = members[rng.choice(len(members), size=n_move, replace=False)]
        targets = rng.choice(empty, size=n_move, replace=False)
        ix[movers] = x0 + targets % block_side
        iy[movers] = y0 + targets // block_side
    return (
        ix,
        iy,
        iz,
        energy,
        {
            "occupied_blocks": n_blocks,
            "disrupted_blocks": n_disrupted,
            "blocks_with_fewer_empty_than_occupied_voxels": short_blocks,
        },
    )


def _disruption_chunk(args: tuple[str, tuple[int, ...], int, int]) -> np.ndarray:
    """Worker: ``(n_events, n_variants, n_phi, 3)`` D_0, D_1, D_2 on W* of the disrupted copies (computed once per event, phi, variant)."""
    key, positions, window_index, block_side = args
    batch, _ = _load_set(key)
    offsets = grid_offsets()
    window = m3.WINDOWS[window_index]
    set_index = [s.key for s in SETS].index(key)
    out = np.empty((len(positions), len(VARIANTS), len(PHI_GRID), 3))
    for e, position in enumerate(positions):
        ix, iy, iz, mev = _event_voxels(batch, position)
        event_index = int(batch.arrays["event_index"][position])
        for v, variant in enumerate(VARIANTS):
            for j, phi in enumerate(PHI_GRID):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [DISRUPTION_SEED, set_index, event_index, v, j]
                    )
                )
                dx, dy, dz, de, _ = plant_disruption(
                    ix, iy, iz, mev, variant, phi, block_side, rng
                )
                profile = offset_averaged_profile(dx, dy, dz, de, offsets)
                out[e, v, j] = window_dimensions(profile, [window])[:, 0]
    return out


# --- exact KS machinery (sections 6.C0, C2, C3) -----------------------------------------------------------------------------------


def ks_numerators(values: np.ndarray, labels: np.ndarray, n_first: int) -> np.ndarray:
    """Two-sample KS numerators ``max |c1 n0 - (pos - c1) n1|`` (integers) for every labelling; KS = numerator / (n1 n0).

    ``values`` is ``(n, k)``, ``labels`` is ``(rows, n)`` with exactly ``n_first`` ones per row (label 1 = first sample). Ties in the
    values are handled (the ECDF difference is only evaluated after the last of a run of equal values). Returns ``(rows, k)`` int64.
    """
    n = values.shape[0]
    n_second = n - n_first
    positions = np.arange(1, n + 1, dtype=np.int64)
    out = np.empty((labels.shape[0], values.shape[1]), dtype=np.int64)
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="stable")
        sorted_values = values[order, j]
        boundary = np.append(sorted_values[1:] != sorted_values[:-1], True)
        cumulative = np.cumsum(labels[:, order], axis=1, dtype=np.int32)[
            :, boundary
        ].astype(np.int64)
        out[:, j] = np.abs(
            cumulative * n_second - (positions[boundary] - cumulative) * n_first
        ).max(axis=1)
    return out


def _permutation_labels(
    n_first: int, n: int, rows: int, rng: np.random.Generator
) -> np.ndarray:
    base = np.r_[np.ones(n_first, dtype=np.int8), np.zeros(n - n_first, dtype=np.int8)]
    return base[np.argsort(rng.random((rows, n)), axis=1)]


def p_values_from_labels(
    values: np.ndarray, permuted_labels: np.ndarray, n_first: int
) -> tuple[np.ndarray, np.ndarray]:
    """KS of the first ``n_first`` rows against the rest and its permutation p = (1 + #{null >= observed}) / (1 + B), per column."""
    n = values.shape[0]
    observed_labels = np.r_[
        np.ones(n_first, dtype=np.int8), np.zeros(n - n_first, dtype=np.int8)
    ][None, :]
    num = ks_numerators(values, np.vstack([observed_labels, permuted_labels]), n_first)
    exceed = (num[1:] >= num[0]).sum(axis=0)
    return num[0] / (n_first * (n - n_first)), (1.0 + exceed) / (
        1.0 + len(permuted_labels)
    )


def permutation_test(
    first: np.ndarray, second: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """KS between two independent samples (columns = statistics) with a label-permutation p. Returns (KS, p) per column."""
    first, second = np.atleast_2d(first.T).T, np.atleast_2d(second.T).T
    values = np.vstack([first, second])
    labels = _permutation_labels(len(first), len(values), permutations, rng)
    return p_values_from_labels(values, labels, len(first))


def pair_swap_test(
    first: np.ndarray, second: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """KS between matched pairs ``(first[i], second[i])`` with the null from swapping the classes inside pairs (S3 stage 3)."""
    first, second = np.atleast_2d(first.T).T, np.atleast_2d(second.T).T
    n = len(first)
    values = np.vstack([first, second])
    observed = ks_numerators(
        values, np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)][None, :], n
    )[0]
    exceed = np.zeros(values.shape[1], dtype=np.int64)
    for start in range(0, permutations, SWAP_CHUNK):
        rows = min(SWAP_CHUNK, permutations - start)
        swap = (rng.random((rows, n)) < 0.5).astype(np.int8)
        num = ks_numerators(values, np.concatenate([1 - swap, swap], axis=1), n)
        exceed += (num >= observed).sum(axis=0)
    return observed / (n * n), (1.0 + exceed) / (1.0 + permutations)


# --- section 6: the checks ------------------------------------------------------------------------------------------------------


def statistic_matrix(d_values: np.ndarray, window_index: int) -> np.ndarray:
    """``(n_events, 2)``: D_2 (primary) and D_0 - D_2 (secondary) on one window from the ``(n, 3, n_windows)`` table."""
    d = d_values[:, :, window_index]
    return np.column_stack([d[:, 2], d[:, 0] - d[:, 2]])


def check_machinery(
    statistics: dict[str, np.ndarray],
    splits: int = C0_SPLITS,
    permutations: int = C0_PERMUTATIONS,
    seed: int = C0_SEED,
) -> dict[str, Any]:
    """C0: random same-class half-splits give a rejection rate of alpha for ANY valid permutation test; sanity only (6.C0)."""
    sigma = float(np.sqrt(ALPHA * (1 - ALPHA) / splits))
    low, high = ALPHA - C0_STANDARD_ERRORS * sigma, ALPHA + C0_STANDARD_ERRORS * sigma
    rng = np.random.default_rng(seed)
    rows = {}
    for key, values in statistics.items():
        half = len(values) // 2
        rejected = np.zeros(values.shape[1])
        for _ in range(splits):
            order = rng.permutation(len(values))
            pair = np.vstack([values[order[:half]], values[order[half : 2 * half]]])
            labels = _permutation_labels(half, 2 * half, permutations, rng)
            _, p = p_values_from_labels(pair, labels, half)
            rejected += p <= ALPHA
        rates = rejected / splits
        rows[key] = {
            "events": len(values),
            "half_size": half,
            "rejection_rate": dict(
                zip(STATISTIC_NAMES, (float(r) for r in rates), strict=True)
            ),
            "sane": bool(np.all((rates >= low) & (rates <= high))),
        }
    return {
        "splits": splits,
        "permutations": permutations,
        "accepted_range": [low, high],
        "per_set": rows,
        "sane": bool(all(r["sane"] for r in rows.values())),
    }


def check_ordering(
    d_values_by_set: dict[str, np.ndarray], window_index: int
) -> dict[str, Any]:
    """C1: fraction of events with D_0 >= D_1 >= D_2 on the window, at least ``MIN_ORDERING_RATE`` in every set."""
    fractions = {}
    for key, d in d_values_by_set.items():
        w = d[:, :, window_index]
        fractions[key] = float(np.mean((w[:, 0] >= w[:, 1]) & (w[:, 1] >= w[:, 2])))
    return {
        "fraction_by_set": fractions,
        "threshold": m3.MIN_ORDERING_RATE,
        "pass": bool(all(f >= m3.MIN_ORDERING_RATE for f in fractions.values())),
    }


def check_matched_null(
    statistics: dict[str, np.ndarray],
    keys: dict[str, np.ndarray],
    pairings: Sequence[tuple[str, str]] = C2_PAIRINGS,
    caliper: float = C2_CALIPER_SD,
    min_pairs: int = C2_MIN_PAIRS,
    permutations: int = C2_PERMUTATIONS,
    seed: int = C2_SEED,
) -> dict[str, Any]:
    """C2: same-class cross-energy matched null (6.C2). ``keys[set]`` is ``(n, 2)``: occupied voxels and prefix deposit (MeV)."""
    from ams_ecal.multiscale_estimators import (
        multiscale_matched_pairs as stage3,
    )  # lazy: heavy imports

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for low_key, high_key in pairings:
        ia, ib = stage3.match_pairs(keys[low_key], keys[high_key], caliper)
        row: dict[str, Any] = {
            "lower_energy_set": low_key,
            "higher_energy_set": high_key,
            "pairs": len(ia),
            "testable": bool(len(ia) >= min_pairs),
        }
        if len(ia):
            row["standardized_mean_difference_log_occupied_log_deposit"] = (
                stage3.balance(keys[low_key], keys[high_key], ia, ib)
            )
        if row["testable"]:
            ks, p = pair_swap_test(
                statistics[low_key][ia], statistics[high_key][ib], permutations, rng
            )
            row["ks"] = dict(zip(STATISTIC_NAMES, (float(v) for v in ks), strict=True))
            row["p_value"] = dict(
                zip(STATISTIC_NAMES, (float(v) for v in p), strict=True)
            )
        rows.append(row)
    testable = [r for r in rows if r["testable"]]
    raw = np.array([r["p_value"][name] for r in testable for name in STATISTIC_NAMES])
    holm = mp.holm(raw) if len(raw) else raw
    for i, r in enumerate(testable):
        r["holm_adjusted_p"] = dict(
            zip(
                STATISTIC_NAMES,
                (float(v) for v in holm[2 * i : 2 * i + 2]),
                strict=True,
            )
        )
    return {
        "pairings": rows,
        "testable_pairings": len(testable),
        "permutations": permutations,
        "caliper_sd": caliper,
        # REGISTRATION AMBIGUITY: no testable pairing is reported as not testable and is not a pass
        "pass": bool(testable and not (holm <= ALPHA).any()),
        "status": "not testable"
        if not testable
        else ("pass" if not (holm <= ALPHA).any() else "fail"),
    }


def arm_sizes(n_events: int) -> list[int]:
    """The realistic arm size and the largest the set allows (up to 300): two disjoint arms need ``2 n <= n_events``."""
    largest = min(C3_MAX_ARM, n_events // 2)
    return sorted({min(C3_REALISTIC_ARM, largest), largest})


def check_power(
    original: np.ndarray,
    disrupted: np.ndarray,
    arm: int,
    splits: int = C3_SPLITS,
    permutations: int = C3_PERMUTATIONS,
    seed: int = C3_SEED,
) -> np.ndarray:
    """C3: power of the KS permutation test between an original arm and a disrupted arm of DIFFERENT events.

    ``original`` is ``(N, 2)`` (D_2, D_0 - D_2 of the undisrupted events); ``disrupted`` is ``(N, n_variants, n_phi, 2)``. Each of the
    ``splits`` random splits draws two disjoint arms of ``arm`` events; the permutation labels are shared by the phi values of a split.
    Returns ``(n_variants, n_phi, 2)`` rejection fractions at alpha.
    """
    rng = np.random.default_rng(seed)
    rejected = np.zeros(disrupted.shape[1:])
    for _ in range(splits):
        order = rng.permutation(len(original))
        a, b = order[:arm], order[arm : 2 * arm]
        labels = _permutation_labels(arm, 2 * arm, permutations, rng)
        for v in range(disrupted.shape[1]):
            for j in range(disrupted.shape[2]):
                _, p = p_values_from_labels(
                    np.vstack([original[a], disrupted[b, v, j]]), labels, arm
                )
                rejected[v, j] += p <= ALPHA
    return rejected / splits


def decide_verdict(
    window_exists: bool,
    c0_sane: bool,
    c1_pass: bool,
    c2_pass: bool,
    c3_pass: bool,
    c2_testable: bool = True,
) -> str:
    """Section 7. C0 not sane stops the run (6.C0); then 'not usable' if no W* or C1 or C3 fails; 'usable' if C2 also passes."""
    if not c0_sane:
        return VERDICT_STOPPED
    if not (window_exists and c1_pass and c3_pass):
        return VERDICT_NOT_USABLE
    if not c2_testable:
        return VERDICT_C2_NOT_TESTABLE
    return VERDICT_USABLE if c2_pass else VERDICT_MATCHING


# --- section 8: synthetic cascades through the estimator of section 4 (secondary, descriptive) -------------------------------------


def _synthetic_task(args: tuple[str, int, int, int]) -> dict[str, Any]:
    kind, n_hits, replicates, seed = args
    from ams_ecal.multiscale_estimators import multiscale_3d_sensitivity as sens

    sampler = {
        "cascade": m3.sample_cascade,
        "B1": m3.sample_smooth,
        "B3": sens.separable_sampler(18.0, 9.0),
    }[kind]
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, n_hits, ["cascade", "B1", "B3"].index(kind)])
    )
    offsets = grid_offsets()
    profiles, fits = [], []
    for _ in range(replicates):
        ix, iy, iz, e = m3._event_from_hits(*sampler(n_hits, rng))
        profile = offset_averaged_profile(ix, iy, iz, e, offsets)
        profiles.append(profile)
        fits.append(
            fit_saturation(profile.mean_occupied_boxes, profile.n_occupied_voxels)
        )
    d = np.array([window_dimensions(p) for p in profiles])
    out: dict[str, Any] = {
        "kind": kind,
        "n_hits": n_hits,
        "replicates": replicates,
        "median_occupied_voxels": float(
            np.median([p.n_occupied_voxels for p in profiles])
        ),
        "fit_accepted_fraction": float(np.mean([f.accepted for f in fits])),
        "median_saturation_side_mm": float(
            np.median([f.saturation_side_mm for f in fits])
        ),
        "ordering_fraction_by_window": [
            float(np.mean((d[:, 0, k] >= d[:, 1, k]) & (d[:, 1, k] >= d[:, 2, k])))
            for k in range(N_WINDOWS)
        ],
    }
    if kind == "cascade":
        rows = []
        for fine, coarse, weights, isotropic in m3.ALIGNED_LEVELS:
            ratio = float(m3.EFFECTIVE_SIDE_MM[coarse] / m3.EFFECTIVE_SIDE_MM[fine])
            for q in ms.Q_VALUES:
                exact = m3.analytic_local_dimension(weights, q, ratio)
                est = np.array(
                    [
                        m3._two_point(p.log_measures()[q], q, fine, coarse)
                        for p in profiles
                    ]
                )
                rel = float((est.mean() - exact) / exact)
                rows.append(
                    {
                        "q": q,
                        "fine_scale": m3.SCALES[fine],
                        "coarse_scale": m3.SCALES[coarse],
                        "isotropic_level": isotropic,
                        "exact": exact,
                        "mean_estimate": float(est.mean()),
                        "relative_error": rel,
                        "within_reference": bool(
                            abs(rel) <= m3.RECOVERY_REFERENCE_TOLERANCE
                        ),
                    }
                )
        out["recovery"] = rows
    return out


def synthetic_recovery(
    hit_counts: Sequence[int] = (1000, 10_000, 30_000),
    replicates: int = 50,
    seed: int = SYNTHETIC_SEED,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Section 8 (optional, descriptive): the S4 cascade (recovery of the known dimensions, 15% line) and the B1/B3 baselines."""
    tasks = [
        (kind, n, replicates, seed)
        for kind in ("cascade", "B1", "B3")
        for n in hit_counts
    ]
    return _map(_synthetic_task, tasks, workers)


# --- reporting -----------------------------------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=1), encoding="utf-8")


def _window_label(k: int) -> str:
    w = m3.WINDOWS[k]
    return f"{m3.SCALES[w[0]]}->{m3.SCALES[w[-1]]}"


def _provenance() -> dict[str, Any]:
    digest = registration_hash()
    if not digest.startswith(REGISTRATION_HASH_PREFIX):
        raise SystemExit(
            f"registration hash {digest[:16]} does not match {REGISTRATION_HASH_PREFIX}; refusing to run"
        )
    return {
        "registration_file": str(REGISTRATION_FILE),
        "registration_sha256": digest,
        "registration_hash_prefix": REGISTRATION_HASH_PREFIX,
        "amendment_1_file": str(AMENDMENT_1_FILE),
        "amendment_1_hash_prefix": AMENDMENT_1_HASH_PREFIX,
        "registration_hash_prefixes": [
            REGISTRATION_HASH_PREFIX,
            AMENDMENT_1_HASH_PREFIX,
        ],
        "git": git_state(),
        "seeds": {
            "grid_offsets (registered)": OFFSET_SEED,
            "disruption": DISRUPTION_SEED,
            "c0_half_splits": C0_SEED,
            "c2_pair_swap": C2_SEED,
            "c3_power_splits": C3_SEED,
            "section_8_synthetic": SYNTHETIC_SEED,
        },
        "grid_offsets_ox_oy_oz": grid_offsets().tolist(),
        "ambiguities": list(AMBIGUITIES),
    }


def _admissibility_report(
    tables: dict[str, dict[str, np.ndarray]], rule: str
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for delta in (DELTA_PRIMARY, DELTA_SENSITIVITY):
        keys, fractions = admissibility_fractions(tables, delta, rule)
        k_star = select_window(fractions)
        report[f"delta_{delta}"] = {
            "windows": [_window_label(k) for k in range(N_WINDOWS)],
            "window_scale_indices": [list(w) for w in m3.WINDOWS],
            "fraction_by_set": {key: fractions[i] for i, key in enumerate(keys)},
            "w_star_index": k_star,
            "w_star": None if k_star is None else _window_label(k_star),
        }
    return report


def _ordering_by_window(d_values: np.ndarray) -> np.ndarray:
    return np.mean(
        (d_values[:, 0] >= d_values[:, 1]) & (d_values[:, 1] >= d_values[:, 2]), axis=0
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {
        f"p{q}": float(np.percentile(finite, q)) for q in (0, 5, 25, 50, 75, 95, 100)
    }


def run_rung0(workers: int = 1) -> dict[str, Any]:
    """Rung 0 (section 9): the first 50 electrons at 400 GeV (by batch position) through section 4 only."""
    t0 = time.perf_counter()
    table = measure_set("electron_400GeV", workers, limit=50, chunk=5)
    wall = time.perf_counter() - t0
    n = len(table["event_index"])
    seconds = table["seconds_profile"] + table["seconds_fit_and_windows"]
    failures = ~table["fit_accepted"]
    report: dict[str, Any] = {
        "registration": "EXP-016 version 2, 2026-11-09, rung 0",
        **_provenance(),
        "rung": 0,
        "events": n,
        "event_selection": "the first 50 events by batch position of E400GeV electrons (event_index "
        f"{int(table['event_index'][0])}..{int(table['event_index'][-1])})",
        "workers": workers,
        "wall_seconds_including_load": wall,
        "fit_failures": int(failures.sum()),
        "fit_failure_rate": float(failures.mean()),
        "fit_non_converged": int((~table["fit_converged"]).sum()),
        "fit_df_outside_range": int(
            (table["fit_converged"] & ~table["fit_accepted"]).sum()
        ),
        "seconds_per_event": {
            "mean": float(seconds.mean()),
            "median": float(np.median(seconds)),
            "max": float(seconds.max()),
            "profile_only_mean": float(table["seconds_profile"].mean()),
            "fit_and_windows_mean": float(table["seconds_fit_and_windows"].mean()),
        },
        "partial_box_energy_fraction": {
            "lateral_only": _quantiles(table["partial_fraction_lateral"]),
            "with_depth_literal": _quantiles(table["partial_fraction_with_depth"]),
            "excluded_lateral_only": int(
                (table["partial_fraction_lateral"] > PARTIAL_BOX_ENERGY_LIMIT).sum()
            ),
            "excluded_with_depth_literal": int(
                (table["partial_fraction_with_depth"] > PARTIAL_BOX_ENERGY_LIMIT).sum()
            ),
        },
        "max_relative_difference_edep_total_vs_fine_sum": float(
            np.max(
                np.abs(table["fine_deposit_sum_mev"] / table["deposit_total_mev"] - 1.0)
            )
        ),
        "occupied_voxels": _quantiles(table["n_occupied_voxels"].astype(float)),
        "saturation_side_mm_distribution": _quantiles(table["saturation_side_mm"]),
        "saturation_side_voxel_equivalent_distribution": _quantiles(
            table["saturation_side_mm"] / m3.EFFECTIVE_SIDE_MM[0]
        ),
        "fractal_dimension_distribution": _quantiles(table["fractal_dimension"]),
        "ordering_fraction_by_window": {
            _window_label(k): float(v)
            for k, v in enumerate(_ordering_by_window(table["d_values"]))
        },
        "admissibility": {},
    }
    for delta in (DELTA_PRIMARY, DELTA_SENSITIVITY):
        ok = admissible_windows(
            table["saturation_side_mm"], table["fit_accepted"], delta
        )
        report["admissibility"][f"delta_{delta}"] = {
            _window_label(k): float(v) for k, v in enumerate(ok.mean(axis=0))
        }
    stop_failures = report["fit_failure_rate"] > 0.10
    stop_time = report["seconds_per_event"]["max"] > 2.0
    report["stop_rule"] = {
        "fit_failures_above_10_percent": bool(stop_failures),
        "an_event_took_more_than_2_s": bool(stop_time),
        "mean_event_time_above_2_s": bool(report["seconds_per_event"]["mean"] > 2.0),
        "rung1_may_proceed_on_these_two_criteria": bool(
            not stop_failures and not stop_time
        ),
        "rung1_blocked_by_partial_box_rule_as_written": bool(
            report["partial_box_energy_fraction"]["excluded_with_depth_literal"] == n
        ),
    }
    report["per_event"] = {
        k: table[k]
        for k in (
            "event_index",
            "n_occupied_voxels",
            "saturation_side_mm",
            "fractal_dimension",
            "fit_accepted",
            "seconds_profile",
            "seconds_fit_and_windows",
        )
    }
    _write_json(RESULT_DIR / "exp016_rung0.json", report)
    return report


def run_rung1(
    workers: int = 6, rule: str = DEFAULT_PARTIAL_BOX_RULE, skip_synthetic: bool = False
) -> dict[str, Any]:
    """Rung 1 (section 9): everything in sections 4-8."""
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    report: dict[str, Any] = {
        "registration": "EXP-016 version 2, 2026-11-09, rung 1",
        **_provenance(),
        "rung": 1,
        "workers": workers,
        "partial_box_rule_applied": rule,
        "status": "running",
    }
    path = RESULT_DIR / "exp016_rung1.json"

    t = time.perf_counter()
    tables = {s.key: measure_set(s.key, workers) for s in SETS}
    timings["section_4_all_sets"] = time.perf_counter() - t
    report["timings_seconds"] = timings
    per_event_seconds = np.concatenate(
        [
            tb["seconds_profile"] + tb["seconds_fit_and_windows"]
            for tb in tables.values()
        ]
    )
    report["per_event_seconds_section_4"] = {
        "mean": float(per_event_seconds.mean()),
        "max": float(per_event_seconds.max()),
    }
    keep = {k: retained(tb, rule) for k, tb in tables.items()}
    report["events"] = {
        k: {
            "total": len(tb["event_index"]),
            "retained": int(keep[k].sum()),
            "excluded_partial_boxes": int((~keep[k]).sum()),
            "excluded_with_depth_literal": int(
                (tb["partial_fraction_with_depth"] > PARTIAL_BOX_ENERGY_LIMIT).sum()
            ),
            "excluded_lateral_only": int(
                (tb["partial_fraction_lateral"] > PARTIAL_BOX_ENERGY_LIMIT).sum()
            ),
            "fit_failures_among_retained": int((~tb["fit_accepted"][keep[k]]).sum()),
            "max_relative_difference_edep_total_vs_fine_sum": float(
                np.max(
                    np.abs(tb["fine_deposit_sum_mev"] / tb["deposit_total_mev"] - 1.0)
                )
            ),
        }
        for k, tb in tables.items()
    }
    _save_event_tables(tables)
    if any(v["retained"] < C2_MIN_PAIRS for v in report["events"].values()):
        report.update(
            status="blocked_by_partial_box_exclusion_rule",
            verdict=None,
            note="a set has fewer than 100 retained events under the applied partial-box rule; nothing further was run",
        )
        report["wall_seconds"] = time.perf_counter() - t_start
        _write_json(path, report)
        return report

    report["admissibility"] = _admissibility_report(tables, rule)
    k_star = report["admissibility"][f"delta_{DELTA_PRIMARY}"]["w_star_index"]
    window_exists = k_star is not None
    if not window_exists:
        _, fractions = admissibility_fractions(tables, DELTA_PRIMARY, rule)
        k_star = fallback_window(fractions)
    report["window"] = {
        "w_star_exists": window_exists,
        "window_index": k_star,
        "scale_indices": list(m3.WINDOWS[k_star]),
        "label": _window_label(k_star),
        "descriptive_only": not window_exists,
        "coarsest_box_side_voxels": m3.SCALES[m3.WINDOWS[k_star][-1]][0],
    }
    block_side = m3.SCALES[m3.WINDOWS[k_star][-1]][0]

    d_by_set = {k: tb["d_values"][keep[k]] for k, tb in tables.items()}
    statistics = {k: statistic_matrix(d, k_star) for k, d in d_by_set.items()}
    t = time.perf_counter()
    report["c0_machinery"] = check_machinery(statistics)
    timings["c0"] = time.perf_counter() - t
    report["c1_ordering"] = check_ordering(d_by_set, k_star)
    if not report["c0_machinery"]["sane"]:
        report.update(status="stopped_at_c0", verdict=VERDICT_STOPPED)
        report["wall_seconds"] = time.perf_counter() - t_start
        _write_json(path, report)
        return report

    t = time.perf_counter()
    c3: dict[str, Any] = {}
    disrupted: dict[str, np.ndarray] = {}
    for key in C3_SETS:
        positions = set_positions(key)
        parts = _map(
            _disruption_chunk,
            [(key, c, k_star, block_side) for c in _chunks(positions, 10)],
            workers,
        )
        disrupted[key] = np.concatenate(parts, axis=0)
    timings["disrupted_copies"] = time.perf_counter() - t
    t = time.perf_counter()
    for key in C3_SETS:
        d = disrupted[key][keep[key]]
        disrupted_stats = np.stack([d[..., 2], d[..., 0] - d[..., 2]], axis=-1)
        c3[key] = {}
        for arm in arm_sizes(int(keep[key].sum())):
            power = check_power(statistics[key], disrupted_stats, arm)
            c3[key][f"arm_{arm}"] = {
                variant: {
                    name: {
                        f"phi_{phi:g}": float(power[v, j, s])
                        for j, phi in enumerate(PHI_GRID)
                    }
                    for s, name in enumerate(STATISTIC_NAMES)
                }
                for v, variant in enumerate(VARIANTS)
            }
    timings["c3_power"] = time.perf_counter() - t
    PHI_GRID.index(PHI_STAR)
    c3_required = {}
    for key in C3_SETS:
        arm = min(C3_REALISTIC_ARM, int(keep[key].sum()) // 2)
        c3_required[key] = c3[key][f"arm_{arm}"][VARIANT_WEIGHTS]["D2"][
            f"phi_{PHI_STAR:g}"
        ]
    report["c3_power"] = {
        "splits": C3_SPLITS,
        "permutations": C3_PERMUTATIONS,
        "phi_star": PHI_STAR,
        "block_side_voxels": block_side,
        "power_by_set": c3,
        "criterion_power_D2_variant_W_at_phi_star_arm_150": c3_required,
        "pass": bool(all(v >= C3_REQUIRED_POWER for v in c3_required.values())),
    }

    t = time.perf_counter()
    keys = {
        k: np.column_stack([tb["n_occupied_voxels"], tb["deposit_total_mev"]])[keep[k]]
        for k, tb in tables.items()
    }
    report["c2_matched_null"] = check_matched_null(statistics, keys)
    timings["c2"] = time.perf_counter() - t

    report["verdict"] = decide_verdict(
        window_exists,
        report["c0_machinery"]["sane"],
        report["c1_ordering"]["pass"],
        report["c2_matched_null"]["pass"],
        report["c3_power"]["pass"],
        c2_testable=report["c2_matched_null"]["status"] != "not testable",
    )
    report["status"] = (
        "complete" if window_exists else "complete_descriptive_only_no_w_star"
    )
    if not skip_synthetic:
        t = time.perf_counter()
        report["section_8_synthetic"] = synthetic_recovery(workers=workers)
        timings["section_8"] = time.perf_counter() - t
    report["wall_seconds"] = time.perf_counter() - t_start
    _write_json(path, report)
    return report


def _save_event_tables(tables: dict[str, dict[str, np.ndarray]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    flat = {
        f"{key}__{name}": array
        for key, table in tables.items()
        for name, array in table.items()
    }
    np.savez_compressed(RESULT_DIR / "exp016_rung1_event_tables.npz", **flat)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rung", choices=("rung0", "rung1"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--partial-box-rule",
        choices=PARTIAL_BOX_RULES,
        default=DEFAULT_PARTIAL_BOX_RULE,
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args(argv)
    result = (
        run_rung0(args.workers)
        if args.rung == "rung0"
        else run_rung1(args.workers, args.partial_box_rule, args.skip_synthetic)
    )
    summary = {
        k: result[k]
        for k in ("rung", "status", "verdict", "stop_rule", "fit_failure_rate")
        if k in result
    }
    print(json.dumps(_jsonable(summary), indent=1))


if __name__ == "__main__":
    main()
