"""3D multiscale estimators on the fine voxels, with synthetic controls and the control-derived eligibility rule (plan item A4, DECIDE-4).

Registration: research/COMPLETED_WORK.md (was 2026-10-16_s4a_3d_estimator_controls_registration.md). Sparse energy deposits on the 216 x 216 x 36 mesh of
3 x 3 x 4.625 mm voxels (flat index (iz * 216 + iy) * 216 + ix); mesh-aligned boxes of (kx, kz) voxels; the same definitions as the 2D
estimator (``multiscale.py``): box probabilities are energy fractions, ``D_q`` is the least-squares slope of ln Z_q (the entropy at
q = 1) against the logarithm of the effective box side (3kx * 3kx * 4.625kz)^(1/3). The DEC-013 acceptance numbers are the project
assumptions of ``multiscale_controls.py``. These are finite-scale descriptors, not evidence of scaling.

    python -m ams_ecal.multiscale_estimators.multiscale_3d
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.validation import sample_definitions as sd

MESH = (216, 216, 36)  # nx, ny, nz
VOXEL_MM = (3.0, 3.0, 4.625)
SCALES = (
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3),
    (6, 4),
    (9, 6),
    (18, 12),
    (27, 18),
    (54, 36),
)  # (kx, kz) voxels, mesh-aligned
EFFECTIVE_SIDE_MM = np.array(
    [(3.0 * kx) ** (2 / 3) * (4.625 * kz) ** (1 / 3) for kx, kz in SCALES]
)
LOG_SIDE = np.log(EFFECTIVE_SIDE_MM)
STAT_NAMES = ("D0", "D1", "D2", "D0-D2")
HIT_COUNTS = (100, 300, 1000, 3000, 10_000, 30_000)
REPLICATES = 1000
RECOVERY_REPLICATES = 200
SEED = 20261016
WORKERS = 6
RESULT_PATH = PROJECT_ROOT / "results" / "multiscale_controls" / "s4a_3d_controls.json"

# DEC-013 project-assumption thresholds (multiscale_controls.py)
BASELINE_INTERVAL = (2.5, 97.5)
MAX_FALSE_POSITIVE_RATE = 0.10
MIN_DETECTION_POWER = 0.90
MIN_ORDERING_RATE = 0.90
RECOVERY_REFERENCE_TOLERANCE = 0.15


def windows() -> list[tuple[int, ...]]:
    """Every run of at least two consecutive scale indices (36 windows of the nine scales)."""
    n = len(SCALES)
    return [tuple(range(i, j)) for i in range(n) for j in range(i + 2, n + 1)]


WINDOWS = windows()


# --- the estimator ------------------------------------------------------------------------------------------------------------


def unravel(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, _ = MESH
    iz, rest = np.divmod(np.asarray(flat, dtype=np.int64), nx * ny)
    iy, ix = np.divmod(rest, nx)
    return ix, iy, iz


def box_shares(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, energy: np.ndarray, kx: int, kz: int
) -> np.ndarray:
    """Energy fractions of the occupied (kx, kx, kz)-voxel boxes (voxels may repeat; their energies add)."""
    nbx = MESH[0] // kx
    nby = MESH[1] // kx
    box = ((iz // kz) * nby + iy // kx) * nbx + ix // kx
    _, inverse = np.unique(box, return_inverse=True)
    sums = np.bincount(inverse, weights=energy)
    return sums[sums > 0] / sums[sums > 0].sum()


def profile(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, energy: np.ndarray
) -> dict[float, np.ndarray]:
    """ln Z_q (the entropy at q = 1) at each of the nine scales, for q = 0, 1, 2."""
    shares = [box_shares(ix, iy, iz, energy, kx, kz) for kx, kz in SCALES]
    return {
        q: np.array([ms.partition_value(p, q) for p in shares]) for q in ms.Q_VALUES
    }


def window_dimension(y: np.ndarray, q: float, window: tuple[int, ...]) -> float:
    idx = list(window)
    x = LOG_SIDE[idx]
    yw = y[idx]
    slope = float(
        ((x - x.mean()) * (yw - yw.mean())).sum() / ((x - x.mean()) ** 2).sum()
    )
    return -slope if q == 1 else slope / (q - 1)


def statistics(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, energy: np.ndarray
) -> np.ndarray:
    """``(4, n_windows)``: D0, D1, D2 and D0 - D2 of every window of one event."""
    prof = profile(ix, iy, iz, energy)
    out = np.empty((4, len(WINDOWS)))
    for k, w in enumerate(WINDOWS):
        d = {q: window_dimension(prof[q], q, w) for q in ms.Q_VALUES}
        out[:, k] = (d[0.0], d[1.0], d[2.0], d[0.0] - d[2.0])
    return out


def occupied_voxels(energy: np.ndarray) -> int:
    return int(np.count_nonzero(energy > 0))


# --- the synthetic controls ---------------------------------------------------------------------------------------------------


def _cdf(weights: np.ndarray) -> np.ndarray:
    c = np.cumsum(weights / weights.sum())
    c[-1] = 1.0
    return c


def _lateral() -> np.ndarray:
    c = np.arange(MESH[0], dtype=float)
    return 0.02 + np.exp(-0.5 * ((c - (MESH[0] - 1) / 2) / 9.0) ** 2)


def _depth() -> np.ndarray:
    r = np.arange(1, MESH[2] + 1, dtype=float)
    return r**2 * np.exp(-r / 6.0)


def sample_smooth(
    n_hits: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hits of the separable smooth map (gamma-like depth x Gaussian lateral core with a floor): equal quanta, per axis independent."""
    lat, dep = _cdf(_lateral()), _cdf(_depth())
    return (
        np.searchsorted(lat, rng.random(n_hits)),
        np.searchsorted(lat, rng.random(n_hits)),
        np.searchsorted(dep, rng.random(n_hits)),
    )


_W_ROOT = np.outer([0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1])  # 4 x 4 x 1
_W2 = np.einsum("i,j,k->ijk", [0.7, 0.3], [0.7, 0.3], [0.7, 0.3])  # 2 x 2 x 2
_W3 = np.einsum(
    "i,j,k->ijk", [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]
)  # 3 x 3 x 3
_W5 = np.einsum("i,j,k->ijk", [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.7, 0.3])  # 3 x 3 x 2
# (weights, child-grid shape, child size in voxels (x, y, z)) from the root down to the voxels
_LEVELS = (
    (_W_ROOT[:, :, None], (4, 4, 1), (54, 54, 36)),
    (_W2, (2, 2, 2), (27, 27, 18)),
    (_W3, (3, 3, 3), (9, 9, 6)),
    (_W3, (3, 3, 3), (3, 3, 2)),
    (_W5, (3, 3, 2), (1, 1, 1)),
)


def sample_cascade(
    n_hits: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hits of the deterministic multiplicative cascade, level by level from the 4 x 4 x 1 root to the voxels."""
    ix = np.zeros(n_hits, dtype=np.int64)
    iy = np.zeros(n_hits, dtype=np.int64)
    iz = np.zeros(n_hits, dtype=np.int64)
    for weights, shape, size in _LEVELS:
        child = np.searchsorted(_cdf(weights.ravel()), rng.random(n_hits))
        cx, cy, cz = np.unravel_index(child, shape)
        ix += cx * size[0]
        iy += cy * size[1]
        iz += cz * size[2]
    return ix, iy, iz


def _event_from_hits(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = (iz * MESH[1] + iy) * MESH[0] + ix
    unique, counts = np.unique(flat, return_counts=True)
    x, y, z = unravel(unique)
    return x, y, z, counts.astype(float)


def analytic_local_dimension(weights: np.ndarray, q: float, ratio: float) -> float:
    """Exact ``D_q`` across one cascade level whose child weights are ``weights`` (side ratio of the two scales ``ratio``)."""
    w = weights.ravel()
    if q == 0:
        return log(w.size) / log(ratio)
    if q == 1:
        return float(-np.sum(w * np.log(w))) / log(ratio)
    return -log(float(np.sum(w**q))) / ((q - 1) * log(ratio))


# (fine scale index, coarse scale index, level weights, isotropic) for the aligned levels of the cascade
ALIGNED_LEVELS = (
    (7, 8, _W2, True),  # (27,18) -> (54,36)
    (5, 7, _W3, True),  # (9,6) -> (27,18)
    (2, 5, _W3, True),  # (3,2) -> (9,6)
    (
        0,
        2,
        _W5,
        False,
    ),  # (1,1) -> (3,2): anisotropic, reported but not part of the rule
)


@dataclass
class ControlResult:
    n_hits: int
    window: tuple[int, ...]
    statistic: str
    baseline_low: float
    baseline_high: float
    false_positive_rate: float
    power: float | None
    ordering_rate: float | None
    median_occupied_voxels: float
    passes_false_positive_limit: bool
    passes_power_and_ordering: bool


def _collect(
    sampler, n_hits: int, rng: np.random.Generator, replicates: int
) -> tuple[np.ndarray, np.ndarray]:
    stats = np.empty((replicates, 4, len(WINDOWS)))
    occupied = np.empty(replicates)
    for r in range(replicates):
        x, y, z, e = _event_from_hits(*sampler(n_hits, rng))
        stats[r] = statistics(x, y, z, e)
        occupied[r] = len(e)
    return stats, occupied


def evaluate(
    n_hits: int, seed: int = SEED, replicates: int = REPLICATES
) -> list[ControlResult]:
    """False-positive rate, detection power and ordering for one hit count, every window and statistic (the 2D logic of
    ``multiscale_controls.evaluate``)."""
    rngs = [
        np.random.default_rng(np.random.SeedSequence([seed, n_hits, k]))
        for k in range(3)
    ]
    base, base_occ = _collect(sample_smooth, n_hits, rngs[0], replicates)
    fresh, _ = _collect(sample_smooth, n_hits, rngs[1], replicates)
    pos, _ = _collect(sample_cascade, n_hits, rngs[2], replicates)
    results = []
    for s, name in enumerate(STAT_NAMES):
        for k, w in enumerate(WINDOWS):
            low, high = np.percentile(base[:, s, k], BASELINE_INTERVAL)
            fpr = float(np.mean((fresh[:, s, k] < low) | (fresh[:, s, k] > high)))
            power = ordering = None
            if name == "D0-D2":
                power = float(np.mean(pos[:, 3, k] > high))
                ordering = float(
                    np.mean(
                        (pos[:, 0, k] >= pos[:, 1, k]) & (pos[:, 1, k] >= pos[:, 2, k])
                    )
                )
            results.append(
                ControlResult(
                    n_hits=n_hits,
                    window=w,
                    statistic=name,
                    baseline_low=float(low),
                    baseline_high=float(high),
                    false_positive_rate=fpr,
                    power=power,
                    ordering_rate=ordering,
                    median_occupied_voxels=float(np.median(base_occ)),
                    passes_false_positive_limit=bool(fpr <= MAX_FALSE_POSITIVE_RATE),
                    passes_power_and_ordering=bool(
                        power is not None
                        and power >= MIN_DETECTION_POWER
                        and ordering is not None
                        and ordering >= MIN_ORDERING_RATE
                    ),
                )
            )
    return results


def _two_point(y: np.ndarray, q: float, fine: int, coarse: int) -> float:
    """``D_q`` from the two scales ``fine`` and ``coarse`` only (the scales between them are skipped)."""
    slope = (y[coarse] - y[fine]) / (LOG_SIDE[coarse] - LOG_SIDE[fine])
    return -slope if q == 1 else slope / (q - 1)


def recovery_table(
    n_hits: int, seed: int = SEED, replicates: int = RECOVERY_REPLICATES
) -> list[dict[str, Any]]:
    """Estimate of ``D_q`` across each aligned cascade level against the exact value (reference tolerance 15%)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n_hits, 9]))
    profiles = [
        profile(*_event_from_hits(*sample_cascade(n_hits, rng)))
        for _ in range(replicates)
    ]
    rows = []
    for fine, coarse, weights, isotropic in ALIGNED_LEVELS:
        ratio = float(EFFECTIVE_SIDE_MM[coarse] / EFFECTIVE_SIDE_MM[fine])
        for q in ms.Q_VALUES:
            exact = analytic_local_dimension(weights, q, ratio)
            est = np.array([_two_point(p[q], q, fine, coarse) for p in profiles])
            rel = float((est.mean() - exact) / exact)
            rows.append(
                {
                    "n_hits": n_hits,
                    "q": q,
                    "fine_scale": SCALES[fine],
                    "coarse_scale": SCALES[coarse],
                    "isotropic_level": isotropic,
                    "exact": exact,
                    "mean_estimate": float(est.mean()),
                    "sd": float(est.std()),
                    "relative_error": rel,
                    "within_reference": bool(abs(rel) <= RECOVERY_REFERENCE_TOLERANCE),
                }
            )
    return rows


def _evaluate_task(args: tuple[int, int, int]) -> dict[str, Any]:
    n_hits, seed, replicates = args
    return {
        "n_hits": n_hits,
        "results": [r.__dict__ for r in evaluate(n_hits, seed, replicates)],
        "recovery": recovery_table(n_hits, seed),
    }


# --- eligibility from the controls and the real events -------------------------------------------------------------------------


def usable_windows(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Per window of the spread statistic D0 - D2: the hit counts at which it is usable and its eligibility threshold (median occupied
    voxels of the control images at the lowest usable hit count)."""
    out: dict[str, Any] = {}
    for w in WINDOWS:
        usable_at = []
        for ev in evaluations:
            for r in ev["results"]:
                if (
                    r["statistic"] == "D0-D2"
                    and tuple(r["window"]) == w
                    and r["passes_false_positive_limit"]
                    and r["passes_power_and_ordering"]
                ):
                    usable_at.append((ev["n_hits"], r["median_occupied_voxels"]))
        out[str(w)] = {
            "scales_mm": [round(float(EFFECTIVE_SIDE_MM[i]), 1) for i in (w[0], w[-1])],
            "usable_at_hits": [n for n, _ in usable_at],
            "eligibility_threshold_occupied_voxels": min(usable_at)[1]
            if usable_at
            else None,
        }
    return out


def occupied_counts(sample: str, energy: float, population: str) -> np.ndarray:
    """Number of occupied fine voxels of every event of one energy of a registered sample."""
    path = sd.locate(sample) / f"E{energy:g}GeV" / "events.npz"
    with np.load(path) as data:
        offsets, edep = data["fine_offsets"], data["fine_edep_mev"]
        keep = sd.population_mask(
            sd.SAMPLES[sample], data["truth_occurred"], population
        )
    positive = np.concatenate([[0], np.cumsum(edep > 0)])
    return (positive[offsets[1:]] - positive[offsets[:-1]])[keep]


def eligibility_on_events(
    thresholds: dict[str, Any], energies=(10.0, 20.0, 50.0, 100.0)
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sample, population in (
        ("electron_development_large", "all"),
        ("proton_development_large", "interacting"),
    ):
        for energy in energies:
            counts = occupied_counts(sample, energy, population)
            result[f"{sample}_{energy:g}"] = {
                "events": len(counts),
                "occupied_voxels_median": float(np.median(counts)),
                "occupied_voxels_5th_percentile": float(np.percentile(counts, 5)),
                "eligible_fraction_by_window": {
                    w: float(
                        np.mean(counts >= t["eligibility_threshold_occupied_voxels"])
                    )
                    for w, t in thresholds.items()
                    if t["eligibility_threshold_occupied_voxels"] is not None
                },
            }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--hit-counts", type=int, nargs="+", default=list(HIT_COUNTS))
    args = parser.parse_args(argv)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool:
        evaluations = list(
            pool.map(
                _evaluate_task, [(n, SEED, args.replicates) for n in args.hit_counts]
            )
        )
    print(f"controls done {time.time() - t0:.0f}s", flush=True)
    thresholds = usable_windows(evaluations)
    usable = {w: t for w, t in thresholds.items() if t["usable_at_hits"]}
    report = {
        "registration": "research/COMPLETED_WORK.md (was 2026-10-16_s4a_3d_estimator_controls_registration.md)",
        "status": "methods development; synthetic controls and exposed development events; no sealed set",
        "scales": [
            {"voxels": s, "effective_side_mm": float(EFFECTIVE_SIDE_MM[i])}
            for i, s in enumerate(SCALES)
        ],
        "replicates": args.replicates,
        "hit_counts": args.hit_counts,
        "usable_windows": usable,
        "n_usable_windows": len(usable),
        "n_windows": len(WINDOWS),
        "recovery": [row for ev in evaluations for row in ev["recovery"]],
        "control_results": [r for ev in evaluations for r in ev["results"]],
        "eligibility_on_real_events": eligibility_on_events(thresholds),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"usable_windows": len(usable), "of": len(WINDOWS)}, indent=1))


if __name__ == "__main__":
    main()
