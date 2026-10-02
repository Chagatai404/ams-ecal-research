"""Synthetic controls and the pre-registered acceptance rules for the estimator-validity gate (EXP-008).

Everything here follows ``research/plans/2026-10-02_multiscale_estimator_validity_preregistration.md``
(acceptance rules confirmed by the researcher on 2026-10-02, record DEC-013). The numbers 95 percent, 10 percent,
90 percent and 15 percent are PROJECT ASSUMPTIONS, not literature values.

The smooth baseline is an ILLUSTRATIVE smooth map (a gamma-like depth profile times a
Gaussian lateral core). It is not a calibrated FastMC shower and says nothing about
showers; it only supplies a structure-free comparison with the same number of hits.

The cascade is a deterministic hierarchy on 18 x 72 with exact aligned-scale answers:
a 1 x 4 split into 18 x 18 blocks, then 2 x 2 (9-boxes), then 3 x 3 (3-boxes), then
3 x 3 (cells). Its Z_q at the aligned sides 18, 9, 3, 1 is known in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from ams_ecal import multiscale as ms

SHAPE = (18, 72)
ALIGNED_SIDES = (1, 3, 9, 18)
HIT_COUNTS = (30, 100, 300, 1000, 3000)
REPLICATES = 1000

# Project-assumption thresholds (DEC-013).
BASELINE_INTERVAL = (2.5, 97.5)
MAX_FALSE_POSITIVE_RATE = 0.10
MIN_DETECTION_POWER = 0.90
MIN_ORDERING_RATE = 0.90  # INTERPRETATION of pre-registration ordering part, added 2026-10-02 after the first run; awaiting the researcher
RECOVERY_REFERENCE_TOLERANCE = 0.15

_W4 = np.array([0.4, 0.3, 0.2, 0.1])
_W2 = np.outer([0.7, 0.3], [0.7, 0.3])
_W3 = np.outer([0.5, 0.3, 0.2], [0.5, 0.3, 0.2])


def smooth_map(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """Illustrative smooth probability map (depth profile x lateral core), sums to 1."""
    h, w = shape
    r = np.arange(1, h + 1, dtype=float)
    depth = r ** 2 * np.exp(-r / 3.0)
    c = np.arange(w, dtype=float)
    lateral = 0.02 + np.exp(-0.5 * ((c - (w - 1) / 2) / 3.0) ** 2)
    p = np.outer(depth / depth.sum(), lateral / lateral.sum())
    return p / p.sum()


def cascade_map() -> np.ndarray:
    """Deterministic hierarchical cascade on 18 x 72 (exact aligned-scale Z_q)."""
    i, j = np.indices(SHAPE)
    p = (_W4[j // 18]
         * _W2[(i % 18) // 9, (j % 18) // 9]
         * _W3[(i % 9) // 3, (j % 9) // 3]
         * _W3[i % 3, j % 3])
    return p / p.sum()


def sample_hits(prob: np.ndarray, n_hits: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial hits (equal quanta) over the cells of a probability map."""
    return rng.multinomial(n_hits, prob.ravel()).reshape(prob.shape).astype(float)


def hierarchy_local_dimension(level: np.ndarray, ratio: int, q: float) -> float:
    """Exact ``D_q`` across one hierarchy level of the cascade (fine side * ratio = coarse side)."""
    w = level.ravel()
    if q == 0:
        return log(w.size) / log(ratio)
    if q == 1:
        return float(-np.sum(w * np.log(w))) / log(ratio)
    return -log(float(np.sum(w ** q))) / ((q - 1) * log(ratio))


def analytic_local_dimensions(q: float) -> dict[tuple[int, int], float]:
    """Exact ``D_q`` for each aligned adjacent pair of the cascade, keyed (fine, coarse)."""
    return {(9, 18): hierarchy_local_dimension(_W2, 2, q),
            (3, 9): hierarchy_local_dimension(_W3, 3, q),
            (1, 3): hierarchy_local_dimension(_W3, 3, q)}


STAT_NAMES = ("D0", "D1", "D2", "D0-D2")
_QLABEL = {"D0": 0.0, "D1": 1.0, "D2": 2.0, "D0-D2": float("nan")}


def image_estimates(image: np.ndarray) -> dict[str, dict[tuple[int, ...], float | None]]:
    """All statistics for one image, from ONE profile: fitted D0, D1, D2 and D0 - D2 per window.

    A window that is saturated for this image (saturation-window rule, ``x > 0``) gives ``None`` for every statistic.
    """
    prof = ms.profile(image)
    ne = ms.n_eff(image)
    out: dict[str, dict[tuple[int, ...], float | None]] = {name: {} for name in STAT_NAMES}
    for w in ms.contiguous_windows():
        if not ms.is_reliable(image.shape, w, ne):
            for name in STAT_NAMES:
                out[name][w] = None
            continue
        d = {q: ms.window_dimension(prof[q], ms.AMS_BOX_SIDES, q, w) for q in ms.Q_VALUES}
        out["D0"][w], out["D1"][w], out["D2"][w] = d[0.0], d[1.0], d[2.0]
        out["D0-D2"][w] = d[0.0] - d[2.0]
    return out


def window_estimates(image: np.ndarray, q: float) -> dict[tuple[int, ...], float | None]:
    """Fitted ``D_q`` for every window of AMS box sides; ``None`` where the window is saturated (saturation-window rule)."""
    return image_estimates(image)[f"D{int(q)}"]


def spread_estimates(image: np.ndarray) -> dict[tuple[int, ...], float | None]:
    """``D_0 - D_2`` per window (the multifractal-spread statistic used for detection power); ``None`` if saturated."""
    return image_estimates(image)["D0-D2"]


@dataclass
class ControlResult:
    n_hits: int
    window: tuple[int, ...]
    q: float
    reliable_fraction: float
    baseline_low: float
    baseline_high: float
    false_positive_rate: float
    power: float | None
    ordering_rate: float | None
    passes_false_positive_limit: bool
    passes_power_and_ordering: bool


def _collect(prob: np.ndarray, n_hits: int, rng: np.random.Generator, replicates: int) -> dict:
    """One image per replicate; every statistic comes from that same image, so lists are paired by index."""
    out = {name: {w: [] for w in ms.contiguous_windows()} for name in STAT_NAMES}
    for _ in range(replicates):
        est = image_estimates(sample_hits(prob, n_hits, rng))
        for name in STAT_NAMES:
            for w, v in est[name].items():
                out[name][w].append(v)
    return out


def evaluate(n_hits: int, seed: int, replicates: int = REPLICATES) -> list[ControlResult]:
    """False-positive rate, detection power and ordering for one hit count over every window, for D0, D1, D2 and the D0 - D2 spread.

    False-positive limit: fresh smooth replicates falling outside the baseline's central 95 percent interval
    (independent seeds for the baseline and the fresh set) must be <= 10 percent.
    Detection power: the cascade's ``D_0 - D_2`` must exceed the baseline's 97.5th percentile in >= 90 percent
    of replicates, AND the ordering D_0 >= D_1 >= D_2 must hold in >= 90 percent of cascade replicates
    (the pre-registered detection-power and ordering parts; the ordering fraction is an interpretation, see ``MIN_ORDERING_RATE``).
    Saturated windows (saturation-window rule) are dropped.
    """
    rng_base, rng_fresh, rng_pos = (np.random.default_rng(np.random.SeedSequence([seed, n_hits, k])) for k in range(3))
    base = _collect(smooth_map(), n_hits, rng_base, replicates)
    fresh = _collect(smooth_map(), n_hits, rng_fresh, replicates)
    pos = _collect(cascade_map(), n_hits, rng_pos, replicates)
    results = []
    for name in STAT_NAMES:
        for w in ms.contiguous_windows():
            b = np.array([v for v in base[name][w] if v is not None])
            if len(b) < 0.95 * replicates:  # window not reliable in >= 95 percent of baseline replicates
                continue
            lo, hi = np.percentile(b, BASELINE_INTERVAL)
            f = np.array([v for v in fresh[name][w] if v is not None])
            fpr = float(np.mean((f < lo) | (f > hi))) if len(f) else float("nan")
            power = ordering = None
            if name == "D0-D2":
                p = np.array([v for v in pos[name][w] if v is not None])
                power = float(np.mean(p > hi)) if len(p) else None
                trio = [(a, b_, c) for a, b_, c in zip(pos["D0"][w], pos["D1"][w], pos["D2"][w])
                        if a is not None and b_ is not None and c is not None]
                ordering = float(np.mean([a >= b_ >= c for a, b_, c in trio])) if trio else None
            results.append(ControlResult(
                n_hits=n_hits, window=w, q=_QLABEL[name], reliable_fraction=len(b) / replicates,
                baseline_low=float(lo), baseline_high=float(hi), false_positive_rate=fpr,
                power=power, ordering_rate=ordering, passes_false_positive_limit=bool(fpr <= MAX_FALSE_POSITIVE_RATE),
                passes_power_and_ordering=bool(power is not None and power >= MIN_DETECTION_POWER
                        and ordering is not None and ordering >= MIN_ORDERING_RATE)))
    return results


def recovery_table(n_hits: int, seed: int, replicates: int = 200) -> list[dict]:
    """Recovery reference: ``D_q`` on the aligned scales against the exact cascade values, per pair and hit count."""
    cascade = cascade_map()
    rng = np.random.default_rng(np.random.SeedSequence([seed, n_hits, 9]))
    images = [sample_hits(cascade, n_hits, rng) for _ in range(replicates)]
    rows = []
    for q in ms.Q_VALUES:
        for pair, d_exact in analytic_local_dimensions(q).items():
            est = np.array([ms.local_dimension(im, q, *pair) for im in images])
            rel = float((est.mean() - d_exact) / d_exact)
            rows.append({"n_hits": n_hits, "q": q, "fine": pair[0], "coarse": pair[1], "exact": d_exact,
                         "mean_estimate": float(est.mean()), "sd": float(est.std()), "relative_error": rel,
                         "within_reference": bool(abs(rel) <= RECOVERY_REFERENCE_TOLERANCE),
                         "saturated_fraction": float(np.mean(
                             [not ms.is_reliable(SHAPE, pair, ms.n_eff(im)) for im in images]))})
    return rows


def verdict(results: list[ControlResult]) -> dict[str, str]:
    """Verdict, per hit count (spread statistic): USABLE only if the false-positive limit, detection power and ordering all hold on a reliable window (saturation-window rule)."""
    out: dict[str, str] = {}
    for n in sorted({r.n_hits for r in results}):
        sel = [r for r in results if r.n_hits == n and np.isnan(r.q)]
        good = [r for r in sel if r.passes_false_positive_limit and r.passes_power_and_ordering]
        out[str(n)] = (f"USABLE on {len(good)} of {len(sel)} reliable windows" if good
                       else ("NOT USABLE (no reliable window passes the false-positive limit, detection power and ordering)" if sel
                             else "NOT USABLE (no window is reliable: all saturated)"))
    return out


def window_reliability(seed: int, replicates: int = 200) -> dict[str, dict[str, float]]:
    """Fraction of smooth-baseline images in which each window is interpretable (x <= 0), per hit count."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    smooth = smooth_map()
    table: dict[str, dict[str, float]] = {}
    for n in HIT_COUNTS:
        images = [sample_hits(smooth, n, rng) for _ in range(replicates)]
        for w in ms.contiguous_windows():
            table.setdefault(str(w), {})[str(n)] = float(
                np.mean([ms.is_reliable(im.shape, w, ms.n_eff(im)) for im in images]))
    return table


def run_controls(out_dir, seed: int = 20261002, replicates: int = REPLICATES) -> dict:
    """Run every control, write ``controls.json``, ``recovery.json``, ``windows.json`` and ``controls.png``.

    Reproduces the 2026-10-02 result record
    ``research/plans/2026-10-02_multiscale_estimator_controls_results.md`` (same seed and replicate count).
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[ControlResult] = []
    for n in HIT_COUNTS:
        results += evaluate(n, seed, replicates)
    verdicts = verdict(results)
    rows = [{"n_hits": r.n_hits, "window": list(r.window), "stat": (f"D{int(r.q)}" if r.q == r.q else "D0-D2"),
             "reliable_fraction": r.reliable_fraction, "baseline": [r.baseline_low, r.baseline_high],
             "false_positive_rate": r.false_positive_rate, "power": r.power, "ordering_rate": r.ordering_rate,
             "false_positive_limit_met": r.passes_false_positive_limit,
             "power_and_ordering_met": r.passes_power_and_ordering} for r in results]
    recovery = [row for n in HIT_COUNTS for row in recovery_table(n, seed)]
    windows = window_reliability(seed)
    (out / "controls.json").write_text(json.dumps(
        {"seed": seed, "replicates": replicates, "verdicts": verdicts, "rows": rows}, indent=1), encoding="utf-8")
    (out / "recovery.json").write_text(json.dumps(recovery, indent=1), encoding="utf-8")
    (out / "windows.json").write_text(json.dumps(windows, indent=1), encoding="utf-8")
    _plot(out / "controls.png", windows, results)
    return {"verdicts": verdicts, "rows": rows, "recovery": recovery, "windows": windows}


def _plot(path, windows: dict, results: list[ControlResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(windows)
    grid = np.array([[windows[w][str(n)] for n in HIT_COUNTS] for w in names])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    image = axs[0].imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    axs[0].set_xticks(range(len(HIT_COUNTS)))
    axs[0].set_xticklabels(HIT_COUNTS)
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names, fontsize=7)
    axs[0].set_xlabel("number of hits")
    axs[0].set_title("Interpretable windows: fraction of images with x <= 0", fontsize=9)
    fig.colorbar(image, ax=axs[0])
    spread = [r for r in results if np.isnan(r.q)]
    for i, r in enumerate(spread):
        axs[1].scatter(HIT_COUNTS.index(r.n_hits) + 0.08 * (i % 5 - 2), r.power, color="#1f4e79", s=14)
    axs[1].axhline(MIN_DETECTION_POWER, color="#c0504d", ls="--", lw=1, label="detection-power threshold")
    axs[1].set_xticks(range(len(HIT_COUNTS)))
    axs[1].set_xticklabels(HIT_COUNTS)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_xlabel("number of hits")
    axs[1].set_ylabel("detection power: cascade D0 - D2 above baseline 97.5th percentile")
    axs[1].set_title("Detection power per interpretable window", fontsize=9)
    axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/multiscale_controls", help="output directory")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args(argv)
    result = run_controls(args.out, args.seed, args.replicates)
    print("Verdict per hit count (D0 - D2 spread statistic):")
    for n, text in result["verdicts"].items():
        print(f"  {n:>5s} hits: {text}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
