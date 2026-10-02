"""Multiscale estimators for non-negative deposit images (estimator-validity gate, EXP-008).

WORDING. These are finite-scale DESCRIPTORS of a pixelised deposit pattern: occupied-box
counts and partition sums over square boxes of side ``s`` cells. A slope computed here is
a property of the pattern AND the scale window AND the number of hits; it is not, by
itself, evidence of fractal or cascade structure. Whether any estimator is usable on the
18 x 72 AMS-like grid is decided by the pre-registered controls
(``research/plans/2026-10-02_multiscale_estimator_validity_preregistration.md``), not by this module.

Definitions (Hentschel and Procaccia, Physica D 8, 435, 1983; secondary confirmation only):
``p_i`` is the fraction of the total energy in box ``i``, ``Z_q(s) = sum_i p_i**q`` over
boxes with ``p_i > 0``, and ``D_q`` is the log-log slope of ``Z_q`` against box side,
``Z_q ~ s**((q - 1) * D_q)``. ``q = 0`` is the occupied-box count, ``q = 1`` uses the
entropy ``S = -sum p ln p`` (``S ~ -D_1 ln s``), ``q = 2`` is concentration.

Scale-window reliability follows Koo and Ju, arXiv:2605.27925 (full text read, pp. 1-4,
a TRANSFERRED approximation from point-sampled trajectories): a window is interpretable
only if ``x = k_mid - k* <= 0``, with ``k* = log2(N_eff / A) / D_f``. On an anisotropic grid
the scale index is taken as ``k(s) = 0.5 * log2(number of available boxes)``, which equals
``log2(L / s)`` on a square grid; ``A = 1`` and ``D_f = 2`` are PROJECT ASSUMPTIONS.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log, log2

import numpy as np

AMS_BOX_SIDES = (1, 2, 3, 6, 9, 18)
"""Square-box sides that divide an 18 x 72 image evenly."""

Q_VALUES = (0.0, 1.0, 2.0)
"""Orders used in the first check (pre-registration section 1)."""

SATURATION_A = 1.0
SATURATION_DF = 2.0


def _checked(image) -> np.ndarray:
    a = np.asarray(image, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {a.shape}")
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("image must be finite and non-negative")
    if a.sum() <= 0:
        raise ValueError("image has no energy")
    return a


def box_shares(image, side: int) -> np.ndarray:
    """Energy fraction of every ``side x side`` box (zeros included), flattened."""
    a = _checked(image)
    h, w = a.shape
    if side < 1 or h % side or w % side:
        raise ValueError(f"box side {side} does not divide shape {a.shape}")
    boxes = a.reshape(h // side, side, w // side, side).sum(axis=(1, 3))
    return (boxes / boxes.sum()).ravel()


def occupied_boxes(image, side: int) -> int:
    return int(np.count_nonzero(box_shares(image, side)))


def partition_value(p: np.ndarray, q: float) -> float:
    """``ln Z_q`` for q != 1 (``ln`` of the occupied count at q = 0); entropy ``S`` at q = 1."""
    p = p[p > 0]
    if q == 1:
        return float(-np.sum(p * np.log(p)))
    if q == 0:
        return float(log(len(p)))
    return float(log(np.sum(p ** q)))


def log_measure(image, sides: Sequence[int], q: float) -> np.ndarray:
    """``ln Z_q`` (or the entropy at q = 1) at each box side."""
    return np.array([partition_value(box_shares(image, s), q) for s in sides])


def _dimension_from_slope(slope: float, q: float) -> float:
    # y = ln Z_q against x = ln s has slope (q - 1) D; at q = 1, y = S has slope -D.
    return -slope if q == 1 else slope / (q - 1)


def local_dimension(image, q: float, fine: int, coarse: int) -> float:
    """``D_q`` from two adjacent scales (fine side < coarse side)."""
    if not fine < coarse:
        raise ValueError("fine side must be smaller than coarse side")
    y = log_measure(image, (fine, coarse), q)
    return _dimension_from_slope((y[1] - y[0]) / (log(coarse) - log(fine)), q)


def fitted_dimension(image, q: float, sides: Sequence[int]) -> float:
    """Least-squares ``D_q`` over a window of box sides (at least two)."""
    sides = tuple(sides)
    if len(sides) < 2:
        raise ValueError("a fit needs at least two scales")
    x = np.log(np.asarray(sides, dtype=float))
    y = log_measure(image, sides, q)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return _dimension_from_slope(slope, q)


def profile(image, sides: Sequence[int] = AMS_BOX_SIDES,
            q_values: Sequence[float] = Q_VALUES) -> dict[float, np.ndarray]:
    """``log_measure`` for every q, computed once per image (shares are the slow part)."""
    shares = [box_shares(image, s) for s in sides]
    return {q: np.array([partition_value(p, q) for p in shares]) for q in q_values}


def window_dimension(y: np.ndarray, sides: Sequence[int], q: float, window: Sequence[int]) -> float:
    """``D_q`` over ``window`` from a precomputed ``log_measure`` array ``y`` over ``sides``."""
    idx = [list(sides).index(s) for s in window]
    x = np.log(np.asarray(window, dtype=float))
    yw = y[idx]
    slope = float(((x - x.mean()) * (yw - yw.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return _dimension_from_slope(slope, q)


def n_eff(image) -> int:
    """Number of distinct occupied cells at the finest scale (the saturation count)."""
    return occupied_boxes(image, 1)


def _scale_index(shape: tuple[int, int], side: int) -> float:
    h, w = shape
    return 0.5 * log2((h // side) * (w // side))


def saturation_x(shape: tuple[int, int], window: Sequence[int], n_effective: int,
                 a: float = SATURATION_A, d_f: float = SATURATION_DF) -> float:
    """``x = k_mid - k*``; the window is interpretable only for ``x <= 0``."""
    if n_effective < 1:
        raise ValueError("n_eff must be at least 1")
    k_mid = float(np.mean([_scale_index(shape, s) for s in window]))
    return k_mid - log2(n_effective / a) / d_f


def is_reliable(shape: tuple[int, int], window: Sequence[int], n_effective: int) -> bool:
    return saturation_x(shape, window, n_effective) <= 0.0


def contiguous_windows(sides: Sequence[int] = AMS_BOX_SIDES) -> list[tuple[int, ...]]:
    """Every run of at least two consecutive box sides."""
    sides = tuple(sides)
    return [sides[i:j] for i in range(len(sides)) for j in range(i + 2, len(sides) + 1)]


def reliable_windows(image, sides: Sequence[int] = AMS_BOX_SIDES) -> list[tuple[int, ...]]:
    a = _checked(image)
    ne = n_eff(a)
    return [w for w in contiguous_windows(sides) if is_reliable(a.shape, w, ne)]
