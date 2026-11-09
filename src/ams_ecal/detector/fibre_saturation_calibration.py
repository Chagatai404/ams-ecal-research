"""EXP-017: an EFFECTIVE fibre-saturation law derived from a published figure.

LABEL (every output): project phenomenological calibration derived from a published figure; not the AMS
parametrisation.

Source figure: Crispoltoni PhD thesis (Perugia), Fig. 2.29 (the same figure as Kounine et al., NIM A 869 (2017),
Fig. 10): maximum cell energy versus shower energy for flight data, MC with saturation and MC without saturation.
AMS reports a light-yield non-linearity "consistent with Birks' law", parametrised as a function of energy density,
but publishes no constants. A classic per-step Birks law cannot produce an energy-DENSITY dependence for near-MIP
shower particles, so this module fits a one-parameter COLLECTIVE law to the digitised curves.

Three parts:

* D0 digitisation of the raster figure (two independent passes, axis calibration from tick marks);
* D1/D2 simulation side: hottest-cell energy from Geant4 per-fibre scintillator deposits, with two candidate laws,
  M1 (per fibre, ``e' = e / (1 + e / e_s)``) and M2 (per cell, ``E' = E / (1 + E / E_s)``);
* the fit (chi-square against the digitised data) and its uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

LABEL = (
    "project phenomenological calibration derived from a published figure; "
    "not the AMS parametrisation"
)

SERIES = ("data", "mc_saturation", "mc_no_saturation")
SERIES_COLOUR = {"data": "red", "mc_saturation": "green", "mc_no_saturation": "blue"}

# Axis ticks that carry labels in the figure (value in the data unit).
X_MAJOR_VALUES = (200.0, 400.0, 600.0, 800.0, 1000.0)
Y_MAJOR_VALUES = (0.0, 20.0, 40.0, 60.0)
X_MINOR_STEP = 50.0
Y_MINOR_STEP = 5.0
X_AXIS_MIN = 100.0

# Legend and its frame occupy the top-left corner of the plot; no data point is there.
LEGEND_BOX = (195, 500, 55, 200)  # x0, x1, y0, y1 (pixels)


# ----------------------------------------------------------------------------------------------------------
# D0: digitisation
# ----------------------------------------------------------------------------------------------------------


def load_rgb(path: str | Path) -> np.ndarray:
    """Read a PNG as a float RGB array ``(rows, cols, 3)``."""

    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=float)
    except ImportError:  # pragma: no cover - PIL is present in the project environment
        import matplotlib.image as mpimg

        arr = mpimg.imread(str(path))[..., :3]
        return arr * 255.0 if arr.max() <= 1.0 else arr.astype(float)


def series_alpha(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel coverage (0..1) of each series colour, assuming anti-aliasing against white.

    Series colours (measured from the figure): red (255, 0, 23), green (0, 152, 46), blue (7, 32, 244).
    """

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = np.where((r > 200) & (np.abs(g - b) < 45), (r - g) / 255.0, 0.0)
    green = np.where((g > r + 10) & (g > b), (g - r) / 152.0, 0.0)
    blue = np.where((b > r + 10) & (b > g + 10), (b - r) / 237.0, 0.0)
    return {
        "data": np.clip(red, 0, 1),
        "mc_saturation": np.clip(green, 0, 1),
        "mc_no_saturation": np.clip(blue, 0, 1),
    }


def _cluster(indices: np.ndarray) -> list[np.ndarray]:
    """Group sorted integer indices into runs of consecutive values."""

    if len(indices) == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return np.split(indices, splits)


def _weighted_centre(profile: np.ndarray, indices: np.ndarray) -> float:
    w = profile[indices]
    return float((indices * w).sum() / w.sum())


@dataclass(frozen=True)
class AxisCalibration:
    """Linear pixel -> data maps for both axes, with the tick-reproduction residuals (pixels)."""

    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    x_major_pixels: tuple[float, ...]
    y_major_pixels: tuple[float, ...]
    x_residual_major_px: float
    y_residual_major_px: float
    x_residual_minor_px: float
    y_residual_minor_px: float
    n_minor_x: int
    n_minor_y: int

    def x_value(self, col):
        return self.x_scale * np.asarray(col, float) + self.x_offset

    def y_value(self, row):
        return self.y_scale * np.asarray(row, float) + self.y_offset

    def x_pixel(self, value):
        return (np.asarray(value, float) - self.x_offset) / self.x_scale

    def y_pixel(self, value):
        return (np.asarray(value, float) - self.y_offset) / self.y_scale


def calibrate_axes(rgb: np.ndarray) -> AxisCalibration:
    """Axis calibration from the tick marks (found in the image, not assumed).

    Frame lines are the rows/columns with a long dark run. The four labelled x ticks (200-800 GeV) and the labelled y
    ticks (20, 40 GeV) are found by tick length; the 1000 GeV, 0 and 60 GeV ticks coincide with the frame lines and
    are taken from the frame centroids. A straight line is fitted through the labelled ticks and then USED TO PREDICT
    every unlabelled (minor) tick: their residuals are the reproduction test.
    """

    dark = 255.0 - rgb.mean(axis=2)
    binary = dark > 128
    row_count = binary.sum(axis=1)
    col_count = binary.sum(axis=0)
    frame_rows = _cluster(np.flatnonzero(row_count > 300))
    frame_cols = _cluster(np.flatnonzero(col_count > 300))
    if len(frame_rows) != 2 or len(frame_cols) != 2:
        raise ValueError("expected exactly two horizontal and two vertical frame lines")

    def line_centre(group: np.ndarray, axis: int) -> float:
        lo, hi = int(group.min()) - 2, int(group.max()) + 3
        profile = (
            dark[lo:hi, :].sum(axis=1) if axis == 0 else dark[:, lo:hi].sum(axis=0)
        )
        return float((np.arange(lo, hi) * profile).sum() / profile.sum())

    y_top = line_centre(frame_rows[0], 0)
    y_bottom = line_centre(frame_rows[1], 0)
    x_left = line_centre(frame_cols[0], 1)
    x_right = line_centre(frame_cols[1], 1)
    bottom_edge = int(frame_rows[1].min())
    left_edge = int(frame_cols[0].max())

    # x ticks: short dark runs rising above the bottom frame line; length separates major from minor.
    near = dark[bottom_edge - 5 : bottom_edge - 1, :].mean(axis=0)
    x_clusters = [
        c
        for c in _cluster(np.flatnonzero(near > 60))
        if c.min() > left_edge + 5 and c.max() < x_right - 5
    ]
    x_pos, x_len = [], []
    for c in x_clusters:
        lo, hi = int(c.min()) - 2, int(c.max()) + 3
        profile = near[lo:hi]
        x_pos.append(float((np.arange(lo, hi) * profile).sum() / profile.sum()))
        centre = round(x_pos[-1])
        n = 0
        for y in range(bottom_edge - 1, 0, -1):
            if binary[y, centre]:
                n += 1
            else:
                break
        x_len.append(n)
    # y ticks: short dark runs to the right of the left frame line.
    nearv = dark[:, left_edge + 3 : left_edge + 7].mean(axis=1)
    y_clusters = [
        c
        for c in _cluster(np.flatnonzero(nearv > 60))
        if c.min() > y_top + 5 and c.max() < y_bottom - 5
    ]
    y_pos, y_len = [], []
    for c in y_clusters:
        lo, hi = int(c.min()) - 2, int(c.max()) + 3
        profile = nearv[lo:hi]
        y_pos.append(float((np.arange(lo, hi) * profile).sum() / profile.sum()))
        centre = round(y_pos[-1])
        n = 0
        for x in range(left_edge + 2, rgb.shape[1] - 1):
            if binary[centre, x]:
                n += 1
            else:
                break
        y_len.append(n)

    x_pos, x_len = np.array(x_pos), np.array(x_len)
    y_pos, y_len = np.array(y_pos), np.array(y_len)
    # the labelled ticks have the longest marks (14 / 20 px against 6 / 9 px)
    x_major = np.sort(x_pos[x_len >= x_len.max() - 2])
    y_major = np.sort(y_pos[y_len >= y_len.max() - 2])
    if len(x_major) != 4 or len(y_major) != 2:
        raise ValueError(
            f"unexpected major tick count: x {len(x_major)}, y {len(y_major)}"
        )

    x_major_px = np.concatenate([x_major, [x_right]])
    y_major_px = np.sort(np.concatenate([[y_top], y_major, [y_bottom]]))[
        ::-1
    ]  # 0, 20, 40, 60 from the bottom
    x_vals, y_vals = np.array(X_MAJOR_VALUES), np.array(Y_MAJOR_VALUES)
    x_scale, x_offset = np.polyfit(x_major_px, x_vals, 1)
    y_scale, y_offset = np.polyfit(y_major_px, y_vals, 1)
    x_res_major = float(np.abs(x_major_px - (x_vals - x_offset) / x_scale).max())
    y_res_major = float(np.abs(y_major_px - (y_vals - y_offset) / y_scale).max())

    # minor ticks: predict from the labelled fit and compare with the measured centroid
    x_minor_values = np.arange(X_AXIS_MIN + X_MINOR_STEP, 1000.0, X_MINOR_STEP)
    x_minor_values = x_minor_values[~np.isin(x_minor_values, X_MAJOR_VALUES)]
    pred_x = (x_minor_values - x_offset) / x_scale
    res_x = [float(np.min(np.abs(x_pos - p))) for p in pred_x]
    y_minor_values = np.arange(Y_MINOR_STEP, 60.0, Y_MINOR_STEP)
    y_minor_values = y_minor_values[~np.isin(y_minor_values, Y_MAJOR_VALUES)]
    pred_y = (y_minor_values - y_offset) / y_scale
    res_y = [float(np.min(np.abs(y_pos - p))) for p in pred_y]
    # the 100 GeV tick coincides with the left frame line
    res_x.append(float(abs(x_left - (X_AXIS_MIN - x_offset) / x_scale)))
    return AxisCalibration(
        x_scale=float(x_scale),
        x_offset=float(x_offset),
        y_scale=float(y_scale),
        y_offset=float(y_offset),
        x_major_pixels=tuple(float(v) for v in x_major_px),
        y_major_pixels=tuple(float(v) for v in y_major_px),
        x_residual_major_px=x_res_major,
        y_residual_major_px=y_res_major,
        x_residual_minor_px=float(max(res_x)),
        y_residual_minor_px=float(max(res_y)),
        n_minor_x=len(res_x),
        n_minor_y=len(res_y),
    )


@dataclass(frozen=True)
class Marker:
    series: str
    col: float
    row: float
    bar_top_row: float
    bar_bottom_row: float
    occluded: bool
    bar_end_occluded: bool
    marker_pass: str
    match_score: float
    bin_index: int


# Marker geometry measured from isolated markers (pixels).
RING_DIAMETER_PX = 9.5
RING_THICKNESS_PX = 1.8
SQUARE_SIDE_PX = 11.5
SQUARE_THICKNESS_PX = 1.6
DISK_DIAMETER_PX = 8.0
TEMPLATE_SIZE = 21
BAR_MIN_RUN_PX = 14


def _template(series: str, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Synthetic marker shape (anti-aliased by 4x oversampling); the central columns (error bar) are blanked."""

    sub = 4
    ax = (np.arange(size * sub) + 0.5) / sub - size / 2.0
    xx, yy = np.meshgrid(ax, ax)
    if series == "mc_no_saturation":
        rad = np.hypot(xx, yy)
        shape = (rad <= RING_DIAMETER_PX / 2) & (
            rad >= RING_DIAMETER_PX / 2 - RING_THICKNESS_PX
        )
    elif series == "mc_saturation":
        half = SQUARE_SIDE_PX / 2
        outer = (np.abs(xx) <= half) & (np.abs(yy) <= half)
        inner = (np.abs(xx) <= half - SQUARE_THICKNESS_PX) & (
            np.abs(yy) <= half - SQUARE_THICKNESS_PX
        )
        shape = outer & ~inner
    else:
        shape = np.hypot(xx, yy) <= DISK_DIAMETER_PX / 2
    tpl = shape.reshape(size, sub, size, sub).mean(axis=(1, 3))
    half_px = size // 2
    tpl[:, half_px - 1 : half_px + 2] = 0.0
    return tpl


def _remove_vertical_runs(
    mask: np.ndarray, min_run: int = BAR_MIN_RUN_PX
) -> np.ndarray:
    """Delete vertical runs (error bars) of at least ``min_run`` pixels; marker edges are shorter."""

    from scipy import ndimage

    labels, count = ndimage.label(
        mask, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    drop = sizes >= min_run
    drop[0] = False
    return mask & ~drop[labels]


def _correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:

    return signal.fftconvolve(image, template[::-1, ::-1], mode="same")


def _parabola(profile: np.ndarray, i: int) -> float:
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return i + 0.5 * (a - c) / denom
    return float(i)


def _plot_alpha(alpha: np.ndarray, frame: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = frame
    out = alpha.copy()
    lx0, lx1, ly0, ly1 = LEGEND_BOX
    out[ly0:ly1, lx0:lx1] = 0.0
    out[:y0] = 0.0
    out[y1:] = 0.0
    out[:, :x0] = 0.0
    out[:, x1:] = 0.0
    return out


def _bin_columns(
    scores: dict[str, np.ndarray],
    frame: tuple[int, int, int, int],
    expected: int | None = 20,
) -> list[float]:
    """Centre columns of the energy bins, from the matched-filter peaks of the red filled circles.

    The three series share the energy bins and the filled disks are never confused with bar fragments (the open
    markers are, at low energy, partly hidden behind them), so the bins come from the data series alone.
    """

    x0, x1, y0, y1 = frame
    score = scores["data"]
    local = ndimage.maximum_filter(score, size=7)
    peaks = np.argwhere((score == local) & (score >= 0.5))
    columns = sorted(float(c) for r, c in peaks if x0 <= c <= x1 and y0 <= r <= y1)
    groups: list[list[float]] = []
    for c in columns:
        if groups and c - groups[-1][-1] <= 5.0:
            groups[-1].append(c)
        else:
            groups.append([c])
    centres = [float(np.median(g)) for g in groups]
    if expected is not None and len(centres) != expected:
        raise ValueError(f"found {len(centres)} energy bins, expected {expected}")
    return centres


def digitise_pass(
    rgb: np.ndarray,
    calibration: AxisCalibration,
    marker_pass: str,
    frame: tuple[int, int, int, int],
    expected_bins: int | None = 20,
) -> list[Marker]:
    """One independent digitisation pass.

    Both passes first remove the long vertical runs (error bars) and locate each marker by a matched filter within the
    column of its energy bin (the three series share the bins; the bins come from the thick blue bars).

    Pass ``A``: strict colour threshold (coverage > 0.6 red, > 0.45 for the thin open markers); marker position = coverage-weighted centroid of the marker
    pixels inside the template support around the matched-filter peak.
    Pass ``B``: loose threshold (coverage > 0.2); marker position = the matched-filter peak refined by parabolas.
    The error-bar ends use their own, lower, thresholds in each pass.
    """

    if marker_pass not in ("A", "B"):
        raise ValueError("marker_pass must be 'A' or 'B'")
    strict = {"data": 0.6, "mc_saturation": 0.45, "mc_no_saturation": 0.45}
    alpha_by_series = {k: _plot_alpha(v, frame) for k, v in series_alpha(rgb).items()}
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    coverage = np.maximum.reduce(list(alpha_by_series.values()))
    # pixels where two series overlap (e.g. the red bar under the green bar) match no pure colour
    mixed = _plot_alpha((chroma > 40).astype(float) * (coverage < 0.6), frame) > 0.5
    masks, templates, scores = {}, {}, {}
    for series in SERIES:
        alpha = alpha_by_series[series]
        masks[series] = _remove_vertical_runs(
            alpha > (strict[series] if marker_pass == "A" else 0.2)
        )
        templates[series] = _template(series)
        scores[series] = (
            _correlate(np.where(masks[series], alpha, 0.0), templates[series])
            / templates[series].sum()
        )
    bins = _bin_columns(scores, frame, expected_bins)
    markers: list[Marker] = []
    for series in SERIES:
        alpha = alpha_by_series[series]
        mask, tpl, score = masks[series], templates[series], scores[series]
        support = tpl > 0.15
        half = tpl.shape[0] // 2
        bar_thr = 0.5 if series == "mc_no_saturation" else 0.12
        if marker_pass == "B":
            bar_thr *= 0.8
        for bin_index, centre_col in enumerate(bins):
            c_lo, c_hi = round(centre_col) - 3, round(centre_col) + 4
            window = score[:, c_lo:c_hi]
            r0, dc = np.unravel_index(int(np.argmax(window)), window.shape)
            c0 = c_lo + dc
            best = float(window[r0, dc])
            if best < 0.1:
                continue
            row_est = _parabola(score[:, c0], r0)
            col_est = _parabola(score[r0, :], c0)
            if marker_pass == "A":
                # dilate the support by one pixel so a half-pixel offset does not clip the marker
                wide = ndimage.binary_dilation(support, iterations=1)
                patch = (
                    np.where(mask, alpha, 0.0)[
                        r0 - half : r0 + half + 1, c0 - half : c0 + half + 1
                    ]
                    * wide
                )
                rows_idx, cols_idx = np.mgrid[-half : half + 1, -half : half + 1]
                total = patch.sum()
                if total > 0:
                    row_est = r0 + float((rows_idx * patch).sum() / total)
                    col_est = c0 + float((cols_idx * patch).sum() / total)
            # error bar: from the marker centre outwards; the first marker-sized stretch may be empty (open markers)
            cc = round(col_est)
            bar_cols = slice(cc - 1, cc + 2)
            column = alpha[:, bar_cols].max(axis=1)
            other = np.zeros_like(column)
            for s2 in SERIES:
                if s2 != series:
                    other = np.maximum(
                        other, alpha_by_series[s2][:, bar_cols].max(axis=1)
                    )
            on = (column > bar_thr) | mixed[:, bar_cols].any(axis=1)
            centre_row = round(row_est)
            ends = []
            for direction in (-1, +1):
                r, last, gap, started = centre_row, centre_row, 0, False
                while 0 <= r < alpha.shape[0]:
                    if on[r]:
                        last, gap, started = r, 0, True
                    else:
                        gap += 1
                        if gap > (2 if started else 9):
                            break
                    r += direction
                ends.append(last)
            top, bottom = ends
            end_occluded = bool(
                (other[max(top - 2, 0) : top + 3] > 0.3).any()
                or (other[bottom - 2 : bottom + 3] > 0.3).any()
            )
            rr, cc2 = round(row_est), round(col_est)
            patch_other = np.zeros(tpl.shape)
            for s2 in SERIES:
                if s2 != series:
                    patch_other = np.maximum(
                        patch_other,
                        alpha_by_series[s2][
                            rr - half : rr + half + 1, cc2 - half : cc2 + half + 1
                        ],
                    )
            covered = float(
                ((tpl > 0.5) & (patch_other > 0.5)).sum() / max((tpl > 0.5).sum(), 1)
            )
            markers.append(
                Marker(
                    series=series,
                    col=float(col_est),
                    row=float(row_est),
                    bar_top_row=float(top) - 0.5,
                    bar_bottom_row=float(bottom) + 0.5,
                    occluded=bool(covered > 0.15 or best < 0.3),
                    bar_end_occluded=end_occluded,
                    marker_pass=marker_pass,
                    match_score=best,
                    bin_index=bin_index,
                )
            )
    return markers


def markers_to_rows(markers: list[Marker], calibration: AxisCalibration) -> list[dict]:
    """Convert pixel markers to ``digitised.csv`` rows (energy, value, error-bar ends in GeV)."""

    rows = []
    for m in sorted(markers, key=lambda m: (SERIES.index(m.series), m.col)):
        rows.append(
            {
                "series": m.series,
                "E_gev": float(calibration.x_value(m.col)),
                "y_gev": float(calibration.y_value(m.row)),
                "y_low": float(calibration.y_value(m.bar_bottom_row)),
                "y_high": float(calibration.y_value(m.bar_top_row)),
                "pass": m.marker_pass,
                "marker_occluded": bool(m.occluded),
                "bar_end_occluded": bool(m.bar_end_occluded),
                "match_score": float(m.match_score),
                "bin": int(m.bin_index),
            }
        )
    return rows


def adopt_curves(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """Combine the two passes into the adopted digitisation (one row per series and energy bin).

    Marker position = mean of the two passes. Error bars are symmetric in the figure, so the half-length is one number:
    for the DATA series (drawn last, never hidden) it is ``min(upper, lower)`` because the only failure there is
    overshoot into colour-mixed pixels of neighbouring bars; for the two MC series, whose bars are partly hidden behind
    the red points, it is the smaller of the two ends unless one end is below 0.6 of the other (truncated by
    occlusion), in which case the longer end is used and the row is flagged ``half_length_unreliable``.
    """

    index_b = {(r["series"], r["bin"]): r for r in rows_b}
    out = []
    for a in rows_a:
        b = index_b[(a["series"], a["bin"])]
        y = 0.5 * (a["y_gev"] + b["y_gev"])
        up = 0.5 * ((a["y_high"] - a["y_gev"]) + (b["y_high"] - b["y_gev"]))
        down = 0.5 * ((a["y_gev"] - a["y_low"]) + (b["y_gev"] - b["y_low"]))
        small, large = min(up, down), max(up, down)
        unreliable = False
        if a["series"] == "data" or small >= 0.6 * large:
            half = small
        else:
            half, unreliable = large, True
        out.append(
            {
                "series": a["series"],
                "bin": a["bin"],
                "E_gev": 0.5 * (a["E_gev"] + b["E_gev"]),
                "y_gev": y,
                "y_low": y - half,
                "y_high": y + half,
                "half_length_gev": half,
                "pass": "AB-mean",
                "marker_occluded": bool(a["marker_occluded"] or b["marker_occluded"]),
                "half_length_unreliable": bool(
                    unreliable
                    or a["bar_end_occluded"]
                    and b["bar_end_occluded"]
                    and a["series"] != "data"
                ),
                "pass_difference_y_gev": abs(a["y_gev"] - b["y_gev"]),
                "pass_difference_E_gev": abs(a["E_gev"] - b["E_gev"]),
            }
        )
    return out


def frame_box(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Inner plotting area (x0, x1, y0, y1) in pixels."""

    binary = (255.0 - rgb.mean(axis=2)) > 128
    rows = _cluster(np.flatnonzero(binary.sum(axis=1) > 300))
    cols = _cluster(np.flatnonzero(binary.sum(axis=0) > 300))
    return (
        int(cols[0].max()) + 2,
        int(cols[1].min()) - 1,
        int(rows[0].max()) + 2,
        int(rows[1].min()) - 1,
    )


def draw_overlay(
    rgb: np.ndarray,
    markers_a: list[Marker],
    markers_b: list[Marker],
    path: str | Path,
    crop: tuple[int, int, int, int] | None = None,
    zoom: float = 2.0,
) -> None:
    """Draw the digitised points of both passes on the source image (pass A: x, pass B: +; bar ends: short ticks)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = crop if crop is not None else (0, rgb.shape[1], 0, rgb.shape[0])
    fig = plt.figure(
        figsize=((x1 - x0) * zoom / 100.0, (y1 - y0) * zoom / 100.0), dpi=100
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgb.astype(np.uint8), interpolation="nearest")
    for markers, style, size in ((markers_a, "x", 60), (markers_b, "+", 60)):
        for m in markers:
            ax.scatter(
                [m.col],
                [m.row],
                marker=style,
                s=size,
                c="black",
                linewidths=0.9,
                zorder=5,
            )
            for end in (m.bar_top_row, m.bar_bottom_row):
                ax.plot(
                    [m.col - 3, m.col + 3],
                    [end, end],
                    color="magenta" if style == "x" else "black",
                    lw=0.7,
                    zorder=5,
                )
    ax.text(
        0.01,
        0.01,
        LABEL,
        transform=ax.transAxes,
        fontsize=6.5 if zoom < 3 else 5,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 1.5, "edgecolor": "none"},
        zorder=10,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------
# D1/D2: hottest-cell energy of Geant4 electrons and the one-parameter saturation laws
# ----------------------------------------------------------------------------------------------------------

LAYERS = 18
CELLS = 72
N_FLAT = LAYERS * CELLS
FIDUCIAL_THETA_MAX_RAD = float(np.radians(12.0))
AMS_ENERGY_RANGE_GEV = (105.0, 950.0)
D1_TOLERANCE = 0.10
MODELS = ("M1", "M2")
CONVENTIONS = ("A", "A_global", "B")


@dataclass
class EventGroup:
    """Electrons of one sample at (about) one energy, reduced to what the hottest-cell analysis needs."""

    name: str
    sample: str
    selection: str
    energy_gev: float
    primary_mev: np.ndarray
    scint_sum_mev: np.ndarray
    edep_total_mev: np.ndarray
    hot_unsaturated_mev: np.ndarray  # re-summed from fibres, no saturation
    event_of_fibre: np.ndarray  # int64, group-local event index repeated per fibre
    cell_of_fibre: np.ndarray  # int64 flat (layer * 72 + cell)
    fibre_mev: np.ndarray  # float64 scintillator deposit of each fibre

    @property
    def n_events(self) -> int:
        return len(self.primary_mev)

    def sampling_fraction(self, convention: str, pooled: float | None = None) -> float:
        """Scintillator-to-calibration factor ``f`` of the convention (state it with every number).

        ``A``: scintillator / total deposit in the prefix (ratio of sums over the group; leakage excluded).
        ``A_global``: the same ratio pooled over every group of the fit (one constant).
        ``B``: scintillator / primary energy of the group (leakage included: the cell sum reconstructs the primary).
        """

        if convention == "A":
            return float(self.scint_sum_mev.sum() / self.edep_total_mev.sum())
        if convention == "A_global":
            if pooled is None:
                raise ValueError("A_global needs the pooled value")
            return pooled
        if convention == "B":
            return float(self.scint_sum_mev.sum() / self.primary_mev.sum())
        raise ValueError(f"unknown convention {convention}")


def _hot_per_event(
    event: np.ndarray, cell: np.ndarray, weight: np.ndarray, n_events: int
) -> np.ndarray:
    grid = np.bincount(
        event * N_FLAT + cell, weights=weight, minlength=n_events * N_FLAT
    )
    return grid.reshape(n_events, N_FLAT).max(axis=1)


def build_group(
    batch, indices: np.ndarray, lookup: tuple[np.ndarray, np.ndarray], **labels
) -> EventGroup:
    """Reduce the selected events of a stored batch (``geant4_backend.Batch``) to an ``EventGroup``."""

    fibre_layer, fibre_cell = lookup
    arrays = batch.arrays
    events, cells, energies = [], [], []
    for local, index in enumerate(indices):
        ids, edep = batch.fibre_deposits(int(index))
        layer, cell = fibre_layer[ids], fibre_cell[ids]
        mapped = (layer >= 0) & (cell >= 0) & (layer < LAYERS)
        cells.append(layer[mapped] * CELLS + cell[mapped])
        events.append(np.full(int(mapped.sum()), local, np.int64))
        energies.append(edep[mapped].astype(np.float64))
    event = np.concatenate(events)
    cell = np.concatenate(cells).astype(np.int64)
    fibre = np.concatenate(energies)
    n = len(indices)
    if "primary_energy_gev" in arrays:
        energy_gev = arrays["primary_energy_gev"][indices]
    else:
        energy_gev = np.full(n, batch.metadata["energy_gev"])
    hot = _hot_per_event(event, cell, fibre, n)
    grid_hot = (
        arrays["readout_grid_mev"][indices]
        .reshape(n, -1)
        .max(axis=1)
        .astype(np.float64)
    )
    if not np.allclose(hot, grid_hot, rtol=2e-4, atol=1e-3):
        raise ValueError(
            "fibre re-summing does not reproduce readout_grid_mev: the fibre-to-cell mapping is wrong"
        )
    return EventGroup(
        primary_mev=np.asarray(energy_gev, float) * 1000.0,
        scint_sum_mev=arrays["readout_grid_mev"][indices]
        .reshape(n, -1)
        .sum(axis=1)
        .astype(np.float64),
        edep_total_mev=arrays["edep_total_mev"][indices].astype(np.float64),
        hot_unsaturated_mev=hot,
        event_of_fibre=event,
        cell_of_fibre=cell,
        fibre_mev=fibre,
        energy_gev=float(np.mean(energy_gev)),
        **labels,
    )


def hot_m1_mev(group: EventGroup, scale_mev: float | None) -> np.ndarray:
    """M1: saturate each fibre (``e' = e / (1 + e / e_s)``), re-sum into cells, take the hottest cell per event."""

    weight = (
        group.fibre_mev
        if scale_mev is None
        else group.fibre_mev / (1.0 + group.fibre_mev / scale_mev)
    )
    return _hot_per_event(
        group.event_of_fibre, group.cell_of_fibre, weight, group.n_events
    )


def saturate_cell_gev(value_gev: np.ndarray, scale_gev: float) -> np.ndarray:
    """M2: saturate a calibrated cell energy, ``E' = E / (1 + E / E_s)`` (monotone, so it commutes with the maximum)."""

    return value_gev / (1.0 + value_gev / scale_gev)


def calibrated_hot_gev(hot_scint_mev: np.ndarray, f: float) -> np.ndarray:
    """Hottest-cell scintillator energy converted to calibrated energy units: ``/ f``, MeV to GeV."""

    return hot_scint_mev / f / 1000.0


# ---- AMS curves as functions of energy ------------------------------------------------------------------------


def _pchip(rows: list[dict], series: str, key: str):
    from scipy.interpolate import PchipInterpolator

    pts = sorted((r["E_gev"], r[key]) for r in rows if r["series"] == series)
    return PchipInterpolator([p[0] for p in pts], [p[1] for p in pts], extrapolate=True)


@dataclass
class AmsCurves:
    """The adopted digitisation as interpolators (value and half-length) at arbitrary energy."""

    rows: list[dict]

    def value(self, series: str, energy_gev):
        return _pchip(self.rows, series, "y_gev")(energy_gev)

    def half_length(self, series: str, energy_gev):
        return _pchip(self.rows, series, "half_length_gev")(energy_gev)

    def in_range(self, energy_gev: float) -> bool:
        return AMS_ENERGY_RANGE_GEV[0] <= energy_gev <= AMS_ENERGY_RANGE_GEV[1]


def implied_saturation_scale(curves: AmsCurves) -> list[dict]:
    """Model-independent check: ``1 / E_s = 1 / y_sat - 1 / y_nosat`` at each AMS energy (calibrated GeV).

    If the one-parameter form ``y_sat = y / (1 + y / E_s)`` held for the AMS curves, E_s would be constant.
    """

    out = []
    for r in sorted(
        (r for r in curves.rows if r["series"] == "mc_no_saturation"),
        key=lambda r: r["E_gev"],
    ):
        e = r["E_gev"]
        no_sat = float(curves.value("mc_no_saturation", e))
        row = {"E_gev": e, "mc_no_saturation_gev": no_sat}
        for series in ("mc_saturation", "data"):
            sat = float(curves.value(series, e))
            gap = 1.0 / sat - 1.0 / no_sat
            row[f"E_s_implied_{series}_gev"] = float(1.0 / gap) if gap > 0 else None
        row["marker_occluded"] = bool(r["marker_occluded"])
        out.append(row)
    return out


# ---- the fit --------------------------------------------------------------------------------------------------


@dataclass
class FitInputs:
    """Everything the chi-square needs for one (sample, convention, digitisation) combination."""

    groups: list[EventGroup]
    convention: str
    curves: AmsCurves
    variant: str = (
        "V0"  # V0 as is; V1 constant ratio to MC-without-saturation; V2 per-point ratio
    )
    scale_grid_mev: np.ndarray | None = None
    m1_hot_mev: np.ndarray | None = (
        None  # (n_scales, n_groups) mean hottest-cell scintillator energy
    )

    def pooled(self) -> float:
        return float(
            sum(g.scint_sum_mev.sum() for g in self.groups)
            / sum(g.edep_total_mev.sum() for g in self.groups)
        )

    def f(self) -> np.ndarray:
        pooled = self.pooled()
        return np.array(
            [g.sampling_fraction(self.convention, pooled) for g in self.groups]
        )

    def energies(self) -> np.ndarray:
        return np.array([g.energy_gev for g in self.groups])

    def no_saturation_mean_gev(self) -> np.ndarray:
        f = self.f()
        return np.array(
            [
                calibrated_hot_gev(g.hot_unsaturated_mev, fi).mean()
                for g, fi in zip(self.groups, f)
            ]
        )

    def stat_error_gev(self) -> np.ndarray:
        f = self.f()
        return np.array(
            [
                calibrated_hot_gev(g.hot_unsaturated_mev, fi).std(ddof=1)
                / np.sqrt(g.n_events)
                for g, fi in zip(self.groups, f)
            ]
        )

    def ratio_to_ams_no_saturation(self) -> np.ndarray:
        return self.no_saturation_mean_gev() / self.curves.value(
            "mc_no_saturation", self.energies()
        )

    def prepare_m1(
        self, n_scales: int = 120, lo_mev: float = 2.0, hi_mev: float = 1.0e5
    ) -> None:
        self.scale_grid_mev = np.geomspace(lo_mev, hi_mev, n_scales)
        self.m1_hot_mev = np.array(
            [
                [hot_m1_mev(g, s).mean() for g in self.groups]
                for s in self.scale_grid_mev
            ]
        )

    def _m1_curve(self, scale_mev: float) -> np.ndarray:
        from scipy.interpolate import CubicSpline

        spline = CubicSpline(np.log(self.scale_grid_mev), self.m1_hot_mev, axis=0)
        return spline(np.log(scale_mev)) / self.f() / 1000.0

    def _m2_curve(self, scale_gev: float) -> np.ndarray:
        f = self.f()
        return np.array(
            [
                saturate_cell_gev(
                    calibrated_hot_gev(g.hot_unsaturated_mev, fi), scale_gev
                ).mean()
                for g, fi in zip(self.groups, f)
            ]
        )

    def predict(self, model: str, parameter: float) -> np.ndarray:
        """Predicted mean hottest-cell energy (calibrated GeV) at each group energy, after the variant correction."""

        raw = self._m1_curve(parameter) if model == "M1" else self._m2_curve(parameter)
        no_sat = self.no_saturation_mean_gev()
        if self.variant == "V0":
            return raw
        if self.variant == "V1":
            return raw / float(np.mean(self.ratio_to_ams_no_saturation()))
        if self.variant == "V2":
            return self.curves.value("mc_no_saturation", self.energies()) * raw / no_sat
        raise ValueError(f"unknown variant {self.variant}")

    def sigma(self, series: str = "data") -> np.ndarray:
        """Uncertainty on the comparison: digitised bar half-length (maybe a distribution width: conservative) plus
        the statistical error of our mean, in quadrature."""

        return np.hypot(
            self.curves.half_length(series, self.energies()), self.stat_error_gev()
        )

    def chi2(self, model: str, parameter: float, series: str = "data") -> float:
        target = self.curves.value(series, self.energies())
        return float(
            (
                ((self.predict(model, parameter) - target) / self.sigma(series)) ** 2
            ).sum()
        )


def fit_parameter(inputs: FitInputs, model: str, bounds: tuple[float, float]) -> dict:
    """Minimise chi-square over the single parameter (log scale); 68 % profile interval from delta chi2 = 1."""

    from scipy.optimize import brentq, minimize_scalar

    lo, hi = np.log(bounds[0]), np.log(bounds[1])

    def chi2_of_log(x: float) -> float:
        return inputs.chi2(model, float(np.exp(x)))

    grid = np.linspace(lo, hi, 241)
    values = np.array([chi2_of_log(x) for x in grid])
    k = int(np.argmin(values))
    left, right = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(
        chi2_of_log, bounds=(left, right), method="bounded", options={"xatol": 1e-9}
    )
    best_x, best = float(res.x), float(res.fun)

    def edge(direction: int) -> float | None:
        limit = hi if direction > 0 else lo
        if chi2_of_log(limit) - best < 1.0:
            return None  # open on this side inside the allowed range
        return float(
            np.exp(
                brentq(lambda x: chi2_of_log(x) - best - 1.0, best_x, limit, xtol=1e-9)
            )
        )

    n_points = len(inputs.groups)
    return {
        "model": model,
        "parameter": float(np.exp(best_x)),
        "interval_68": [edge(-1), edge(+1)],
        "chi2": best,
        "n_points": n_points,
        "dof": n_points - 1,
        "chi2_per_dof": best / (n_points - 1) if n_points > 1 else None,
        "at_edge_of_range": bool(k in (0, len(grid) - 1)),
        "predicted_gev": [
            float(v) for v in inputs.predict(model, float(np.exp(best_x)))
        ],
    }


# ----------------------------------------------------------------------------------------------------------
# Orchestration: D0 -> D1 -> D2 -> outputs
# ----------------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THESIS_PDF = Path(
    "D:/Projects/QML Research/papers/01 Detector Hardware and Calibration/TesiPhDCrispoltoni.pdf"
)
THESIS_FIGURE_PAGE = 67
OUTPUT_DIR = PROJECT_ROOT / "results" / "detector_response" / "fibre_saturation"
NORMAL_BATCHES = (
    ("data/geant4_component_electron/train/E50GeV", "component_electron"),
    ("data/geant4_component_electron/train/E100GeV", "component_electron"),
    (
        "data/geant4_highenergy_pilot_electron/pilot/E200GeV",
        "highenergy_pilot_electron",
    ),
    (
        "data/geant4_highenergy_pilot_electron/pilot/E400GeV",
        "highenergy_pilot_electron",
    ),
    (
        "data/geant4_highenergy_pilot_electron/pilot/E800GeV",
        "highenergy_pilot_electron",
    ),
)
REALISTIC_BATCHES = tuple(
    f"data/geant4_realistic_regime_pilot_electron/pilot/E{energy}GeV"
    for energy in (50, 100, 200, 400)
)
SOURCE_FIGURE = "Crispoltoni PhD thesis (Perugia), Fig. 2.29, PDF page 67 (same figure as Kounine et al., NIM A 869 (2017), Fig. 10)"


def extract_figure(pdf: Path, page: int, workdir: Path) -> Path:
    """Extract the embedded raster image of one PDF page with ``pdfimages`` (poppler)."""

    import subprocess

    workdir.mkdir(parents=True, exist_ok=True)
    prefix = workdir / "fig229"
    subprocess.run(
        ["pdfimages", "-png", "-f", str(page), "-l", str(page), str(pdf), str(prefix)],
        check=True,
    )
    found = sorted(workdir.glob("fig229-*.png"))
    if len(found) != 1:
        raise RuntimeError(
            f"expected one raster image on page {page}, found {len(found)}"
        )
    return found[0]


def digitise_figure(figure_png: Path) -> dict:
    """D0: both passes, the adopted curves, and the D0 gate numbers."""

    rgb = load_rgb(figure_png)
    calibration = calibrate_axes(rgb)
    frame = frame_box(rgb)
    markers_a = digitise_pass(rgb, calibration, "A", frame)
    markers_b = digitise_pass(rgb, calibration, "B", frame)
    rows_a = markers_to_rows(markers_a, calibration)
    rows_b = markers_to_rows(markers_b, calibration)
    adopted = adopt_curves(rows_a, rows_b)
    index_b = {(m.series, m.bin_index): m for m in markers_b}
    d_col = np.array(
        [abs(m.col - index_b[(m.series, m.bin_index)].col) for m in markers_a]
    )
    d_row = np.array(
        [abs(m.row - index_b[(m.series, m.bin_index)].row) for m in markers_a]
    )
    tick_worst = max(
        calibration.x_residual_major_px,
        calibration.y_residual_major_px,
        calibration.x_residual_minor_px,
        calibration.y_residual_minor_px,
    )
    gate = {
        "tick_positions_reproduced_within_1px": bool(tick_worst < 1.0),
        "worst_tick_residual_px": float(tick_worst),
        "x_major_residual_px": calibration.x_residual_major_px,
        "y_major_residual_px": calibration.y_residual_major_px,
        "x_minor_residual_px": calibration.x_residual_minor_px,
        "y_minor_residual_px": calibration.y_residual_minor_px,
        "n_minor_ticks_checked": [calibration.n_minor_x, calibration.n_minor_y],
        "passes_agree_within_marker_size": bool(
            max(d_col.max(), d_row.max()) <= DISK_DIAMETER_PX
        ),
        "marker_size_px": DISK_DIAMETER_PX,
        "max_pass_difference_px": float(max(d_col.max(), d_row.max())),
        "median_pass_difference_px": float(np.median(np.concatenate([d_col, d_row]))),
        "max_pass_difference_gev_y": float(
            max(r["pass_difference_y_gev"] for r in adopted)
        ),
        "markers_per_series": {
            s: sum(1 for r in adopted if r["series"] == s) for s in SERIES
        },
        "markers_flagged_occluded": {
            s: [
                round(r["E_gev"], 1)
                for r in adopted
                if r["series"] == s and r["marker_occluded"]
            ]
            for s in SERIES
        },
    }
    return {
        "rgb": rgb,
        "calibration": calibration,
        "markers_a": markers_a,
        "markers_b": markers_b,
        "rows_a": rows_a,
        "rows_b": rows_b,
        "adopted": adopted,
        "gate": gate,
    }


def write_digitised_csv(
    path: Path, rows_a: list[dict], rows_b: list[dict], adopted: list[dict]
) -> None:
    import csv

    fields = [
        "series",
        "E_gev",
        "y_gev",
        "y_low",
        "y_high",
        "pass",
        "bin",
        "marker_occluded",
        "bar_end_occluded",
        "half_length_gev",
        "half_length_unreliable",
        "match_score",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        for block in (rows_a, rows_b, adopted):
            for row in block:
                writer.writerow(
                    {
                        **{
                            k: (f"{v:.5g}" if isinstance(v, float) else v)
                            for k, v in row.items()
                            if k in fields
                        },
                        "label": LABEL,
                    }
                )


def load_event_groups(root: Path) -> tuple[list[EventGroup], list[EventGroup]]:
    """Normal-incidence groups (one per batch) and theta <= 12 deg fiducial groups of the realistic-regime batches."""

    from ams_ecal.detector.geometry import load_geometry
    from ams_ecal.detector.transport_geometry import FibreLayout
    from ams_ecal.geant4_simulation.geant4_backend import GEOMETRY_CONFIG, load_batch
    from ams_ecal.multiscale_estimators.realistic_regime_analysis import fiducial_mask

    geometry = load_geometry(GEOMETRY_CONFIG)
    lookup = FibreLayout(geometry).readout_lookup(geometry.number_of_superlayers)
    normal = []
    for relative, sample in NORMAL_BATCHES:
        directory = root / relative
        if not (directory / "events.npz").exists():
            continue
        batch = load_batch(directory)
        if len(batch) != batch.metadata["n_events"]:
            raise ValueError(
                f"{relative}: array length does not match n_events (batch still being written?)"
            )
        normal.append(
            build_group(
                batch,
                np.arange(len(batch)),
                lookup,
                name=Path(relative).name,
                sample=sample,
                selection="normal incidence, all events",
            )
        )
    inclined = []
    for relative in REALISTIC_BATCHES:
        directory = root / relative
        if not (directory / "events.npz").exists():
            continue
        batch = load_batch(directory)
        if len(batch) != batch.metadata["n_events"]:
            raise ValueError(
                f"{relative}: array length does not match n_events (batch still being written?)"
            )
        a = batch.arrays
        keep = fiducial_mask(
            a["entry_x_mm"],
            a["entry_y_mm"],
            a["primary_theta_rad"],
            a["primary_phi_rad"],
        ) & (a["primary_theta_rad"] <= FIDUCIAL_THETA_MAX_RAD)
        low, high = batch.metadata["energy_band_gev"]
        edges = (low, float(np.sqrt(low * high)), high * 1.0001)
        for k in range(2):
            inside = (
                keep
                & (a["primary_energy_gev"] >= edges[k])
                & (a["primary_energy_gev"] < edges[k + 1])
            )
            if inside.sum() >= 5:
                inclined.append(
                    build_group(
                        batch,
                        np.flatnonzero(inside),
                        lookup,
                        name=f"E{low:.0f}-{high:.0f}GeV_bin{k}",
                        sample="realistic_regime_pilot_electron",
                        selection=f"theta <= 12 deg, fiducial |x|,|y| <= 297 mm at z = 0 and 166.5 mm; "
                        f"E in [{edges[k]:.1f}, {edges[k + 1]:.1f}) GeV",
                    )
                )
    return normal, inclined


def d1_table(groups: list[EventGroup], curves: AmsCurves) -> list[dict]:
    """D1: our hottest cell without saturation against the digitised MC without saturation."""

    pooled = float(
        sum(g.scint_sum_mev.sum() for g in groups)
        / sum(g.edep_total_mev.sum() for g in groups)
    )
    rows = []
    for g in groups:
        extrapolated = (
            not curves.in_range(g.energy_gev)
            and g.energy_gev >= 0.92 * AMS_ENERGY_RANGE_GEV[0]
        )
        comparison = (
            "interpolated"
            if curves.in_range(g.energy_gev)
            else (
                "extrapolated_below_lowest_marker"
                if extrapolated
                else "none_outside_ams_range"
            )
        )
        usable = comparison != "none_outside_ams_range"
        row = {
            "group": g.name,
            "sample": g.sample,
            "selection": g.selection,
            "n_events": g.n_events,
            "E_gev": g.energy_gev,
            "ams_range": bool(curves.in_range(g.energy_gev)),
            "ams_comparison": comparison,
            "ams_mc_no_saturation_gev": float(
                curves.value("mc_no_saturation", g.energy_gev)
            )
            if usable
            else None,
            "ams_mc_no_saturation_half_length_gev": (
                float(curves.half_length("mc_no_saturation", g.energy_gev))
                if usable
                else None
            ),
            "ams_data_gev": float(curves.value("data", g.energy_gev))
            if usable
            else None,
        }
        for convention in CONVENTIONS:
            f = g.sampling_fraction(convention, pooled)
            hot = calibrated_hot_gev(g.hot_unsaturated_mev, f)
            row[f"conv_{convention}"] = {
                "f": f,
                "mean_gev": float(hot.mean()),
                "median_gev": float(np.median(hot)),
                "std_gev": float(hot.std(ddof=1)),
                "std_over_mean": float(hot.std(ddof=1) / hot.mean()),
                "mean_error_gev": float(hot.std(ddof=1) / np.sqrt(len(hot))),
                "ratio_mean_to_ams": float(hot.mean() / row["ams_mc_no_saturation_gev"])
                if usable
                else None,
                "ratio_median_to_ams": float(
                    np.median(hot) / row["ams_mc_no_saturation_gev"]
                )
                if usable
                else None,
            }
        row["ams_bar_over_value"] = (
            row["ams_mc_no_saturation_half_length_gev"]
            / row["ams_mc_no_saturation_gev"]
            if usable
            else None
        )
        rows.append(row)
    return rows


def run_fits(
    fit_groups: list[EventGroup],
    adopted: list[dict],
    rows_a: list[dict],
    rows_b: list[dict],
    inclined_fit_groups: list[EventGroup],
) -> dict:
    """D2/D3: central fits, profile intervals, and the shifts of every systematic variation."""

    curves = AmsCurves(adopted)
    bounds = {"M1": (3.0, 5.0e4), "M2": (2.0, 5.0e3)}

    def fit(groups, convention, curves_used, variant, models=MODELS, grid=None):
        inputs = FitInputs(groups, convention, curves_used, variant)
        if grid is None:
            inputs.prepare_m1()
        else:
            inputs.scale_grid_mev, inputs.m1_hot_mev = grid
        return inputs, {m: fit_parameter(inputs, m, bounds[m]) for m in models}

    central_inputs, central = fit(fit_groups, "A", curves, "V0")
    grid = (central_inputs.scale_grid_mev, central_inputs.m1_hot_mev)
    variations = {}
    for name, convention, curves_used, variant in (
        ("pass_A_only", "A", AmsCurves(adopt_curves(rows_a, rows_a)), "V0"),
        ("pass_B_only", "A", AmsCurves(adopt_curves(rows_b, rows_b)), "V0"),
        ("ratio_constant_V1", "A", curves, "V1"),
        ("ratio_per_point_V2", "A", curves, "V2"),
        ("convention_A_global", "A_global", curves, "V0"),
        ("convention_B_V0", "B", curves, "V0"),
        ("convention_B_with_constant_ratio_V1", "B", curves, "V1"),
    ):
        _, variations[name] = fit(
            fit_groups, convention, curves_used, variant, grid=grid
        )
    realistic = None
    if len(inclined_fit_groups) >= 3:
        _, realistic = fit(inclined_fit_groups, "A", curves, "V0")

    summary = {}
    for model in MODELS:
        centre = central[model]["parameter"]
        shifts = {
            name: variations[name][model]["parameter"] / centre - 1.0
            for name in variations
        }
        source = {
            "digitisation_pass": max(
                abs(shifts["pass_A_only"]), abs(shifts["pass_B_only"])
            ),
            "d1_ratio_or_offset": max(
                abs(shifts["ratio_constant_V1"]), abs(shifts["ratio_per_point_V2"])
            ),
            "sampling_fraction_convention": max(
                abs(shifts["convention_A_global"]),
                abs(shifts["convention_B_with_constant_ratio_V1"]),
            ),
        }
        extra = {
            "inclined_theta12_sample_shift": (
                realistic[model]["parameter"] / centre - 1.0 if realistic else None
            ),
            "convention_B_uncorrected_shift": shifts["convention_B_V0"],
            "convention_B_uncorrected_note": "convention B fails the D1 gate (up to +32% above AMS MC without saturation), so this shift mostly reflects the calibration offset; reported, not added to the quadrature total",
        }
        summary[model] = {
            "not_in_quadrature": extra,
            "relative_shift_of_parameter": shifts,
            "relative_systematic_by_source": source,
            "relative_systematic_total_quadrature": float(
                np.sqrt(sum(v**2 for v in source.values()))
            ),
        }
    return {
        "central_inputs": central_inputs,
        "central": central,
        "variations": variations,
        "realistic_theta12_sample": realistic,
        "systematics": summary,
        "bounds": bounds,
    }


def saturated_curve_for_group(
    group: EventGroup, model: str, parameter: float, convention: str, pooled: float
) -> float:
    """Mean saturated hottest cell (calibrated GeV, same ``f`` as the unsaturated curve) of one group."""

    f = group.sampling_fraction(convention, pooled)
    if model == "M1":
        return float(calibrated_hot_gev(hot_m1_mev(group, parameter), f).mean())
    return float(
        saturate_cell_gev(
            calibrated_hot_gev(group.hot_unsaturated_mev, f), parameter
        ).mean()
    )


def hottest_cell_figure(
    path: Path,
    curves: AmsCurves,
    normal: list[EventGroup],
    inclined: list[EventGroup],
    fits: dict,
    pooled: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 8.6), gridspec_kw={"height_ratios": [3, 1.6]}, sharex=True
    )
    style = {
        "data": ("o", "tab:red", "AMS data (digitised)", True),
        "mc_saturation": (
            "s",
            "tab:green",
            "AMS MC with saturation (digitised)",
            False,
        ),
        "mc_no_saturation": (
            "o",
            "tab:blue",
            "AMS MC without saturation (digitised)",
            False,
        ),
    }
    for series, (marker, colour, label, filled) in style.items():
        rows = sorted(
            (r for r in curves.rows if r["series"] == series), key=lambda r: r["E_gev"]
        )
        e = np.array([r["E_gev"] for r in rows])
        y = np.array([r["y_gev"] for r in rows])
        h = np.array([r["half_length_gev"] for r in rows])
        top.errorbar(
            e,
            y,
            yerr=h,
            fmt=marker,
            color=colour,
            mfc=colour if filled else "none",
            ms=4,
            lw=0.8,
            capsize=0,
            label=label,
            alpha=0.9,
        )
    pooled_sets = (
        ("ours, no saturation, normal incidence", normal, None, None, "k", "D", "-"),
        (
            "ours, no saturation, theta <= 12 deg",
            inclined,
            None,
            None,
            "0.45",
            "^",
            ":",
        ),
    )
    for label, groups, _model, _par, colour, marker, ls in pooled_sets:
        if not groups:
            continue
        convention_f = [g.sampling_fraction("A") for g in groups]
        e = np.array([g.energy_gev for g in groups])
        y = np.array(
            [
                calibrated_hot_gev(g.hot_unsaturated_mev, f).mean()
                for g, f in zip(groups, convention_f)
            ]
        )
        order = np.argsort(e)
        top.plot(
            e[order],
            y[order],
            marker=marker,
            color=colour,
            ls=ls,
            ms=5,
            lw=1.0,
            label=label,
        )
    colours = {"M1": ("tab:purple", "-"), "M2": ("tab:orange", "--")}
    for model in MODELS:
        parameter = fits["central"][model]["parameter"]
        e = np.array([g.energy_gev for g in normal])
        y = np.array(
            [
                saturated_curve_for_group(g, model, parameter, "A", pooled)
                for g in normal
            ]
        )
        order = np.argsort(e)
        colour, ls = colours[model]
        name = (
            f"ours + M1 per-fibre, e_s = {parameter:.0f} fibre-MeV"
            if model == "M1"
            else f"ours + M2 per-cell, E_s = {parameter:.0f} GeV"
        )
        top.plot(
            e[order],
            y[order],
            color=colour,
            ls=ls,
            marker="x",
            ms=6,
            lw=1.4,
            label=name,
        )
        bottom.plot(
            e[order],
            (
                y
                / np.array(
                    [
                        calibrated_hot_gev(
                            g.hot_unsaturated_mev, g.sampling_fraction("A")
                        ).mean()
                        for g in normal
                    ]
                )
            )[order],
            color=colour,
            ls=ls,
            marker="x",
            label=f"ours {model} / ours no saturation",
        )
    e_grid = np.linspace(106, 945, 200)
    bottom.plot(
        e_grid,
        curves.value("data", e_grid) / curves.value("mc_no_saturation", e_grid),
        color="tab:red",
        lw=1.2,
        label="AMS data / AMS MC without saturation",
    )
    bottom.plot(
        e_grid,
        curves.value("mc_saturation", e_grid)
        / curves.value("mc_no_saturation", e_grid),
        color="tab:green",
        lw=1.2,
        label="AMS MC with saturation / without",
    )
    top.set_ylabel("maximum cell energy [calibrated GeV]")
    bottom.set_ylabel("saturated / unsaturated")
    bottom.set_xlabel("shower energy [GeV] (ours: primary energy)")
    top.set_xscale("log")
    top.set_xlim(40, 1100)
    top.legend(fontsize=7, loc="upper left")
    bottom.legend(fontsize=7, loc="lower left")
    top.grid(alpha=0.3)
    bottom.grid(alpha=0.3)
    fig.suptitle(
        "EXP-017: effective fibre saturation\n" + LABEL,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def run(
    figure_png: Path, output_dir: Path = OUTPUT_DIR, root: Path = PROJECT_ROOT
) -> dict:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    d0 = digitise_figure(figure_png)
    write_digitised_csv(
        output_dir / "digitised.csv", d0["rows_a"], d0["rows_b"], d0["adopted"]
    )
    draw_overlay(
        d0["rgb"],
        d0["markers_a"],
        d0["markers_b"],
        output_dir / "digitisation_overlay.png",
        zoom=1.6,
    )
    draw_overlay(
        d0["rgb"],
        d0["markers_a"],
        d0["markers_b"],
        output_dir / "digitisation_overlay_zoom_low_energy.png",
        crop=(120, 380, 330, 500),
        zoom=4,
    )
    draw_overlay(
        d0["rgb"],
        d0["markers_a"],
        d0["markers_b"],
        output_dir / "digitisation_overlay_zoom_high_energy.png",
        crop=(560, 900, 60, 330),
        zoom=4,
    )
    curves = AmsCurves(d0["adopted"])

    normal, inclined = load_event_groups(root)
    d1 = d1_table(normal + inclined, curves)
    d1_normal = [r for r in d1 if r["sample"] != "realistic_regime_pilot_electron"]
    gated = [r for r in d1_normal if r["ams_range"]]
    d1_gate = {
        "tolerance": D1_TOLERANCE,
        "per_convention": {
            c: {
                "points": [
                    {"E_gev": r["E_gev"], "ratio": r[f"conv_{c}"]["ratio_mean_to_ams"]}
                    for r in gated
                ],
                "max_abs_deviation": max(
                    abs(r[f"conv_{c}"]["ratio_mean_to_ams"] - 1.0) for r in gated
                ),
                "passed": all(
                    abs(r[f"conv_{c}"]["ratio_mean_to_ams"] - 1.0) < D1_TOLERANCE
                    for r in gated
                ),
            }
            for c in CONVENTIONS
        },
        "gated_energies_gev": [r["E_gev"] for r in gated],
        "note": "100 GeV lies about 5% below the lowest AMS marker (extrapolated) and 50 GeV outside the AMS range: not gated",
    }

    fit_groups = [g for g in normal if curves.in_range(g.energy_gev)]
    inclined_fit_groups = [g for g in inclined if curves.in_range(g.energy_gev)]
    fits = run_fits(
        fit_groups, d0["adopted"], d0["rows_a"], d0["rows_b"], inclined_fit_groups
    )
    pooled = fits["central_inputs"].pooled()

    central = fits["central"]
    mc_sat_check = {}
    for model in MODELS:
        inputs = fits["central_inputs"]
        chi2 = inputs.chi2(model, central[model]["parameter"], series="mc_saturation")
        mc_sat_check[model] = {
            "chi2": chi2,
            "n_points": len(fit_groups),
            "chi2_per_point": chi2 / len(fit_groups),
            "note": "no parameter is fitted to this curve; sigma = digitised MC-with-saturation bar half-length plus our statistical error",
        }
    gates_d2 = {
        m: bool(
            central[m]["chi2_per_dof"] is not None and central[m]["chi2_per_dof"] < 2.0
        )
        for m in MODELS
    }
    preferred = "M1" if gates_d2["M1"] else ("M2" if gates_d2["M2"] else None)
    hottest_cell_figure(
        output_dir / "hottest_cell_curves.png", curves, normal, inclined, fits, pooled
    )

    f_central = fits["central_inputs"].f()
    result = {
        "experiment": "EXP-017",
        "label": LABEL,
        "source_figure": SOURCE_FIGURE,
        "provenance": {
            "fit_target": "digitised AMS DATA curve (maximum cell energy vs energy); not an AMS constant",
            "ams_statement": "AMS reports fibre saturation consistent with Birks' law, parametrised vs energy density; no constants published",
            "reason_for_effective_law": "classic per-step Birks cannot give an energy-density dependence for near-MIP shower particles; "
            "e_s (M1) applies to each fibre's full-length integrated deposit, not to a local density",
            "error_bars": "digitised bars are symmetric; they may be distribution widths rather than errors of the mean, so chi-square is conservative",
            "x_axis": "ours: primary energy (AMS: reconstructed energy, close to primary for contained showers)",
            "sampling_fraction_central": "A: f = sum(readout_grid_mev) / sum(edep_total_mev), ratio of sums per energy group; "
            "calibrated cell energy = scintillator MeV / f",
        },
        "d0": {
            "calibration": d0["calibration"].__dict__,
            "gate": d0["gate"],
            "files": ["digitised.csv", "digitisation_overlay.png"],
        },
        "shape_check_saturated_over_unsaturated": {
            "energies_gev": [g.energy_gev for g in fit_groups],
            "ours_M1": [
                float(p / n)
                for p, n in zip(
                    central["M1"]["predicted_gev"],
                    fits["central_inputs"].no_saturation_mean_gev(),
                    strict=True,
                )
            ],
            "ours_M2": [
                float(p / n)
                for p, n in zip(
                    central["M2"]["predicted_gev"],
                    fits["central_inputs"].no_saturation_mean_gev(),
                    strict=True,
                )
            ],
            "ams_data_over_mc_no_saturation": [
                float(
                    curves.value("data", g.energy_gev)
                    / curves.value("mc_no_saturation", g.energy_gev)
                )
                for g in fit_groups
            ],
            "ams_mc_saturation_over_mc_no_saturation": [
                float(
                    curves.value("mc_saturation", g.energy_gev)
                    / curves.value("mc_no_saturation", g.energy_gev)
                )
                for g in fit_groups
            ],
            "note": "a one-parameter law tuned near 800 GeV over-suppresses by about 4-6% at 200-400 GeV; the implied E_s of the AMS curves falls monotonically with energy",
        },
        "caveats": [
            "D2 gate (chi2/dof < 2) passes but has little power: 3 energy points, digitised bars about 10-15% of the value, possibly distribution widths",
            "the two digitisation passes share bin finding, bar removal, template and half-length rule; the pass systematic understates digitisation uncertainty",
            "the data half-length uses the smaller of the two bar ends; variation (a) does not test this choice",
            "the blue (MC without saturation) series has 19 of 20 markers and the six lowest-energy ones, including 189 and 212 GeV, are flagged occluded; the D1 agreement at 200 GeV rests on them",
            "primary fit uses no AMS point above 800 GeV (our simulated energies stop at 800 GeV)",
            "normal incidence versus AMS acceptance: the theta <= 12 deg sample shifts the parameters (see systematics not_in_quadrature)",
            "x is primary energy; AMS x is reconstructed energy",
            "e_s applies to a fibre's full-length integrated deposit, not to a local energy density",
            "the overlay PNGs embed the thesis raster figure: check redistribution rights before committing or sharing the results directory",
        ],
        "implied_saturation_scale_from_ams_curves": implied_saturation_scale(curves),
        "d1": {"table": d1, "gate": d1_gate},
        "d2": {
            "fit_groups": [g.name for g in fit_groups],
            "fit_group_energies_gev": [g.energy_gev for g in fit_groups],
            "f_central": list(f_central),
            "ams_data_at_fit_energies_gev": list(
                curves.value("data", fits["central_inputs"].energies())
            ),
            "ams_data_sigma_gev": list(fits["central_inputs"].sigma("data")),
            "central": central,
            "gate_chi2_per_dof_below_2": gates_d2,
            "preferred_model": preferred,
            "variations": fits["variations"],
            "realistic_theta12_sample": fits["realistic_theta12_sample"],
            "systematics": fits["systematics"],
            "check_against_ams_mc_with_saturation": mc_sat_check,
            "parameter_bounds": fits["bounds"],
        },
        "units": {
            "M1_parameter": "fibre-MeV (e_s applied to each fibre's scintillator deposit)",
            "M2_parameter": "calibrated cell GeV (E_s applied to the cell energy after division by f)",
        },
    }
    (output_dir / "exp017_result.json").write_text(
        json.dumps(jsonable(result), indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--figure",
        type=Path,
        help="PNG of the thesis figure; extracted from the PDF when omitted",
    )
    args = parser.parse_args(argv)
    if args.figure is None:
        import tempfile

        figure = extract_figure(
            THESIS_PDF, THESIS_FIGURE_PAGE, Path(tempfile.mkdtemp(prefix="exp017_"))
        )
    else:
        figure = args.figure
    run(figure)


if __name__ == "__main__":
    main()
