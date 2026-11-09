"""EXP-017 tests: digitisation machinery on synthetic figures, the M1/M2 laws, the fit, and the optional readout step."""

import itertools

import numpy as np
import pytest

from ams_ecal.detector import fibre_saturation_calibration as F
from ams_ecal.detector.readout_response import (
    FIBRE_SATURATION_SCALE_EXP017_MEV,
    PROVENANCE,
    ReadoutResponseConfig,
    apply_fibre_saturation,
    apply_readout_response,
)

# ---- synthetic figure -------------------------------------------------------------------------------------------

LEFT, RIGHT, TOP, BOTTOM = 40, 440, 20, 320  # frame line centres (pixels)
X_PER_GEV = (RIGHT - LEFT) / 900.0  # 100 .. 1000 GeV
Y_PER_GEV = (BOTTOM - TOP) / 60.0  # 0 .. 60 GeV


def col_of(energy: float) -> float:
    return LEFT + (energy - 100.0) * X_PER_GEV


def row_of(value: float) -> float:
    return BOTTOM - value * Y_PER_GEV


def synthetic_figure(
    markers: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """White image with a 2-px frame, minor/major ticks, and red disks with error bars (energy, value, half-length)."""

    img = np.full((360, 480, 3), 255.0)
    img[TOP - 1 : TOP + 1, LEFT : RIGHT + 1] = 0.0
    img[BOTTOM : BOTTOM + 2, LEFT : RIGHT + 1] = 0.0
    img[TOP : BOTTOM + 1, LEFT - 1 : LEFT + 1] = 0.0
    img[TOP : BOTTOM + 1, RIGHT : RIGHT + 2] = 0.0
    for energy in np.arange(150.0, 1000.0, 50.0):
        c = round(col_of(energy))
        length = 14 if energy % 200 == 0 else 6
        img[BOTTOM - length : BOTTOM, c : c + 2] = 0.0
    for value in np.arange(5.0, 60.0, 5.0):
        r = round(row_of(value))
        length = 20 if value % 20 == 0 else 9
        img[r : r + 2, LEFT + 1 : LEFT + 1 + length] = 0.0
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    for energy, value, half in markers or []:
        c, r = col_of(energy), row_of(value)
        h = half * Y_PER_GEV
        img[int(r - h) : int(r + h) + 1, int(c) : int(c) + 2] = (255.0, 0.0, 23.0)
        disk = (xx - (c + 0.5)) ** 2 + (yy - r) ** 2 <= 4.0**2
        img[disk] = (255.0, 0.0, 23.0)
    return img


def test_axis_calibration_recovers_the_tick_scale_and_reproduces_minor_ticks() -> None:
    cal = F.calibrate_axes(synthetic_figure())
    assert cal.x_value(col_of(600.0) + 0.5) == pytest.approx(600.0, abs=1.5)
    assert cal.y_value(row_of(30.0) + 0.5) == pytest.approx(30.0, abs=0.3)
    assert cal.x_residual_minor_px < 1.0
    assert cal.y_residual_minor_px < 1.0
    assert cal.x_scale == pytest.approx(1.0 / X_PER_GEV, rel=0.01)


def test_series_alpha_separates_the_three_colours() -> None:
    rgb = np.array(
        [
            [
                [255.0, 0.0, 23.0],
                [0.0, 152.0, 46.0],
                [7.0, 32.0, 244.0],
                [255.0, 255.0, 255.0],
            ]
        ]
    )
    a = F.series_alpha(rgb)
    assert a["data"][0, 0] == pytest.approx(1.0) and a["data"][0, 1:].max() < 0.5
    assert (
        a["mc_saturation"][0, 1] == pytest.approx(1.0)
        and a["mc_saturation"][0, 0] == 0.0
    )
    assert (
        a["mc_no_saturation"][0, 2] == pytest.approx(1.0)
        and a["mc_no_saturation"][0, 3] == 0.0
    )


def test_digitisation_of_red_disks_recovers_value_and_error_bar_in_both_passes() -> (
    None
):
    truth = [(300.0, 10.0, 1.5), (500.0, 20.0, 3.0), (800.0, 22.0, 1.0)]
    rgb = synthetic_figure(truth)
    cal = F.calibrate_axes(rgb)
    frame = F.frame_box(rgb)
    for marker_pass in ("A", "B"):
        markers = [
            m
            for m in F.digitise_pass(rgb, cal, marker_pass, frame, expected_bins=3)
            if m.series == "data"
        ]
        rows = F.markers_to_rows(markers, cal)
        assert len(rows) == 3
        for row, (energy, value, half) in zip(rows, truth, strict=True):
            assert row["E_gev"] == pytest.approx(energy, abs=3.0)
            assert row["y_gev"] == pytest.approx(value, abs=0.25)
            assert row["y_high"] - row["y_low"] == pytest.approx(2 * half, abs=0.6)


def test_adopt_curves_uses_the_smaller_end_for_data_and_flags_truncated_mc_bars() -> (
    None
):
    def row(series, y, low, high, pass_):
        return {
            "series": series,
            "bin": 0,
            "E_gev": 400.0,
            "y_gev": y,
            "y_low": low,
            "y_high": high,
            "pass": pass_,
            "marker_occluded": False,
            "bar_end_occluded": True,
        }

    data = F.adopt_curves(
        [row("data", 20.0, 17.5, 25.0, "A")], [row("data", 20.0, 17.5, 25.0, "B")]
    )[0]
    assert data["half_length_gev"] == pytest.approx(
        2.5
    )  # the upper end overshoots into mixed colours
    mc = F.adopt_curves(
        [row("mc_saturation", 20.0, 19.9, 22.0, "A")],
        [row("mc_saturation", 20.0, 19.9, 22.0, "B")],
    )[0]
    assert mc["half_length_unreliable"] is True and mc[
        "half_length_gev"
    ] == pytest.approx(2.0)


# ---- the two laws -----------------------------------------------------------------------------------------------


def group_of(
    fibres_per_event: list[list[tuple[int, float]]], energy_gev: float, name: str = "g"
) -> F.EventGroup:
    """Synthetic group: each event is a list of (flat cell, fibre MeV)."""

    event, cell, mev = [], [], []
    for k, fibres in enumerate(fibres_per_event):
        for c, e in fibres:
            event.append(k)
            cell.append(c)
            mev.append(e)
    event, cell, mev = np.array(event), np.array(cell), np.array(mev, float)
    n = len(fibres_per_event)
    hot = F._hot_per_event(event, cell, mev, n)
    scint = np.array([sum(e for _, e in fibres) for fibres in fibres_per_event])
    return F.EventGroup(
        name=name,
        sample="synthetic",
        selection="synthetic",
        energy_gev=energy_gev,
        primary_mev=np.full(n, energy_gev * 1000.0),
        scint_sum_mev=scint,
        edep_total_mev=scint / 0.07,
        hot_unsaturated_mev=hot,
        event_of_fibre=event,
        cell_of_fibre=cell,
        fibre_mev=mev,
    )


def test_m1_saturates_each_fibre_and_m2_saturates_the_cell_sum() -> None:
    g = group_of([[(5, 100.0), (5, 100.0), (9, 30.0)]], 100.0)
    e_s = 200.0
    m1 = F.hot_m1_mev(g, e_s)[0]
    assert m1 == pytest.approx(2 * 100.0 / (1 + 100.0 / e_s))
    m2 = F.saturate_cell_gev(np.array([200.0]), e_s)[0]
    assert m2 == pytest.approx(200.0 / (1 + 200.0 / e_s))
    assert (
        m1 > m2
    )  # splitting the same energy over several fibres saturates less than saturating the sum
    assert F.hot_m1_mev(g, None)[0] == pytest.approx(200.0)
    assert F.hot_m1_mev(g, 1e12)[0] == pytest.approx(200.0, rel=1e-6)


def test_sampling_fraction_conventions() -> None:
    g = group_of([[(0, 70.0)], [(0, 70.0)]], 1.0)
    assert g.sampling_fraction("A") == pytest.approx(
        0.07
    )  # scintillator / total deposit
    assert g.sampling_fraction("B") == pytest.approx(
        140.0 / 2000.0
    )  # scintillator / primary (leakage included)
    assert g.sampling_fraction("A_global", pooled=0.05) == 0.05
    with pytest.raises(ValueError):
        g.sampling_fraction("C")


def curve_rows(
    values: dict[str, np.ndarray], energies, half: float = 0.5
) -> F.AmsCurves:
    return F.AmsCurves(
        [
            {
                "series": series,
                "E_gev": float(e),
                "y_gev": float(y),
                "half_length_gev": half,
                "marker_occluded": False,
            }
            for series, ys in values.items()
            for e, y in zip(energies, ys, strict=True)
        ]
    )


def test_fit_recovers_a_known_per_cell_scale_from_noise_free_curves() -> None:
    f = 0.07
    energies = (200.0, 400.0, 800.0)
    groups = []
    for energy in energies:
        hot_gev = energy / 20.0  # calibrated GeV
        spread = [hot_gev * (1.0 + 0.05 * k) for k in (-1, 0, 1)]
        g = group_of([[(3, h * f * 1000.0)] for h in spread], energy, f"E{energy:.0f}")
        g.edep_total_mev = g.scint_sum_mev / f
        groups.append(g)
    true_scale = 90.0
    target = np.array(
        [
            F.saturate_cell_gev(
                F.calibrated_hot_gev(g.hot_unsaturated_mev, f), true_scale
            ).mean()
            for g in groups
        ]
    )
    no_sat = np.array(
        [F.calibrated_hot_gev(g.hot_unsaturated_mev, f).mean() for g in groups]
    )
    curves = curve_rows(
        {"data": target, "mc_saturation": target, "mc_no_saturation": no_sat},
        energies,
        half=0.3,
    )
    result = F.fit_parameter(F.FitInputs(groups, "A", curves), "M2", (2.0, 5e3))
    assert result["parameter"] == pytest.approx(true_scale, rel=1e-3)
    assert result["chi2"] < 1e-6
    lo, hi = result["interval_68"]
    assert lo < true_scale < hi


def test_m1_curve_decreases_monotonically_with_the_scale() -> None:
    groups = [
        group_of([[(1, 80.0), (1, 60.0), (2, 20.0)], [(4, 120.0), (4, 10.0)]], 400.0)
    ]
    energies = (400.0, 500.0, 600.0, 700.0, 800.0)
    curves = curve_rows({s: np.linspace(5, 30, 5) for s in F.SERIES}, energies)
    inputs = F.FitInputs(groups, "A", curves)
    inputs.prepare_m1(n_scales=30, lo_mev=5.0, hi_mev=1e4)
    values = [inputs.predict("M1", s)[0] for s in (10.0, 50.0, 200.0, 1000.0)]
    assert all(a < b for a, b in itertools.pairwise(values))


def test_implied_saturation_scale_is_constant_for_an_exact_one_parameter_law() -> None:
    energies = np.linspace(100.0, 900.0, 9)
    no_sat = energies / 20.0
    sat = no_sat / (1 + no_sat / 120.0)
    curves = curve_rows(
        {"mc_no_saturation": no_sat, "data": sat, "mc_saturation": sat}, energies
    )
    out = F.implied_saturation_scale(curves)
    assert all(x["E_s_implied_data_gev"] == pytest.approx(120.0, rel=1e-6) for x in out)


def test_the_label_is_the_required_wording() -> None:
    assert (
        F.LABEL
        == "project phenomenological calibration derived from a published figure; not the AMS parametrisation"
    )


# ---- optional step in readout_response ---------------------------------------------------------------------------


def test_fibre_saturation_step_is_off_by_default_and_matches_the_law() -> None:
    fibres = np.array([0.0, 10.0, 260.0, 1000.0])
    assert apply_fibre_saturation(fibres) is fibres  # default: nothing happens
    out = apply_fibre_saturation(fibres, FIBRE_SATURATION_SCALE_EXP017_MEV)
    assert out[0] == 0.0
    assert out[2] == pytest.approx(130.0)  # e = e_s halves
    assert np.all(np.diff(out) > 0) and np.all(out <= fibres)
    with pytest.raises(ValueError):
        apply_fibre_saturation(fibres, 0.0)


def test_adding_the_step_does_not_change_the_existing_defaults() -> None:
    config = ReadoutResponseConfig()
    grid = np.random.default_rng(0).exponential(2.0, size=(3, 18, 72))
    assert np.array_equal(
        apply_readout_response(grid, config, np.random.default_rng(1)), grid
    )
    assert PROVENANCE["fibre_saturation_scale_mev"].startswith(
        "project phenomenological calibration"
    )
    assert set(config.labelled()) == {
        "pe_per_mip",
        "gain_sigma",
        "crosstalk_fraction",
        "threshold_mips",
        "threshold_before_gain",
    }
