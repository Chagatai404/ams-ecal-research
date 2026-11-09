"""Fast synthetic tests of the EXP-015 realistic-regime pilot analysis (no Geant4 data needed)."""

import json
import math

import numpy as np
import pytest

from ams_ecal.detector.readout_response import MIP_FIBRE_MEV, photoelectron_smearing
from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.multiscale_estimators import realistic_regime_analysis as rr

BANDS = ((50.0, 100.0), (100.0, 200.0))


def synthetic_class(name, seed, per_band=180, hits=900, interacting=True):
    """Log-uniform energies in two bands, a central fiducial entry, and dense random deposit images."""

    rng = np.random.default_rng(seed)
    lower, upper, energy, grids = [], [], [], []
    for lo, hi in BANDS:
        e = np.exp(rng.uniform(math.log(lo), math.log(hi), per_band))
        energy.append(e)
        lower += [lo] * per_band
        upper += [hi] * per_band
        for _ in range(per_band):
            grid = np.zeros((18, 72))
            cells = rng.integers(0, 18 * 72, hits + int(rng.integers(-50, 50)))
            np.add.at(grid.ravel(), cells, rng.exponential(1.0, len(cells)))
            grids.append(grid)
    energy = np.concatenate(energy)
    n = len(energy)
    grids = np.array(grids, dtype=np.float32)
    return rr.ClassEvents(
        name=name,
        readout_grid_mev=grids,
        edep_scintillator_mev=grids.sum(axis=(1, 2)).astype(float),
        edep_total_mev=energy * 900.0,
        entry_x_mm=rng.uniform(-100, 100, n),
        entry_y_mm=rng.uniform(-100, 100, n),
        primary_energy_gev=energy,
        primary_theta_rad=np.arccos(
            np.sqrt(rng.uniform(math.cos(rr.THETA_MAX_RAD) ** 2, 1, n))
        ),
        primary_phi_rad=rng.uniform(0, 2 * np.pi, n),
        truth_occurred=np.full(n, interacting),
        band_lower_gev=np.array(lower),
        band_upper_gev=np.array(upper),
        fine_voxel_count=(grids > 0).reshape(n, -1).sum(axis=1) * 3,
        generated_per_band={lo: per_band for lo, _ in BANDS},
    )


# --- 1. fiducial geometry -----------------------------------------------------------------------------------------------------------


def test_fiducial_geometry_on_hand_made_cases():
    x = np.array([0.0, 296.0, 298.0, 0.0, 290.0, -296.5])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta = np.array([0.0, 0.0, 0.0, 0.4, 0.4, 0.0])
    phi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    got = rr.fiducial_mask(x, y, theta, phi)
    # 3: x at exit = 166.5 tan 0.4 = 70.3 mm, inside; 4: 290 + 70.3 exceeds 297 at the exit; 5: -296.5 is inside the 297 limit
    assert got.tolist() == [True, True, False, True, False, True]


def test_fiducial_uses_the_exit_plane_and_the_azimuth():
    # a track entering well inside but leaving through the +y side at phi = 90 degrees
    entry = np.array([0.0])
    ok_exit = rr.fiducial_mask(
        entry, entry + 250.0, np.array([0.5]), np.array([math.pi / 2])
    )
    out_exit = rr.fiducial_mask(
        entry, entry + 250.0, np.array([0.5]), np.array([-math.pi / 2])
    )
    assert (
        not ok_exit[0] and out_exit[0]
    )  # +y: 250 + 166.5 tan(0.5) = 340 > 297; -y: 160 inside
    x, y = rr.projected_axis(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.3]),
        np.array([math.pi / 2]),
        100.0,
    )
    assert abs(x[0]) < 1e-9 and y[0] == pytest.approx(100 * math.tan(0.3))


def test_distance_to_nearest_edge():
    assert rr.distance_to_nearest_edge(np.array([300.0]), np.array([-10.0]))[
        0
    ] == pytest.approx(24.0)


# --- 3. weights ---------------------------------------------------------------------------------------------------------------------


def log_uniform_bands(rng, bands, per_band):
    lower, upper, energy = [], [], []
    for lo, hi in bands:
        energy.append(np.exp(rng.uniform(math.log(lo), math.log(hi), per_band)))
        lower += [lo] * per_band
        upper += [hi] * per_band
    return np.concatenate(energy), np.array(lower), np.array(upper)


def test_electron_weights_reproduce_the_target_spectrum():
    bands = (
        (50.0, 100.0),
        (100.0, 200.0),
        (200.0, 400.0),
        (400.0, 800.0),
        (800.0, 2000.0),
    )
    n = 200_000
    energy, lower, upper = log_uniform_bands(np.random.default_rng(3), bands, n)
    w = rr.physical_spectrum_weights(
        energy, lower, upper, {lo: n for lo, _ in bands}, rr.electron_density
    )
    assert w.sum() == pytest.approx(1.0, rel=0.01)  # 1 in expectation
    edges = np.geomspace(50, 2000, 15)
    dn = np.histogram(energy, bins=edges, weights=w)[0] / np.diff(edges)
    centres = np.sqrt(edges[1:] * edges[:-1])
    slope = np.polyfit(np.log(centres), np.log(dn), 1)[0]
    assert slope == pytest.approx(-3.18, abs=0.06)
    # band totals follow the integral of E^-3.18 over each band
    totals = np.array([w[lower == lo].sum() for lo, _ in bands])
    integral = np.array([(lo**-2.18 - hi**-2.18) / 2.18 for lo, hi in bands])
    assert totals == pytest.approx(integral / integral.sum(), rel=0.02)


def test_proton_weights_follow_the_rigidity_spectrum_and_its_break():
    bands = (
        (50.0, 100.0),
        (100.0, 200.0),
        (200.0, 400.0),
        (400.0, 800.0),
        (800.0, 2000.0),
    )
    n = 200_000
    energy, lower, upper = log_uniform_bands(np.random.default_rng(4), bands, n)
    w = rr.physical_spectrum_weights(
        energy, lower, upper, {lo: n for lo, _ in bands}, rr.proton_density
    )
    edges = np.geomspace(50, 2000, 15)
    dn = np.histogram(energy, bins=edges, weights=w)[0] / np.diff(edges)
    centres = np.sqrt(edges[1:] * edges[:-1])
    expected = rr.proton_density(centres)
    ratio = dn / expected
    assert ratio / ratio[0] == pytest.approx(np.ones_like(ratio), rel=0.05)
    # the spectrum hardens above the 331 GV break: the local slope at 1 TeV is shallower than at 60 GeV
    low = np.polyfit(np.log(centres[:3]), np.log(dn[:3]), 1)[0]
    high = np.polyfit(np.log(centres[-3:]), np.log(dn[-3:]), 1)[0]
    assert high > low and low == pytest.approx(-2.85, abs=0.15)
    assert rr.proton_rigidity_gv(np.array([1.0]))[0] == pytest.approx(
        math.sqrt(1 + 2 * 0.938272)
    )


def test_kish_effective_sample_size():
    assert rr.kish_effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert rr.kish_effective_sample_size(
        np.array([1.0, 0.0, 0.0, 0.0])
    ) == pytest.approx(1.0)
    assert rr.kish_effective_sample_size(np.array([2.0, 1.0, 1.0])) == pytest.approx(
        16 / 6
    )


# --- 4. offset averaging ------------------------------------------------------------------------------------------------------------


def test_a_single_zero_offset_reproduces_the_existing_estimator():
    rng = np.random.default_rng(1)
    image = rng.exponential(1.0, (18, 72)) * (rng.random((18, 72)) < 0.6)
    averaged = rr.offset_partition_profile(image, [(0, 0)])
    reference = ms.profile(image)
    for q in ms.Q_VALUES:
        assert averaged[q] == pytest.approx(reference[q])


def test_offset_average_over_all_offsets_is_translation_invariant():
    rng = np.random.default_rng(2)
    image = np.zeros((18, 72))
    image[6:11, 20:30] = rng.exponential(1.0, (5, 10)) * (rng.random((5, 10)) < 0.7)
    moved = np.zeros_like(image)
    moved[6:11, 21:31] = image[
        6:11, 20:30
    ]  # translate one cell along the cell axis, away from every edge
    everything = [(lo, co) for lo in range(18) for co in range(18)]
    a, b = (
        rr.offset_partition_profile(image, everything),
        rr.offset_partition_profile(moved, everything),
    )
    for q in ms.Q_VALUES:
        assert a[q] == pytest.approx(b[q], rel=1e-9, abs=1e-12)
    # a single fixed grid is NOT translation invariant for the same pair
    single_a, single_b = (
        rr.offset_partition_profile(image, [(0, 0)]),
        rr.offset_partition_profile(moved, [(0, 0)]),
    )
    assert not np.allclose(single_a[0.0], single_b[0.0])


def test_offsets_are_reproducible_and_inside_the_declared_range():
    first, second = rr.draw_offsets(17), rr.draw_offsets(17)
    assert (
        (first == second).all()
        and first.shape == (8, 2)
        and first.min() >= 0
        and first.max() < 18
    )
    assert not (rr.draw_offsets(18) == first).all()


def test_dimension_row_has_three_orders_per_registered_window_and_marks_ineligible_events():
    rng = np.random.default_rng(5)
    dense = rng.exponential(1.0, (18, 72)) * (rng.random((18, 72)) < 0.7)
    row = rr.dimension_row(dense, rr.draw_offsets(1))
    assert (
        len(row) == 3 * len(rr.WINDOWS) == len(rr.dimension_names())
        and np.isfinite(row).all()
    )
    sparse = np.zeros((18, 72))
    sparse[3, 3:13] = 1.0
    assert np.isnan(
        rr.dimension_row(sparse, rr.draw_offsets(1))
    ).all()  # fewer occupied cells than any window needs


# --- 5. matching --------------------------------------------------------------------------------------------------------------------


def test_matching_respects_the_caliper_and_is_one_to_one():
    rng = np.random.default_rng(6)
    cells_e, cells_p = rng.lognormal(6.4, 0.3, 400), rng.lognormal(6.2, 0.35, 500)
    energy_e, energy_p = (
        cells_e * rng.lognormal(3.0, 0.2, 400),
        cells_p * rng.lognormal(2.7, 0.3, 500),
    )
    ie, ip = rr.match_on_occupancy_and_energy(
        cells_e, energy_e, cells_p, energy_p, rr.CALIPER_SD
    )
    assert len(ie) > 50 and len(set(ie)) == len(ie) and len(set(ip)) == len(ip)
    ze = np.log(np.column_stack([cells_e, energy_e]))
    zp = np.log(np.column_stack([cells_p, energy_p]))
    pooled = np.vstack([ze, zp])
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0)
    distance = np.linalg.norm((ze[ie] - mean) / sd - (zp[ip] - mean) / sd, axis=1)
    assert (distance <= rr.CALIPER_SD + 1e-12).all()


def test_pairings_cover_the_same_band_and_one_or_two_steps_above():
    pairs = rr.band_pairings([50.0, 100.0, 200.0], [50.0, 100.0, 200.0, 400.0])
    assert (
        (50.0, 50.0, 0) in pairs
        and (50.0, 100.0, 1) in pairs
        and (50.0, 200.0, 2) in pairs
    )
    assert (200.0, 400.0, 1) in pairs and (200.0, 800.0, 2) not in pairs


# --- 6. statistics ------------------------------------------------------------------------------------------------------------------


def test_holm_matches_the_textbook_values_and_leaves_untestable_out():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.005])
    got = rr.holm_adjust(p)
    # sorted finite: 0.005*4, 0.01*3, 0.03*2, 0.04*1 -> 0.02, 0.03, 0.06, 0.06 (running max)
    assert got[[4, 0, 2, 1]] == pytest.approx([0.02, 0.03, 0.06, 0.06]) and np.isnan(
        got[3]
    )


def test_fewer_than_100_pairs_is_not_testable_never_no_effect():
    rng = np.random.default_rng(7)
    result = rr.run_pairing(
        50.0,
        50.0,
        0,
        np.arange(40),
        np.arange(40),
        np.ones((40, 2)),
        np.ones((40, 2)),
        lambda i: rng.random((len(i), 18)),
        lambda i: rng.random((len(i), 18)),
        None,
        None,
        200,
        rng,
    )
    assert not result.testable and result.rows == [] and "not testable" in result.reason


def test_pair_swap_p_values_are_not_small_for_identical_populations_and_small_for_shifted_ones():
    rng = np.random.default_rng(8)
    same_a, same_b = rng.normal(size=(150, 3)), rng.normal(size=(150, 3))
    null = rr.compare_matched_pairs(same_a, same_b, None, None, 300, rng)
    shifted = rr.compare_matched_pairs(same_a + 1.5, same_b, None, None, 300, rng)
    assert (null["p"] > 0.001).all() and (shifted["p"] < 0.01).all()


# --- 7. photoelectron scan ----------------------------------------------------------------------------------------------------------


def test_photoelectron_smearing_removes_faint_cells_and_preserves_the_mean():
    rng = np.random.default_rng(9)
    faint = np.full(
        (200, 18, 72), 0.1 * MIP_FIBRE_MEV
    )  # mean 0.5 p.e. at 5 p.e. per MIP
    smeared = photoelectron_smearing(faint, 5.0, rng)
    assert (smeared > 0).mean() < 0.55  # P(N >= 1) = 1 - exp(-0.5) = 0.39
    assert smeared.mean() == pytest.approx(faint.mean(), rel=0.02)
    bright = np.full((50, 18, 72), 20 * MIP_FIBRE_MEV)
    assert (photoelectron_smearing(bright, 5.0, rng) > 0).mean() > 0.999


def test_smeared_images_are_built_with_fixed_seeds_and_rematched_keys():
    events = synthetic_class("electron", 10, per_band=20)
    selected = np.ones(len(events), bool)
    a = rr.build_class_images(events, selected, "electron", 1, 5.0, draw=1)
    b = rr.build_class_images(events, selected, "electron", 1, 5.0, draw=1)
    c = rr.build_class_images(events, selected, "electron", 1, 5.0, draw=2)
    plain = rr.build_class_images(events, selected, "electron", 1, None)
    assert (a.grids == b.grids).all() and not (a.grids == c.grids).all()
    assert a.cells.mean() < plain.cells.mean()  # occupancy falls at 5 p.e. per MIP
    assert a.energy_mev == pytest.approx(
        a.grids.sum(axis=(1, 2))
    )  # keys are recomputed from the smeared image


# --- end to end ---------------------------------------------------------------------------------------------------------------------


def test_end_to_end_on_synthetic_events_writes_every_output(tmp_path):
    electrons = synthetic_class("electron", 11)
    protons = synthetic_class("proton", 12)
    protons.truth_occurred[::12] = False  # some crossing protons
    summary = rr.run_analysis(
        electrons, protons, tmp_path, data_root=None, permutations=60, run_scan=False
    )
    for name in (
        "F1_generation",
        "F2_occupancy",
        "F3_acceptance_missing_energy",
        "F4_conventional",
        "F5_matched_dimensions",
    ):
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (
            tmp_path / f"{name}.json"
        ).exists()
    for name in (
        "matched_pair_statistics.csv",
        "fiducial_acceptance.csv",
        "effective_sample_size.csv",
    ):
        assert (tmp_path / name).exists()
    assert "F5" in summary["stages_run"]
    tested = [p for p in summary["matched_pairs"] if p["testable"]]
    assert tested and all(p["pairs"] >= rr.MIN_PAIRS for p in tested)
    assert rr.EXPLORATORY_LABEL in (tmp_path / "F5_matched_dimensions.json").read_text(
        encoding="utf-8"
    )


def test_hook_for_the_three_dimensional_statistic_wires_matching_and_marks_small_samples_untestable():
    electrons, protons = (
        synthetic_class("electron", 13, per_band=10),
        synthetic_class("proton", 14, per_band=10),
    )
    out = rr.three_dimensional_hook(
        electrons, protons, lambda idx, edep, window: 0.0, (1, 2, 3)
    )
    assert out["testable"] is False and out["pairs"] < rr.MIN_PAIRS


def test_photoelectron_scan_runs_every_point_with_fixed_draws_and_draws_its_figure(
    tmp_path,
):
    electrons, protons = (
        synthetic_class("electron", 15, per_band=20),
        synthetic_class("proton", 16, per_band=20),
    )
    scan = rr.photoelectron_scan(
        electrons, protons, protons.truth_occurred.astype(bool), 20
    )
    points = [(p["pe_per_mip"], p["draw"]) for p in scan]
    assert points == [(None, 0)] + [
        (pe, d) for pe in (10.0, 8.0, 5.0) for d in range(3)
    ]
    assert all(
        not pairing["testable"] for p in scan for pairing in p["pairings"]
    )  # 20 events: never "no effect"
    rr.figure_f5b_scan(scan, tmp_path)
    assert (tmp_path / "F5b_photoelectron_scan.png").stat().st_size > 0


# --- secondary view: no DEC-013 eligibility cut ----------------------------------------------------------------------------------


def test_secondary_dimension_row_differs_from_primary_only_by_the_absence_of_the_cut():
    rng = np.random.default_rng(5)
    dense = rng.exponential(1.0, (18, 72)) * (rng.random((18, 72)) < 0.7)
    offsets = rr.draw_offsets(1)
    np.testing.assert_array_equal(  # an eligible event is identical in both views
        rr.dimension_row(dense, offsets, apply_eligibility=True),
        rr.dimension_row(dense, offsets, apply_eligibility=False),
    )
    sparse = np.zeros((18, 72))
    sparse[3, 3:13] = 1.0
    assert np.isnan(rr.dimension_row(sparse, offsets)).all()
    uncut = rr.dimension_row(sparse, offsets, apply_eligibility=False)
    assert len(uncut) == len(rr.dimension_names())
    assert not np.isnan(uncut).all()  # the cut, not the estimator, produced the NaNs
    assert np.isnan(rr.dimension_row(np.zeros((18, 72)), offsets, False)).all()


def test_secondary_view_is_labelled_has_no_holm_and_is_written(tmp_path):
    electrons = synthetic_class("electron", 11)
    protons = synthetic_class("proton", 12)
    summary = rr.run_analysis(
        electrons, protons, tmp_path, permutations=60, run_scan=False
    )
    label = "secondary, descriptive, outside DEC-013 controls"
    assert rr.SECONDARY_LABEL == label
    assert "F5 secondary" in summary["stages_run"]
    assert ("matched_pairs_" + label) in summary
    assert (tmp_path / "F5_matched_dimensions_secondary.png").stat().st_size > 0
    figure = (tmp_path / "F5_matched_dimensions_secondary.json").read_text(
        encoding="utf-8"
    )
    assert label in figure
    payload = json.loads(
        (tmp_path / "matched_pair_statistics.json").read_text(encoding="utf-8")
    )
    primary_pairs = payload["pairings"]
    secondary_pairs = payload[label]["pairings"]
    assert len(primary_pairs) == len(secondary_pairs) > 0
    for a, b in zip(primary_pairs, secondary_pairs, strict=True):
        assert a["pairs"] == b["pairs"]  # identical matching
        for row in b["rows"]:
            assert (
                row["p_holm_family"] is None
            )  # raw p only in the secondary view (NaN)
    assert label in (tmp_path / "matched_pair_statistics.csv").read_text(
        encoding="utf-8"
    )


def test_theta12_stratum_selects_exactly_the_fiducial_events_with_theta_up_to_12_degrees():
    events = synthetic_class("electron", 11)
    limit = math.radians(12.0)
    assert rr.SENSITIVITY_THETA_MAX_RAD == pytest.approx(limit)
    events.primary_theta_rad[:4] = [0.0, limit, limit + 1e-6, rr.THETA_MAX_RAD]
    everything = np.ones(len(events), bool)
    expected = np.flatnonzero(
        rr.fiducial_of(events) & (events.primary_theta_rad <= limit)
    )
    full = rr.build_class_images(events, everything, "electron", 1)
    stratum = rr.build_class_images(
        events, everything, "electron", 1, theta_max_rad=limit
    )
    assert np.array_equal(stratum.event_keys, expected + 10_000_000)
    assert 0 < len(stratum.event_keys) < len(full.event_keys)
    rows = rr.acceptance_table({"electron": (events, everything)}, limit)
    assert sum(r["accepted"] for r in rows) == len(expected)
    assert sum(r["events"] for r in rows) == int(
        (events.primary_theta_rad <= limit).sum()
    )
    assert all(r["view"] == rr.SENSITIVITY_LABEL for r in rows)


def test_theta12_sensitivity_is_labelled_written_and_leaves_the_primary_untouched(
    tmp_path,
):
    electrons = synthetic_class("electron", 11)
    protons = synthetic_class("proton", 12)
    summary = rr.run_analysis(
        electrons, protons, tmp_path, permutations=60, run_scan=False
    )
    label = "sensitivity, theta <= 12 deg"
    assert rr.SENSITIVITY_LABEL == label
    assert (tmp_path / "F5_matched_dimensions_theta12.png").stat().st_size > 0
    assert label in (tmp_path / "F5_matched_dimensions_theta12.json").read_text(
        encoding="utf-8"
    )
    assert ("matched_pairs_" + label) in summary
    assert ("fiducial_acceptance_" + label) in summary
    payload = json.loads(
        (tmp_path / "matched_pair_statistics.json").read_text(encoding="utf-8")
    )
    assert len(payload[label]["pairings"]) > 0
    assert label in (tmp_path / "matched_pair_statistics.csv").read_text(
        encoding="utf-8"
    )
    assert label in (tmp_path / "fiducial_acceptance_theta12.csv").read_text(
        encoding="utf-8"
    )
    # the full-sample primary uses every fiducial event, more than the stratum
    assert (
        summary["matched_pairs"][0]["pairs"]
        >= summary["matched_pairs_" + label][0]["pairs"]
    )


# --- POST HOC pooled-proton matching ------------------------------------------------------------------------------------------------


def test_pooled_matching_uses_all_proton_bands_and_is_one_to_one_within_each_pairing():
    e_img = rr.build_class_images(
        synthetic_class("electron", 11), np.ones(360, bool), "electron", 1
    )
    p_img = rr.build_class_images(
        synthetic_class("proton", 12), np.ones(360, bool), "proton_interacting", 2
    )
    pooled = rr.analyse_pooled(
        e_img, p_img, 30, np.random.default_rng(0), with_excess=False
    )
    assert [p.label for p in pooled] == ["P-band e50", "P-band e100", "P-all"]
    for p in pooled:
        assert len(set(p.proton_index.tolist())) == len(
            p.proton_index
        )  # each proton at most once
        assert len(set(p.electron_index.tolist())) == len(p.electron_index)
        assert p.result.pairs == len(p.electron_index)
        assert p.energy_ratio["pairs"] == len(p.electron_index)
    # the pool reaches protons of BOTH bands (the pre-stated pairings would have used one band only)
    for p in pooled:
        assert set(p_img.band_lower_gev[p.proton_index]) == {50.0, 100.0}
    all_pairing = pooled[-1]
    assert sum(c["pairs"] for c in all_pairing.combinations) == all_pairing.result.pairs
    assert {
        (c["electron_band_gev"], c["proton_band_gev"]) for c in all_pairing.combinations
    } == {(50.0, 50.0), (50.0, 100.0), (100.0, 50.0), (100.0, 100.0)}
    assert set(e_img.band_lower_gev[pooled[0].electron_index]) == {50.0}


def test_post_hoc_pooled_outputs_carry_the_label_and_write_nothing_of_the_pre_stated_analysis(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rr, "PHOTOELECTRON_SCAN", (None, 10.0))
    monkeypatch.setattr(rr, "SCAN_DRAWS", 1)
    electrons = synthetic_class("electron", 11, per_band=45)
    protons = synthetic_class("proton", 12, per_band=45)
    pooled_dir = tmp_path / "pooled"
    summary = rr.run_post_hoc_pooled(
        electrons, protons, pooled_dir, permutations=40, scan_permutations=20
    )
    label = "POST HOC, EXPLORATORY: pooled-proton matching chosen after the pre-stated pairings were not testable; no claim"
    assert rr.POST_HOC_LABEL == label and summary["status"] == label
    for name in (
        "pooled_F5_matched_dimensions",
        "pooled_F5_matched_dimensions_secondary",
        "pooled_F5_matched_dimensions_theta12",
        "pooled_balance_energy_ratio",
        "pooled_F5b_photoelectron_scan",
    ):
        assert (pooled_dir / f"{name}.png").stat().st_size > 0
        assert label in (pooled_dir / f"{name}.json").read_text(encoding="utf-8")
    for name in (
        "pooled_matched_pair_statistics.json",
        "pooled_summary.json",
        "pooled_photoelectron_scan.json",
    ):
        assert label in (pooled_dir / name).read_text(encoding="utf-8")
    assert label in (pooled_dir / "pooled_matched_pair_statistics.csv").read_text(
        encoding="utf-8"
    )
    payload = json.loads(
        (pooled_dir / "pooled_matched_pair_statistics.json").read_text(encoding="utf-8")
    )
    assert [p["pairing"] for p in payload[rr.SENSITIVITY_LABEL]["pairings"]] == [
        "P-all"
    ]
    for p in payload[rr.SECONDARY_LABEL]["pairings"]:
        assert all(row["p_holm_family"] is None for row in p["rows"])
    assert not (pooled_dir / "matched_pair_statistics.json").exists()
    assert not (pooled_dir / "F5_matched_dimensions.png").exists()
    assert not (pooled_dir / "photoelectron_scan.json").exists()


def test_cli_flag_runs_only_the_pooled_analysis_and_is_off_by_default(
    tmp_path, monkeypatch
):
    events = {
        "electron": synthetic_class("electron", 11, per_band=45),
        "proton": synthetic_class("proton", 12, per_band=45),
    }
    monkeypatch.setattr(
        rr, "find_class_directory", lambda root, name, rung: tmp_path / name
    )
    monkeypatch.setattr(rr, "load_class_events", lambda directory, name: events[name])
    flagged, plain = tmp_path / "flag", tmp_path / "plain"
    base = [
        "--data-root",
        str(tmp_path),
        "--rung",
        "rung1",
        "--permutations",
        "30",
        "--skip-scan",
    ]
    rr.main([*base, "--out", str(flagged), "--post-hoc-pooled"])
    assert (flagged / "pooled_summary.json").exists() and not (
        flagged / "F1_generation.png"
    ).exists()
    rr.main([*base, "--out", str(plain)])
    assert (plain / "F1_generation.png").exists() and not (
        plain / "pooled_summary.json"
    ).exists()
