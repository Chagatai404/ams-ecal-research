"""The pilot report runs end to end on small synthetic batches.

The synthetic events are crude on purpose - a straight track for protons that
do not interact, a longitudinal blob that grows with remaining depth for those
that do - so the report's plumbing and its known answers can be checked
without Geant4: e.g. the fitted interaction length and the sign of the depth
correlations.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.pilot_report import build_report

LAMBDA_MM = 270.0


def synthetic_batch(directory: Path, energy: float, n: int, seed: int, layers: int, physics: str) -> None:
    rng = np.random.default_rng(seed)
    depth = rng.exponential(LAMBDA_MM, n)
    occurred = depth < 166.5 if layers == 18 else depth < 2497.5
    grid = np.zeros((n, layers, 72), np.float32)
    grid[:, :, 36] = rng.gamma(4.0, 0.15, (n, layers))  # through-going track
    for i in np.flatnonzero(occurred):
        start = int(depth[i] // 9.25)
        amount = energy * 3.0 * rng.uniform(0.5, 1.5)
        grid[i, start : start + 6, 35:38] += amount / 18.0
    readout = grid[:, :18]
    arrays = {
        "event_index": np.arange(n),
        "seed": np.arange(n),
        "entry_x_mm": rng.uniform(0, 9, n),
        "entry_y_mm": rng.uniform(0, 9, n),
        "readout_grid_mev": readout,
        "edep_scintillator_mev": readout.sum(axis=(1, 2)).astype(float),
        "edep_total_mev": 13.0 * readout.sum(axis=(1, 2)).astype(float),
        "truth_occurred": occurred & (depth < 166.5) if layers == 18 else occurred,
        "truth_z_mm": np.where(occurred, depth, np.nan),
        "truth_material": np.where(occurred, "ECAL_matrix", ""),
        "truth_leading_kinetic_energy_mev": np.full(n, 0.3 * energy * 1000),
        "truth_primary_kinetic_energy_before_mev": np.full(n, energy * 1000),
        "truth_primary_elastic_scatters": np.zeros(n, int),
        "truth_n_secondaries": np.full(n, 30),
    }
    if layers > 18:
        arrays["extended_readout_grid_mev"] = grid
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "events.npz", **arrays)
    meta = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "energy_gev": energy,
        "materials": {"prefix_depth_lambda_i": 0.618, "fibre_volume_fraction": 0.314,
                      "fibre_mass_fraction": 0.049},
        "git": {"commit": "synthetic", "tracked_changes": False},
        "geant4_version": "synthetic",
        "physics_list": physics,
        "n_events": n,
        "configuration_sha256": "0" * 64,
        "production_cut_mm": 0.7,
        "created_utc": "2026-09-29T00:00:00Z",
    }
    (directory / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    for energy in (10.0, 50.0):
        synthetic_batch(data / "baseline" / f"E{energy:g}GeV", energy, 3000, 1, 18, "FTFP_BERT")
        synthetic_batch(data / "alternate" / f"E{energy:g}GeV", energy, 800, 2, 18, "QBBC")
        synthetic_batch(data / "fixed_entry" / f"E{energy:g}GeV", energy, 800, 3, 18, "FTFP_BERT")
        synthetic_batch(data / "extended" / f"E{energy:g}GeV", energy, 600, 4, 270, "FTFP_BERT")
    out = tmp_path_factory.mktemp("results")
    return build_report(data, out), out


def test_every_section_and_output_is_written(summary) -> None:
    result, out = summary
    for section in ("interaction", "detector", "p7", "truncation", "physics_list",
                    "fixed_entry", "threshold_sensitivity", "analysis_choices", "provenance",
                    "model_inputs"):
        assert section in result
    assert (out / "summary.json").is_file()
    for name in result["figures"]:
        assert (out / name).is_file()
    assert len(result["figures"]) == 9


def test_the_interaction_length_is_recovered(summary) -> None:
    result, _ = summary
    fit = result["interaction"]["10"]["effective_interaction_length_mm"]
    assert fit["estimate"] == pytest.approx(LAMBDA_MM, abs=4 * fit["standard_error"])
    p = result["interaction"]["10"]["p_no_inelastic"]
    assert p["low"] < np.exp(-166.5 / LAMBDA_MM) < p["high"]


def test_depth_drives_the_synthetic_longitudinal_centre(summary) -> None:
    result, _ = summary
    cog = result["p7"]["10"]["observables"]["long_cog_mm"]
    assert cog["s_d_epsilon_squared"]["estimate"] > 0.5
    assert cog["spearman_with_depth"] > 0.5


def test_truncation_compares_the_same_events(summary) -> None:
    result, _ = summary
    entry = result["truncation"]["10"]
    assert entry["n_interacting_in_prefix"] > 50
    corr = entry["depth_correlations"]["long_cog_mm"]
    assert corr["spearman_thin"]["estimate"] > 0 and corr["spearman_full"]["estimate"] > 0
