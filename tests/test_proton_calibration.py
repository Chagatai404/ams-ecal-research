"""The proton model calibration / validation split, crossing table and artifact."""

import json

import numpy as np
import pytest

from ams_ecal.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.proton_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CrossingTable,
    ProtonCalibration,
    build_calibration,
    calibration_mask,
    content_sha256,
    validation_mask,
)

DEPTH_MM = 166.5
TRUE_LENGTH_MM = 255.0


# --- the split ---------------------------------------------------------


def test_the_split_partitions_every_event_exactly_once() -> None:
    index = np.arange(4000)

    validation = validation_mask(index)
    calibration = calibration_mask(index)

    assert not np.any(validation & calibration)
    assert np.all(validation | calibration)


def test_a_quarter_of_the_events_are_held_out() -> None:
    assert validation_mask(np.arange(4000)).sum() == 1000


def test_the_rule_depends_only_on_the_event_index() -> None:
    index = np.array([0, 1, 2, 3, 4, 7, 11, 12])

    assert validation_mask(index).tolist() == [
        False, False, False, True, False, True, True, False,
    ]
    # The same event lands on the same side whatever else is in the batch.
    assert validation_mask(index[[3]]).tolist() == [True]
    assert validation_mask(index[::-1]).tolist() == validation_mask(index).tolist()[::-1]


def test_a_shorter_batch_keeps_the_status_of_the_events_it_shares() -> None:
    # A fixed-entry control reuses the leading seeds of the baseline batch, so
    # its events must be held out exactly when their baseline twins are.
    baseline = validation_mask(np.arange(4000))
    fixed_entry = validation_mask(np.arange(1000))

    assert np.array_equal(baseline[:1000], fixed_entry)


def test_the_split_is_interleaved_not_blocked() -> None:
    validation = validation_mask(np.arange(4000))

    assert validation[:2000].sum() == validation[2000:].sum() == 500


def test_a_non_integer_index_is_rejected() -> None:
    with pytest.raises(TypeError):
        validation_mask(np.array([0.0, 1.0, 3.0]))


def test_a_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        validation_mask(np.array([-1, 0, 1]))


# --- the crossing table -------------------------------------------------


def make_table(**overrides) -> CrossingTable:
    levels = np.array([0.0, 0.5, 1.0])
    base = np.array([[0.1, 0.5, 1.0], [0.2, 0.6, 1.2], [0.3, 0.7, 1.4]])
    quantiles = np.stack([base, 2.0 * base])  # (energy, chord bin, level)
    fields = {
        "energies_gev": np.array([10.0, 100.0]),
        "chord_edges_mm": np.array([2.0, 3.0]),
        "levels": levels,
        "quantiles_mev": {"readout": quantiles, "deposition": 10.0 * quantiles},
        "counts": np.full((2, 3), 100),
    }
    fields.update(overrides)
    return CrossingTable(**fields)


def test_energy_weights_interpolate_in_the_logarithm_of_the_energy() -> None:
    table = make_table()

    assert table.energy_weights(10.0) == (0, 1, 0.0)
    assert table.energy_weights(100.0) == (0, 1, 1.0)
    assert table.energy_weights(float(np.sqrt(1000.0)))[2] == pytest.approx(0.5)


@pytest.mark.parametrize("energy", [9.99, 100.01, 0.0, 1e6])
def test_energies_outside_the_calibrated_range_raise(energy) -> None:
    with pytest.raises(ValueError, match="nothing is extrapolated"):
        make_table().energy_weights(energy)


def test_chord_bins_follow_the_interior_edges() -> None:
    bins = make_table().chord_bin(np.array([1.0, 1.999, 2.0, 2.5, 3.0, 4.0]))

    assert bins.tolist() == [0, 0, 1, 1, 2, 2]


def test_the_quantile_function_is_interpolated_value_by_value() -> None:
    vector = make_table().quantile_vector("readout", float(np.sqrt(1000.0)), 2.5)

    assert np.allclose(vector, 1.5 * np.array([0.2, 0.6, 1.2]))


def test_sampling_interpolates_between_levels_and_is_monotone_in_the_variate() -> None:
    table = make_table()
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    chord = np.full(5, 2.5)

    energies = table.sample_layer_energies("readout", 10.0, chord, u)

    assert np.allclose(energies, [0.2, 0.4, 0.6, 0.9, 1.2])
    assert np.all(np.diff(energies) > 0)


def test_the_two_representations_use_their_own_tables() -> None:
    table = make_table()
    args = (10.0, np.array([2.5]), np.array([0.5]))

    assert table.sample_layer_energies("deposition", *args)[0] == pytest.approx(
        10.0 * table.sample_layer_energies("readout", *args)[0]
    )


def test_the_draw_stays_inside_the_calibrated_support() -> None:
    table = make_table()
    u = np.random.default_rng(0).uniform(size=500)
    energies = table.sample_layer_energies("readout", 40.0, np.full(500, 3.5), u)

    vector = table.quantile_vector("readout", 40.0, 3.5)
    assert energies.min() >= vector[0] and energies.max() <= vector[-1]


def test_a_variate_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="uniform"):
        make_table().sample_layer_energies("readout", 10.0, np.array([2.5]), np.array([1.2]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"energies_gev": np.array([10.0])}, "two"),
        ({"chord_edges_mm": np.array([3.0, 2.0])}, "chord edges"),
        ({"levels": np.array([0.1, 0.5, 1.0])}, "levels"),
        ({"counts": np.zeros((2, 2))}, "counts"),
    ],
)
def test_malformed_tables_are_rejected(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_table(**overrides)


def test_non_monotone_quantiles_are_rejected() -> None:
    bad = np.zeros((2, 3, 3))
    bad[..., 1] = 1.0  # falls back to 0 at the last level
    with pytest.raises(ValueError, match="non-decreasing"):
        make_table(quantiles_mev={"readout": bad, "deposition": bad})


# --- the builder on synthetic pilot batches --------------------------------


def write_pilot_batch(root, energy_gev, n=1200, seed=0, poison=False) -> None:
    rng = np.random.default_rng(seed)
    depth = rng.exponential(TRUE_LENGTH_MM, n)
    occurred = depth < DEPTH_MM
    index = np.arange(n)
    readout = np.zeros((n, 18, 72), dtype=np.float32)
    deposit = np.zeros((n, 18, 72), dtype=np.float32)
    scale = energy_gev / 10.0
    readout[:, :, 36] = rng.gamma(4.0, 0.15 * scale**0.1, size=(n, 18))
    deposit[:, :, 36] = rng.gamma(8.0, 1.0, size=(n, 18))
    if poison:  # any leak of a validation event would show up as 1e6 MeV
        held_out = validation_mask(index)
        readout[held_out] = 1.0e6
        deposit[held_out] = 1.0e6
    directory = root / "baseline" / f"E{energy_gev:g}GeV"
    directory.mkdir(parents=True)
    np.savez(
        directory / "events.npz",
        event_index=index,
        seed=index + 1000,
        entry_x_mm=rng.uniform(0, 9, n),
        entry_y_mm=rng.uniform(0, 9, n),
        truth_occurred=occurred,
        truth_z_mm=np.where(occurred, depth, 0.0),
        readout_grid_mev=readout,
        deposit_grid_mev=deposit,
    )
    metadata = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "energy_gev": energy_gev,
        "physics_list": "FTFP_BERT",
        "configuration_sha256": "0" * 64,
        "geant4_version": "test-11.4",
        "geant4_pybind_version": "0.0",
        "geant4_datasets": {},
        "sample": "baseline",
        "particle": "proton",
        "production_cut_mm": 0.7,
        "geometry_variant": "ams",
        "theta_rad": 0.0,
        "phi_rad": 0.0,
        "entry_spot_mm": {"x_min_mm": 0.0, "x_max_mm": 9.0, "y_min_mm": 0.0, "y_max_mm": 9.0},
        "materials": {
            "matrix_constraint": "average_density",
            "composite_density_g_cm3": 6.8,
            "prefix_depth_lambda_i": 0.618,
            "prefix_depth_x0": 16.68,
        },
        "base_seed": 1,
        "git": {"commit": "0" * 40, "tracked_changes": False},
        "created_utc": "2026-01-01T00:00:00Z",
        "geometry_approximations": ["test"],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot")
    write_pilot_batch(root, 10.0, seed=1, poison=True)
    write_pilot_batch(root, 100.0, seed=2, poison=True)
    return root


@pytest.fixture(scope="module")
def calibration(pilot) -> ProtonCalibration:
    # the table-only build: these synthetic batches have no bursts to calibrate a structure on
    return build_calibration(pilot, energies_gev=(10.0, 100.0), structure=False)


def test_no_validation_event_reaches_a_table(calibration) -> None:
    # validation events carry 1e6 MeV per cell; a single leak would set the maximum
    for table in calibration.crossing.quantiles_mev.values():
        assert table.max() < 1000.0


def test_the_pooled_interaction_length_recovers_the_truth(calibration) -> None:
    assert calibration.effective_length_mm == pytest.approx(TRUE_LENGTH_MM, rel=0.10)
    assert calibration.effective_length_error_mm > 0
    assert calibration.depth_mm == pytest.approx(DEPTH_MM)


def test_the_build_is_deterministic(pilot, calibration) -> None:
    again = build_calibration(pilot, energies_gev=(10.0, 100.0), structure=False)

    assert again.content_sha256 == calibration.content_sha256
    for name, array in calibration.arrays().items():
        assert np.array_equal(array, again.arrays()[name])


def test_the_manifest_records_the_provenance_the_plan_asks_for(calibration) -> None:
    manifest = calibration.manifest

    assert manifest["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert manifest["physics_list"] == "ftfp_bert"
    assert manifest["energies_gev"] == [10.0, 100.0]
    assert manifest["split_rule"] == {
        "validation_modulus": 4,
        "validation_residue": 3,
        "used": "calibration events only",
    }
    assert manifest["source"]["geant4_version"] == "test-11.4"
    assert set(manifest["source"]["pilot_batches"]) == {"10", "100"}
    assert manifest["source"]["pilot_batches"]["10"]["configuration_sha256"] == "0" * 64
    assert manifest["counts"]["10"]["calibration_events"] == 900
    assert manifest["counts"]["10"]["crossing"] + manifest["counts"]["10"]["interacting"] == 900
    assert manifest["interaction"]["effective_length_mm"] == calibration.effective_length_mm
    assert manifest["builder"]["module"] == "ams_ecal.proton_calibration"
    assert "created_utc" in manifest and "wording" in manifest


def test_the_content_hash_matches_the_arrays(calibration) -> None:
    assert content_sha256(calibration.arrays()) == calibration.content_sha256


def test_an_artifact_survives_a_save_and_load(calibration, tmp_path) -> None:
    calibration.save(tmp_path / "artifact")

    loaded = ProtonCalibration.load(tmp_path / "artifact")

    assert loaded.content_sha256 == calibration.content_sha256
    assert loaded.effective_length_mm == calibration.effective_length_mm
    assert np.array_equal(
        loaded.crossing.quantiles_mev["readout"], calibration.crossing.quantiles_mev["readout"]
    )


def test_a_tampered_array_is_detected_on_load(calibration, tmp_path) -> None:
    directory = calibration.save(tmp_path / "artifact")
    with np.load(directory / "calibration.npz") as data:
        arrays = {name: data[name] for name in data.files}
    arrays["effective_length_mm"] = np.asarray(1.0)
    np.savez_compressed(directory / "calibration.npz", **arrays)

    with pytest.raises(ValueError, match="content hash"):
        ProtonCalibration.load(directory)


def test_an_unsupported_schema_version_is_rejected(calibration, tmp_path) -> None:
    directory = calibration.save(tmp_path / "artifact")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        ProtonCalibration.load(directory)


def test_a_mismatched_physics_list_is_refused(pilot) -> None:
    with pytest.raises(ValueError, match="physics list"):
        build_calibration(pilot, energies_gev=(10.0, 100.0), physics_list="qgsp_bert")


def test_too_little_data_in_a_chord_bin_is_refused(pilot) -> None:
    with pytest.raises(ValueError, match="crossing layers"):
        build_calibration(
            pilot, energies_gev=(10.0, 100.0), min_layers_per_bin=10_000, structure=False
        )


# --- the builder with the crossing structure ------------------------------------------------


def write_burst_batch(root, energy_gev, n=1600, seed=0) -> None:
    """A pilot batch whose crossing events carry planted bursts (20% of events)."""

    write_pilot_batch(root, energy_gev, n=n, seed=seed, poison=True)
    path = root / "baseline" / f"E{energy_gev:g}GeV" / "events.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    rng = np.random.default_rng(seed + 100)
    deposit = np.zeros((n, 18, 72), dtype=np.float32)
    readout = np.zeros((n, 18, 72), dtype=np.float32)
    deposit[:, :, 36] = rng.gamma(25.0, 8.0 / 25.0, (n, 18))
    readout[:, :, 36] = rng.gamma(40.0, 0.6 / 40.0, (n, 18))
    has = rng.random(n) < 0.2
    onset = rng.integers(0, 18, n)
    offset = np.arange(18)[None, :] - onset[:, None]
    shape = np.where(offset >= 0, (offset + 1.0) * np.exp(-offset / 3.0), 0.0)
    burst = np.where(has[:, None], rng.uniform(80.0, 240.0, n)[:, None] * shape / shape.max(), 0.0)
    deposit[:, :, 36] += burst
    readout[:, :, 36] += 0.06 * burst
    held_out = validation_mask(arrays["event_index"])
    deposit[held_out] = 1.0e6
    readout[held_out] = 1.0e6
    arrays["deposit_grid_mev"], arrays["readout_grid_mev"] = deposit, readout
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def burst_pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("burst_pilot")
    write_burst_batch(root, 10.0, seed=1)
    write_burst_batch(root, 100.0, seed=2)
    return root


@pytest.fixture(scope="module")
def structured(burst_pilot) -> ProtonCalibration:
    return build_calibration(burst_pilot, energies_gev=(10.0, 100.0), n_chord_bins=3)


def test_a_pilot_with_bursts_builds_a_structured_artifact(structured) -> None:
    assert structured.manifest["schema_version"] == CALIBRATION_SCHEMA_VERSION == 2
    assert structured.structure is not None
    assert structured.structure.burst.probability == pytest.approx([0.2, 0.2], abs=0.05)
    assert structured.manifest["crossing"]["layers_independent"] is False
    assert structured.manifest["crossing"]["table_built_from"] == (
        "layers of crossing events without a burst"
    )
    assert structured.manifest["structure"]["burst_ratio"] == 3.0
    assert set(structured.manifest["structure"]["burst_probability"]) == {"10", "100"}


def test_no_validation_event_reaches_the_structure(structured) -> None:
    arrays = structured.arrays()

    for name, array in arrays.items():
        if array.dtype.kind == "f":
            assert np.nanmax(array) < 1000.0, name  # a leaked 1e6 MeV event would set the maximum


def test_the_structured_build_is_deterministic(burst_pilot, structured) -> None:
    again = build_calibration(burst_pilot, energies_gev=(10.0, 100.0), n_chord_bins=3)

    assert again.content_sha256 == structured.content_sha256


def test_the_structure_survives_a_save_and_load(structured, tmp_path) -> None:
    structured.save(tmp_path / "artifact")

    loaded = ProtonCalibration.load(tmp_path / "artifact")

    assert loaded.structure is not None
    assert loaded.content_sha256 == structured.content_sha256
    assert np.array_equal(
        loaded.structure.burst.amplitude_quantiles_mev,
        structured.structure.burst.amplitude_quantiles_mev,
    )
    assert np.array_equal(loaded.structure.bulk_coupling["readout"], structured.structure.bulk_coupling["readout"])


def test_a_tampered_structure_array_is_detected_on_load(structured, tmp_path) -> None:
    directory = structured.save(tmp_path / "artifact")
    with np.load(directory / "calibration.npz") as data:
        arrays = {name: data[name] for name in data.files}
    arrays["burst_probability"] = np.asarray(arrays["burst_probability"]) * 0.5
    np.savez_compressed(directory / "calibration.npz", **arrays)

    with pytest.raises(ValueError, match="content hash"):
        ProtonCalibration.load(directory)


def test_a_table_only_artifact_loads_without_a_structure(calibration, tmp_path) -> None:
    calibration.save(tmp_path / "artifact")

    loaded = ProtonCalibration.load(tmp_path / "artifact")

    assert loaded.structure is None
    assert loaded.manifest["crossing"]["layers_independent"] is True


def test_a_pilot_without_bursts_cannot_calibrate_a_structure(pilot) -> None:
    with pytest.raises(ValueError, match="too few to calibrate"):
        build_calibration(pilot, energies_gev=(10.0, 100.0))
