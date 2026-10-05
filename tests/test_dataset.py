"""Dataset assembly and the dataset-level checks: each check must see the fault it exists for."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.dataset import (
    ELECTRON,
    PROTON,
    SPLITS,
    Dataset,
    DatasetSpec,
    assemble,
    check_distribution_identity,
    check_nuisance_leakage,
    check_provenance,
    check_regeneration,
    check_seed_isolation,
    check_split_integrity,
    check_validity,
    distinct_seeds,
    electron_grid,
    generate_electrons,
    load_electron_model,
    nuisance_material,
    plan_events,
    save_dataset,
    split_sizes,
)
from ams_ecal.tracking import TrackState

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"


def provenance_of(spec: DatasetSpec) -> dict[str, np.ndarray]:
    plan = plan_events(spec)
    n = len(plan)
    return {
        "label": plan.label,
        "seed": plan.seed,
        "energy_mev": plan.energy_mev,
        "entry_x_mm": plan.entry_x_mm,
        "entry_y_mm": plan.entry_y_mm,
        "split": plan.split,
        "interacting": np.zeros(n, dtype=bool),
        "interaction_depth_mm": np.full(n, np.nan),
    }


# ---------------------------------------------------------------- the sampling plan


def test_seeds_are_distinct_deterministic_and_stable_when_more_are_asked_for() -> None:
    seeds = distinct_seeds(7, 120_000)  # a raw 32-bit stream this long repeats a seed by chance

    assert len(np.unique(seeds)) == 120_000
    assert np.array_equal(seeds, distinct_seeds(7, 120_000))
    assert np.array_equal(seeds[:5000], distinct_seeds(7, 5000))
    assert not np.array_equal(seeds[:100], distinct_seeds(8, 100))


def test_the_plan_is_balanced_deterministic_and_class_independent() -> None:
    spec = DatasetSpec(events_per_class=4000, base_seed=11)

    plan = plan_events(spec)

    assert np.array_equal(plan.seed, plan_events(spec).seed)
    assert (plan.label == ELECTRON).sum() == (plan.label == PROTON).sum() == 4000
    assert np.all((plan.energy_mev >= 10_000) & (plan.energy_mev <= 100_000))
    assert np.all((plan.entry_x_mm >= 0) & (plan.entry_x_mm <= 9))
    for klass in (ELECTRON, PROTON):
        sizes = [int(((plan.split == s) & (plan.label == klass)).sum()) for s in range(3)]
        assert sizes == split_sizes(spec) == [2800, 600, 600]
    # log-uniform: the median is the geometric mean of the range, not the arithmetic one
    assert np.median(plan.energy_mev) == pytest.approx(np.sqrt(10_000 * 100_000), rel=0.05)


def test_the_plan_passes_its_own_sampling_checks() -> None:
    spec = DatasetSpec(events_per_class=6000, base_seed=12)
    provenance = provenance_of(spec)

    assert not check_distribution_identity(provenance)["material"]
    assert not check_seed_isolation(provenance)["material"]
    assert not check_split_integrity(spec, provenance)["material"]


def test_a_spec_must_be_sensible() -> None:
    with pytest.raises(ValueError, match="too small"):
        DatasetSpec(events_per_class=3, base_seed=1)
    with pytest.raises(ValueError, match="sum to one"):
        DatasetSpec(100, 1, split_fractions=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="increasing"):
        DatasetSpec(100, 1, energy_range_gev=(100.0, 10.0))
    with pytest.raises(ValueError, match="representation"):
        DatasetSpec(100, 1, representation="raw")


# ---------------------------------------------------------------- each check sees its fault


def test_a_class_dependent_energy_distribution_is_flagged() -> None:
    provenance = provenance_of(DatasetSpec(3000, 13))
    proton = provenance["label"] == PROTON
    provenance["energy_mev"] = np.where(
        proton, provenance["energy_mev"] * 1.5, provenance["energy_mev"]
    )

    row = check_distribution_identity(provenance)

    assert row["material"] and row["ks"]["energy_mev"] > 0.15
    assert row["ks"]["entry_x_mm"] < 0.05


def test_a_repeated_or_shared_seed_is_flagged() -> None:
    spec = DatasetSpec(1000, 14)
    provenance = provenance_of(spec)
    repeated = dict(provenance, seed=provenance["seed"].copy())
    repeated["seed"][1] = repeated["seed"][0]  # same split, same class
    shared = dict(provenance, seed=provenance["seed"].copy())
    electron_test = np.where((shared["label"] == ELECTRON) & (shared["split"] == 2))[0][0]
    proton_train = np.where((shared["label"] == PROTON) & (shared["split"] == 0))[0][0]
    shared["seed"][proton_train] = shared["seed"][electron_test]  # across class AND split

    assert check_seed_isolation(repeated)["repeated_seeds"] == 1
    row = check_seed_isolation(shared)
    assert row["material"] and row["shared_across_classes"] == 1 and row["shared_across_splits"] == 1
    assert not check_seed_isolation(provenance)["material"]


def test_a_seed_repeated_inside_one_class_and_split_is_flagged() -> None:
    provenance = provenance_of(DatasetSpec(1000, 22))
    same_group = np.where((provenance["label"] == PROTON) & (provenance["split"] == 0))[0][:2]
    broken = dict(provenance, seed=provenance["seed"].copy())
    broken["seed"][same_group[1]] = broken["seed"][same_group[0]]

    row = check_seed_isolation(broken)

    assert row["material"] and row["repeated_seeds"] == 1
    assert row["shared_across_classes"] == 0 and row["shared_across_splits"] == 0


def test_a_broken_split_is_flagged() -> None:
    spec = DatasetSpec(3000, 15)
    provenance = provenance_of(spec)
    unbalanced = dict(provenance, split=provenance["split"].copy())
    unbalanced["split"][np.where(provenance["label"] == PROTON)[0][:50]] = 2
    sorted_by_energy = dict(provenance)
    order = np.argsort(provenance["energy_mev"])
    sorted_by_energy["split"] = np.empty_like(provenance["split"])
    sorted_by_energy["split"][order] = np.sort(provenance["split"])  # test split = highest energies

    assert check_split_integrity(spec, unbalanced)["material"]
    row = check_split_integrity(spec, sorted_by_energy)
    assert row["material"] and any("energy KS" in v for v in row["violations"])


def test_nuisance_leakage_sees_energy_that_predicts_the_class() -> None:
    provenance = provenance_of(DatasetSpec(1500, 16))
    clean = check_nuisance_leakage(provenance, n_permutations=10)
    leaky = dict(provenance)
    proton = provenance["label"] == PROTON
    leaky["energy_mev"] = np.where(proton, provenance["energy_mev"] * 4, provenance["energy_mev"])

    leak = check_nuisance_leakage(leaky, n_permutations=10)

    assert not clean["material"] and clean["auc"] < 0.55
    assert leak["material"] and leak["auc"] > 0.8


def test_the_nuisance_verdict_needs_the_effect_size_and_the_null() -> None:
    assert nuisance_material(0.70, 0.52)
    assert not nuisance_material(0.54, 0.52)  # above the null, too small
    assert not nuisance_material(0.60, 0.62)  # large, inside the null
    assert nuisance_material(0.55, 0.52)  # the threshold itself counts


def test_validity_flags_nonfinite_negative_and_counts_empty_events() -> None:
    good = np.ones((4, 18, 72), dtype=np.float32)
    empty = good.copy()
    empty[2] = 0
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    negative = good.copy()
    negative[1, 1, 1] = -1

    assert not check_validity(good)["material"]
    assert check_validity(empty)["zero_energy_events"] == 1 and not check_validity(empty)["material"]
    assert check_validity(bad)["material"] and check_validity(negative)["material"]
    assert check_validity(good[0])["material"]


def test_provenance_completeness_is_checked() -> None:
    spec = DatasetSpec(100, 17)
    provenance = provenance_of(spec)
    n = len(provenance["seed"])
    assert not check_provenance(provenance, n)["material"]

    missing = {k: v for k, v in provenance.items() if k != "split"}
    assert "missing split" in check_provenance(missing, n)["problems"]
    assert check_provenance(provenance, n + 1)["material"]
    no_depth = dict(provenance, interacting=provenance["interacting"].copy())
    no_depth["interacting"][np.where(provenance["label"] == PROTON)[0][0]] = True
    problems = check_provenance(no_depth, n)["problems"]
    assert "an interacting proton has no interaction depth" in problems
    electron_interacting = dict(provenance, interacting=provenance["interacting"].copy())
    electron_interacting["interacting"][0] = True
    problems = check_provenance(electron_interacting, n)["problems"]
    assert "an electron is marked interacting" in problems


# ---------------------------------------------------------------- generation


def test_electron_generation_does_not_depend_on_chunking_or_workers() -> None:
    plan = plan_events(DatasetSpec(12, 18))
    e, x, y, s = plan.energy_mev[:12], plan.entry_x_mm[:12], plan.entry_y_mm[:12], plan.seed[:12]

    whole = generate_electrons("deposition", e, x, y, s, workers=1, chunk=12)
    cut = generate_electrons("deposition", e, x, y, s, workers=2, chunk=5)

    assert whole.shape == (12, 18, 72) and whole.dtype == np.float32
    assert np.array_equal(whole, cut)
    with pytest.raises(ValueError, match="positive"):
        generate_electrons("deposition", e, x, y, s, workers=0)


def test_an_electron_grid_matches_generate_event() -> None:
    model = load_electron_model("deposition")
    plan = plan_events(DatasetSpec(6, 19))
    track = TrackState(
        x0_mm=plan.entry_x_mm[0], y0_mm=plan.entry_y_mm[0], z0_mm=0.0, theta_rad=0.0, phi_rad=0.0
    )
    event = model.generate_event(
        event_id="e",
        primary_energy_mev=plan.energy_mev[0],
        track=track,
        random_seed=int(plan.seed[0]),
        simulation_version="test",
        configuration_sha256="a" * 64,
    )

    grid = electron_grid(
        model, plan.energy_mev[0], plan.entry_x_mm[0], plan.entry_y_mm[0], int(plan.seed[0])
    )

    assert np.array_equal(grid, np.array(event.cell_energies_mev, dtype=np.float32))


@pytest.mark.skipif(not ARTIFACT.is_dir(), reason="the calibration artifact has not been built")
def test_an_assembled_dataset_regenerates_bit_identically_and_saves_with_its_hash(tmp_path) -> None:
    spec = DatasetSpec(events_per_class=20, base_seed=20)

    dataset = assemble(spec, workers=2)

    assert isinstance(dataset, Dataset) and len(dataset) == 40
    assert dataset.grids_mev.shape == (40, 18, 72) and dataset.grids_mev.dtype == np.float32
    assert not check_validity(dataset.grids_mev)["material"]
    assert not check_provenance(dataset.provenance, 40)["material"]
    assert dataset.provenance["interacting"][:20].sum() == 0  # electrons
    assert dataset.provenance["interacting"][20:].sum() > 0  # some protons interact
    row = check_regeneration(dataset, n_events=40)
    assert row["n_checked"] == 40 and not row["material"]
    assert dataset.sample(PROTON, 0).grids.shape[0] == split_sizes(spec)[0]

    checks = {"flagged": ["example"], "rows": {}}
    path = save_dataset(dataset, checks, tmp_path / "ds", status="TEST")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    events = tmp_path / "ds" / "events.npz"
    assert manifest["events_sha256"] == hashlib.sha256(events.read_bytes()).hexdigest()
    assert manifest["flagged_checks"] == ["example"] and manifest["status"] == "TEST"
    assert "smooth by design" in manifest["not_comparable_between_classes"]
    loaded = np.load(events)
    assert np.array_equal(loaded["grids_mev"], dataset.grids_mev)
    # the training file holds no label proxy: neither the interaction flag, nor the depth, nor the seeds
    assert set(loaded.files) == {"grids_mev", "label", "split", "energy_mev", "entry_x_mm", "entry_y_mm"}
    provenance = np.load(tmp_path / "ds" / "provenance.npz")
    assert set(provenance.files) == {"seed", "interacting", "interaction_depth_mm"}
    assert manifest["provenance_sha256"] == hashlib.sha256(
        (tmp_path / "ds" / "provenance.npz").read_bytes()
    ).hexdigest()
    assert np.array_equal(provenance["seed"], dataset.provenance["seed"])
    assert SPLITS == ("train", "validation", "test")


@pytest.mark.skipif(not ARTIFACT.is_dir(), reason="the calibration artifact has not been built")
def test_regeneration_catches_a_changed_event() -> None:
    dataset = assemble(DatasetSpec(events_per_class=12, base_seed=21), workers=1)
    grids = dataset.grids_mev.copy()
    grids[3, 5, 5] += 1.0
    altered = Dataset(dataset.spec, grids, dataset.provenance, dataset.model_versions)

    row = check_regeneration(altered, n_events=len(altered))

    assert row["material"] and row["different"] == [3]
