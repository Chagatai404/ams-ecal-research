"""Assembly of the FastMC electron/proton dataset and its dataset-level checks.

The design and the checks are registered in
``research/plans/2026-10-04_dataset_level_checks_preregistration.md``. Version v0 is the
perfect-event dataset: both classes in the ``deposition`` representation (DEC-006), normal
incidence, one energy and one entry-point distribution for both classes.

``plan_events`` fixes everything random about the SAMPLING (energies, entry points, seeds, splits,
labels) without generating a single shower; ``assemble`` then generates every event from its own
row of the plan, so any event can be regenerated from its provenance alone (check D8).

Seeds are 32-bit, so a large dataset would draw repeated ones by chance; the seed stream is
extended and de-duplicated deterministically until it holds enough distinct seeds, in order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.added_checks import EventSample, classifier_two_sample_row, event_features
from ams_ecal.fastmc_config import config_digest, load_fastmc_config
from ams_ecal.geant4_backend import PROJECT_ROOT
from ams_ecal.geometry import load_geometry
from ams_ecal.lateral import AMSLateralShowerModel
from ams_ecal.longitudinal import AMSLongitudinalGammaModel
from ams_ecal.proton import PROTON_CONFIG, ProtonShowerModel
from ams_ecal.proton_batch import generate_batch_parallel
from ams_ecal.stochastic import StochasticEMShowerModel
from ams_ecal.tracking import TrackState

GEOMETRY_CONFIG = PROJECT_ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG = PROJECT_ROOT / "configs" / "fastmc.yaml"

ELECTRON, PROTON = 0, 1
CLASS_NAMES = {ELECTRON: "electron", PROTON: "proton"}
SPLITS = ("train", "validation", "test")
KS_MATERIAL = 0.05
NUISANCE_AUC_MATERIAL = 0.55
REGENERATION_EVENTS = 100
DEFAULT_CHUNK = 250

TRAINING_FIELDS = ("label", "split", "energy_mev", "entry_x_mm", "entry_y_mm")
"""Stored beside the grids in ``events.npz``. Everything else is reproducibility information and is
written to ``provenance.npz``: ``interacting`` is a perfect proxy for the label (no electron
interacts), so a training file that carried it would hand the label to any model that read it."""

PROVENANCE_FIELDS = (
    "label",
    "seed",
    "energy_mev",
    "entry_x_mm",
    "entry_y_mm",
    "split",
    "interacting",
    "interaction_depth_mm",
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Everything that defines the sampling of a dataset."""

    events_per_class: int
    base_seed: int
    representation: str = "deposition"
    energy_range_gev: tuple[float, float] = (10.0, 100.0)
    entry_spot_mm: tuple[float, float] = (0.0, 9.0)
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)

    def __post_init__(self) -> None:
        if self.events_per_class < len(SPLITS) * 2:
            raise ValueError("events_per_class is too small to fill three splits")
        if abs(sum(self.split_fractions) - 1.0) > 1e-9 or min(self.split_fractions) <= 0:
            raise ValueError("split_fractions must be positive and sum to one")
        low, high = self.energy_range_gev
        if not 0 < low < high:
            raise ValueError("energy_range_gev must be an increasing positive pair")
        if self.representation not in ("deposition", "readout"):
            raise ValueError("representation must be 'deposition' or 'readout'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "events_per_class": self.events_per_class,
            "base_seed": self.base_seed,
            "representation": self.representation,
            "energy_range_gev": list(self.energy_range_gev),
            "entry_spot_mm": list(self.entry_spot_mm),
            "split_fractions": list(self.split_fractions),
        }


@dataclass(frozen=True, slots=True)
class EventPlan:
    """The sampling of a dataset: row ``i`` is one event, electrons first, then protons."""

    label: np.ndarray
    seed: np.ndarray
    energy_mev: np.ndarray
    entry_x_mm: np.ndarray
    entry_y_mm: np.ndarray
    split: np.ndarray  # index into SPLITS

    def __len__(self) -> int:
        return len(self.seed)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def distinct_seeds(base_seed: int, count: int) -> np.ndarray:
    """``count`` distinct 32-bit seeds from one base seed, deterministic and in stream order."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    seen: set[int] = set()
    seeds: list[int] = []
    batch = count + count // 20 + 64  # room for the expected collisions; doubled if still short
    while len(seeds) < count:
        seen.clear()
        seeds.clear()
        for child in np.random.SeedSequence(base_seed).spawn(batch):
            value = int(child.generate_state(1, dtype=np.uint32)[0])
            if value not in seen:
                seen.add(value)
                seeds.append(value)
                if len(seeds) == count:
                    break
        batch *= 2
    return np.array(seeds, dtype=np.int64)


def split_sizes(spec: DatasetSpec) -> list[int]:
    """Events per class in each split, from the design."""

    n = spec.events_per_class
    cuts = np.cumsum(np.round(np.array(spec.split_fractions) * n).astype(int))
    cuts[-1] = n
    return np.diff(np.concatenate([[0], cuts])).tolist()


def plan_events(spec: DatasetSpec) -> EventPlan:
    """Fix the energies, entry points, seeds, splits and labels of a dataset."""

    n = spec.events_per_class
    total = 2 * n
    seeds = distinct_seeds(spec.base_seed, total)
    # One stream of energies and entry points serves both classes: it cannot depend on the label.
    rng = np.random.default_rng([spec.base_seed, 1])
    low, high = spec.energy_range_gev
    energy_gev = np.exp(rng.uniform(np.log(low), np.log(high), total))
    entry_x = rng.uniform(*spec.entry_spot_mm, total)
    entry_y = rng.uniform(*spec.entry_spot_mm, total)
    label = np.concatenate([np.full(n, ELECTRON), np.full(n, PROTON)])
    split = np.empty(total, dtype=np.int64)
    split_rng = np.random.default_rng([spec.base_seed, 2])
    for klass in (ELECTRON, PROTON):
        order = split_rng.permutation(n)
        assigned = np.empty(n, dtype=np.int64)
        start = 0
        for index, size in enumerate(split_sizes(spec)):
            assigned[order[start : start + size]] = index
            start += size
        split[klass * n : (klass + 1) * n] = assigned
    return EventPlan(label, seeds, 1000.0 * energy_gev, entry_x, entry_y, split)


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


def load_electron_model(representation: str = "deposition") -> StochasticEMShowerModel:
    geometry = load_geometry(GEOMETRY_CONFIG)
    config = load_fastmc_config(FASTMC_CONFIG)
    model = StochasticEMShowerModel(
        config=config.stochastic_em,
        longitudinal=AMSLongitudinalGammaModel(
            config=config.longitudinal_em, geometry=geometry, regime=config.regime
        ),
        lateral=AMSLateralShowerModel(config=config.lateral_em, geometry=geometry),
    )
    return model.as_regime("deposition" if representation == "deposition" else "sampling")


def electron_grid(
    model: StochasticEMShowerModel, energy_mev: float, x_mm: float, y_mm: float, seed: int
) -> np.ndarray:
    """One electron event's cell energies, drawn as ``generate_event`` draws them."""

    track = TrackState(x0_mm=x_mm, y0_mm=y_mm, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    cells = model.sample_cell_energies_mev(energy_mev, track, np.random.default_rng(int(seed)))
    return np.array(cells, dtype=np.float32)


_ELECTRON_MODEL: StochasticEMShowerModel | None = None


def _initialise_electrons(representation: str) -> None:
    global _ELECTRON_MODEL
    _ELECTRON_MODEL = load_electron_model(representation)


def _electron_chunk(arguments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    if _ELECTRON_MODEL is None:
        raise RuntimeError("worker model not initialised")
    energies, x, y, seeds = arguments
    return np.stack(
        [
            electron_grid(_ELECTRON_MODEL, e, a, b, s)
            for e, a, b, s in zip(energies, x, y, seeds, strict=True)
        ]
    )


def generate_electrons(
    representation: str,
    energies_mev: np.ndarray,
    entry_x: np.ndarray,
    entry_y: np.ndarray,
    seeds: np.ndarray,
    *,
    workers: int = 4,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Electron grids ``(n, layers, cells)``, identical however they are chunked."""

    if workers < 1 or chunk < 1:
        raise ValueError("workers and chunk must be positive")
    pieces = [
        (
            energies_mev[a : a + chunk],
            entry_x[a : a + chunk],
            entry_y[a : a + chunk],
            seeds[a : a + chunk],
        )
        for a in range(0, len(seeds), chunk)
    ]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialise_electrons, initargs=(representation,)
    ) as pool:
        return np.concatenate(list(pool.map(_electron_chunk, pieces)))


@dataclass(frozen=True, slots=True)
class Dataset:
    spec: DatasetSpec
    grids_mev: np.ndarray  # (n, layers, cells) float32
    provenance: dict[str, np.ndarray]
    model_versions: dict[str, str]

    def __len__(self) -> int:
        return len(self.grids_mev)

    def sample(self, label: int, split: int | None = None) -> EventSample:
        mask = self.provenance["label"] == label
        if split is not None:
            mask &= self.provenance["split"] == split
        return EventSample(
            self.grids_mev[mask],
            self.provenance["entry_x_mm"][mask],
            self.provenance["entry_y_mm"][mask],
        )


def assemble(
    spec: DatasetSpec, *, workers: int = 4, config_path: str | Path = PROTON_CONFIG
) -> Dataset:
    """Generate every event of the plan; electrons then protons, each from its own seed."""

    plan = plan_events(spec)
    is_electron = plan.label == ELECTRON
    electron_grids = generate_electrons(
        spec.representation,
        plan.energy_mev[is_electron],
        plan.entry_x_mm[is_electron],
        plan.entry_y_mm[is_electron],
        plan.seed[is_electron],
        workers=workers,
    )
    protons = generate_batch_parallel(
        spec.representation,  # type: ignore[arg-type]
        plan.energy_mev[~is_electron],
        plan.entry_x_mm[~is_electron],
        plan.entry_y_mm[~is_electron],
        plan.seed[~is_electron],
        workers=workers,
        config_path=config_path,
    )
    n_electrons = int(is_electron.sum())
    provenance = {
        "label": plan.label,
        "seed": plan.seed,
        "energy_mev": plan.energy_mev,
        "entry_x_mm": plan.entry_x_mm,
        "entry_y_mm": plan.entry_y_mm,
        "split": plan.split,
        "interacting": np.concatenate([np.zeros(n_electrons, dtype=bool), protons.interacting]),
        "interaction_depth_mm": np.concatenate(
            [np.full(n_electrons, np.nan), protons.interaction_depth_mm]
        ),
    }
    proton = ProtonShowerModel.from_config(config_path)
    return Dataset(
        spec=spec,
        grids_mev=np.concatenate([electron_grids, protons.grids_mev]),
        provenance=provenance,
        model_versions={
            "proton_model_version": proton.model_version,
            "proton_calibration_content_sha256": proton.calibration.content_sha256,
            "electron_config_sha256": config_digest([GEOMETRY_CONFIG, FASTMC_CONFIG]),
        },
    )


# ----------------------------------------------------------------------
# Dataset-level checks D1-D9
# ----------------------------------------------------------------------


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ks_2samp(a, b).statistic)


def check_distribution_identity(provenance: dict[str, np.ndarray]) -> dict[str, Any]:
    """D1 and D2: the two classes have the same energy and entry-point distributions."""

    electron = provenance["label"] == ELECTRON
    ks = {
        name: _ks(provenance[name][electron], provenance[name][~electron])
        for name in ("energy_mev", "entry_x_mm", "entry_y_mm")
    }
    return {"ks": ks, "material": any(v >= KS_MATERIAL for v in ks.values())}


def check_seed_isolation(provenance: dict[str, np.ndarray]) -> dict[str, Any]:
    """D3: no seed repeats anywhere, so none is shared across splits or classes."""

    seeds = provenance["seed"]
    _, counts = np.unique(seeds, return_counts=True)
    repeated = int((counts > 1).sum())
    cross_class = len(
        set(seeds[provenance["label"] == ELECTRON].tolist())
        & set(seeds[provenance["label"] == PROTON].tolist())
    )
    cross_split = 0
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            cross_split += len(
                set(seeds[provenance["split"] == a].tolist())
                & set(seeds[provenance["split"] == b].tolist())
            )
    return {
        "repeated_seeds": repeated,
        "shared_across_classes": cross_class,
        "shared_across_splits": cross_split,
        "material": bool(repeated or cross_class or cross_split),
    }


def check_split_integrity(spec: DatasetSpec, provenance: dict[str, np.ndarray]) -> dict[str, Any]:
    """D4: sizes equal the design, classes balanced in every split, energies agree across splits."""

    expected = split_sizes(spec)
    violations: list[str] = []
    sizes: dict[str, list[int]] = {}
    for klass in (ELECTRON, PROTON):
        mask = provenance["label"] == klass
        got = [int(((provenance["split"] == s) & mask).sum()) for s in range(len(SPLITS))]
        sizes[CLASS_NAMES[klass]] = got
        if got != expected:
            violations.append(f"{CLASS_NAMES[klass]} split sizes {got} != design {expected}")
    if not np.isin(provenance["split"], np.arange(len(SPLITS))).all():
        violations.append("an event has no valid split")
    energy = provenance["energy_mev"]
    pairwise = {}
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            ks = _ks(energy[provenance["split"] == a], energy[provenance["split"] == b])
            pairwise[f"{SPLITS[a]}-{SPLITS[b]}"] = ks
            if ks >= KS_MATERIAL:
                violations.append(f"energy KS {ks:.3f} between {SPLITS[a]} and {SPLITS[b]}")
    return {
        "sizes": sizes,
        "energy_ks": pairwise,
        "violations": violations,
        "material": bool(violations),
    }


def nuisance_material(auc: float, null_threshold: float) -> bool:
    """D5 verdict: AUC at least ``NUISANCE_AUC_MATERIAL`` AND above the permutation null."""

    return bool(auc >= NUISANCE_AUC_MATERIAL and auc > null_threshold)


def check_nuisance_leakage(
    provenance: dict[str, np.ndarray], *, n_permutations: int = 200, seed: int = 0
) -> dict[str, Any]:
    """D5: can energy and entry point alone tell the classes apart?"""

    electron = provenance["label"] == ELECTRON
    columns = np.column_stack(
        [provenance["energy_mev"], provenance["entry_x_mm"], provenance["entry_y_mm"]]
    )
    row = classifier_two_sample_row(
        columns[electron], columns[~electron], n_permutations=n_permutations, seed=seed
    )
    row["material"] = nuisance_material(row["auc"], row["null_threshold"])
    row["features"] = ["energy_mev", "entry_x_mm", "entry_y_mm"]
    return row


def check_validity(grids: np.ndarray) -> dict[str, Any]:
    """D6: finite, non-negative, right shape; zero-energy events are counted, not gated."""

    problems = []
    if grids.ndim != 3:
        problems.append("grids are not (events, layers, cells)")
    if not np.isfinite(grids).all():
        problems.append("non-finite values")
    if (grids < 0).any():
        problems.append("negative values")
    zero = int((grids.reshape(len(grids), -1).sum(axis=1) <= 0).sum())
    return {
        "shape": list(grids.shape),
        "zero_energy_events": zero,
        "problems": problems,
        "material": bool(problems),
    }


def check_provenance(provenance: dict[str, np.ndarray], n_events: int) -> dict[str, Any]:
    """D7: every field present, one entry per event; interacting protons carry a depth."""

    missing = [f for f in PROVENANCE_FIELDS if f not in provenance]
    wrong_length = [
        f for f in PROVENANCE_FIELDS if f in provenance and len(provenance[f]) != n_events
    ]
    problems = [f"missing {f}" for f in missing] + [f"wrong length {f}" for f in wrong_length]
    if not missing and not wrong_length:
        proton = provenance["label"] == PROTON
        interacting = provenance["interacting"].astype(bool)
        if np.isnan(provenance["interaction_depth_mm"][proton & interacting]).any():
            problems.append("an interacting proton has no interaction depth")
        if interacting[~proton].any():
            problems.append("an electron is marked interacting")
        for name in ("seed", "energy_mev", "entry_x_mm", "entry_y_mm"):
            if not np.isfinite(provenance[name].astype(float)).all():
                problems.append(f"non-finite {name}")
    return {"problems": problems, "material": bool(problems)}


def regenerate(
    spec: DatasetSpec,
    row: dict[str, Any],
    proton_model: ProtonShowerModel,
    electron_model: StochasticEMShowerModel,
) -> np.ndarray:
    """One event from its provenance row alone."""

    energy, x, y, seed = (
        float(row["energy_mev"]),
        float(row["entry_x_mm"]),
        float(row["entry_y_mm"]),
        int(row["seed"]),
    )
    if int(row["label"]) == ELECTRON:
        return electron_grid(electron_model, energy, x, y, seed)
    track = TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    variant = proton_model.as_representation(spec.representation)  # type: ignore[arg-type]
    grid, _, _ = variant.generate_grid(energy, track, seed)
    return np.asarray(grid, dtype=np.float32)


def check_regeneration(
    dataset: Dataset,
    *,
    n_events: int = REGENERATION_EVENTS,
    seed: int = 0,
    config_path: str | Path = PROTON_CONFIG,
) -> dict[str, Any]:
    """D8: events regenerated from their own provenance row are bit-identical."""

    proton = ProtonShowerModel.from_config(config_path)
    electron = load_electron_model(dataset.spec.representation)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), size=min(n_events, len(dataset)), replace=False)
    different = []
    for i in chosen:
        row = {name: values[i] for name, values in dataset.provenance.items()}
        if not np.array_equal(regenerate(dataset.spec, row, proton, electron), dataset.grids_mev[i]):
            different.append(int(i))
    return {"n_checked": len(chosen), "different": different, "material": bool(different)}


def report_native_separability(dataset: Dataset, geometry, *, seed: int = 0) -> dict[str, Any]:
    """D9, REPORTED NOT GATED: how well generator-native features separate the classes."""

    electron_features, names = event_features(dataset.sample(ELECTRON), geometry)
    proton_features, _ = event_features(dataset.sample(PROTON), geometry)
    row = classifier_two_sample_row(electron_features, proton_features, n_permutations=20, seed=seed)
    row.pop("material")
    row["features"] = names
    row["status"] = (
        "REPORTED, NOT GATED: the electron generator is smooth by design (DEC-001) and the proton "
        "generator is not; separability is a statement about the generators, not about physics."
    )
    return row


def run_checks(
    dataset: Dataset, *, n_permutations: int = 200, config_path: str | Path = PROTON_CONFIG
) -> dict[str, Any]:
    geometry = load_geometry(GEOMETRY_CONFIG)
    rows = {
        "D1_D2_distribution_identity": check_distribution_identity(dataset.provenance),
        "D3_seed_isolation": check_seed_isolation(dataset.provenance),
        "D4_split_integrity": check_split_integrity(dataset.spec, dataset.provenance),
        "D5_nuisance_leakage": check_nuisance_leakage(
            dataset.provenance, n_permutations=n_permutations
        ),
        "D6_validity": check_validity(dataset.grids_mev),
        "D7_provenance": check_provenance(dataset.provenance, len(dataset)),
        "D8_regeneration": check_regeneration(dataset, config_path=config_path),
    }
    return {
        "rows": rows,
        "flagged": [name for name, row in rows.items() if row["material"]],
        "D9_native_separability": report_native_separability(dataset, geometry),
    }


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dataset(dataset: Dataset, checks: dict[str, Any], directory: Path, *, status: str) -> Path:
    """Write ``events.npz`` (grids + ``TRAINING_FIELDS``), ``provenance.npz`` (the rest),
    ``checks.json`` and ``manifest.json`` (with both files' SHA-256)."""

    directory.mkdir(parents=True, exist_ok=True)
    events = directory / "events.npz"
    provenance = directory / "provenance.npz"
    training = {k: v for k, v in dataset.provenance.items() if k in TRAINING_FIELDS}
    reproducibility = {k: v for k, v in dataset.provenance.items() if k not in TRAINING_FIELDS}
    np.savez_compressed(events, grids_mev=dataset.grids_mev, **training)
    np.savez_compressed(provenance, **reproducibility)
    (directory / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    manifest = {
        "status": status,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec": dataset.spec.as_dict(),
        "n_events": len(dataset),
        "classes": CLASS_NAMES,
        "splits": list(SPLITS),
        "events_sha256": _sha256(events),
        "provenance_sha256": _sha256(provenance),
        "training_file_fields": ["grids_mev", *TRAINING_FIELDS],
        "provenance_file_fields": sorted(reproducibility),
        "model_versions": dataset.model_versions,
        "flagged_checks": checks["flagged"],
        "not_comparable_between_classes": (
            "Electron events are smooth by design (deterministic lateral profile, DEC-001); proton "
            "events carry bursts, lateral spill and cell-level fluctuations. A classifier can "
            "separate the classes by that asymmetry alone; its accuracy is not a physics result."
        ),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events-per-class", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--representation", default="deposition")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--status", default="DEVELOPMENT: dataset-level checks not yet confirmed")
    args = parser.parse_args(argv)
    spec = DatasetSpec(args.events_per_class, args.base_seed, args.representation)
    dataset = assemble(spec, workers=args.workers)
    checks = run_checks(dataset, n_permutations=args.permutations)
    print(save_dataset(dataset, checks, args.out, status=args.status))
    print("flagged:", checks["flagged"])


if __name__ == "__main__":
    main()
