"""Batch generation of proton events: arrays instead of canonical events, over a process pool.

WHY. A dataset needs many events and the per-event cost of the canonical ``ECALEvent`` (validating
1296 values in Python) dominates. ``ProtonShowerModel.generate_grid`` returns the same cell
energies without it, event for event, and every event depends only on its own seed, so a batch
can be cut into chunks and spread over processes. The result of a batch is therefore
IDENTICAL however it is chunked or parallelised, and identical to ``generate_event`` per seed;
the tests check both.

SEEDS. The caller supplies one seed per event (``spawn_event_seeds`` derives independent ones
from a base seed). Nothing here draws a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ams_ecal.detector.tracking import TrackState
from ams_ecal.proton_model.proton import PROTON_CONFIG, ProtonShowerModel
from ams_ecal.proton_model.proton_config import ProtonRepresentation

DEFAULT_CHUNK = 250


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Events of one batch; row ``i`` is the event of ``seeds[i]``."""

    grids_mev: np.ndarray  # (n, layers, cells) float32
    interacting: np.ndarray  # (n,) bool
    interaction_depth_mm: np.ndarray  # (n,) NaN for a crossing proton
    energies_mev: np.ndarray  # (n,)
    entry_xy_mm: np.ndarray  # (n, 2)
    seeds: np.ndarray  # (n,)

    def __len__(self) -> int:
        return len(self.seeds)


def _check(
    energies_mev: np.ndarray, entry_x: np.ndarray, entry_y: np.ndarray, seeds: np.ndarray
) -> None:
    n = len(seeds)
    if not (len(energies_mev) == len(entry_x) == len(entry_y) == n):
        raise ValueError("energies, entry coordinates and seeds must have the same length")
    if len({int(s) for s in seeds}) != n:
        raise ValueError("every event needs its own seed")


def generate_batch(
    model: ProtonShowerModel,
    energies_mev: Sequence[float],
    entry_x_mm: Sequence[float],
    entry_y_mm: Sequence[float],
    seeds: Sequence[int],
) -> BatchResult:
    """Generate one event per seed, serially, at normal incidence from ``(entry_x, entry_y)``."""

    energies = np.asarray(energies_mev, dtype=float)
    x, y = np.asarray(entry_x_mm, dtype=float), np.asarray(entry_y_mm, dtype=float)
    seed_array = np.asarray(seeds, dtype=np.int64)
    _check(energies, x, y, seed_array)
    layers, cells = model.geometry.number_of_layers, model.geometry.cells_per_layer
    grids = np.empty((len(seed_array), layers, cells), dtype=np.float32)
    interacting = np.zeros(len(seed_array), dtype=bool)
    depth = np.full(len(seed_array), np.nan)
    for i, seed in enumerate(seed_array):
        track = TrackState(
            x0_mm=float(x[i]), y0_mm=float(y[i]), z0_mm=0.0, theta_rad=0.0, phi_rad=0.0
        )
        grid, status, latent = model.generate_grid(float(energies[i]), track, int(seed))
        grids[i] = grid
        interacting[i] = status == "interacting"
        if interacting[i] and latent is not None:
            depth[i] = float(latent["interaction_depth_mm"])
    return BatchResult(grids, interacting, depth, energies, np.stack([x, y], axis=1), seed_array)


_WORKER_MODEL: ProtonShowerModel | None = None


def _initialise(representation: ProtonRepresentation, config_path: str) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = ProtonShowerModel.from_config(config_path).as_representation(representation)


def _run_chunk(arguments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> BatchResult:
    if _WORKER_MODEL is None:
        raise RuntimeError("worker model not initialised")
    energies, x, y, seeds = arguments
    return generate_batch(_WORKER_MODEL, energies, x, y, seeds)


def generate_batch_parallel(
    representation: ProtonRepresentation,
    energies_mev: Sequence[float],
    entry_x_mm: Sequence[float],
    entry_y_mm: Sequence[float],
    seeds: Sequence[int],
    *,
    workers: int = 4,
    chunk: int = DEFAULT_CHUNK,
    config_path: str | Path = PROTON_CONFIG,
) -> BatchResult:
    """The same events as ``generate_batch``, spread over ``workers`` processes in chunks."""

    energies = np.asarray(energies_mev, dtype=float)
    x, y = np.asarray(entry_x_mm, dtype=float), np.asarray(entry_y_mm, dtype=float)
    seed_array = np.asarray(seeds, dtype=np.int64)
    _check(energies, x, y, seed_array)
    if workers < 1 or chunk < 1:
        raise ValueError("workers and chunk must be positive")
    pieces = [
        (energies[a : a + chunk], x[a : a + chunk], y[a : a + chunk], seed_array[a : a + chunk])
        for a in range(0, len(seed_array), chunk)
    ]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialise, initargs=(representation, str(config_path))
    ) as pool:
        results = list(pool.map(_run_chunk, pieces))
    return BatchResult(
        grids_mev=np.concatenate([r.grids_mev for r in results]),
        interacting=np.concatenate([r.interacting for r in results]),
        interaction_depth_mm=np.concatenate([r.interaction_depth_mm for r in results]),
        energies_mev=energies,
        entry_xy_mm=np.stack([x, y], axis=1),
        seeds=seed_array,
    )
