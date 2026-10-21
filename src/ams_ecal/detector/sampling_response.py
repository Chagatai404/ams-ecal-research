"""R-A, the sampling response: all-material deposit per cell -> scintillating-fibre energy per cell.

Registered in ``research/COMPLETED_WORK.md (was 2026-10-12_detector_response_model_proposal_DRAFT.md)`` (D6.1, D6.7, D6.8).

The Geant4 files hold two grids: ``deposit_grid_mev`` (energy in all material, truth accounting) and
``readout_grid_mev`` (energy in the scintillating fibres only, the detector image before any electronics).
The model maps the first to the second, cell by cell:

    R = D * rho,    rho drawn from the empirical distribution of R / D in the cell's (layer, deposit-size) class.

Each class table is a quantile table of the fitted ratios, rescaled to the class mean ratio, so the layer
means close in expectation. Cell-level scatter is reproduced without a parametric form.

Scope: the ratio depends on the shower content (a hadronic shower deposits a different share in lead than an
electromagnetic one), so one model is fitted **per particle class** and must not be applied to another.
``results/detector_response/r_a_closure.json`` documents that an electron-fitted model is wrong by tens of
percent on protons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_LAYERS = 18
N_QUANTILES = 101
MIN_CLASS_CELLS = 400
"""A (layer, size) class with fewer fitted cells falls back to the table pooled over layers."""
SIZE_EDGES_MEV = np.concatenate([[0.0], np.logspace(-2, 4, 25)])
"""Deposit-size classes: (0, 0.01], then 24 logarithmic classes up to 10^4 MeV; larger values join the last class."""
EVENT_EDGES_MEV = np.logspace(np.log10(5.0e3), np.log10(2.0e5), 9)
"""Event-size classes, from the event's total deposit (a proxy for the shower energy, read from the input itself)."""
_QUANTILE_GRID = np.linspace(0.0, 1.0, N_QUANTILES)


def size_class(deposit_mev: np.ndarray) -> np.ndarray:
    """Index of the deposit-size class of each value (zero deposits fall in class 0)."""

    return np.clip(
        np.searchsorted(SIZE_EDGES_MEV, deposit_mev, side="left") - 1,
        0,
        len(SIZE_EDGES_MEV) - 2,
    )


def event_class(event_total_mev: np.ndarray) -> np.ndarray:
    """Index of the event-size class of each event total (values outside the edges join the end classes)."""

    return np.clip(
        np.searchsorted(EVENT_EDGES_MEV, event_total_mev, side="right") - 1,
        0,
        len(EVENT_EDGES_MEV) - 2,
    )


@dataclass(frozen=True)
class SamplingResponse:
    """Fitted R-A model for one particle class.

    ``tables[layer, size_class, event_class]`` is a quantile table of the ratio R/D; ``mean_ratio`` is the class
    value ``sum R / sum D`` that the drawn ratios are rescaled to. A class with too few fitted cells uses the
    table pooled over layers (and failing that over event classes) for its shape, with its level rescaled by the
    layer's overall mean ratio, so the layer means are not mixed across layers.
    """

    particle: str
    tables: np.ndarray  # (layers, size classes, event classes, quantiles)
    mean_ratio: np.ndarray  # (layers, size classes, event classes)
    n_fitted_cells: np.ndarray  # (layers, size classes, event classes)
    fitted_from: str

    @classmethod
    def fit(
        cls, deposit: np.ndarray, fibre: np.ndarray, *, particle: str, fitted_from: str
    ) -> SamplingResponse:
        """Fit from matching ``(events, layers, cells)`` arrays (MeV)."""

        if (
            deposit.shape != fibre.shape
            or deposit.ndim != 3
            or deposit.shape[1] != N_LAYERS
        ):
            raise ValueError(
                "deposit and fibre must be matching (events, 18, cells) arrays"
            )
        n_size = len(SIZE_EDGES_MEV) - 1
        n_event = len(EVENT_EDGES_MEV) - 1
        events = event_class(deposit.sum(axis=(1, 2), dtype=np.float64))
        sizes = size_class(deposit)
        layers = np.broadcast_to(np.arange(N_LAYERS)[None, :, None], deposit.shape)
        evs = np.broadcast_to(events[:, None, None], deposit.shape)
        positive = deposit > 0
        d = deposit[positive].astype(np.float64)
        r = fibre[positive].astype(np.float64)
        layer_key, size_key, event_key = (
            layers[positive],
            sizes[positive],
            evs[positive],
        )

        def grouped(
            key: np.ndarray, n_groups: int
        ) -> dict[int, tuple[np.ndarray, float, int]]:
            order = np.argsort(key, kind="stable")
            bounds = np.searchsorted(key[order], np.arange(n_groups + 1))
            out = {}
            for g in range(n_groups):
                idx = order[bounds[g] : bounds[g + 1]]
                if len(idx):
                    out[g] = (
                        np.quantile(r[idx] / d[idx], _QUANTILE_GRID),
                        float(r[idx].sum() / d[idx].sum()),
                        len(idx),
                    )
            return out

        full = grouped(
            (layer_key * n_size + size_key) * n_event + event_key,
            N_LAYERS * n_size * n_event,
        )
        by_size_event = grouped(size_key * n_event + event_key, n_size * n_event)
        by_size = grouped(size_key, n_size)
        layer_factor = np.array(
            [
                r[layer_key == layer].sum() / d[layer_key == layer].sum()
                for layer in range(N_LAYERS)
            ]
        ) / (r.sum() / d.sum())
        tables = np.zeros((N_LAYERS, n_size, n_event, N_QUANTILES))
        mean_ratio = np.zeros((N_LAYERS, n_size, n_event))
        counts = np.zeros((N_LAYERS, n_size, n_event), dtype=np.int64)
        for layer in range(N_LAYERS):
            for c in range(n_size):
                for e in range(n_event):
                    key = (layer * n_size + c) * n_event + e
                    if key in full and full[key][2] >= MIN_CLASS_CELLS:
                        (
                            tables[layer, c, e],
                            mean_ratio[layer, c, e],
                            counts[layer, c, e],
                        ) = full[key]
                    elif (
                        c * n_event + e in by_size_event
                        and by_size_event[c * n_event + e][2] >= MIN_CLASS_CELLS
                    ):
                        tables[layer, c, e], mean_ratio[layer, c, e], _ = by_size_event[
                            c * n_event + e
                        ]
                        mean_ratio[layer, c, e] *= layer_factor[layer]
                    elif c in by_size:
                        tables[layer, c, e], mean_ratio[layer, c, e], _ = by_size[c]
                        mean_ratio[layer, c, e] *= layer_factor[layer]
        return cls(particle, tables, mean_ratio, counts, fitted_from)

    def apply(self, deposit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw fibre energy for ``(events, layers, cells)`` deposits. Zero deposit gives zero."""

        if deposit.ndim != 3 or deposit.shape[1] != N_LAYERS:
            raise ValueError("deposit must be (events, 18, cells)")
        sizes = size_class(deposit)
        layers = np.broadcast_to(np.arange(N_LAYERS)[None, :, None], deposit.shape)
        events = np.broadcast_to(
            event_class(deposit.sum(axis=(1, 2), dtype=np.float64))[:, None, None],
            deposit.shape,
        )
        u = rng.random(deposit.shape) * (N_QUANTILES - 1)
        low = np.floor(u).astype(np.intp)
        high = np.minimum(low + 1, N_QUANTILES - 1)
        weight = u - low
        ratio = (
            self.tables[layers, sizes, events, low] * (1 - weight)
            + self.tables[layers, sizes, events, high] * weight
        )
        table_mean = np.trapezoid(self.tables, _QUANTILE_GRID, axis=-1)
        scale = np.divide(
            self.mean_ratio,
            table_mean,
            out=np.ones_like(table_mean),
            where=table_mean > 0,
        )
        ratio = ratio * scale[layers, sizes, events]
        return np.where(deposit > 0, deposit * ratio, 0.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "particle": self.particle,
            "fitted_from": self.fitted_from,
            "size_edges_mev": SIZE_EDGES_MEV.tolist(),
            "event_edges_mev": EVENT_EDGES_MEV.tolist(),
            "tables": self.tables.tolist(),
            "mean_ratio": self.mean_ratio.tolist(),
            "n_fitted_cells": self.n_fitted_cells.tolist(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SamplingResponse:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            data["particle"],
            np.array(data["tables"]),
            np.array(data["mean_ratio"]),
            np.array(data["n_fitted_cells"], dtype=np.int64),
            data["fitted_from"],
        )
