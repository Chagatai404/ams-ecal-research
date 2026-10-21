"""Component-wise mean sampling response (design: research/COMPLETED_WORK.md (was 2026-10-12_particle_agnostic_RA_design_DRAFT.md)).

The fibre energy of a cell is predicted from the deposit split by component (em, had_low, had_mid, had_high, ion,
neutron, other) with a class-mean ratio ``sum R / sum D`` per (component, layer, deposit-size class). No particle label
and no beam energy enters. All fits and tests are table algebra on sufficient statistics accumulated per
``event_index % 4`` part, so any fit/test split of the registered criteria is exact and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ams_ecal.detector.sampling_response import (
    MIN_CLASS_CELLS,
    N_LAYERS,
    SIZE_EDGES_MEV,
)

COMPONENTS = ("em", "had_low", "had_mid", "had_high", "ion", "neutron", "other")
N_COMPONENTS = len(COMPONENTS)
N_BINS = len(SIZE_EDGES_MEV) - 1
N_PARTS = 4
SHAPE = (N_COMPONENTS, N_LAYERS, N_BINS)


@dataclass
class Statistics:
    """Sums per (part, component, layer, size class); ``orphan`` is fibre energy where the component deposit is 0."""

    sum_deposit: np.ndarray  # (parts, c, l, b)
    sum_fibre: np.ndarray  # (parts, c, l, b)
    count: np.ndarray  # (parts, c, l, b) cells with D > 0
    orphan_fibre: np.ndarray  # (parts, c, l)
    n_events: np.ndarray  # (parts,)

    @classmethod
    def zeros(cls) -> Statistics:
        return cls(
            np.zeros((N_PARTS, *SHAPE)),
            np.zeros((N_PARTS, *SHAPE)),
            np.zeros((N_PARTS, *SHAPE)),
            np.zeros((N_PARTS, N_COMPONENTS, N_LAYERS)),
            np.zeros(N_PARTS),
        )

    def add_events(
        self,
        deposit: np.ndarray,
        fibre: np.ndarray,
        event_index: np.ndarray,
    ) -> None:
        """``deposit`` and ``fibre`` have shape (events, components, layers, cells)."""

        size = N_COMPONENTS * N_LAYERS * N_BINS
        for part in range(N_PARTS):
            select = event_index % N_PARTS == part
            if not select.any():
                continue
            d = deposit[select].astype(np.float64)
            r = fibre[select].astype(np.float64)
            self.n_events[part] += select.sum()
            positive = d > 0
            bins = np.clip(np.digitize(d, SIZE_EDGES_MEV) - 1, 0, N_BINS - 1)
            comp = np.broadcast_to(
                np.arange(N_COMPONENTS)[None, :, None, None], d.shape
            )
            lay = np.broadcast_to(np.arange(N_LAYERS)[None, None, :, None], d.shape)
            flat = ((comp * N_LAYERS + lay) * N_BINS + bins)[positive]
            for target, values in (
                (self.sum_deposit, d[positive]),
                (self.sum_fibre, r[positive]),
                (self.count, np.ones(flat.size)),
            ):
                target[part] += np.bincount(
                    flat, weights=values, minlength=size
                ).reshape(SHAPE)
            self.orphan_fibre[part] += np.where(~positive, r, 0.0).sum(axis=(0, 3))

    def parts(self, parts: tuple[int, ...]) -> Statistics:
        """Same statistics with every part outside ``parts`` set to zero (the part axis keeps its length)."""

        keep = np.zeros(N_PARTS, dtype=bool)
        keep[list(parts)] = True
        return Statistics(
            self.sum_deposit * keep[:, None, None, None],
            self.sum_fibre * keep[:, None, None, None],
            self.count * keep[:, None, None, None],
            self.orphan_fibre * keep[:, None, None],
            self.n_events * keep,
        )

    def __add__(self, other: Statistics) -> Statistics:
        return Statistics(
            self.sum_deposit + other.sum_deposit,
            self.sum_fibre + other.sum_fibre,
            self.count + other.count,
            self.orphan_fibre + other.orphan_fibre,
            self.n_events + other.n_events,
        )


@dataclass(frozen=True)
class RatioTable:
    """ratio[c, l, b] with the fallback level used per class (0 own, 1 pooled over layers, 2 component)."""

    ratio: np.ndarray
    level: np.ndarray

    @classmethod
    def fit(cls, fit: Statistics) -> RatioTable:
        sd = fit.sum_deposit.sum(axis=0)
        sf = fit.sum_fibre.sum(axis=0)
        n = fit.count.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            own = sf / sd
            pooled = sf.sum(axis=1) / sd.sum(axis=1)  # (c, b)
            overall = sf.sum(axis=(1, 2)) / sd.sum(axis=(1, 2))  # (c,)
        n_pooled = n.sum(axis=1)
        ratio = np.empty(SHAPE)
        level = np.empty(SHAPE, dtype=np.int8)
        for c in range(N_COMPONENTS):
            for lay in range(N_LAYERS):
                for b in range(N_BINS):
                    if n[c, lay, b] >= MIN_CLASS_CELLS and np.isfinite(own[c, lay, b]):
                        ratio[c, lay, b], level[c, lay, b] = own[c, lay, b], 0
                    elif n_pooled[c, b] >= MIN_CLASS_CELLS and np.isfinite(
                        pooled[c, b]
                    ):
                        ratio[c, lay, b], level[c, lay, b] = pooled[c, b], 1
                    else:
                        ratio[c, lay, b], level[c, lay, b] = (
                            overall[c] if np.isfinite(overall[c]) else 0.0,
                            2,
                        )
        return cls(ratio, level)

    def predict_layers(self, test: Statistics) -> np.ndarray:
        """Predicted total fibre energy per layer (summed over events and cells)."""

        sd = test.sum_deposit.sum(axis=0)
        return (self.ratio * sd).sum(axis=(0, 2))

    def fallback_share(self, test: Statistics) -> list[float]:
        contribution = self.ratio * test.sum_deposit.sum(axis=0)
        total = contribution.sum()
        return [float(contribution[self.level == k].sum() / total) for k in range(3)]


def actual_layers(test: Statistics) -> np.ndarray:
    """Geant4 total fibre energy per layer, orphan energy included."""

    return test.sum_fibre.sum(axis=(0, 1, 3)) + test.orphan_fibre.sum(axis=(0, 1))


def closure(table: RatioTable, test: Statistics) -> np.ndarray:
    """Predicted over Geant4 minus one, per layer (the rule of ``sampling_response_validation.layer_closure``)."""

    return table.predict_layers(test) / actual_layers(test) - 1.0
