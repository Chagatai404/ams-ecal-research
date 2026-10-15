"""R-B, the reduced instrumental-effect ladder B1-B4, applied to fibre energy per cell.

Registered in ``research/plans/2026-10-12_detector_response_model_proposal_DRAFT.md`` (D6.2-D6.7); evidence in
``research/plans/2026-10-12_s5a_detector_response_evidence_record.md``.

Only effects that AMS cannot invert are modelled. Attenuation, the raw cell-to-cell gain spread, the HG/LG ratio,
temperature and the PMT position effect are corrected by AMS before shower variables exist, so they are NOT added.

Order of the default chain: B1, B2, B4, B3. ``threshold_before_gain`` runs B1, B3, B2, B4 instead (the D6.6 check
of the undocumented ordering of the gain error and the threshold).

* B1 photoelectron statistics: ``N ~ Poisson(pe_per_mip * E / MIP)``, returned as ``N * MIP / pe_per_mip``.
  8 p.e. per MIP cell is documented (Spinella et al. 2010, Sec. II).
* B2 residual gain error: one factor ``1 + s z`` per cell, drawn once per detector realisation
  (``draw_gain_map``) and applied to every event of that realisation. 2.4%, 7.2% and 1.5% are documented
  equalization precisions (Sun et al. 2021; Hu et al. 2023); using them as a Gaussian spread is an assumption.
* B4 cross-talk: a fraction ``x`` of each cell moves to its two neighbours in the same layer (``x/2`` each).
  Portions that would leave the grid stay in the cell, so the total is conserved. No source for the value.
* B3 readout threshold: values below ``threshold_mips * MIP`` are set to zero. No source for the value.

``MIP_FIBRE_MEV`` is the fibre-only energy of a MIP in one cell in this repository's readout grid; the 7 MeV
quoted in the instrument paper is the loss in all material and must not be used on fibre energy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

MIP_FIBRE_MEV = 0.5596007704734802
"""Fibre-only MIP energy per cell (equals ``validation.added_checks.MIP_CELL_MEV``; a test asserts the equality)."""
N_LAYERS = 18

PROVENANCE = {
    "pe_per_mip": "documented (about 8 p.e. per MIP cell; position along the cell not stated)",
    "gain_sigma": "documented as an equalization precision (2.4%, 7.2%, 1.5%); Gaussian spread is an assumption",
    "crosstalk_fraction": "no source",
    "threshold_mips": "no source",
}


@dataclass(frozen=True)
class ReadoutResponseConfig:
    """One point of the degradation ladder. All defaults switch the step off (identity)."""

    pe_per_mip: float | None = None
    """B1 photoelectrons per fibre-MIP cell; ``None`` switches B1 off."""
    gain_sigma: float = 0.0
    """B2 relative standard deviation of the per-cell gain factor."""
    crosstalk_fraction: float = 0.0
    """B4 fraction of a cell's value moved to its two neighbours."""
    threshold_mips: float = 0.0
    """B3 threshold in units of ``MIP_FIBRE_MEV``."""
    threshold_before_gain: bool = False
    """D6.6: apply B3 before B2 (default is gain error first)."""

    def __post_init__(self) -> None:
        if self.pe_per_mip is not None and self.pe_per_mip <= 0:
            raise ValueError("pe_per_mip must be positive")
        if not 0.0 <= self.gain_sigma < 0.3:
            raise ValueError("gain_sigma must be in [0, 0.3)")
        if not 0.0 <= self.crosstalk_fraction <= 1.0:
            raise ValueError("crosstalk_fraction must be in [0, 1]")
        if self.threshold_mips < 0:
            raise ValueError("threshold_mips must not be negative")

    def labelled(self) -> dict[str, dict[str, object]]:
        """Every parameter with its value and provenance label (acceptance check 7)."""

        return {
            f.name: {
                "value": getattr(self, f.name),
                "provenance": PROVENANCE.get(f.name, "ordering choice"),
            }
            for f in fields(self)
        }


def draw_gain_map(sigma: float, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """One detector realisation of the residual gain factors, shape ``(layers, cells)``."""

    if not 0.0 <= sigma < 0.3:
        raise ValueError("sigma must be in [0, 0.3)")
    return 1.0 + sigma * rng.standard_normal((N_LAYERS, n_cells))


def photoelectron_smearing(
    fibre_mev: np.ndarray, pe_per_mip: float, rng: np.random.Generator
) -> np.ndarray:
    """B1: Poisson photoelectron statistics, mean preserving."""

    mean_pe = pe_per_mip * fibre_mev / MIP_FIBRE_MEV
    return rng.poisson(mean_pe) * (MIP_FIBRE_MEV / pe_per_mip)


def crosstalk(values: np.ndarray, fraction: float) -> np.ndarray:
    """B4: move ``fraction`` of each cell to its two neighbours along the cell axis, conserving the total."""

    if fraction == 0.0:
        return values
    half = 0.5 * fraction * values
    out = (1.0 - fraction) * values
    out[..., 1:] += half[..., :-1]
    out[..., :-1] += half[..., 1:]
    out[..., 0] += half[..., 0]  # the share that would leave the grid stays in the cell
    out[..., -1] += half[..., -1]
    return out


def apply_readout_response(
    fibre_mev: np.ndarray,
    config: ReadoutResponseConfig,
    rng: np.random.Generator,
    gain_map: np.ndarray | None = None,
) -> np.ndarray:
    """Run B1-B4 on ``(events, 18, cells)`` fibre energy.

    ``gain_map`` must be the same array for every call of one detector realisation. It is required when
    ``gain_sigma > 0`` and ignored otherwise.
    """

    if fibre_mev.ndim != 3 or fibre_mev.shape[1] != N_LAYERS:
        raise ValueError("fibre_mev must be (events, 18, cells)")
    values = np.asarray(fibre_mev, dtype=np.float64)
    if config.pe_per_mip is not None:
        values = photoelectron_smearing(values, config.pe_per_mip, rng)
    if config.gain_sigma > 0 and (
        gain_map is None or gain_map.shape != (N_LAYERS, values.shape[2])
    ):
        raise ValueError(
            "gain_map of shape (18, cells) is required when gain_sigma > 0"
        )
    cut = config.threshold_mips * MIP_FIBRE_MEV

    def gain(v: np.ndarray) -> np.ndarray:
        return v * gain_map[None, :, :] if config.gain_sigma > 0 else v

    def threshold(v: np.ndarray) -> np.ndarray:
        return np.where(v < cut, 0.0, v) if cut > 0 else v

    if config.threshold_before_gain:
        return crosstalk(gain(threshold(values)), config.crosstalk_fraction)
    return threshold(crosstalk(gain(values), config.crosstalk_fraction))
