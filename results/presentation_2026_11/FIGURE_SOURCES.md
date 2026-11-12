# Figures and equations of the group talk, 13 November 2026

Every figure on the slides, where its code lives, how to regenerate it, what it reads, and the first 16 hex digits of the
sha256 of the exact image used in the deck (deck version 20). Run commands from the repository root with `PYTHONPATH=src`.

## Presentation figures (one module, Ilimni palette on white)

Module: `src/ams_ecal/geant4_simulation/presentation_figures_2026_11.py`
Command: `PYTHONPATH=src .venv/Scripts/python -m ams_ecal.geant4_simulation.presentation_figures_2026_11 --theme ilimni`
Output: `results/presentation_2026_11/ilimni/` (without `--theme ilimni`: the original navy palette in `results/presentation_2026_11/`).
All numbers the figures report are written to `numbers.json` in the same folder; the theme changes colours only.

| Slide | Figure | Function | Inputs | sha256 (16) |
|---|---|---|---|---|
| Electron and proton, same visible energy | P5 event displays per view | `fig_event_displays` | `data/geant4_realistic_regime_powered_{electron,proton}` | da0f00b203415118 |
| Check 1: where electron showers peak | P1 longitudinal profile and shower maximum | `fig_longitudinal` | `data/geant4_electron_sample/baseline/E{10,20,50,100}GeV` | 8b9e7b61042d5136 |
| Check 2: energy response and leakage | P2 resolution and visible fraction | `fig_resolution` | same electrons; Poisson seed 20261112 | 8f76f62d9103e520 |
| Check 3: proton interactions (left) | P3 first-interaction depth | `fig_proton_interaction` | `data/geant4_proton_pilot/baseline/E{10,20,50,100}GeV` | 4b7d0aa3e7e88417 |
| Check 3: proton interactions (right) | P4 visible fraction by band | `fig_visible_fraction_bands` | realistic-regime pilot and powered samples, fiducial | 9eed1cd0c9ac8b0d |
| The 2D estimator mixed the two projections | P6 view offset | `fig_view_offset` | same realistic-regime samples | a53f714145c1a80c |
| A finite-range slope, not a fractal dimension | P7 box counting, one event | `fig_box_counting` | one 0.8–2 TeV powered electron | 943929c0a86d5dac |
| Look-alikes still differ in D₁ and D₂ | P8 pooled matched dimensions (post hoc) | `fig_pooled_dimensions` | `data/geant4_realistic_regime_pilot_{electron,proton}/pilot`; recomputes the EXP-015 pooled matching (234 pairs, 183 valid; KS 0.087, 0.459, 0.497) | d88184fd20221e28 |
| Every interval of the test | P9 EXP-019 intervals | `fig_intervals` | `results/realistic_regime/exp019_powered_incremental_classifier/exp019_result.json` (no refit) | 4a80a35c79f8e976 |

P8 replaces the earlier one-off script `make_pooled_D.py` (session scratch, not in the repository); the same calls and seed
reproduce its numbers exactly. P9 re-plots the same rows as `powered_incremental_classifier.figure_intervals`.

## Figures produced by the experiment code (unchanged images)

| Slide | Figure | Module and function | Command | Output | sha256 (16) |
|---|---|---|---|---|---|
| Controls first: find planted structure | estimator controls | `ams_ecal.multiscale_estimators.multiscale_controls._plot` | `python -m ams_ecal.multiscale_estimators.multiscale_controls --out results/multiscale_controls` | `results/multiscale_controls/controls.png` | 65af1fdc534147d0 |
| Fibre saturation: calibrated, not validated | hottest-cell curves (EXP-017) | `ams_ecal.detector.fibre_saturation_calibration` (writes `hottest_cell_curves.png` in `main`) | `python -m ams_ecal.detector.fibre_saturation_calibration` | `results/detector_response/fibre_saturation/hottest_cell_curves.png` | d569d7c46c0de650 |
| Compare look-alikes | F2 occupancy | `ams_ecal.multiscale_estimators.realistic_regime_analysis.figure_f2_occupancy` | the rung-1 run of `realistic_regime_analysis` (`--data-root data --out results/realistic_regime/rung1`) | `results/realistic_regime/rung1/F2_occupancy.png` | 544001b848435c83 |
| Pilot: 1,500 electrons, 6,000 protons | F3 acceptance and missing energy | `...realistic_regime_analysis.figure_f3_acceptance` | same run | `results/realistic_regime/rung1/F3_acceptance_missing_energy.png` | d42ce9b2056e6a70 |

These four keep their original labels, which still contain plain-text symbols ("theta <= 12 deg", "x <= 0",
"D0 (window ...)"). Re-rendering them needs their full analyses; the slide titles above each image say what is plotted.

## Equations

Module: `src/ams_ecal/geant4_simulation/presentation_equations_2026_11.py` (LaTeX `standalone` class, then
`dvisvgm --no-fonts`; needs MiKTeX `latex` and `dvisvgm`).
Output: `results/presentation_2026_11/equations/<name>.svg` and `sizes.json`.
Each equation is written from the code that produced the number (see the module docstring). One correction against
earlier versions of the deck: the AMS resolution is `σ_E/E = 10.5%/√(E/GeV) ⊕ 1.4%`, the two terms added in quadrature
(Kounine et al., NIM A 869 (2017), Fig. 14: σ = √(10.5²/E + 1.4²) %), not a linear sum. The figure code always used
the quadrature form.
