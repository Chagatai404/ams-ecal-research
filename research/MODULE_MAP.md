# MODULE MAP — what each file is for in the model

One page that says, for every module in `src/ams_ecal/`, what part of the FastMC (or of the evidence about it) it is, which test
file checks it, and where its results and decisions live. It is a lookup table, not a plan: reasons stay in `DECISIONS.md` and the
dated plans in `plans/`. Old labels in file names are translated in `NAMES.md`.

## Folder layout

Source and tests use the same folders, so a module and its test are always at the same relative path.

```text
src/ams_ecal/                      tests/
├── detector/                      ├── detector/
├── electron_model/                ├── electron_model/
├── electron_studies/              ├── electron_studies/
├── proton_model/                  ├── proton_model/
├── multiscale_estimators/         ├── multiscale_estimators/
├── validation/                    ├── validation/
└── geant4_simulation/             └── geant4_simulation/
```

Import as `ams_ecal.<folder>.<module>`; run as `python -m ams_ecal.<folder>.<module>`. The top-level `ams_ecal` still exports the
event, configuration and geometry classes (`from ams_ecal import ECALEvent, load_geometry, ...`).

**Status words** (used in the last column, so "calibration" never reads as "validated"):

- **generator** — code that produces events; part of the model.
- **calibration** — fits parameters of a model on *exposed development* data. Not a validation.
- **analysis only** — measures Geant4 or the model to inform a decision; changes nothing in the generator.
- **control** — a planted-answer or null check that tells us whether an estimator can be trusted.
- **validation** — compares a frozen model with *held-out or sealed* data. Sealed sets are opened only through `validation/sealed_set.py`.
- **infrastructure** — geometry, I/O, configuration, guards.

## 1. `detector/` — detector and event foundations (shared by electrons and protons)

| Module | Test | What it is in the model | Status |
|---|---|---|---|
| `geometry.py` | `test_geometry.py` | Idealised AMS-02 ECAL: 9 superlayers, 18 layers, 72 cells per layer, alternating orientation. Config `configs/geometry.yaml`. | infrastructure |
| `crossing.py` | `test_crossing.py` | Exact fibre path length of a straight track through the cells. | infrastructure |
| `projection.py` | `test_projection.py` | Fine energy deposits to the canonical alternating readout. | infrastructure |
| `tracking.py` | `test_tracking.py` | `TrackState` and the projection of a track to a z plane. | infrastructure |
| `readout.py` | `test_readout.py` | Fibre orientation, coordinate to cell index, track projection to cell indices. | infrastructure |
| `event.py` | `test_event.py` | The canonical event record (`ECALEvent`) and its provenance (`EventProvenance`) that every generator fills. | infrastructure |
| `transport_geometry.py` | `test_transport_geometry.py` | Detailed-transport geometry derived from the canonical ECAL. | infrastructure |

## 2. `electron_model/` — the electron / EM generators (DEC-001 baseline; em_production Slice 1 candidate)

| Module | Test | What it is in the model | Status |
|---|---|---|---|
| `longitudinal.py` | `test_longitudinal.py` | Mean longitudinal profile: gamma form, shower maximum depth, deposition and sampling regimes. | generator |
| `lateral.py` | `test_lateral.py` | Mean lateral profile of an EM shower (deterministic in the scaled depth). | generator |
| `stochastic.py` | `test_stochastic.py` | Event generation: lognormal T0, alpha = 1 + 0.65 T0, lateral spots. This is **DEC-001**: kept as the historical baseline and null control; not rewritten. | generator |
| `fastmc_config.py` | `test_fastmc_config.py` | Validated loading of `configs/fastmc.yaml` (electron model parameters, E_c = 7.6 MeV). | infrastructure |
| `em_production.py` | `test_em_production.py` | **em_production Slice 1**: the gamma-family longitudinal generator (calibrated origin convention and beta, empirical in-prefix tail correction, covariant widths, skew-normal ln T with a Gaussian copula) on DEC-001's lateral grid; the verified parameter artifact. Candidate, development. | generator |
| `em_production_calibration.py` | `test_em_production_calibration.py` | Calibrates the Slice 1 parameters on the exposed baseline electrons (origin fixed at the front face; contained-fraction tolerance) → `data/calibration/em_production/gamma_baseline_v1/parameters.json`. | calibration |

## 3. `electron_studies/` — evidence about the generator, not the generator

These are the files that grew fastest. They run on the **exposed** Geant4 electrons only; the sealed electron set has not been opened.
Results: `results/em_generator/`. Plan and notes: `research/plans/2026-10-05_em_*`, `2026-10-06_em_*`. Nothing here writes into `electron_model/`.

| Order | Module | Test | Question it answers | Status | Output in `results/em_generator/` |
|---|---|---|---|---|---|
| 1 | `em_exploration_check.py` | `test_em_exploration_check.py` | How does the DEC-001 generator compare with the exploration Geant4 electrons, row by row of the validation contract? (strong disagreement) | analysis only | `exploration_comparison.json` |
| 2 | `em_longitudinal_fluctuations.py` | `test_em_longitudinal_fluctuations.py` | Do per-event (ln T, ln alpha) fluctuate like the Grindhammer–Peters forms? Includes the planted-correlation control. Its helpers (`layer_bounds_x0`, per-event fits) are reused by 3 and 4. | analysis + control | `longitudinal_fluctuations.json` |
| 3 | `em_longitudinal_structure_analysis.py`, `em_longitudinal_structure_plots.py` | `test_em_longitudinal_structure_analysis.py` | Longitudinal structure of Geant4 electrons: joint T/alpha, event-level beta, how beta and rho depend on the depth-origin convention, counterfactual ensembles, confounders of the lateral–longitudinal coupling. (Decision rows F1-F5.) | analysis only | `longitudinal_structure_*.json/csv`, `longitudinal_structure/*.png` |
| 4 | `em_depth_origin_calibration.py`, `em_depth_origin_calibration_plots.py` | `test_em_depth_origin_calibration.py` | Can one common depth origin z0 plus one mean beta explain the Geant4 mean profile at 10–100 GeV? Identifiability surface, bootstrap, per-energy fits, calibrated-coordinate GP checks, the first-layer-floor gate, deposition versus readout. (Decision rows 2026-10-06.) | calibration + control | `depth_origin_calibration*.json/csv`, `depth_origin_calibration/*.png` |
| 5 | `em_profile_shape_study.py`, `em_profile_shape_study_plots.py` | `test_em_profile_shape_study.py` | Why can the single gamma not reach the 100 GeV profile? Residual structure of the per-event fits, held-out score of the extensions (energy-dependent beta or origin, depth law, a tail component with the literature decay length), the ln T skewness controls and consequence for the leakage, the AMS b = 0.65 table on deposition and readout, the literature formulas against the electrons. | analysis + control | `profile_shape_study*.json/csv`, `profile_shape_study/*.png` |
| 6 | `em_extended_tail_study.py`, `em_extended_tail_study_plots.py` | `test_em_extended_tail_study.py` | What does the extended-depth sample (`configs/geant4_electron_extended.yaml`, 270 layers) say about the tail: is it the same physics as the baseline, did the prefix-only fits predict the energy behind layer 17, the decay length and its energy dependence, how much of the event-level leakage the prefix shape explains. | analysis + control | `extended_tail_study*.json/csv`, `extended_tail_study/*.png` |
| 7 | `em_production_development_check.py`, `em_production_development_check_plots.py` | `test_em_production_development_check.py` | How close is Slice 1 to Geant4 (calibration set and a development hold-out) and to DEC-001, against the Geant4 sample-to-sample floor? Not a validation; the sealed set is not read. | analysis + control | `em_production_development_check*.json`, `em_production_development_check/*.png` |

Reading order for a newcomer: 1 → 2 → 3 → 4 → 5 → 6 → 7, then `research/plans/2026-10-05_em_production_generator_plan.md` and `DECISIONS.md` (DEC-014 and the F-rows).

## 4. `proton_model/` — the proton model

| Module | Test | What it is in the model | Status |
|---|---|---|---|
| `proton_config.py` | `test_proton_config.py` | Validated `configs/fastmc_proton.yaml`. | infrastructure |
| `proton.py` | `test_proton.py` | Interaction draw and the crossing branch (shared draw, DEC-006). | generator |
| `proton_structure.py` | `test_proton_structure.py` | Burst latent and lateral spill of crossing protons (DEC-008, DEC-009). | generator + calibration |
| `proton_interacting.py` | `test_proton_interacting.py`, `test_proton_interacting_branch.py` | Layer energies of interacting protons; full two-branch generator. | generator |
| `proton_lateral.py` | `test_proton_lateral.py` | How a layer's energy is spread over cells (scale, quanta). | generator |
| `proton_batch.py` | `test_proton_batch.py` | Batched, process-pool generation; identical to the per-event generator. | infrastructure |
| `proton_calibration.py` | `test_proton_calibration.py` | **The proton calibration**: data split, tables, artifact → `results/proton_model/`. | calibration |
| `proton_dependency.py`, `proton_checks.py` | `test_proton_dependency.py`, `test_proton_checks.py` | Dependency analysis (step 0) and paired variant checks that decided what the generator must keep explicit. | analysis only |
| `proton_structure_check.py` | *(no test file of its own; exercised through `validation/added_checks.py` and `proton_model/physics_list_systematic.py`)* | In-sample fit-quality check of the crossing structure and the interacting model. | analysis only |
| `proton_validation.py` | `test_proton_validation.py` | Comparison with **held-out** Geant4 events. | validation |
| `physics_list_systematic.py` | `test_physics_list_systematic.py` | Proton model against other hadronic physics lists: a systematic, not a validation. | analysis only |

## 5. `multiscale_estimators/` — the estimator-validity gate

| Module | Test | What it is | Status |
|---|---|---|---|
| `multiscale.py` | `test_multiscale.py` | Multiscale estimators of non-negative deposit images (EXP-008, DEC-010). | analysis only |
| `multiscale_controls.py` | `test_multiscale_controls.py` | Synthetic controls and pre-registered acceptance rules; run with `python -m ams_ecal.multiscale_estimators.multiscale_controls` → `results/multiscale_controls/`. | control |

## 6. `validation/` — dataset, added checks and the sealed-set guard

| Module | Test | What it is | Status |
|---|---|---|---|
| `dataset.py` | `test_dataset.py` | Assembly of the electron/proton dataset and the dataset-level checks. | infrastructure + control |
| `added_checks.py` | `test_added_checks.py` | The added validation checks as pre-registered; each must see a planted difference. → `results/added_checks/`. | control |
| `sealed_set.py` | `test_sealed_set.py` | Hash manifest, final-only guarded reader and opening ledger for the sealed proton and electron sets. Neither sealed set has been opened. | infrastructure (protocol guard) |

## 7. `geant4_simulation/` — Geant4 truth, backend and pilot

| Module | Test | What it is | Status |
|---|---|---|---|
| `geant4_backend.py` | `test_geant4_backend.py` | Geant4 transport backend (pure helpers always tested; transport only where Geant4 is installed). | infrastructure |
| `geant4_truth.py` | `test_geant4_truth.py` | Truth-level first inelastic interaction from Geant4 records. | infrastructure |
| `pilot_analysis.py`, `pilot_report.py` | `test_pilot_analysis.py`, `test_pilot_report.py` | Observables, statistics and report of the Geant4 proton pilot → `results/geant4_proton_pilot/`. | analysis only |

## Test conventions (what a passing test means here)

- **Planted answer:** synthetic data with a known truth; the code must recover it (and must *not* find an effect when none is planted).
- **Null / control:** a case with no effect, to show the estimator does not invent one.
- **Exact identity:** algebra that must hold to numerical precision.
- **End to end:** the real analysis on a small slice of the exposed data, writing tables and plots to a temporary directory.
- No test opens a sealed set; `validation/test_sealed_set.py` and the exploration and physics-list tests check the refusal.

## Old paths that are kept on purpose

On 2026-10-06 the flat `src/ams_ecal/<module>.py` layout was grouped into the folders above, and the slice-labelled modules were renamed
(`em_f5_analysis` → `em_longitudinal_structure_analysis`, `em_f5_plots` → `em_longitudinal_structure_plots`,
`em_origin_beta_calibration` → `em_depth_origin_calibration`, `em_origin_beta_plots` → `em_depth_origin_calibration_plots`; results `f5_*`/`f5/` →
`longitudinal_structure_*`, `f6_*`/`f6/` → `depth_origin_calibration_*`). Every reference in code, tests, notebooks and the research records
was rewritten. Only these still show the **old** paths, because they feed provenance, and they are not edited:

- comments in `configs/*.yaml` (the configuration files are hashed into digests);
- `data/calibration/*/manifest.json` (`"module": "ams_ecal.proton_calibration"`, built at commit 8c8a5b9; the manifest hash covers the arrays,
  not this string; new builds record `ams_ecal.proton_model.proton_calibration`).

To translate an old path: `ams_ecal.<name>` is `ams_ecal.<folder>.<name>` (folder from the tables above).
`results/block6b/` keeps an old label on purpose (see `NAMES.md`).
