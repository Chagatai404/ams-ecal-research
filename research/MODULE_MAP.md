# MODULE MAP — what each file is for in the model

One page that says, for every module in `src/ams_ecal/`, what part of the FastMC (or of the evidence about it) it is, which test
file checks it, and where its results and decisions live. It is a lookup table, not a plan: reasons stay in `COMPLETED_WORK.md` and the
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
├── benchmarks/                    ├── benchmarks/
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
| `sampling_response.py` | `test_sampling_response.py` | R-A: the sampling response from true deposit to fibre energy as class tables of the ratio R/D, fitted per particle class (electron tables pass the registered 3% closure at 10-100 GeV; proton tables do not). | calibration |
| `sampling_response_validation.py` | `test_sampling_response_proton_split.py` | The registered closure rule and fits of R-A (electron, proton, split proton) → `results/detector_response/`. | control |
| `component_sampling_response.py`, `component_sampling_response_validation.py` | `test_component_sampling_response.py` | The component-wise (particle-agnostic) mean model on the tagged Geant4 sets and its registered criteria (not met). | analysis only |
| `sampling_response_em_share.py` | `test_sampling_response_em_share.py` | Proton tables conditioned on the electromagnetic share of the deposit (registered, not met; data-starved on the small set). | analysis only |
| `readout_response.py` | `test_readout_response.py` | R-B: Poisson photoelectrons, gain map, cross-talk, threshold with provenance labels; sensitivity scans only. | infrastructure |

## 2. `electron_model/` — the electron / EM generators (DEC-001 baseline; em_production Slice 1 candidate)

| Module | Test | What it is in the model | Status |
|---|---|---|---|
| `longitudinal.py` | `test_longitudinal.py` | Mean longitudinal profile: gamma form, shower maximum depth, deposition and sampling regimes. | generator |
| `lateral.py` | `test_lateral.py` | Mean lateral profile of an EM shower (deterministic in the scaled depth). | generator |
| `stochastic.py` | `test_stochastic.py` | Event generation: lognormal T0, alpha = 1 + 0.65 T0, lateral spots. This is **DEC-001**: kept as the historical baseline and null control; not rewritten. | generator |
| `fastmc_config.py` | `test_fastmc_config.py` | Validated loading of `configs/fastmc.yaml` (electron model parameters, E_c = 7.6 MeV). | infrastructure |
| `em_production.py` | `test_em_production.py` | **em_production Slice 1**: the gamma-family longitudinal generator (calibrated origin convention and beta, empirical in-prefix tail correction, covariant widths, skew-normal ln T with a Gaussian copula) on DEC-001's lateral grid; the verified parameter artifact. Candidate, development. | generator |
| `em_production_calibration.py` | `test_em_production_calibration.py` | Calibrates the Slice 1 parameters on the exposed baseline electrons (origin fixed at the front face; contained-fraction tolerance) → `data/calibration/em_production/gamma_baseline_v1/parameters.json`. | calibration |
| `em_production_lateral.py` | `test_em_production_lateral.py` | **em_production Slice 2**: the Grindhammer-Peters lateral spot model on Slice 1's longitudinal draws (two-component radial law at each event's own tau_i, R_M = 18 mm, equal-energy spots as a multinomial over closed-form strip fractions); the verified parameter artifact that records the Slice 1 digest. Candidate, development. | generator |
| `em_production_lateral_calibration.py` | `test_em_production_lateral_calibration.py` | Calibrates the 12 radial and 4 spot constants on the exposed baseline electrons (energy-weighted ensemble mean; spot structure fitted on layers 2-17) → `data/calibration/em_production/gp_lateral_v1/parameters.json`. | calibration |

## 3. `electron_studies/` — evidence about the generator, not the generator

These are the files that grew fastest. They run on the **exposed** Geant4 electrons only; the sealed electron set has not been opened.
Results: `results/em_generator/`. Plan and notes: `research/COMPLETED_WORK.md (was 2026-10-05_em_)*`, `2026-10-06_em_*`. Nothing here writes into `electron_model/`.

| Order | Module | Test | Question it answers | Status | Output in `results/em_generator/` |
|---|---|---|---|---|---|
| 1 | `em_exploration_check.py` | `test_em_exploration_check.py` | How does the DEC-001 generator compare with the exploration Geant4 electrons, row by row of the validation contract? (strong disagreement) | analysis only | `exploration_comparison.json` |
| 2 | `em_longitudinal_fluctuations.py` | `test_em_longitudinal_fluctuations.py` | Do per-event (ln T, ln alpha) fluctuate like the Grindhammer–Peters forms? Includes the planted-correlation control. Its helpers (`layer_bounds_x0`, per-event fits) are reused by 3 and 4. | analysis + control | `longitudinal_fluctuations.json` |
| 3 | `em_longitudinal_structure_analysis.py`, `em_longitudinal_structure_plots.py` | `test_em_longitudinal_structure_analysis.py` | Longitudinal structure of Geant4 electrons: joint T/alpha, event-level beta, how beta and rho depend on the depth-origin convention, counterfactual ensembles, confounders of the lateral–longitudinal coupling. (Decision rows F1-F5.) | analysis only | `longitudinal_structure_*.json/csv`, `longitudinal_structure/*.png` |
| 4 | `em_depth_origin_calibration.py`, `em_depth_origin_calibration_plots.py` | `test_em_depth_origin_calibration.py` | Can one common depth origin z0 plus one mean beta explain the Geant4 mean profile at 10–100 GeV? Identifiability surface, bootstrap, per-energy fits, calibrated-coordinate GP checks, the first-layer-floor gate, deposition versus readout. (Decision rows 2026-10-06.) | calibration + control | `depth_origin_calibration*.json/csv`, `depth_origin_calibration/*.png` |
| 5 | `em_profile_shape_study.py`, `em_profile_shape_study_plots.py` | `test_em_profile_shape_study.py` | Why can the single gamma not reach the 100 GeV profile? Residual structure of the per-event fits, held-out score of the extensions (energy-dependent beta or origin, depth law, a tail component with the literature decay length), the ln T skewness controls and consequence for the leakage, the AMS b = 0.65 table on deposition and readout, the literature formulas against the electrons. | analysis + control | `profile_shape_study*.json/csv`, `profile_shape_study/*.png` |
| 6 | `em_extended_tail_study.py`, `em_extended_tail_study_plots.py` | `test_em_extended_tail_study.py` | What does the extended-depth sample (`configs/geant4_electron_extended.yaml`, 270 layers) say about the tail: is it the same physics as the baseline, did the prefix-only fits predict the energy behind layer 17, the decay length and its energy dependence, how much of the event-level leakage the prefix shape explains. | analysis + control | `extended_tail_study*.json/csv`, `extended_tail_study/*.png` |
| 7 | `em_production_development_check.py`, `em_production_development_check_plots.py` | `test_em_production_development_check.py` | How close is Slice 1 to Geant4 (calibration set and a development hold-out) and to DEC-001, against the Geant4 sample-to-sample floor? Not a validation; the sealed set is not read. | analysis + control | `em_production_development_check*.json`, `em_production_development_check/*.png` |
| 8 | `em_lateral_structure_study.py` | `test_em_lateral_structure_study.py` | Which radial law describes the mean cell profile of the Geant4 electrons (two-component Grindhammer-Peters / AMS law, a one-component law, DEC-001, the published constants), and what spot-number fluctuation structure must a lateral generator reproduce (relative variance before and after conditioning on the event's centre of gravity)? Analysis for the lateral slice; nothing is generated. | analysis | `lateral_structure_study.json`, `lateral_structure_study_layers.csv` |
| 9 | `em_production_lateral_check.py`, `em_production_lateral_check_plots.py` | `test_em_production_lateral_check.py` | How close is Slice 2 (the lateral spot model) to Geant4 on the calibration set and the extended-prefix hold-out, against Slice 1 with DEC-001's lateral law (control) and the Geant4 sample-to-sample floor? Lateral observables, cell profile, lateral-longitudinal coupling, spot numbers; the proposed signs are reported, not gated. Not a validation; the sealed set is not read. | analysis + control | `em_production_lateral_check.json`, `em_production_lateral_check/*.png` |
| 10 | `em_production_lateral_e1_decision.py` | `test_em_production_lateral_e1_decision.py` | The pre-stated adoption rule for extension E1 (benefit on the hold-out for the width and the hit cells, no overfitting, at most two added parameters, no regression), written before the extension was calibrated; applied to the Slice 2 check, the E1 check and the E1 calibration record. Verdict: E1 not adopted. | control (decision rule) | `em_production_lateral_e1_decision.json` |
| 11 | `em_lateral_width_diagnostic.py` | `test_em_lateral_width_diagnostic.py` | Where does the lateral width excess of Slice 2 come from? Decomposes the width^2 of Geant4 (baseline and hold-out), of Slice 2 with its spots and of its expectation (infinitely many spots) by distance from the track and by layer, with the energy beyond several distances: separates the radial-law tail from spot granularity. Result: the far tail of the mean radial law. | analysis | `em_production_lateral_width_diagnostic.json`, `.png` |
| 12 | `em_production_lateral_tail_decision.py` | `test_em_production_lateral_tail_decision.py` | The pre-stated adoption rule for the tail-sensitive radial calibration (the width improves, no overfitting, no added parameter, no regression), written before the first run; applied to the Slice 2 check, the candidate check and the candidate calibration record. Verdict for both weights: not adopted. With `--max-added-parameters 1` it also judged the faster-tail candidate (not adopted, one sub-criterion short). | control (decision rule) | `em_production_lateral_tail_decision.json`, `em_production_lateral_tail_balanced_decision.json`, `em_production_lateral_faster_tail_decision.json` |
| 13 | `em_lateral_metric_resolution.py` | `test_em_lateral_metric_resolution.py` | What can the layers 3-17 profile-error metric resolve? Paired bootstrap of the difference of two models' profile errors over the Geant4 hold-out events, the reference-only floor and the model seed spread; it showed the 0.3-point tolerance was not derived and that +0.41 is real but concentrated at low energy. | analysis | `em_production_lateral_metric_resolution.json` |
| 14 | `em_lateral_distinguishability.py` | `test_em_lateral_distinguishability.py` | Can the registered classifier two-sample test tell a lateral candidate from Geant4? AUC with the permutation null, feature-group ablations, the single-feature separation, and the Geant4 baseline-against-hold-out floor. A diagnostic, never a tuning target. Result: AUC 1.000 for both candidates, separated by the cell occupancy. | analysis + control | `em_production_lateral_distinguishability.json`, `em_production_lateral_single_features.json` |
| 15 | `em_lateral_occupancy_diagnostic.py` | `test_em_lateral_occupancy_diagnostic.py` | What carries the missing cell occupancy? Occupied fraction, empty fraction and energy per occupied cell by distance from the track and by layer group, for Geant4 and both candidates. Result: a constant 1.2-1.5 MeV outer quantum in Geant4 against 7.7-9.1 MeV in the model. | analysis | `em_production_lateral_occupancy_diagnostic.json` |

Reading order for a newcomer: 1 → 2 → 3 → 4 → 5 → 6 → 7, then `research/COMPLETED_WORK.md (Appendix D)` and `COMPLETED_WORK.md` (DEC-014 and the F-rows).

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
| `ladder_sanity_audit.py` | `test_ladder_sanity_audit.py` | Slice S0: is the near-perfect electron/proton separation an implementation error? Same-class nulls, label permutation, fold-seed spread, duplicates and seed overlap, a duplicate-injection positive control, energy-only and logistic baselines, the eligibility-indicator-only rung, feature-family and layer-block ablations, the energy-blind pooled test. Expectations are fixed in the module docstring before any run. → `results/sanity_audit/`; record `research/COMPLETED_WORK.md (was 2026-10-11_s0_sanity_audit.md)`. | analysis + control |
| `multiscale_verdict_reread.py` | `test_multiscale_verdict_reread.py` | S3 stage 1: vectorized stratified-KS permutation engine and the valid re-read of the 192 development KS verdicts → `results/multiscale_controls/s3_reread_*`. | analysis only |
| `multiscale_shuffle_excess.py` | `test_multiscale_shuffle_excess.py` | S3 stage 2: cell-shuffle excess of the 24 statistics (full-layer and 9-cell-block surrogates). | analysis only |
| `multiscale_matched_pairs.py` | `test_multiscale_matched_pairs.py` | S3 stage 3: caliper matching on occupied cells and deposited energy, pair-swap permutation test, same-class guard. | analysis only |
| `multiscale_3d.py` | `test_multiscale_3d.py` | S4 part 1: the 3D multiscale estimator on the fine voxels with synthetic controls and a control-derived eligibility (negative under the registered rules). | control |

## 6. `validation/` — dataset, added checks and the sealed-set guard

| Module | Test | What it is | Status |
|---|---|---|---|
| `dataset.py` | `test_dataset.py` | Assembly of the electron/proton dataset and the dataset-level checks. | infrastructure + control |
| `added_checks.py` | `test_added_checks.py` | The added validation checks as pre-registered; each must see a planted difference. → `results/added_checks/`. | control |
| `sealed_set.py` | `test_sealed_set.py` | Hash manifest, final-only guarded reader and opening ledger for the sealed proton and electron sets. Neither sealed set has been opened. | infrastructure (protocol guard) |
| `sample_definitions.py` | `test_sample_definitions.py` | The one definition of every exposed Geant4 sample: where it lives (ordered data roots, nothing copied), which events are held out, which population (`all`, `interacting`, `crossing`) an analysis reads, a refusal to load sealed sets, and the data manifest `research/data_manifest.json` (`write`, `verify`). Older multiscale modules still carry their own sample definitions until S1. | infrastructure |

## 7. `geant4_simulation/` — Geant4 truth, backend and pilot

| Module | Test | What it is | Status |
|---|---|---|---|
| `geant4_backend.py` | `test_geant4_backend.py` | Geant4 transport backend (pure helpers always tested; transport only where Geant4 is installed). | infrastructure |
| `geant4_truth.py` | `test_geant4_truth.py` | Truth-level first inelastic interaction from Geant4 records. | infrastructure |
| `pilot_analysis.py`, `pilot_report.py` | `test_pilot_analysis.py`, `test_pilot_report.py` | Observables, statistics and report of the Geant4 proton pilot → `results/geant4_proton_pilot/`. | analysis only |
| `em_variant_initialisation_dump.py` | `test_em_variant_initialisation_dump.py` | Slice S0: what Geant4 sets up for an EM physics list at initialisation, with no events (EM parameters, process names), and the differences between two lists. Its scalar-getter `dump` is blind to enums and tuples (msc step limit, step function, fluctuation model); the complete comparison is `compare-logs` on the saved standard output (Geant4's printed table). Used on `FTFP_BERT_EMV`: at least seven differing settings. → `results/em_variants/`. | analysis only |

## 8. `inference/` — statistical foundations (slice S1, gates SV-1..SV-8)

Record: `research/COMPLETED_WORK.md (was 2026-10-11_s1_statistical_foundations.md)`. None of these decides physics; they replace known problems of the development ladder (review C1-C7, C12, C13). Each is validated by `coverage_simulations.py` (→ `results/inference/coverage.json`).

| Module | Test | What it is | Status |
|---|---|---|---|
| `rejection_limits.py` | `test_rejection_limits.py` | Exact Clopper-Pearson limits; rejection as a censored lower limit (k = 0 never reports N); sample sizes; four disjoint data roles; order-statistic threshold with its Beta uncertainty; `evaluate_at_efficiency`. | analysis tool |
| `cross_fitting.py` | `test_cross_fitting.py` | Repeated cross-fits with seed spread; role-separated fit/calibrate/evaluate; repeated role splits; energy-only and logistic baselines; the same-class null. | analysis tool |
| `missingness_controls.py` | `test_missingness_controls.py` | Indicator-only separation, explicit missing indicators, eligible-in-both matching, "not testable" verdicts. | analysis tool |
| `permutation_nulls.py` | `test_permutation_nulls.py` | Permutation tests within strata (+1 rule), stratified KS with counts, matched-size split floors, the cell-shuffle null (plan item A3). | analysis tool |
| `multiplicity.py` | `test_multiplicity.py` | Benjamini-Hochberg, Holm, Westfall-Young step-down, one named primary per family. | analysis tool |
| `claim_specification.py` | `test_claim_specification.py` | The estimand-to-failure claim template, with a digest. | process tool |
| `coverage_simulations.py` | `test_coverage_simulations.py` | Coverage and type-I-error simulations of all of the above, including real development-electron halves. | control |

## 9. `benchmarks/` — classifier benchmark harness

| Module | Test | What it is | Status |
|---|---|---|---|
| `learning_curves.py` | `test_learning_curves.py` | Role-separated learning curves of electron/proton classifiers: class-balanced log-loss, proton rejection with exact limits, boosted trees, logistic regression and a small GPU network; registered DRY RUN on existing development data, no claims. | infrastructure + analysis only |

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
