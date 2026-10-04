# DECISIONS — master list

One line per decision, named by what it decides. Reasons and evidence are NOT here: follow the link. Newest first inside each table.
Record IDs (DEC-, EXP-, RQ-, H-) belong to the research-record format and are given in the last column only.
Old labels (Block 6B, D5, C2, ...) are translated in `research/NAMES.md`.
**To add a decision:** one new row, same columns, link to the plan or vault note. Status is `accepted` only if the researcher accepted it.

## Multiscale analysis and the proton model

| Date | Decision | Status | Details | Record |
|---|---|---|---|---|
| 2026-10-02 | Proton steps 4-8 (crossing repair, interacting model, batch path, freeze) started on branch `fastmc-validated-dataset`, reading the chat message "plan and implement the remaining part of the stochastic event generation" as the "approval to start" of item 7 of the weekend plan | accepted 2026-10-02 (researcher: "Yes to all") | `research/plans/2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md` section 7 | |
| 2026-10-04 | Implementation formulation of the interacting model (DEC-005 realised as per-offset empirical ln-energy quantile tables + Gaussian copula with a common factor and AR chain + back-edge shift + log-space amplitude mapping, not the literal amplitude x profile product) and of the lateral model (quanta, core-fraction logit, Pareto weights) | **interpretation, awaiting the researcher** (the researcher asked to implement DEC-005 and to continue; the change of formulation was not separately accepted) | `research/STATE.md` branch section, 2026-10-04 | DEC-005 |
| 2026-10-02 | Crossing repair refinements: shared burst latent across readout and deposition; downstream extent as a latent; three-parameter bulk coupling (a third element beyond the two accepted repairs) | accepted 2026-10-02 (researcher: "Yes to all"; Q6, Q7) | same plan, section 5c | |
| 2026-10-02 | Acceptance rules for the multiscale estimator controls: at most 10% false positives against the baseline's 95% interval; detection power at least 90%; only unsaturated windows count; recovery shown against a 15% reference line; verdict USABLE only if all rules hold | accepted | `research/plans/2026-10-02_multiscale_estimator_validity_preregistration.md` | DEC-013 |
| 2026-10-02 | Fraction of cascade replicates that must keep the ordering D0 >= D1 >= D2: 0.90 | **interpretation, awaiting the researcher** (the verdict does not depend on it) | same file, section 7 | DEC-013 |
| 2026-10-02 | The first multiscale check uses orders q = 0, 1, 2 only, and box sides 1, 2, 3, 6, 9, 18 on the 18 x 72 image | accepted | same file | DEC-013 |
| 2026-10-02 | The combined 18 x 72 image is primary; per-view images are secondary | accepted | `research/plans/2026-10-02_weekend_multiscale_check_and_proton_model_completion_plan.md` | DEC-012 |
| 2026-10-02 | Order of the first multiscale check: estimator controls, then a proton first look on calibration events only, then electron versus proton | accepted | same plan | DEC-012 |
| 2026-09-30 | Estimator-validity gate: no multiscale or fractal claim until the estimators pass controls; FastMC need not reproduce fractal structure | accepted | `research/STATE.md` | DEC-010 |
| 2026-09-30 | Lateral spill: crossing protons get a per-layer lateral spill | accepted | `research/STATE.md` | DEC-009 |
| 2026-09-30 | Burst latent: a structural event-level burst (type, onset, energy, downstream extent), not a scale multiplier | accepted | `research/STATE.md` | DEC-008 |
| 2026-09-30 | Validation contract plus added checks (layer correlations, classifier two-sample test, sparsity, ungated multiscale panel), registered before the fresh test set | accepted | `research/STATE.md` | DEC-007 |
| 2026-09-30 | Shared interaction draw; deposition and readout fluctuations may differ; deposition is the common electron/proton representation | accepted | `research/STATE.md` | DEC-006 |
| 2026-09-30 | Interacting-proton factorization: amplitude x universal profile + upstream albedo latent + lateral-width latent, with a back-edge factor, a correlated residual in log-energy space (normality tested first) and a shared albedo latent | accepted | `research/STATE.md` | DEC-005 |
| 2026-09-29 | Proton model: analytic exponential interaction depth + empirical conditional distributions; two representations; QGSP_BERT only as a systematic | accepted | `research/plans/2026-09-29_proton_shower_model_implementation_plan.md` | DEC-004 |
| 2026-09-28 | Proton model before detector response | accepted | `research/STATE.md` | DEC-003 |
| 2026-09-28 | EM event generator regime switch (`sampling` default, `deposition`) | implemented; acceptance not separately recorded | `research/STATE.md` | |
| 2026-09-21 | First publication is the multiscale study; FastMC is a smooth control; QML is downstream | accepted | `research/PUBLICATION_ROADMAP.md` | DEC-002 |
| 2026-09-21 | EM event generator: one lognormal depth variable T0 per event, beta fixed at 0.65, entry-referenced origin | accepted | vault `04 Decisions/DEC-001 ...` | DEC-001 |

## Detector, geometry and representation (collected from notebooks 00-04)

Stated in the notebooks, collected here. "Fact" rows are documented AMS values used as constants, not choices.

| Source | Decision | Status | Details |
|---|---|---|---|
| nb 00 | Origin at the front-face centre, +z into the ECAL | convention | `configs/geometry.yaml` |
| nb 00 | Geometry constants from AMS documentation: 648 x 648 mm, 166.5 mm deep, 17 X0, 9 superlayers, 18 readout layers, 72 cells per layer, 9 mm pitch | fact | `configs/geometry.yaml` |
| nb 00 | The 18 layer reference depths are uniformly spaced (mean slice 9.25 mm) | simplifying assumption | nb 00 |
| nb 00 | No effective E_c or Moliere radius assigned to the lead-fibre composite until sourced | stated; check against current `configs/geometry.yaml` | nb 00 |
| nb 00, 02 | The 21-cell track-centred crop is a planned preprocessing hypothesis that needs a width ablation | open | nb 00, nb 02 |
| nb 01 | Track direction: theta in [0, pi/2), phi in [0, 2 pi); theta = pi/2 excluded; reference point need not lie inside the ECAL | convention | nb 01 |
| nb 02 | Cell index = floor((u - u_min) / 9 mm), u_min = -324 mm, grid half-open [-324, 324) | convention | nb 02 |
| nb 02 | The fibre axis is not the measured coordinate; each layer supplies ONE transverse coordinate (alternating views) | convention | nb 02 |
| nb 02 | The 18 x 72 image is **not** a 72 x 72 pixel plane; a per-view image is 9 x 72 | convention | nb 02 |
| nb 03 | One canonical `ECALEvent` for FastMC and Geant4, same 18 x 72 energy-grid convention | accepted | `src/ams_ecal/event.py` |
| nb 03 | Not defined by the event model: shower development, random generation, noise, thresholds, saturation, cropping, ML inputs, labels, serialization | scope | nb 03 |
| nb 04 | Ideal geometry kept separate from detector response; noise, thresholds, saturation, dead channels, attenuation, gain, calibration, alignment excluded until the detector response model | scope | nb 04 |

## Shower model conventions (collected from notebooks 05-07)

| Source | Decision | Status | Details |
|---|---|---|---|
| nb 05 | Mean longitudinal profile is a deterministic gamma, integrated over the 18 finite layers, with explicit longitudinal leakage | accepted | `src/ams_ecal/longitudinal.py` |
| nb 06 | Mean lateral profile is deterministic; calibrated on 3-180 GeV test-beam data, higher energies are extrapolation | accepted, limitation | `src/ams_ecal/lateral.py` |
| nb 06 | Electron and positron mean profiles treated identically | assumption | nb 06 |
| nb 06 | Cell marginal assumes an effectively infinite fibre direction, so fibre-end leakage is not modelled | limitation | nb 06 |
| nb 06 | Fluctuations belong to stochastic generation; detector response comes later | scope | nb 06 |
| nb 07 | The EM event generator fluctuates only the shape, through one variable T0; lateral profile deliberately deterministic | accepted | nb 07 |
| nb 07 | Excluded from the EM event generator: fluctuating beta, explicit shower-start sampling, independent per-layer draws, a two-variable (T, alpha) model, a stochastic lateral model, microscopic transport, proton showers | scope | nb 07 |
| nb 07 | The test suite does not assert that the ensemble mean equals the deterministic profile (it holds for T0 only) | accepted | nb 07 |

## Geant4 pilot analysis choices (collected from notebook 08 and `research/STATE.md`)

Fixed before the production data was read.

| Source | Decision | Status | Details |
|---|---|---|---|
| nb 08 | Truth first interaction = first step of the primary whose defining process is hadronic and named `*Inelastic`; a secondary-based finder is the cross-check | operational definition | nb 08 |
| nb 08 | MIP scale measured from truth-non-interacting events | analysis choice | nb 08 |
| nb 08 | MIP band = 99% upper quantile of non-interacting scintillator energy (95% and 99.9% as variations) | analysis choice | nb 08 |
| STATE | Thresholds 0.5 x MIP (varied 0.25-1) | analysis choice | `research/STATE.md` |
| nb 08 | Depth-dominance statistic S_D = bias-adjusted epsilon^2 over 10 equal-count depth bins (5 and 20 as variations); 95% bootstrap / Wilson intervals; an association, not a causal index | analysis choice | nb 08 |
| STATE | Detector image is scintillator only; entry point spread uniformly over one 9 x 9 mm cell; production cut 0.7 mm (0.1 mm checked); normal incidence only | analysis choice | `configs/geant4_proton_pilot.yaml` |
| STATE | The proton model decision belongs to the researcher; the pilot proposal is not an implementation | accepted | `research/STATE.md` |

## Data and experiments

| Date | Decision | Status | Details | Record |
|---|---|---|---|---|
| 2026-10-02 | Electron sample: 10, 20, 50, 100 GeV x 1000 events, FTFP_BERT, AMS-only, base seed 20270000 | accepted; finished | `configs/geant4_electron_sample.yaml` | DEC-012 / EXP-009 |
| 2026-10-02 | Electron timing probe: 20 events at 10 and 100 GeV, cost only | accepted; done | `configs/geant4_electron_probe.yaml` | DEC-012 / EXP-009 |
| 2026-10-02 | Fresh proton test set generated EARLY and SEALED (never opened, hashed, disjoint seeds), opened once at final validation; the added validation checks are registered first | accepted | weekend plan, section 5 | DEC-012 |
| 2026-09-30 | Proton model sequence, steps 1-10; claim order: calibration observation, FastMC hypothesis, fresh-set validation, physics-list robustness, interpretation | accepted | `research/STATE.md` | DEC-011 |
| 2026-09-29 | Held-out events `event_index % 4 == 3` are not used for fitting or exploratory analysis | accepted | `research/plans/2026-09-29_proton_dependency_analysis.md` | |

## Process and tools

| Date | Decision | Status | Details |
|---|---|---|---|
| 2026-10-02 | Names describe the work. No "Block N", "Slice N", or numbered decision/criterion labels in new work; chronology is kept by dates, notebook numbers, step numbers and the plan, experiment and decision records | accepted | `research/NAMES.md` |
| 2026-10-02 | Notebooks are written only after a model is fully implemented, to explain the final model. Decisions, reasoning and experiment results are recorded in this file, the dated plans and the results records, not in notebooks | accepted | `research/NAMES.md` |
| 2026-10-02 | Research OS v0.6.2: installed Obsidian adapters (Excalidraw, Bases, Breadcrumbs) are used by default | accepted, released | Research OS `references/V0.6_IMPLEMENTATION.md` |
| 2026-10-01 | Research OS v0.6.1: at least one visual per tutoring block; verification tiers (producer agent, separate reviewer, human) | accepted, released | same file |
| 2026-10-01 | The producing agent verifies teaching visuals by independent recomputation | accepted | same file |
| 2026-10-02 | Delayed learning check scheduled for 2026-10-04 09:00 | scheduled | vault `Tutor Sessions` |
| 2026-09-30 | Obsidian holds curated knowledge; the repo is the source of truth for the current implementation | accepted | `CLAUDE.md` |

## Open and pending (nothing here is decided)

| Item | Needed from |
|---|---|
| Confirm 0.90 as the ordering fraction (the verdict does not depend on it) | researcher |
| Whether to add a window rule stricter than the saturation-window rule (it alone was too lenient on six scales) | researcher |
| Eligibility count (occupied cells per event) before any multiscale value is read on Geant4 events | researcher |
| Detector response stays blocked until its evidence and adversarial pass runs after proton model step 10 | plan after step 10 |
| Approval for the burst latent and lateral spill repair (EXP-005), the interacting model (EXP-006), the alternative physics-list calibration (EXP-007) | researcher, per item |
| Backsplash, incidence angle, material systematic: before or after the proton model | researcher |
| Which statistic `T_bar` represents; whether `deposition` should use G&P homogeneous constants | researcher |
| Snippet-level sources in the estimator pre-registration still to be verified | source-verifier pass |
| Production-cut sensitivity before any sub-cell multiscale claim | follow-up to the first check |
| Numbers of the added validation checks (`2026-10-02_added_validation_checks_preregistration.md`) and scikit-learn as a `validation` dependency group | ACCEPTED 2026-10-02 ("Yes to all"); the sealed Geant4 sets may now be generated, after the frozen artifact is rebuilt from a clean commit |
| First validated dataset = perfect-event (`deposition`) version, detector-level version second | ACCEPTED 2026-10-02 (\"Yes to all\") |
| Sealed sets: add off-anchor energies 14, 30, 70 GeV; add an electron sealed set; use the electron sample for the EM contract | ACCEPTED 2026-10-02 (\"Yes to all\") |
| If the EM generator separates from Geant4 electrons at cell level, extend it with a calibrated cell-level fluctuation (reverses DEC-001's exclusion) | researcher, after the electron contract (Q4) |
| Split the detector response into a Geant4-calibrated sampling response (R-A) and the evidence-gated instrumental effects (R-B) | ACCEPTED 2026-10-02 (\"Yes to all\") |
| Detector-response evidence pass is incomplete: the discovery scout's result is thin and unreviewed; per-paper outcomes (Research OS v0.7 rule) are not recorded | source-verification pass, then adversarial pass |
