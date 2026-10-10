# Datasets for the multiscale test — PLAN (for the researcher's approval; nothing here is built)

_Drafted 2026-10-09 at the researcher's request: "make a complete plan, verified by the literature and Geant4 simulations we made to produce the final datasets to start testing our multiscale hypothesis on; we have been spending a week on this, we should speed up."_

Status: **a plan.** Section 8 lists the decisions that block it. Reads with `research/PUBLICATION_ROADMAP.md` (Paper 1), `research/STATE.md` (RQ-001, H-001),
`2026-10-02_multiscale_estimator_validity_preregistration.md`, `2026-10-02_multiscale_estimator_controls_results.md`,
`2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md`, `research/sealed/README.md`.

## 1. The finding that sets the order: FastMC fidelity is not on the critical path of the first test

The registered core evidence path of Paper 1 is **fine-grained Geant4 -> AMS-like projection -> does the signal survive -> incremental information -> classical ML**.
The roadmap states that FastMC is "a smooth phenomenological control and development simulator" that "cannot by itself establish or falsify microscopic
fractal/multifractal cascade structure". The week of lateral and longitudinal electron work was FastMC fidelity. It is needed for one arm (does FastMC preserve the
multiscale observables) and for cheap large samples, **not** for the first test of H-001.

Geant4 is cheap here: 4000 electrons took about 51 minutes and 4000 protons at 100 GeV 710 s on 12 workers (plan of 2026-10-02, section 2). Every Geant4 run stores the
deposition grid, the readout grid and, for the sealed configs, the fine prefix deposits (`store_fine_prefix_deposits: true`). **The fastest honest route is to produce
the test datasets with Geant4 and to run the FastMC arm in parallel.** I have not independently re-derived the timings this session; they are the recorded values.

## 2. What exists (verified by listing the repository today)

| set | content | status |
|---|---|---|
| `geant4_electron_sample` | e-, 10/20/50/100 GeV x 1000, base seed 20270000 | exposed: development |
| `geant4_electron_extended` | e-, same energies x 1000, 270 layers (prefix = hold-out for FastMC) | exposed: development |
| `geant4_proton_pilot` | p, 10/20/50/100 GeV x ~4000, normal incidence; events with `event_index % 4 == 3` are held out | calibration and a held-out quarter |
| `geant4_electron_sealed` | e-, 7 energies (10, 14, 20, 30, 50, 70, 100) x 1000, base seed 20271002, fine prefix stored | sealed, never opened |
| `geant4_proton_sealed` | p, same 7 energies x 3000, base seed 20271001, fine prefix stored | sealed, never opened |

Fixed by the existing configs: FTFP_BERT, AMS-only geometry, 0.7 mm production cut, entry spot over one 9 mm cell, **normal incidence only**.

## 3. What the literature and the controls already support, and what they do not

| claim the test depends on | support | gap |
|---|---|---|
| Showers are cascades with scale-dependent structure worth testing | the repository's cascade controls (EXP-008) recover the exact cascade value | EXP-008 is synthetic; **no claim about real showers exists** |
| The estimators work at AMS granularity | usable only with about 1000-3000 hits (2 of 13 and 4 of 14 windows); coarse pair (9, 18) from 100 hits, fine pair (1, 3) needs 3000 | **the eligibility count (occupied cells per event) of real Geant4 events was never run** |
| Multifractal / intermittency analysis of showers in the literature | the estimator pre-registration cites it | its snippet-level sources are **still unverified** (open item in DECISIONS); not read in full text |
| EM lateral/longitudinal reference (Grindhammer-Peters, Kounine, AMS papers, PDG) | read in full text, with locators (2026-10-06/07 notes) | concerns FastMC, not the estimators |
| The Geant4 simulation is a trustworthy truth | one physics list, one cut, normal incidence; `cut_0p1mm` and `QGSP_BERT` variants exist for protons; the electron EM-option variant (EXP-007) was **never run** | systematics before any sub-cell claim |

Consequence: the plan has an explicit **literature verification gate** (phase 1) and a **systematics gate** (phase 3). Under Research OS v0.8, a source is labelled by what
was read (full text, abstract, snippet); a snippet is a gap, and nothing downloads without your permission (I give links).

## 4. The datasets to produce

All with provenance: clean commit, per-file SHA-256, disjoint seeds, a manifest; `deposition` is the common representation (DEC-006, F3), `readout` is stored alongside.

| id | purpose | content | source |
|---|---|---|---|
| G-dev | design and debug the observables, choose windows, set tolerances | existing exposed electron and proton Geant4 (section 2) | exists |
| G-train | the classical-ML ladder and the sample-efficiency curves (they need statistics the exposed sets do not have) | e- and p, 10/20/50/100 GeV, **10,000 per class per energy**, fine deposits stored, entry spot as the pilot | NEW Geant4 run, about 4-6 h on 12 workers (scaled from the recorded timings) |
| G-final | the one registered test: survival through readout, incremental information, e/p separation | the **sealed** sets (7 energies; the 14, 30, 70 GeV interpolation points included) | exists, unopened: opened ONCE, through the opening ledger |
| G-syst | is the structure a Geant4 artefact | electrons with `FTFP_BERT_EMZ` and `FTFP_BERT_EMV` and with a 0.1 mm cut, at 20 and 100 GeV x 2000 each; the proton variants (QBBC, QGSP_BERT, 0.1 mm) already exist | NEW, small; needs approval to run Geant4 |
| F-v0 | the FastMC arm: does a smooth, calibrated generator preserve the observables | FastMC electrons (Slice 2 candidate or its successor) and protons at the G-train energies and counts, perfect-event version | uses existing generators; labelled candidate, not validated |

Matching rules, written before any look: equal events per class per energy; the same entry spot; energy matched by construction (the same grid); containment and
total deposited energy reported per class so that a width or depth difference is not read as multiscale structure. Normal incidence only, stated on every figure.

## 5. Phases, gates and time

| phase | work | gate (what must be true to continue) | time |
|---|---|---|---|
| 0 | decisions in section 8; park the lateral extension (E2 stays a recorded result; the checks running now are finished and logged, not extended) | your approval of this plan | today |
| 1 | **eligibility count** on G-dev (occupied cells per event, deposition and readout, fine vs 18 x 72), per class and energy; literature verification pass of the multiscale/intermittency sources the pre-registration cites (full text where the library has it, links to you where it does not) | the eligible fraction is stated per energy; every source has an outcome (verified, gap, rejected) | 1-2 days |
| 2 | **pre-register the test** (observables, windows, controls, tolerances, the confound controls of section 6, the analysis tree) before G-train exists | you confirm the numbers; they are not changed after | 1 day, overlaps phase 3 start |
| 3 | generate G-train; generate G-syst; run the systematics check on G-dev-size subsets | provenance complete; systematics reported per observable | 1-2 days (mostly compute) |
| 4 | development analysis on G-dev and G-train only: estimator behaviour, survival through readout, ladder baselines | all development decisions frozen in a tagged commit | 2-3 days |
| 5 | **one opening of G-final** through `sealed_set.open_set`; the registered tests run once | the verdict is reported row by row, including the negative outcomes the roadmap lists as publishable | 1 day |
| P | in parallel, off the critical path: F-v0 and the FastMC arm; the lateral extension work resumes only if you ask | none blocks phases 1-5 | as available |

About **8-10 working days to the first registered verdict**, against an open-ended FastMC refinement. This is an estimate from the recorded run times and the work above, not a measurement.

## 6. Controls that keep the result honest

1. **Conventional-feature control.** Every multiscale effect is reported conditional on width, depth, total energy, containment and hit count (the ladder step "conventional + multiscale"). Differences that reduce to width or depth are an outcome of the paper, not a failure.
2. **Estimator validity gate (DEC-010).** No multiscale value is read on events or windows the controls rule unusable; the eligible fraction is part of every result.
3. **Fine vs projected.** The same events analysed at the fine scale and after the 18 x 72 projection, so "present in transport, destroyed by segmentation" is measurable.
4. **Smooth null.** `em_smooth_null` (DEC-001) and the smooth baseline of EXP-008 are run through the same pipeline.
5. **Systematics.** The G-syst variants, per observable, before any sub-cell statement.
6. **No classifier tuning.** The classifier two-sample test stays an omnibus diagnostic; no generator or observable is tuned to it.
7. **Opening discipline.** G-final is opened once; every threshold and tolerance is fixed in code before; none relaxed afterwards.

## 7. Stated limitations that travel with the datasets

Normal incidence only (no angular calibration exists). One physics list and one cut in the main sets. Perfect-event (deposition) data first; the detector-level version
needs the instrumental effects (R-B), which are still blocked on the evidence pass and are **not** part of this plan. FastMC electrons are distinguishable from Geant4
(classifier AUC 1.000, measured today); they are a smooth control, so that is a documented property, not a defect of the first test.

## 8. Decisions I need from you (in the order they block work)

1. **Approve the order**: Geant4-first for the multiscale test; FastMC refinement moves to a parallel, non-blocking track. (If you disagree because the paper must use FastMC samples, the plan changes: say which claim needs them.)
2. **Approve generating G-train** (10,000 per class per energy; Geant4 compute) and **G-syst** (needs your approval to run Geant4).
3. **Confirm the sealed-set allocation**: the sealed e/p sets serve as G-final for the multiscale test AND as the validation of FastMC. They can be opened once only. Proposal: the multiscale test opens them first; FastMC validation then uses the opening's fine-grained comparison, or you authorise new sealed electron/proton sets for FastMC (cheap, about a day of compute).
4. **Confirm the pre-registration numbers** in phase 2 when drafted (orders q = 0, 1, 2; box sides 1, 2, 3, 6, 9, 18; the 0.90 ordering fraction is still an open interpretation).
5. **Literature**: permission to run the verification pass of the multiscale sources; I give you links for papers I cannot read, and add nothing to the library myself.

## 9. What I will not do without your word

Open any sealed set; build anything in phases 1-5 before approval; change DEC-001, the proton model, or the sealed-set rules; call anything validated; download literature.

## 10. Phase 1 result: the eligibility count (2026-10-09, exposed events only, no multiscale value computed)

Code `src/ams_ecal/multiscale_estimators/eligibility_count.py`, result `results/multiscale_controls/eligibility_count.json`. Electrons: the 4 x 1000 baseline sample; protons: the calibration events of the pilot baseline (`event_index % 4 != 3`). The conversion from the controls' "hits" to occupied cells (200 control images each): **100 hits = 81 cells, 300 = 176, 1000 = 336, 3000 = 572.** The controls found the estimators usable from the 1000-hit images (2 of 13 windows) and better at 3000.

| class, GeV | deposition cells (median) | readout cells (median) | events at or above 336 cells (deposition / readout) | at or above 572 |
|---|---|---|---|---|
| electron 10 | 299 | 165 | 0.03 / 0.00 | 0.00 / 0.00 |
| electron 20 | 359 | 209 | 0.90 / 0.00 | 0.00 / 0.00 |
| electron 50 | 450 | 284 | 1.00 / 0.03 | 0.00 / 0.00 |
| electron 100 | 526 | 350 | 1.00 / 0.72 | 0.05 / 0.00 |
| proton 10-100 (all) | 30-45 | 22-27 | 0.32-0.40 / 0.19-0.33 | 0.11-0.30 / 0.00-0.17 |
| proton interacting | 452-659 | 302-492 | (about 47-48% of protons) | |
| proton crossing | 21-22 | 18-19 | 0 | 0 |

What this changes (measured, not yet interpreted physically):
1. **At the AMS readout (18 x 72), most electron events are below the controls' usable regime** (165-350 cells against 336 for the 1000-hit control): the "does the structure survive the readout" question is expected to be answered mostly by a lack of resolving power at the grid level, and the estimators must not be read on events below the cell threshold. This is a finding about the measurement, not about showers.
2. **About half of the protons are crossing particles with about 20 cells**, far below any usable regime and trivially separated from electrons by the cell count. The e/p multiscale comparison is meaningful only for **interacting protons** (and electrons), matched on occupied cells and deposited energy. Every analysis is stratified by interaction.
3. **The saturation rule alone is useless as an eligibility rule:** it leaves every event "a reliable window" (1.00), as the controls' addendum already warned. The operational eligibility rule proposed for the pre-registration is the control-derived one: an event is eligible for a window only if its occupied cells are at least those of the control image at which that window passed (336 for the 1000-hit windows, 572 for the 3000-hit windows).
4. **The finer stored scales have the counts the estimators need** (electrons: 1000-3100 fibres and 4000-16000 fine voxels per event; protons about 100, crossing ones minimal), but the controls were run on the 18 x 72 grid only. **A new work item: re-run the estimator controls on the fine-scale geometry** (fibre / fine-voxel positions) before any fine-scale value is read. This is the same pre-registered procedure on a new geometry and needs the pre-registration addendum (phase 2).
5. **G-train protons are raised to 20,000 per energy** (about 48% interact, so about 9,600 interacting per energy matches the 10,000 electrons); the cost is about one hour at 100 GeV from the recorded timing (estimate).

## 11. First development reading of the registered statistics (2026-10-09; exposed events; NOT the registered test)

Code `multiscale_test.py` (statistics) and `multiscale_development.py` (driver), result `results/multiscale_controls/multiscale_development.json` (60 bootstraps; electrons: the 1000 baseline events per energy; protons: truth-interacting calibration events, 1408-1453 per energy). Sealed sets unread.
- **Deposition grid, electrons against interacting protons, matched on occupied cells (five strata):** at 20, 50 and 100 GeV every one of the 28 registered statistic-window rows at the two largest windows is above the registered rule (matched KS 0.27-0.63 against a split-half floor of 0.05); for example D0 - D2 over all six scales: 0.37 / 0.51 / 0.63 at 20 / 50 / 100 GeV. At 10 GeV only 3% of electrons are eligible, so nothing is compared.
- **Readout grid:** nothing at 10 and 20 GeV (0% of electrons reach 336 cells); 50 GeV compares only the 3% of electrons that are eligible; 100 GeV (72% eligible) gives matched KS 0.31-0.57. This is the plan's expected loss of resolving power at the AMS readout, now measured.
- **Matching on the deposited energy is not usable** (no common support: electrons deposit nearly all of the beam energy, interacting protons are spread; matched KS is NaN at 20 GeV and 0.78-0.96 at 100 GeV where only the thin overlap remains), so the occupied-cell matching carries the comparison.
- **What this does NOT show:** (1) the difference may reduce to conventional shape (width, depth, hit count) that the cells matching does not remove; that is the ladder's job (Q-D) and has not been run; (2) five cell strata can leave residual confounding (D0 depends on the cell count by construction); (3) electrons are 1000 events, protons are interacting-truth only, a selection that real data cannot make; (4) exposed development data, used for design, so it is not evidence for H-001; (5) the electron samples are Geant4 FTFP_BERT, one list, one cut, normal incidence.
- Next: the conventional-feature ladder and the proton-rejection metric on G-dev, then on G-train when it exists; the fine-scale estimator controls.

## 12. First development reading of the information ladder (2026-10-09; exposed events; NOT the registered test)

Code `multiscale_ladder.py`, result `results/multiscale_controls/multiscale_ladder_development.json` (100 bootstraps; the two raw-grid rungs not fitted in this run; electrons 1000, interacting protons 1408-1453 or all calibration protons 3000 per energy).
- **The conventional features already separate the classes perfectly:** AUC 1.000 (0.999 at the lowest) at every energy, both grids, both proton sets. Proton rejection at 90% and at 99% electron efficiency equals the number of protons available (no proton passes), so the rejection is only bounded from below (greater than 1400-3000).
- **Consequently the registered added-information rule cannot be met:** the gain of conventional + multiscale over conventional is 0.0000 (interval within +-0.003) everywhere, "adds information" is False everywhere, and the 0.01 AUC criterion has no headroom. This does NOT show the multiscale features add nothing: it shows this setting cannot ask the question.
- **Multiscale features alone** reach AUC 0.83 / 0.99 / 1.00 / 1.00 (deposition, interacting protons, 10-100 GeV) and 0.70 / 0.77 / 0.82 / 0.95 (readout); at low energies on the readout much of that is the eligibility indicator itself (NaN versus not NaN follows the cell count), not a dimension.
- **Why it is easy here (hypothesis, not tested):** the samples have fixed beam energies, so the total deposited energy and the layer profile separate electrons (deposit nearly all of the beam energy) from protons (spread, often leaving early) almost completely; AMS does not know the energy this way (it has the tracker rigidity, which would enter an E/p cut).
- **Consequence for the pre-registration (an amendment to decide before the final opening; NOT made):** the primary quantity should be the proton rejection at fixed electron efficiency, which needs far more protons than 3000 to resolve (a rejection of 1e3-1e5), and/or a harder setting for the conventional rung (for example without the total deposited energy, or within narrow energy windows, or with the energy treated as unknown). G-train (10,000 electrons and 20,000 protons per energy) lifts the resolvable rejection to about 2e4 at best; reaching AMS-relevant rejections needs a large simulated sample, which is the purpose of the FastMC arm. The 0.01 AUC rule is kept as registered until the researcher decides.

## 13. Large-sample development reading (2026-10-10; exposed Geant4, 10 / 20 / 50 GeV; NOT the registered test; sealed sets unread)

Code `multiscale_gtrain_development.py`, result `results/multiscale_controls/multiscale_gtrain_development_10_20_50GeV.json`. Electrons: 10,000 per energy; protons: 20,000 per energy, of which 9,647-9,709 truth-interacting are compared (the others are crossing and ineligible). 100 bootstraps. The 100 GeV electrons were still generating.
- **The registered statistics, matched on occupied cells (deposition grid):** D0, D1, D2 and D0 - D2 over all six scales differ between electrons and interacting protons by matched KS 0.33-0.55 at 10, 20 and 50 GeV (intervals within about +-0.05; split-half floors 0.02-0.10), every one material by the registered rule. The eligible fraction of electrons is 0.03 at 10 GeV (so the 10 GeV value rests on 3% of the electrons), 0.91 at 20 and 1.00 at 50 GeV; of interacting protons 0.68-0.80.
- **Readout grid:** nothing eligible at 10 GeV (0% of electrons) and at 20 GeV (0%); at 50 GeV only 3% of electrons are eligible but the matched KS of those is 0.27-0.56, material by the rule.
- **The ladder has no headroom, even with the harder rungs:** on the deposition grid the conventional rung gives AUC 1.0000 and no proton passes at 99% electron efficiency (rejection above 9,600, bounded by the sample); dropping the total energy gives AUC 0.9999-1.0000 and a rejection of 4,800-9,700; dropping the energy and the occupied-cell count too gives AUC 0.9999-1.0000 and a 99% rejection of 1,900-4,800 (a false-positive rate of about 2-5 x 10^-4, i.e. 2-4 protons in 9,700). The paired AUC gain from adding the multiscale features is +0.0000 (intervals within +-0.0003) in all three, at all energies, on both grids. The multiscale features alone reach AUC 0.84 / 0.99 / 1.00 (10 / 20 / 50 GeV, deposition).
- **The sample-efficiency curves are flat at the ceiling:** with only 100 events per class the harder rungs already reach AUC 0.996-0.999, and 0.9998-0.9999 from 1,000; adding the multiscale features changes nothing at any size.
- **What this means (measured, then interpretation):** in this simulation (normal incidence, perfect events, fixed beam energy, no noise, no backsplash) the conventional shower-shape features separate electrons from truth-interacting protons almost completely, so the question "do the multiscale features add information" cannot be answered by electron-versus-proton classification in this setting; the matched statistics say the dimensions differ strongly between the classes, but a perfectly separating conventional classifier shows that this difference is not independent evidence of added information. The pre-registered added-information design needs an amendment (researcher's decision) before the final opening: candidate designs are (i) the high-rejection regime, which needs about 10^6 protons (a FastMC-scale sample), (ii) a matched design that restricts to events the conventional features cannot separate, (iii) a detector-level setting (noise, angles, unknown energy), (iv) reporting the question as unanswerable here, which is a publishable negative outcome of the roadmap. None is adopted.
- **Disclosed:** the two harder rungs were added after seeing AUC 1.000 and are exploratory; all variants are reported, none selected.
