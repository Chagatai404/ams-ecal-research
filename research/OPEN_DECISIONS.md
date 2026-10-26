# What is waiting to be decided, and what comes next

_Consolidated 2026-10-21; updated 2026-10-22 with the decisions and results of that day (R-A round, EXP-010, EXP-011 pilot). Companion file: `COMPLETED_WORK.md` (what has been done, tested and decided). Status words: DECIDE (waits for the researcher) · TODO (planned, not run) · BLOCKED (named blocker) · GAP (fact not verified). Nothing here is accepted until the researcher says so; every duration and cost is an estimate, not a measurement._

_Design and evidence records behind each item are in the Obsidian vault, `01 Projects/AMS ECAL QML/Research Records/` (names in brackets below)._

## 1. Decisions waiting for the researcher

Listed in the order they block work. The agent's recorded advice is shown where it exists; every choice is the researcher's.

### 1. Push the local commits
Commits `daa0007` and `42410c6` (branch `statistical-foundations`) and `70cfedf` and `d97c2f4` (branch `exp-010-011-ra-capped-round`, branched from it) are not pushed. The researcher pushes or allows it. Git will convert line endings of the new `.sh` driver to CRLF the next time it rewrites it, which would break it under bash; a `.gitattributes` line `*.sh text eol=lf` is the usual fix (not made).

### 2. The 3D result (S4 part 1 was negative) — (b) and (c) done on 2026-10-22, the next step is a decision
The researcher chose to do both (b) and (c). Results are in `COMPLETED_WORK.md` section 4.4.
- **(b) DONE (EXP-010, EXPLORATORY, hash 5ec9ca672a04f1c0).** No window is usable under all three baselines and both cascades, so the original NEGATIVE verdict stands as the primary result. The Geant4-mean baseline cannot be tested at a realistic occupancy within the registered hit grid (6,737 occupied voxels at 30,000 hits against 16,505 for real 100 GeV electrons), and windows starting at the four finest scales have reversed ordering (D0 < D1) even when the saturation rule (a = 1, d_f = 3) calls them reliable.
- **(c) pilot DONE (EXP-011).** Electrons at 400 GeV reach the 23,721-voxel threshold in 98% of events (41.7% at 200 GeV); interacting protons at 0.5 to 2 TeV overlap the occupancy of 100 to 400 GeV electrons; leakage behind the prefix at 1 TeV is a median 20.9%. The production for P1 (EXP-013) is **blocked** on the decision below and on the amendment edits (EXP-014).
- **DECIDE: the control design.** The 23,721 threshold comes from the window (4,5,6,7), which is not robust across baselines, so occupancy is measured but eligibility is not established. Options: **(i)** register a new control design (a Geant4-mean baseline evaluated at a realistic hit number, and a saturation rule that explains the ordering reversal), reported beside the original verdict; **(ii)** accept the negative result and drop the claim that the fine-scale route is available (original option (a)); **(iii)** leave the finite-scale question to the readout-level statistics (original option (d)). (i) costs one registration and a few CPU-hours; (ii) and (iii) cost nothing but shrink Paper 1's P1 and P2.
[S4 part 1 record, section 3]

### 3. R-A (deposit to fibre energy) — CLOSED at 8 registered versions (2026-10-22)
The researcher asked for the untried candidates first. They were registered (hash 417e7a22681d944d) and run on the old odd events (EXP-012): the crossing flag with finer share bins, the energy scale jointly with the share, and the slow-hadron share each came out worse than the unsplit baseline at two or three energies, so none was eligible and the fresh confirmation was not needed. **The unsplit class tables are the working R-A** with their disclosed error (closure 3.5 to 7.1% on one draw, 4 to 5.5% on another; the draw-to-draw spread is about 1 point). The noise floor (hash d996fae236341618) is 1.2 to 1.4% for electrons and 2.7 to 4.4% for protons, so the 3% criterion is unmeetable for interacting protons under the registered reading. Still the researcher's:
- whether to re-state the 3% criterion in noise terms (the floor compares two independent halves and may overstate by up to about 1.4x against a nearly noise-free table); a new, separately dated registration;
- whether to validate the electron tables above 200 GeV on the pilot electrons (200, 400, 800 GeV, tagged); the lookup size axis ends at 10^4 MeV, so TeV core cells may extrapolate in deposit size too;
- whether a closure error of the adopted size moves the P2 degradation-curve statistic (materiality; not yet registered).
[R-A design record, section 20 (in the backup, see `COMPLETED_WORK.md` header); registrations in `~/.research-os-backups/registrations/`]

### 4. Energies to test (researcher requirement, 2026-10-21) — pilot done, production needs sizes
Also test at AMS-relevant energies: flight 50 to 70 GeV, and Monte Carlo 200 GeV to 2 TeV as in Hashmani et al. 2024 (the energies are from a paper not re-read here; GAP). The researcher wants higher energies tested for sure. The pilot gives measured costs for 200 to 800 GeV electrons, 1 TeV extended electrons and 0.5 to 2 TeV protons (CPU-s per event 42 to 135 for electrons, 8 to 23 for protons; `COMPLETED_WORK.md` section 4.4). DECIDE: the production sizes and energies (EXP-013); a tenfold pilot (about 3,000 electrons per energy and 10,000 protons per energy) gives roughly 1,500 matched pairs at about 1e6 CPU-s, one day on 14 workers (an extrapolation). R-A above 200 GeV stays an unvalidated extrapolation unless the electron tables are validated on the tagged pilot electrons. Feeds DECIDE-9.

### 5. The DECIDE-1 amendment: named primaries P1 to P5, freezing rule, null headline
The single 0.01 AUC-gain rule is replaced by five endpoint families, each with ONE named primary (DRAFT; the researcher said "Rest seems okay to proceed" on 2026-10-12, recorded as acceptance as the WORKING version only; it becomes binding when the dated addendum is committed after S4 and before any confirmatory look). Choices that are the researcher's:

| primary | what it tests | named primary (summary) | decision rule / failure meaning |
|---|---|---|---|
| P1 physics difference | interacting protons differ from electrons in the finite-scale concentration of energy at equal occupancy and equal deposited energy | D2 (q = 2) of the energy measure on 3D fine voxels, window and box sizes in mm frozen by a dated addendum after S4; electrons against truth-interacting protons, 50 GeV; stratum-weighted KS within five equal-count occupancy-and-energy strata; matched design with different primary energies, 200 GeV protons added | adjusted p at most 0.05 AND same sign under FTFP_BERT, QGSP_BERT and the 0.1 mm cut AND the planted control recovers; a null is not called "no effect" without a power statement |
| P2 readout survival | the P1 separation persists as the fine voxels are degraded to the AMS-like readout | ratio of KS at the readout to KS at the fine level, 100 GeV, over four levels (fine voxel, 9 mm cells, fibre energy via R-A, one view per layer); paired bootstrap | "survived" if KS exceeds the cell-shuffle surrogate's upper 95% limit with the fine level's sign; levels with fewer than 100 eligible events per class are "not testable", never "destroyed" |
| P3 simulator fidelity | a generator can match Geant4 on standard metrics and still differ in the finite-scale statistic | KS of D2 between generator and Geant4 events relative to the 95th percentile of the Geant4 split floor; INACTIVE until DECIDE-3 releases generator work | the multiscale statistic adds a diagnostic only if it exceeds the floor while the standard set does not flag the defect |
| P4 realistic performance | finite-scale features raise proton rejection at fixed 90% electron efficiency in an AMS-like regime | paired count of protons passing at the same electron efficiency, gradient boosting with and without multiscale features, all protons, spectrum-weighted energies and angles; INACTIVE until realistic-regime data exist | exact paired test plus a censored Clopper-Pearson lower limit; at least 3 x 10^4 evaluation protons per claimed 10^4 rejection; never a statement of actual AMS performance |
| P5 efficiency | finite-scale features let a classifier reach the same performance with fewer training events, or a smaller or faster model | difference of cross-fitted log-loss at 300 training events per class on an observable-defined hard subset of the 100 GeV readout events, averaged over 10 repeated role splits; full learning curve at 100, 300, 1,000, 3,000 per class | upper 95% limit below zero AND negative at three of four sizes; model and compute efficiency are pre-specified SECONDARIES |

Decided 2026-10-22: q = 2 for P1 is acceptable for now, but higher energies must be tested (a q scan and the other energies form a separate secondary generalisation family with its own Westfall-Young or Holm correction, nothing promoted); KS as the statistic; alpha = 0.05. The P1 energy is set by a rule, not named: the lowest energy at which at least 50% of electrons reach the eligibility threshold, fixed before any D statistic (2D at 50 GeV as a secondary). With the pilot and the 23,721-voxel threshold that energy is 400 GeV, but the threshold is not established (see item 2), so the rule stays unapplied until the control design is decided.
Still to confirm or change: 100 GeV for P2 (a 3D statistic is untestable at 100 GeV, where only 18,219 occupied voxels is the 99th percentile); electrons against truth-interacting protons for P1 but all protons for P4; a same-class cross-energy matched null (electron against electron, proton against proton) that P1 must pass; the estimand stated as conditional on eligibility; the planted control must establish the power of D2 itself (the eligibility was derived from D0 minus D2).
- **Freezing rule.** Window, box sizes, strata and eligibility thresholds are fixed in a dated addendum written after S4 and before the first look at any confirmatory event, from rules not from data (the widest window passing the 3D controls, box sizes in mm, equal-count quintile strata). After the tag and hash the primaries cannot change; a changed primary is a new registration and the set that was looked at is consumed.
- **Planted positive control (P1).** Before P1 runs on confirmatory data the statistic must recover a planted loss of lateral organisation at fixed occupancy (a fraction phi of cells permuted within each layer; phi = 1 is the full cell-shuffle): power at least 0.90 at the planned sample size, and a false-positive rate of at most 0.10 on two independent halves of one class.
- **Multiplicity (update needed).** The draft still says Benjamini-Hochberg for secondaries; the researcher decided on 2026-10-15 Westfall-Young where the guard passes, otherwise Holm, B = 10,000, with BH only exploratory. The text must be brought in line.
- **Is a null headline acceptable for Paper 1?** The literature makes a null plausible (HEGRA precedent; EM nearly monofractal); the roadmap already lists negative outcomes as results; P5 keeps the paper reportable whatever the finite-scale features do.
- **A sixth family for model and compute efficiency, or a secondary?** The agent advises a secondary (as drafted).
[Amendment draft record, sections 4 to 5]

### 6. DECIDE-2: the sealed sets
Recorded and accepted: the existing sealed sets stay closed (an untouched reserve; opened for FastMC validation only when a frozen candidate exists); new sealed sets per purpose with 63-bit seeds, including protons at 200 GeV. Still open:
- the **exit date**: if no frozen FastMC candidate exists by that date, the purpose is re-scoped in writing;
- the **two sealed-proton seed collisions** with exposed events: regenerate those events or record them in the opening ledger (the agent advises a ledger).

### 7. DECIDE-9: the realistic regime
Incidence angle (required for a realistic regime; today only normal incidence and a 9 x 9 mm entry spot), backsplash and what lies behind the real AMS ECAL, material ordering, and the spectra and sizes (electron spectrum, proton spectrum, angular range, full-face entry, ECAL-only). Also rename the misleading "G-train" label.

### 8. Classifier list for the benchmarks
Candidates: gradient-boosted trees and logistic regression on conventional features, the same with multiscale features, a small CNN, and larger image models (MLP, CNN, ResNet, CvT-like in the style of Hashmani et al.). The harness defects must be fixed first (see section 3).

### 9. Release the independent reviewers
The agent advises a statistics review of the S1 and S3 tools soon (SV-9); the others (ML, simulation, physics, reproducibility) after the R-A choice and dataset v1.

### Smaller open items
- Confirm 0.90 as the ordering fraction (the verdict does not depend on it); whether a window rule stricter than the saturation-window rule is wanted.
- Which statistic `T_bar` represents; whether the `deposition` regime should use the Grindhammer-Peters homogeneous constants. Configuration level.
- The pre-stated signs of "gamma family inadequate" (before any sealed electron read) await confirmation.
- Re-register the KS thresholds from a split-Geant4 floor before any sealed opening (a universal KS 0.05 was not confirmed).
- Promote the fractal-inspired architectures and the three-efficiency study into the roadmap (lower urgency; waits for the hard-regime evidence).
- Sync the Research OS repository with the installed version.
- Researcher review of the agent-introduced unreviewed choices (tolerances, moment weights, spot structure, doubling constant, tail exponent, Grindhammer-Peters assumptions; listed in the Slice 2 lateral note).
- Whether to ask for a citation trace of Ruan 2014 (INSPIRE `refersto:arxiv:1312.7662`) and add the paywalled references to the library.
- Eligibility count: the 3D rule replaces the 336/572-cell thresholds only if the controls pass; otherwise the 2D thresholds stay (64 of 192 tests testable). After EXP-010 the controls do not pass under all baselines, so the 2D thresholds stay for now.
- GAP: the Obsidian folder `01 Projects/AMS ECAL QML/Research Records/` cited in the headers of both record files and in `COMPLETED_WORK.md` section 9 was not found in the vault (only `AMS ECAL QML.md` was); the R-A design record survives only in the backup `~/.research-os-backups/ams-ecal-docs-before-consolidation-2026-10-21`. Either copy the records into the vault or correct the pointers.
- Stale count: `COMPLETED_WORK.md` section 9 gives 916 tests passed on 2026-10-11; the full suite has not been rerun since the inference, benchmark, R-A-round, EXP-010 and seed-check tests were added (1,088 tests are now collected, 2026-10-26).
- Sync `research/graph/` with the prose records: it stops at DEC-013 and EXP-009, while the records cite DEC-014 and EXP-010 to EXP-014 (the machine-readable graph lags; the prose files are current).
- Repository `README.md`, `research/README.md`, `MODULE_MAP.md` and `PUBLICATION_ROADMAP.md` were brought up to date on 2026-10-26 (status, results, plan, glossary, missing modules). Keep the README's sections 5 and 6 in step with these two files when they change.
- Hashmani et al. 2024 (the source of the 200 GeV to 2 TeV and 50 to 70 GeV energies) has not been re-read; the energies are as relayed by the previous agent.
- Ollama and its `llama-server` runner were stopped on 2026-10-22 at the researcher's instruction to free memory (they respawned while the Geant4 pilot ran); start Ollama again if a tool needs a local model.

## 2. Plan: the road to the machine-learning benchmarks

**Status 2026-10-22.** Done: the R-A capped round (closed, unsplit tables), the 3D control sensitivity check (EXP-010, negative-consistent) and the high-energy pilot (EXP-011). Blocked: the production for P1 (EXP-013) on the control-design decision (item 2) and on the amendment edits (EXP-014: P1 energy rule, generalisation family, cross-energy null, the Westfall-Young and Holm text). Not started: the P2 materiality registration; validating the electron R-A tables above 200 GeV on the tagged pilot electrons; the harness defects of section 3.

**"Ready to start" (proposed).**
- *Exploratory dry run (no claims)*: done on the existing ideal-regime development data (see `COMPLETED_WORK.md` section 4.6).
- *Registered benchmarks (the Paper 1 numbers)*: conventional-feature gradient boosting, raw-image models, and the same with finite-scale features, at detector level in the AMS-like regime, evaluated for P4 and P5. All entry conditions required: (1) the amendment signed and frozen; (2) the benchmark protocol registered before any evaluation event is looked at; (3) detector-level dataset v1 produced with passing checks; (4) R-A and R-B in place with sensitivity scans; (5) harness reviewed; (6) independent reviews done; (7) role-disjoint sets and electron-calibration thresholds fixed.

| work package | depends on | what is done | researcher input | estimate |
|---|---|---|---|---|
| Housekeeping | nothing | push the commits; register component and large-proton samples in the manifest (done in code); proton reproducibility caveat recorded; unit test for `run_proton_split`; rerun the full suite; refresh these two files and the graph | push or allow | 0.5 to 1 day |
| Re-read of existing evidence (S3) | S1 tools | remaining stages: cut and EMZ variant comparison (EMV after its diagnosis), occupancy- and morphology-matched hard subsets, proton smooth-baseline look, view-aware readout statistics after DECIDE-4 box sizes; register which statistics use B = 10,000 and which a smaller B | list of secondary statistics | 3 to 5 days |
| Fine-scale and degradation analysis (S4) | S3, the 3D decision | planted-structure recovery, depth-resolved statistics, paired degradation curves fine voxel, 9 mm 3D, fibre energy (R-A), one view per layer | the 3D option (section 1, item 2) | 4 to 6 days |
| Detector-response completion | the R-A choice | R-A decision; R-B scan design (gain, photoelectrons, cross-talk, threshold) with provenance labels, no value without evidence | R-A choice; scan ranges | 2 to 3 days |
| Amendment, freeze and review | the amendment draft, S4 | update the multiplicity text; planted-control criteria; the dated addendum after S4; statistics review of the frozen code; tag and hash | items 5 and 6 above | 2 to 3 days plus review latency |
| Realistic-regime design and generator | DECIDE-9, detector-response completion | spectra, angular distribution, full-face entry, ECAL-only, backsplash and material; the backend today has normal incidence and a 9 x 9 mm spot only, so angles and full-face entry are backend changes with a control run | DECIDE-9; spectra and angular range | 3 to 5 days |
| Production | the generator and control | protons at least 10^6 (P4 needs at least 3 x 10^4 evaluation protons per claimed 10^4 rejection; about 10^6 for a measured 10^4 at 10%), electrons 10^5 to 3 x 10^5; roles train / validation / calibration / evaluation with 63-bit seeds; new sealed sets with protons at 200 GeV (about 9 h per 100,000 events, extrapolated). Estimated cost under proton E^-2.7 and electron E^-3 between 10 and 1000 GeV: 10^6 protons about 12 h, 3 x 10^5 electrons about 33 h on 14 workers | spectra, sample sizes | 2 to 4 days wall |
| Detector-level dataset v1 | production, R-A, R-B | apply the response with the scans, roles and manifest, checks D1 to D9, label-leakage check, KS thresholds re-registered | thresholds | 2 to 3 days |
| Benchmark protocol and harness | the amendment text | registered protocol: classifier list and budgets, tuning discipline, energy-only, logistic and same-class baselines, metrics (censored Clopper-Pearson rejection, log-loss learning curves, parameters, time), nested or whole-pipeline bootstrap uncertainty with a coverage test, leakage controls; the machine has an RTX 4050 (6 GB) and 15 GB RAM; 10^6 events x 18 x 72 is 5.2 GB in float32 | the classifier list | 4 to 6 days |
| Independent reviews | the frozen code and protocol | statistics (SV-9), ML (leakage, fair comparison), simulation, physics, reproducibility | the command to release them | latency unknown |
| Learning and defence track | throughout | sampling calorimetry, the meaning of the ratios, fractals from zero, permutation and multiplicity, one claim card per result, spaced retrieval; a defence viva before writing | start T1 | in parallel |

**Critical path.** Housekeeping, then S3 and detector-response completion in parallel, then S4, then the amendment and freeze with review, then realistic-regime design, production, dataset v1, and the registered benchmarks. The protocol and harness run in parallel once the amendment text is settled. If decisions arrive promptly the chain is about 4 to 6 weeks of working time, 3 to 6 days of it compute; the estimate is dominated by the decisions and the review latency.

**What would change the plan.**
- A null result in S3 or S4 (the literature-plausible outcome): the benchmarks still run because P5 and efficiency curves are reportable; the claims shrink, not the programme.
- R-A not meeting 3% closure in the realistic regime: report the disclosed systematic in the degradation curves; a new R-A version only by the researcher's go.
- Electron energies above 200 GeV: R-A is an extrapolation; cap at 200 GeV or tag a higher-energy sample.
- Cost: cloud or fewer events change the schedule, not the design; the 6 GB GPU needs small batches and float16 for ResNet or CvT-like models.
- The reviews may invalidate a result; their findings are fixed before the registration, not after.

## 3. Harness defects to fix before the benchmark protocol is registered

1. Same-class null: at least 2,500 per class and a tolerance stated in standard errors (3 standard errors of the AUC), with independent pools per seed.
2. Evaluation set: at least 3 x 10^4 evaluation protons so rejection is not capped by the evaluation size; report the censored lower limit as such.
3. The ideal regime cannot discriminate rungs at 1,000 or more events per class: the registered learning curves belong to the detector-level, realistic-regime dataset, with the P5 hard subset defined by observables and frozen in the amendment addendum.
4. At least 10 repeats and an uncertainty procedure that accounts for fold correlation (nested or whole-pipeline bootstrap), not the spread over seeds.
5. Re-check the unexplained +0.0085 logistic offset on independent pools.

## 4. Planned work never started (or open)

- Electron generator: layer-to-layer residual noise; occupancy mismatch against Geant4 (one time-boxed attempt only after the multiscale-fidelity endpoint is defined).
- Proton model: final validation on the sealed set (step 10); QGSP_BERT calibration; final report; teaching notebook; backsplash.
- Estimator controls: a cell-shuffle control inside the estimator module and controls on a finer 2D grid (the 3D route changes how the second is needed).
- Fractal-preservation gate comparing fine Geant4, the 18 x 72 projection and FastMC (needs the fine-image definition and a FastMC at fidelity).
- Scale-removal ablations; raw 18 x 72 baselines in the style of Hashmani et al.; learnable multiscale representation and fractal-inspired architectures (gated, not in the roadmap papers); QML (Paper 2, gated, nothing started).
- Production-cut sensitivity before any sub-cell claim; the EMV anomaly (seven differing settings; further swaps not approved).
- Research OS v0.7 rule: per-paper outcomes for the detector-response literature pass.

## 5. Learning and ownership

- T1 (sampling calorimetry from zero) waits for the researcher to start. A decision-readiness diagnostic and curriculum are in the Obsidian vault (`Tutor Sessions`); its first run (2026-10-21) showed strong procedures and recurring gaps in what an absent or bounded result licenses and in writing the working down.
- A defence viva (Research OS `defend-research`) before writing, after the curriculum.
