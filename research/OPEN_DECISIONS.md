# What is waiting to be decided, and what comes next

_Consolidated 2026-10-21. Companion file: `COMPLETED_WORK.md` (what has been done, tested and decided). Status words: DECIDE (waits for the researcher) · TODO (planned, not run) · BLOCKED (named blocker) · GAP (fact not verified). Nothing here is accepted until the researcher says so; every duration and cost is an estimate, not a measurement._

_Design and evidence records behind each item are in the Obsidian vault, `01 Projects/AMS ECAL QML/Research Records/` (names in brackets below)._

## 1. Decisions waiting for the researcher

Listed in the order they block work. The agent's recorded advice is shown where it exists; every choice is the researcher's.

### 1. Push the two local commits
Commits `daa0007` and `42410c6` are not pushed (the permission check blocked the agent's push). The researcher pushes or allows it.

### 2. The 3D result (S4 part 1 was negative)
No development event is eligible for the one usable 3D window (see `COMPLETED_WORK.md` section 4.4). The P1 and P2 primaries of the amendment assume a 3D estimator that passes the controls, so the draft must name this result.
- **(a)** Accept the negative result and drop the claim that the fine-scale route is available below about 150 GeV; keep P4 and P5 as the paper's anchors.
- **(b)** Register a sensitivity check of the controls (a less concentrated smooth baseline; the ordering rule restricted to windows without saturated scales), reported beside the original result and never as a replacement chosen after the fact.
- **(c)** Look at higher-energy electrons (occupied voxels grow roughly with energy: about 16,500 at 100 GeV, so a few hundred GeV is needed), which fits an AMS-like spectrum up to the TeV but needs a production not yet approved.
- **(d)** Leave the finite-scale question to the readout-level statistics already tested and treat it as a secondary result.
[S4 part 1 record, section 3]

### 3. R-A (deposit to fibre energy)
Accept the unsplit class tables as the working version (closure 3 to 8% on the development set, 4 to 5.5% on the large tagged set; the 3% criterion is not met), disclosing the error in the degradation curves, or register one more version. Untried candidates: the crossing flag with finer share bins below 0.3, the energy scale entering jointly with the share, a component deposit (the slow-hadron part). All five registered variants failed. Electrons above 200 GeV are an unvalidated extrapolation. [R-A design record, section 20]

### 4. Energies to test (researcher requirement, 2026-10-21)
Also test at AMS-relevant energies: flight 50 to 70 GeV, and Monte Carlo 200 GeV to 2 TeV as in Hashmani et al. 2024. Development data cover 10 to 100 GeV only; R-A is fitted at 10 to 100 GeV and a tagged 200 GeV sample exists. Options: cap the electron spectrum at 200 GeV (cheapest honest choice) or tag a higher-energy electron sample. Feeds DECIDE-9.

### 5. The DECIDE-1 amendment: named primaries P1 to P5, freezing rule, null headline
The single 0.01 AUC-gain rule is replaced by five endpoint families, each with ONE named primary (DRAFT; the researcher said "Rest seems okay to proceed" on 2026-10-12, recorded as acceptance as the WORKING version only; it becomes binding when the dated addendum is committed after S4 and before any confirmatory look). Choices that are the researcher's:

| primary | what it tests | named primary (summary) | decision rule / failure meaning |
|---|---|---|---|
| P1 physics difference | interacting protons differ from electrons in the finite-scale concentration of energy at equal occupancy and equal deposited energy | D2 (q = 2) of the energy measure on 3D fine voxels, window and box sizes in mm frozen by a dated addendum after S4; electrons against truth-interacting protons, 50 GeV; stratum-weighted KS within five equal-count occupancy-and-energy strata; matched design with different primary energies, 200 GeV protons added | adjusted p at most 0.05 AND same sign under FTFP_BERT, QGSP_BERT and the 0.1 mm cut AND the planted control recovers; a null is not called "no effect" without a power statement |
| P2 readout survival | the P1 separation persists as the fine voxels are degraded to the AMS-like readout | ratio of KS at the readout to KS at the fine level, 100 GeV, over four levels (fine voxel, 9 mm cells, fibre energy via R-A, one view per layer); paired bootstrap | "survived" if KS exceeds the cell-shuffle surrogate's upper 95% limit with the fine level's sign; levels with fewer than 100 eligible events per class are "not testable", never "destroyed" |
| P3 simulator fidelity | a generator can match Geant4 on standard metrics and still differ in the finite-scale statistic | KS of D2 between generator and Geant4 events relative to the 95th percentile of the Geant4 split floor; INACTIVE until DECIDE-3 releases generator work | the multiscale statistic adds a diagnostic only if it exceeds the floor while the standard set does not flag the defect |
| P4 realistic performance | finite-scale features raise proton rejection at fixed 90% electron efficiency in an AMS-like regime | paired count of protons passing at the same electron efficiency, gradient boosting with and without multiscale features, all protons, spectrum-weighted energies and angles; INACTIVE until realistic-regime data exist | exact paired test plus a censored Clopper-Pearson lower limit; at least 3 x 10^4 evaluation protons per claimed 10^4 rejection; never a statement of actual AMS performance |
| P5 efficiency | finite-scale features let a classifier reach the same performance with fewer training events, or a smaller or faster model | difference of cross-fitted log-loss at 300 training events per class on an observable-defined hard subset of the 100 GeV readout events, averaged over 10 repeated role splits; full learning curve at 100, 300, 1,000, 3,000 per class | upper 95% limit below zero AND negative at three of four sizes; model and compute efficiency are pre-specified SECONDARIES |

Choices to confirm or change: q = 2 for P1; 50 GeV for P1 and 100 GeV for P2; electrons against truth-interacting protons for P1 but all protons for P4; KS as the statistic; alpha = 0.05.
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
- Eligibility count: the 3D rule replaces the 336/572-cell thresholds only if the controls pass; otherwise the 2D thresholds stay (64 of 192 tests testable).

## 2. Plan: the road to the machine-learning benchmarks

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
