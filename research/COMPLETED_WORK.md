# What has been done

_Consolidated 2026-10-21 from `STATE.md`, `DECISIONS.md`, `RESEARCH_PLAN.md` and 55 dated plans and reviews, which were then removed. The companion file is `OPEN_DECISIONS.md` (what is waiting for the researcher and what is planned next)._

_The originals are kept in three places: the git tag `docs-before-consolidation-2026-10-21`, a backup in `~/.research-os-backups/ams-ecal-docs-before-consolidation-2026-10-21` (with a SHA-256 list), and the Obsidian vault folder `01 Projects/AMS ECAL QML/Research Records/` (evidence and design records)._

## 0. How to read this file

Every result carries one evidence label.

| label | meaning |
|---|---|
| ESTABLISHED | checked against a primary source, or passed a test that was registered before it was run |
| DEVELOPMENT | measured on exposed (development) data; not validated; never used as a final claim |
| EXPLORATORY | analysis not powered or registered after seeing data; no claim follows |
| NEGATIVE | a registered test that did not give a usable result |
| CLOSED | tried under a pre-stated rule and not adopted; kept as a result |
| GAP | a literature fact or number not verified |

Nothing in this file is called "validated". The sealed Geant4 sets are unopened. Everything below is simulation (Geant4 11.4.1 or FastMC), never AMS data.

## 1. The project in one page

**Question (RQ-001).** To what extent do electron/positron and proton showers show discriminative multiscale spatial structure, how much of it survives AMS-02 ECAL-like readout, and what does it add beyond conventional calorimetric observables and standard machine learning?

**Program order (accepted).** Establish shower structure, test detector survival, quantify information content, classical ML benchmarks, and only then QML. Paper 1 is the multiscale shower-information study; FastMC is a smooth control; QML is downstream (Paper 2).

**Scope (researcher, 2026-10-10).** (1) Does a fractal feature exist and survive the readout; (2) efficiency in all three senses (sample efficiency, model/compute efficiency, electron selection at fixed rejection) together with information beyond conventional features; (3) fractal-inspired CNN and vision-transformer architectures (gated candidates, not yet in the roadmap papers); (4) detector effects (R-A sampling response and R-B instrumental effects).

**Where the evidence stands (honest summary).**
- In ideal Geant4 (fixed known energy, normal incidence, zero-threshold readout) electron against proton is trivially separable: AUC 0.9999 to 1.0000, and at least 0.95 from any single feature family. No implementation error was found.
- Multiscale features add nothing detectable beyond conventional features there. The registered 0.01 AUC-gain rule is unmeetable by construction (its largest possible gain is at most 1e-4).
- Every proton-rejection number so far is a zero-failure floor, not a measurement. The valid 95% lower limit is about N/3.
- The finite-scale statistics do separate the classes at the 2D-testable level, but after matching on occupied cells and deposited energy the difference is not detectable with 100 to 136 pairs (exploratory).
- The 3D estimator is not usable on any development event under the registered rules (negative).
- The literature makes a null or small-gain result plausible (see section 8). Survival through the AMS readout is untested.

## 2. Standing rules

- One researcher decides what becomes accepted knowledge. Do not turn an analogy into a mechanism, a hypothesis into a fact, or an exploratory result into a conclusion.
- Sealed sets are opened once, through `sealed_set.open_set`, with a registered harness. Rules and thresholds are fixed in code before a run and never relaxed after it.
- The classifier AUC is a diagnostic, never a tuning target.
- Nothing is called validated before the sealed verdict; say calibrated, candidate or development.
- No literature PDF is downloaded without permission. Commit only when asked; never push unasked.
- Names describe the work (no "Block N" or "Slice N" in new work). Notebooks are written only after a model is final.
- Obsidian holds curated knowledge; the repository is the source of truth for the implementation.

## 3. Data inventory

| set | content | status |
|---|---|---|
| Proton pilot | 10/20/50/100 GeV x 4,000 protons, normal incidence, FTFP_BERT, plus variants (fixed entry, QBBC, QGSP_BERT, extended depth, 0.1 mm cut at 10 and 100 GeV, material) | development; the held-out quarter `event_index % 4 == 3` was consumed by the crossing look |
| Electron sample | 10/20/50/100 GeV x 1,000, base seed 20270000 | development |
| Extended-depth electrons | 4 x 1,000, 270 layers, base seed 20271010 | development (measures the longitudinal tail) |
| Development-large sets | electrons 4 x 10,000, protons 4 x 20,000 (seeds 20272001/2, commit `48a4a78`); cut, EMZ and EMV electron variants at 20 and 100 GeV | development; located through `research/data_manifest.json` and `validation/sample_definitions.py` (files not moved) |
| Component-tagged sets | five particle classes, and 4 x 20,000 protons (base seed 20274001) with the Geant4 7-channel component tags | development; proton reproducibility caveat: 6 of 300 events at 10 GeV, and three same-configuration seed collisions, recorded in the sample notes |
| Sealed protons | 7 energies (10, 14, 20, 30, 50, 70, 100 GeV) x 3,000, base seed 20271001 | UNOPENED; two seed collisions with exposed events are known (derived from the seed policy, no sealed read) |
| Sealed electrons | 7 energies x 1,000, base seed 20271002 | UNOPENED; the final fresh electron validation set |

Details of the sealed sets (generation incident, what was read before sealing, the opening guard) are in `research/sealed/README.md`.

## 4. Results by area

### 4.1 FastMC electron generator

- **DEC-001 generator (smooth null), ESTABLISHED as implemented, DEVELOPMENT as a model.** Gamma longitudinal profile with one lognormal depth variable T0, beta fixed at 0.65, entry-referenced origin, deterministic lateral profile. A regime switch (`sampling` default, `deposition`) pairs mean depth and fluctuation width coherently. AMS publishes no mean-depth formula (verified in two AMS papers), so the PDG offset is our choice. At 100 GeV: s = 0.1069, T-bar = 8.985 X0 (deposition) or 8.319 X0 (sampling). It is kept permanently as `em_smooth_null`.
- **First out-of-sample comparison with Geant4 electrons (2026-10-05): disagreement.** KS 0.6 to 1.0 on most contract observables, classifier AUC 1.00 at every energy, mostly from cell occupancy. DEC-001 was not rewritten; a new generator `em_production` supersedes it for production (DEC-014).
- **em_production Slice 1 (longitudinal), DEVELOPMENT.** Locked gamma family, origin fixed at the front face as a coordinate convention, beta 0.525, an empirical in-prefix tail correction, Grindhammer-Peters covariant widths, skew-normal ln T with a Gaussian copula. Layers 0 to 3 error 2 to 5% against 42 to 62% for DEC-001; mean leakage error 0.2 to 0.85 points against 3.4 to 5.0. Not at the Geant4 sample-to-sample floor; contained fraction 0.4 to 0.9 points high (the 0.3-point tolerance was not met).
- **Analyses behind it (DEVELOPMENT).** Calibrated origin z0 = -0.98 X0 and beta = 0.571 on the deposition profile; AMS b = 0.65 is an origin convention, not a different shower; ln T skewness (+0.8 to +1.1) is real; the measured tail beyond the 18-layer prefix is exponential with attenuation length 3.6 X0 (5.1, 6.7, 9.2, 11.3% of E behind the prefix at 10, 20, 50, 100 GeV); tail components fitted on the prefix are an empirical shape correction, not a physical tail.
- **em_production Slice 2 (lateral spots), DEVELOPMENT.** Grindhammer-Peters two-component radial law with 12 calibrated constants (R_M = 18 mm), per-event tau_i, equal-energy spots. Cell profile in layers 3 to 17 within 1.2 to 2.2 points; the lateral-longitudinal coupling has the right sign and size. Not achieved: width (KS 0.7) and hit-cell count (182 against 334 occupied cells at 20 GeV).
- **Four lateral extensions, all CLOSED under pre-stated rules:** E1 (spot cap and quantum), tail-sensitive radial calibration (two weights), faster-falling tail (missed one sub-criterion by 0.11 points; the 0.3-point tolerance was the agent's analogy to the 0.5-point leakage floor, not derived), and E2 (fixed absolute tail-spot energy 1.36 MeV: fixed occupancy within 3% but regressed width, coupling and spot number).
- **Open:** occupancy mismatch against Geant4; layer-to-layer residual noise never started; unverified Grindhammer-Peters assumptions (E in GeV, Z = 82); R_M 18 mm against 27.6 mm (Rossi).
- **Under DECIDE-3 refinement is paused.** No FastMC sample is used in any performance or confirmatory claim.

### 4.2 Geant4 proton pilot and the FastMC proton model

- **Pilot (ESTABLISHED for Geant4 11.4.1 in our material model).** Infrastructure with a scintillator-only detector image, step-based truth, a geometry audit (43,155 fibres, no overlaps, mass 475.4153 kg) and bit-for-bit reproducibility. P(no inelastic interaction in the prefix) = 0.512 to 0.530 against exp(-0.6) = 0.549; effective interaction length about 250 to 262 mm. Crossing protons give 11.4 to 13.1 MeV of scintillator energy. Interacting visible energy is broad and left-tailed. Depth dominates where the energy is deposited but not how much, and deep-calorimeter correlations do not transfer to the prefix. FTFP_BERT, QBBC and QGSP_BERT agree on interaction probability and depth law; the visible-energy scale at 20 to 100 GeV is model-dependent (QGSP_BERT medians about 13 to 28% lower).
- **Proton model (DEVELOPMENT, in-sample only).** Analytic exponential interaction depth plus empirical conditional distributions; crossing branch with burst latent and lateral spill; interacting branch as per-offset empirical ln-energy tables with a Gaussian copula (a calibrated phenomenological surrogate, not a derived hadronic model); batch path about 18 ms per event. The crossing branch was validated once on held-out events and failed event-total, hit-multiplicity, max-cell and containment rows (KS 0.10 to 0.37), which the repairs addressed; the sealed validation (step 10) has not run. Residuals named: lowest 5% of interacting event totals 20 to 40% high; front-layer lateral law too narrow; a smooth model has not come close to a classifier two-sample test.
- **Physics-list systematic (2026-10-05).** Interacting events at 20 to 50 GeV have a visible-energy median 22 to 26% lower in QGSP_BERT; the model's own distance (KS 0.12 to 0.13) equals the list effect, so a validation tolerance tighter than that cannot be read as a statement about hadronic physics.
- **Open physics questions:** backsplash from material behind the ECAL (prefix signal 26 to 45 MeV against 11 to 13 MeV), incidence angle (only normal incidence studied), entry-phase effect, composition-matched material variant, production-cut sensitivity before any sub-cell claim.

### 4.3 Detector response (R-A and R-B)

- **R-A (deposit to fibre energy), DEVELOPMENT.** Electron class tables pass the registered 3% per-layer closure at 10 to 100 GeV. Every particle-agnostic or conditioned variant failed its registered test: the component-wise model, the split proton table, and the electromagnetic-share-conditioned table on the small and large tagged proton sets. The class-table error follows the electromagnetic share of the deposit (sign-changing, 5 to 15%) and early interactions, not the crossing/interacting split. The working R-A (unsplit class tables) has closure 3 to 8% on the development set and 4 to 5.5% on the large tagged set, so it does not meet the 3% criterion. Untried candidates: crossing flag with finer share bins, a joint energy-scale covariate, a component deposit (slow-hadron part). Electrons above 200 GeV are an unvalidated extrapolation (fitted 10 to 100 GeV; a tagged 200 GeV sample exists).
- **R-A noise floor (2026-10-22, DEVELOPMENT, registered before the run, hash in section 5).** Maximum over 18 layers of a model-free half-against-half closure, 95th percentile: electrons 1.2 to 1.4% at 10 to 100 GeV; crossing protons 2.7 to 3.2%; interacting protons 3.2 to 4.4%. Under the registered reading the 3% criterion is structurally unmeetable for interacting protons at all four energies and for crossing protons at 100 GeV (and at 50 GeV in the tagged set), partly noise for the rest, and a real-error region for electrons. The electron class tables, which pass 3%, are consistent with this. The proton closure of 4 to 5.5% on the large tagged set is above the floor, so some real error probably remains; the data do not say how much. Caveat (not registered, noted after the numbers): the floor compares two independent halves, so it carries both halves' noise; against a fitted table that is nearly noise-free the matching floor could be smaller by up to a factor of about 1.4 (protons then roughly 1.9 to 3.1%). The registered reading is kept as it stands; whether to re-state the rule is the researcher's decision. Code: `src/ams_ecal/detector/sampling_response_noise_floor.py`; result: `results/detector_response/r_a_noise_floor.json`.
- **R-A capped round, Step A (2026-10-22, DEVELOPMENT, NEGATIVE; hash 417e7a22681d944d).** Fit on the even events of the large tagged proton set, validated on its odd events (already seen by the five earlier variants), maximum per-layer closure at 10, 20, 50, 100 GeV: unsplit baseline 5.9, 3.5, 7.1, 4.6%; V1 crossing flag with finer share bins 2.8, 8.5, 10.0, 11.4%; V2 share jointly with energy 3.8, 5.1, 13.5, 17.0%; V3 slow-hadron share 3.9, 7.0, 6.1, 6.6%. The pooled-proton noise floor (95th percentile) is 3.8, 3.3, 3.1, 2.9%. Under the registered rule no candidate is eligible (each is worse than the baseline at two or three energies), so none is selected, the confirmation on a fresh tagged set is not run and not needed, and the unsplit proton tables are the working R-A with their disclosed error (the baseline here is 3.5 to 7.1% on one random draw; the earlier 4.2 to 5.5% was another draw of the same stochastic prediction, so the draw-to-draw spread is itself about 1 point). V2 leave-one-energy-out: 5.1% at 20 GeV, 13.6% at 50 GeV (no threshold). V1 improves 10 GeV only (5.9 to 2.8%). Thin bins (more than 20% of cells on a pooled fallback): V1 6 of 35 class-energy cells, V2 12 of 23, V3 0 of 20; no V2 cell was empty. The search for an R-A version is closed at 8 registered versions; a further one needs a new registration by the researcher. Code: `src/ams_ecal/detector/sampling_response_capped_round.py`; result: `results/detector_response/r_a_capped_round_selection.json`.
- **R-B evidence (literature, S5a).** AMS corrects fibre attenuation, MIP equalization, HG/LG gain (ratio about 33) and temperature before clustering, so gain-type effects must not be added as distortions. Documented stochastic effects: about 8 photoelectrons per MIP, pedestal noise with zero suppression; residual gain errors 2.4% (proton MIP) and 1.5% (helium). NOT FOUND: the zero-suppression threshold, cross-talk, the AMS Birks constant, saturation level. R-B is built as sensitivity scans only.
- **EMV anomaly (OPEN).** `FTFP_BERT_EMV` electrons have a 37% lower scintillator share at the same total deposit; EMV differs from FTFP_BERT in at least seven settings (not two, as first written). The approved swap of `ApplyCuts` and `MscRangeFactor` restored only 41% of the share (closure test failed); further swaps are not approved.

### 4.4 Multiscale estimators and controls

- **2D estimators (DONE).** D0, D1, D2 on the 18 x 72 grid, box sides 1, 2, 3, 6, 9, 18, orders q = 0, 1, 2; DEC-013 acceptance rules (at most 10% false positives against the baseline interval, power at least 90%, recovery within a 15% reference line). Synthetic controls: NOT USABLE at 300 hits or fewer; usable on 2 of 13 windows at 1,000 hits and 4 of 14 at 3,000 hits (cell-equivalent eligibility thresholds 336 and 572, flagged weak).
- **Eligibility on real events.** On the readout most electrons have 165 to 350 occupied cells (below 336) at 10 to 50 GeV; about half the protons are crossing particles with about 20 cells.
- **3D estimator on fine voxels (S4 part 1), NEGATIVE under the registered rules.** On the 216 x 216 x 36 mesh with nine scales (3.5 to 164 mm) only 1 of 36 windows is usable (18 to 82 mm), only at 30,000 hits (median 23,721 occupied voxels). Development events have a median 3,922 (electrons, 10 GeV) to 16,505 (100 GeV) occupied voxels and 1,988 to 7,112 for interacting protons, so no development event is eligible. The ordering rule D0 >= D1 >= D2 fails for windows with saturated fine scales. This is a statement about the estimator under the registered controls, not about showers. The controls depend on a smooth baseline, an equal-quanta hit model, and the ordering rule; none was changed after the numbers.
- **3D control sensitivity check (EXP-010, 2026-10-22, EXPLORATORY, registered before running, hash 5ec9ca672a04f1c0).** Beside the original NEGATIVE verdict (1 of 36 windows, (4,5,6,7) at 30,000 hits, threshold 23,721 voxels, no eligible development event), which stands as the primary result. Three baselines (original, Geant4 ensemble mean of the even 100 GeV electrons, broader analytic map), two cascades (equal hits; occupancy-matched), 8 hit counts to 30,000, 1,000 replicates, the saturation rule with a = 1, d_f = 3 (saturated windows dropped, ordering never waived). Result: no window is usable under all three, so "not usable". Under the original baseline alone the check reproduces (4,5,6,7) at 30,000 hits (threshold 23,724) and adds (5,6,7) at 15,000 and 20,000 hits (13,180 and 16,921 voxels), which fails at 30,000 (power 0.23). Under the broader map alone several coarse windows pass from 3,000 hits ((6,7), (5,6,7,8)). Under the Geant4 mean profile nothing passes at any hit count: power is 0.00 in every reliable window at 30,000 hits, because the mean profile is itself concentrated (median 6,737 occupied voxels at 30,000 hits, against 16,505 for real 100 GeV electrons) so its D0 - D2 spread already exceeds the planted cascade's, and the registered grid stops at 30,000 hits, far below the hit number that would give this baseline a realistic occupancy. The noise reference (original configuration, new seed 20261023) reproduces the original verdict, (4,5,6,7) at 30,000 hits only. Ordering: windows starting at the four finest scales are reversed (D0 < D1) in every replicate even when the rule calls them reliable, so the saturation rule does not explain the ordering failure. These are statements about what the controls can detect, not about showers. Code: `src/ams_ecal/multiscale_estimators/multiscale_3d_sensitivity.py`; result: `results/multiscale_controls/exp010_full.json`.
- **High-energy pilot (EXP-011, 2026-10-22, DEVELOPMENT, never used for P1; configs `configs/geant4_highenergy_pilot_*.yaml`, data in `data/geant4_highenergy_pilot_*`, commit 70cfedf, no tracked changes).** Fine voxels on; component tags on except the extended run. Gates written before the run: **G1 cost within 2x: passes** (CPU-s per event, mean: protons 8.2, 15.2, 22.5 at 0.5, 1, 2 TeV; electrons 42.4, 66.3, 134.7 at 200, 400, 800 GeV; extended 1 TeV electrons 258.5; total 1.5e5 CPU-s against 1.5e5 to 2.4e5 projected, about 4 h of wall time on 14 workers). **G2 memory: passes** (peak 220 to 310 MB per worker, 14 workers about 3 to 4 GB; files 45 to 107 MB per 100 to 1000 events). **G3 (lowest energy with at least 50% of electrons at or above 23,721 occupied voxels): 400 GeV** (41.7% at 200, 98.0% at 400, 100% at 800 GeV; median occupied voxels 23,334, 32,572, 44,710; the exponent falls from 0.56 to 0.62 at 10 to 100 GeV to about 0.5 at 100 to 800 GeV). **G4 (feasibility, not a power statement): pairs exist.** Interacting protons at 0.5, 1, 2 TeV have median occupancy 14,340, 16,818, 19,177, a 95th percentile 36,356, 48,515, 61,740, and 24%, 38%, 45% of them reach 23,721 voxels; they are electromagnetic-rich (median share 0.87 to 0.92). Pooling the eligible pilot events, 719 electrons and 527 interacting protons give 153 one-to-one pairs within 0.25 sd on occupied voxels and prefix deposit (442 on occupancy alone); the pilot's sample sizes are the limit and a tenfold production would give roughly 1,500, at about 1e6 CPU-s (one day on 14 workers, under the 3-day stop). No power requirement is registered for the pair count, so G4 is a cost and feasibility check only. **Leakage at 1 TeV, measured** with the extended geometry: the energy behind the 18-layer prefix is a median 20.9% of E (5th to 95th percentile 15.1 to 34.5%), against 11.3% at 100 GeV; prefix plus extension contain 99.7% of E. The 23,721 threshold comes from the S4 window (4,5,6,7), which EXP-010 showed is not usable under all baselines, so G3 measures occupancy, not an established eligibility rule.
- **Two planned controls never implemented:** a cell-shuffle negative control inside the estimator module (the shuffle surrogate was later built for S3 stage 2) and controls on a finer 2D grid.

### 4.5 Audit and statistical foundations

- **S0 sanity audit (2026-10-11).** No implementation error: same-class nulls 0.483 to 0.515 (32 pairs), permutation nulls 0.488 to 0.513, fold-seed sd at most 8.4e-5, no duplicates or shared seeds among baseline pilot and development-large sets; a duplicate-injection positive control moves the AUC from 0.50 to 0.80 and 0.91. The 0.991 against 0.998 difference is readout against deposition. The problem is overdetermined: every feature family alone gives AUC of at least 0.95; the energy-blind pooled test gives 0.99995. The registered multiscale rung equals the eligibility indicator within 0.014 at 10 GeV and on the readout at 20 to 50 GeV (missingness) and exceeds it by 0.22 to 0.30 elsewhere (real but redundant information). The held-out-quarter rule changes no AUC by more than 0.0038.
- **S1 statistical tools (2026-10-11), ESTABLISHED on simulations.** Package `src/ams_ecal/inference/` (56 tests): censored exact Clopper-Pearson rejection limits, four data roles, repeated cross-fitting, missingness controls, stratified permutation nulls and the cell-shuffle null, Holm, Benjamini-Hochberg and Westfall-Young, the claim template. 27 coverage and type-I checks pass (lowest Clopper-Pearson coverage 0.955; permutation type-I 0.030 to 0.060).
- **S3 decisions (researcher, 2026-10-15).** B = 10,000 permutations; secondaries by Westfall-Young step-down where the joint permutation draws exist (guard: a 1,000-trial same-class family-wise check per family), otherwise Holm; Holm always reported as the sensitivity; Benjamini-Hochberg only as an exploratory view; five cross-fit seeds as a project choice. Evidence: permutation p-values must use (b+1)/(B+1); Holm needs B of at least 20m - 1 (3,839 for m = 192); seed spread is not the AUC uncertainty; BH is proven under positive dependence only. Eight open-access papers were read in the library `Stats/`.
- **S3 stage 1 (re-read of the 192 development KS verdicts), EXPLORATORY.** Only 64 of 192 tests are testable (the others had a median of 2 eligible electrons); all 64 are significant and are exactly the 64 testable old "material" verdicts; 54 to 55 of 64 stay significant after matching on occupancy and energy, but only 1 to 2 of 25 joint cells hold both classes (thin common support).
- **S3 stage 2 (cell-shuffle excess), EXPLORATORY.** Both classes depart from their scrambled images almost everywhere; the electron-proton difference of the excess is significant in 64 of 64 (full shuffle) and 61 to 62 of 64 (nine-cell block shuffle) tests, with occupancy matched only.
- **S3 stage 3 (matched pairs), EXPLORATORY.** Electrons and interacting protons of different primary energies matched 1:1 on occupied cells and deposited energy (caliper 0.25 sd; pairings 20/50, 20/100, 50/100 GeV; 100 to 136 pairs, 1 to 2% of each class; no readout pairing testable). Only 8 of 24 statistics are testable. The stage-2 block-shuffle difference is significant in 0 of 24 matched tests, so occupancy and deposited energy explain it as far as 100 to 136 pairs can say (detectable KS about 0.17 to 0.19 or larger). The full-shuffle excess is inconsistent across pairings. The guards for Westfall-Young and the excess statistic passed. Protons at 200 GeV were added to the planned sealed sets so that matching has overlap.
- **Existing-data overlap studies (P1).** At a fixed primary energy the classes do not overlap in deposited energy; cross-energy matching is needed.

### 4.6 Benchmark harness dry run (2026-10-16)

Role-separated learning curves, class-balanced log-loss, exact rejection limits, boosted trees, logistic regression and a small GPU network (PyTorch 2.14.1, CUDA 13.0, RTX 4050; dependency group `ml`). 313 s for 11 models x 3 blocks x 4 sizes x 5 repeats. A statement about the harness only: the ideal regime is saturated (AUC 0.995 to 1.000 for all, energy-only included) and rejection is capped at 1,336 by 4,000 evaluation protons. Defects to fix before a registered protocol: same-class tolerance of 0.02 is only 1.5 standard errors at 1,000 per class (two checks failed by chance); need at least 3 x 10^4 evaluation protons, at least 10 repeats with fold-correlation-aware uncertainty, independent pools, and a re-check of the unexplained +0.0085 logistic offset.

## 5. Registered tests

Binding registrations were written before the numbers they govern. Their standalone text has left the repository at the researcher's request; the hash is the SHA-256 (first 16 hex digits) of the original text, kept in the git tag and the backup so each can be checked.

| registered | what was fixed in advance | outcome | hash |
|---|---|---|---|
| 2026-10-02 estimator validity | DEC-013 rules (10% false positives, 90% power, 15% recovery line), orders q = 0, 1, 2, box sides 1, 2, 3, 6, 9, 18; ordering fraction 0.90 (interpretation, awaiting confirmation) | not usable at 300 hits or fewer; usable 2 of 13 windows at 1,000 hits | 8c3299598918ce33 |
| 2026-10-02 added validation checks (amended 2026-10-05) | layer correlations, classifier two-sample test, sparsity, ungated multiscale panel, before the fresh test set; AUC 0.55 confirmed; KS 0.05 as a universal threshold not confirmed (to be re-registered from a split-Geant4 floor) | dry run became a development pass (2026-10-05) | 1867424aa9ac33a7 |
| 2026-10-04 dataset-level checks (amended 2026-10-05) | checks D1 to D9 for the e/p dataset | assembler exists; no dataset assembled | 04509055f67f381c |
| 2026-10-09 multiscale test | cells 336 and 572, KS 0.10, five strata, AUC gain 0.01, training sizes 100 to 10,000, ordering fraction 0.90, proton rejection at 90% and 99% electron efficiency (confirmed "for now" by the researcher) | the 0.01 gain rule is unmeetable in the ideal regime (largest possible gain at most 1e-4); to be amended (see OPEN_DECISIONS.md) | f9c92989d881694e |
| 2026-10-15 S3 re-read | permutation within strata, 10,000 draws, Westfall-Young and Holm, family-wise guard | guard passed (WY 0.051 to 0.058, Holm 0.038 to 0.041); 64 of 192 testable | 3bd67f174e99b96b |
| 2026-10-15 S3 stage 2 | cell-shuffle excess, full-layer and nine-cell-block surrogates | guard passed on six testable blocks (WY 0.042 to 0.059) | f28292437cd417c4 |
| 2026-10-16 S3 stage 3 | 1:1 caliper matching (0.25 sd), pair-swap permutation, not testable below 100 pairs | guard passed (WY 0.039, Holm 0.024); 0 of 24 | 9b2c394ec8f433de |
| 2026-10-16 S4 part 1 | 3D estimator controls on the 216 x 216 x 36 mesh, nine scales, DEC-013 numbers unchanged | NEGATIVE (no eligible event) | 7b816ca7c1a7ccb2 |
| 2026-10-16 harness dry run | role-separated learning curves; same-class AUC must be 0.5 within 0.02 | 2 of 9 checks failed by chance; tolerance miscalibrated | 9a6d73cb0d2053f6 |
| 2026-10-22 EXP-010 3D control sensitivity check (option b) | REGISTERED, NOT YET RUN: 3D saturation rule (k = log2(boxes)/3, a = 1, d_f = 3, saturated windows dropped, ordering never waived); baselines original, Geant4 ensemble-mean profile from even events at 100 GeV, broader analytic map; usable only if usable under all three and under both the equal-hits and the occupancy-matched cascade; hit grid plus 15,000 and 20,000; seed 20261023 with the original configuration rerun as noise reference; EXPLORATORY, the original NEGATIVE verdict stays primary | rung 1 run 2026-10-22 (B1, 30,000 hits, 100 replicates; 60 s): only window (4,5,6,7) usable, as in S4 part 1; the ordering diagnostic shows D0 < D1 in every replicate for all windows starting at the four finest scales, including windows the saturation rule calls reliable, so the rule with a = 1, d_f = 3 does not explain the ordering failure (as registered, nothing is changed; a different rule is a new registration). A first attempt matched the cascade's occupancy one-sidedly (it needs fewer hits, not more, at 30,000: 28,289 against 23,713 voxels); fixed to a two-sided search and rerun, first output kept as `exp010_rung1_first_attempt_one_sided_matching.json`. Full grid run 2026-10-22 (438 s on 6 workers): NO window usable under all three baselines and both cascades, so the sensitivity check is "not usable" and the original NEGATIVE verdict stands; EXPLORATORY; see section 4.4 | 5ec9ca672a04f1c0 |
| 2026-10-22 R-A capped round (EXP-012 steps 2-5) | REGISTERED, NOT YET FITTED: three candidates (crossing flag with finer share bins below 0.3; energy scale jointly with share, log-energy interpolation; slow-hadron share), fit even, select on old odd events (eligible = no worse than the unsplit baseline anywhere and 1 point better at two or more energies), one confirmation on a fresh tagged set of 10,000 per energy at 10 to 200 GeV (needs production approval), adopted if within max(3%, fresh-set floor) and no worse than the baseline; otherwise unsplit tables with disclosed error; the search closes at 8 registered versions. Conditional on truth tags, not a standalone R-A | Step A run 2026-10-22: NO candidate eligible (V1 worse than the baseline at 20, 50, 100 GeV; V2 worse at 50 and 100; V3 worse at 20, 50, 100), so nothing is selected, the fresh production is not needed, the round ends and the unsplit proton tables stand with their disclosed error | 417e7a22681d944d |
| 2026-10-22 R-A noise floor (EXP-012 step 1) | model-free half-against-half floor of the maximum per-layer closure, 1,000 splits, seed 20261022; reading: p95 at least 3% unmeetable, 1.5 to 3% partly noise, below 1.5% real error | electrons p95 1.2 to 1.4% (real model error region); crossing protons 2.7 to 3.2%, interacting protons 3.2 to 4.4% (mostly unmeetable); DEVELOPMENT; see section 4.3 | d996fae236341618 |

Pre-stated adoption rules for the electron lateral extensions and the R-A fit rules (registered before each fit, in the R-A design record) are in the Obsidian records and in the code that implements them. Rules and thresholds were never relaxed after a result.

## 6. Accepted decisions (summary)

Decisions are the researcher's, in their own words where recorded. Record IDs (DEC-, EXP-, H-) refer to the machine-readable graph in `research/graph/`.

**Program and scope**
- 2026-09-21 First publication is the multiscale study; FastMC is a smooth control; QML downstream (DEC-002).
- 2026-10-10 Scope includes efficiency in three senses, fractal-inspired architectures as gated candidates, and detector effects as an accepted track.
- 2026-10-11 Redesigned slice program S0 to S8 and validity gates SV-1 to SV-10 accepted as the work order, run one slice at a time ("Yes decisions are valid, and we can go for the slices in the plan 1 by one now").
- 2026-10-11 DECIDE-1: the single 0.01 AUC-gain rule is replaced by endpoint families with ONE named primary each, through a dated amendment before any opening (text pending, see OPEN_DECISIONS.md). DECIDE-2: existing sealed sets stay closed; new sealed sets per purpose with 63-bit seeds; the 2026-10-09 agreement that the multiscale test opens the existing sets first is withdrawn. DECIDE-3: no override of the 0.3-point rule; FastMC refinement paused. DECIDE-4: two representations (view-aware readout at detector level; 3D sparse voxels as the truth-level fine reference).

**Statistics (2026-10-15)**
- B = 10,000; Westfall-Young where joint draws exist and the guard passed, otherwise Holm; Holm always as sensitivity; Benjamini-Hochberg exploratory only.
- S3 stages 1 to 3 and the larger tagged proton run and proton class table approved and run; P1 keeps the matching of the amendment draft and 200 GeV protons join the new sealed sets.
- 2026-10-16 PyTorch installed as the `ml` dependency group.

**Electron generator (2026-10-05 to 2026-10-09)**
- DEC-014: `em_production` supersedes DEC-001 for production; DEC-001 stays as `em_smooth_null`. Grindhammer-Peters is the structural prior for longitudinal fluctuations; its spot model is the lateral mechanism (not the proton quanta). Calibrated and validated in `deposition` first. A per-layer mean correction is not the first response to the early-layer deficit.
- The depth origin is a calibrated coordinate convention, not a physical shower start. Slice 1 locks the gamma family; Slice 2 uses the Grindhammer-Peters lateral structure with R_M = 18 mm. Extensions were tried only "if it doesn't cause overfitting and doesn't complicate the parameterization".
- The never-opened sealed electron set is the final fresh electron validation set.

**Proton model (2026-09-28 to 2026-10-05)**
- Proton model before detector response; analytic interaction depth plus empirical conditional distributions; two representations; QGSP_BERT only as a systematic (DEC-003, DEC-004).
- Interacting-proton factorization with a back-edge factor, a correlated residual in log-energy space (normality tested first) and a shared albedo latent; shared interaction draw; validation contract with added checks; burst latent; lateral spill (DEC-005 to DEC-009). The interacting model was realised as per-offset tables plus a copula, accepted as a calibrated phenomenological surrogate; its structure is frozen and parameters are not added to chase individual discrepancies.
- 2026-09-30 Estimator-validity gate: no multiscale claim until the estimators pass controls (DEC-010).

**Process**
- Names describe the work; notebooks only after a model is final; Obsidian adapters used by default; teaching visuals verified by independent recomputation.
- Held-out events `event_index % 4 == 3` are not used for fitting.

**Detector, geometry and conventions (facts and conventions used as constants).** Origin at the front-face centre, +z into the ECAL; 648 x 648 mm, 166.5 mm deep, 17 X0, 9 superlayers, 18 readout layers, 72 cells per layer, 9 mm pitch; the 18 x 72 image is a stack of alternating one-dimensional views, not a 72 x 72 plane; one canonical `ECALEvent` for FastMC and Geant4; geometry kept separate from detector response. See `configs/geometry.yaml`.

## 7. Corrections of earlier statements (kept for honesty)

- Fractal architectures are in the repository (gated candidates in `CHAOS_FRACTALS_QML.md`), though not in the roadmap papers. Detector effects are an accepted track, not something needing promotion.
- Generation timings were underestimated (100 GeV electrons took about 6 h, not 4).
- 2026-10-11 "FastMC fidelity is on the critical path because about 10^6 events are needed" was withdrawn: 10^6 normal-incidence protons are about one day of Geant4 wall time.
- Every proton-rejection number in the development readings is the zero-failure floor, not a measurement.
- The EMV anomaly involves at least seven settings, not two.
- The 2026-09-20 claim that the ensemble mean reproduces the deterministic profile holds for T0 only (Jensen's inequality); the "x27" amplification constant is layer 1's sensitivity (13.2) times |z| = 2.041.

## 8. Literature picture

- Calorimeter fractal-dimension work (Ruan 2014) is simulation-only, support-based, and its fractal dimension is tied to the hit count. The closest precedent (HEGRA, Schaefer 2001) found multifractal parameters strongly correlated with conventional shape variables, a slight combined gain, and pixels "much too coarse". EM showers are near-monofractal in the one study found (Anchordoqui 2000), which is counter-indicative. No multiscale fidelity metric exists in the fast-simulation literature. Geant4 physics lists differ from data and each other by 10 to 20%.
- Published proton rejection (read by the previous agent, not re-read here): ECAL-only classifiers 10^2 to 10^3 at 90% electron efficiency; with energy-momentum match about 10^4; ECAL plus tracker exceeds 10^4 on flight data (Kounine et al. 2017); a deep-learning study on the 18 x 72 image used Monte Carlo at 200 GeV to 2 TeV and flight data at 50 to 70 GeV (Hashmani et al. 2024), reported without uncertainties.
- Gaps: Hentschel-Procaccia 1983 and Bialas-Peschanski 1986 (not in the library), q < 0 snippets, an INSPIRE citation trace of Ruan 2014. Reports from literature subagents were found unreliable; only claims read directly in the PDFs with a locator are used.

## 9. Where things live

| what | where |
|---|---|
| Source code, grouped by role | `src/ams_ecal/` (module guide: `research/MODULE_MAP.md`, name translations: `research/NAMES.md`) |
| Tests | `tests/` (mirrors the source tree); last full run 916 passed (2026-10-11) |
| Results (JSON, CSV, figures) | `results/` |
| Decision-check scripts and their outputs | `results/decision_checks/` |
| Data manifest | `research/data_manifest.json`; sealed sets: `research/sealed/` |
| Machine-readable research graph (questions, hypotheses, experiments, decisions) | `research/graph/` |
| Publication roadmap, chaos/fractal/QML note | `research/PUBLICATION_ROADMAP.md`, `research/CHAOS_FRACTALS_QML.md` |
| Evidence and design records (literature verification, R-A design, proton dependency analysis, program redesign review, reviewer round 1) | Obsidian `01 Projects/AMS ECAL QML/Research Records/` |
| Everything removed on 2026-10-21 | git tag `docs-before-consolidation-2026-10-21`; backup `~/.research-os-backups/ams-ecal-docs-before-consolidation-2026-10-21` |
