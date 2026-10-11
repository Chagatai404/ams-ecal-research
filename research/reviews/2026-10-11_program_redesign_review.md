# Program redesign review, 2026-10-11 — critical assessment, statistical audit, AMS realism, decision memo

> **Accepted 2026-10-11** by the researcher ("Yes decisions are valid, and we can go for the slices in the plan 1 by one now"): the recommendations of section 5 and the slice program. Open sub-items: the DECIDE-1 amendment text, the DECIDE-2 exit date, regenerate-or-ledger for the seed collisions. The text below is kept as written for the decision (history).

_Evidence record, not a plan. The single plan is `research/RESEARCH_PLAN.md`; its section 18 (PROPOSED redesign) cites this file. Requested by the researcher on 2026-10-11 (the brief "Scientific Redesign, Statistical Audit, and Roadmap Revision"). Nothing here is an accepted decision: every recommendation is **PROPOSED / DECIDE** until the researcher accepts it. No sealed set was read, no simulation run, no model trained, no pre-registered number changed._

## 0. How this record was produced (provenance)

| reviewer | role | inputs | access to repository | output |
|---|---|---|---|---|
| A | shower physics, hypotheses (physics-reviewer) | first principles, standard references | allowed, not used | report in chat, summarised in sections 1, 2 |
| B | independent literature discovery (literature-scout) | neutral question only; NO repository, NO local library | none | source map; **no full text read, every number UNVERIFIED** |
| C | AMS-02 detector realism and benchmarks (literature-scout) | local library (inherited project citations) + open web | configs and plan only | report in chat; benchmarks re-checked by me in the PDFs (section 4) |
| D | statistics (statistics-reviewer) | ladder, test, controls code; results JSON | read-only | report in chat; key items re-checked by me in the code (section 3) |
| E | simulation and ML methodology (ml-reviewer) | code, configs, exposed data (read-only checks) | read-only | session scratchpad `review/E_simulation_ml.md` |
| author (Claude) | synthesis, verification | all of the above | | this record |

Adversarial stage: section 9 (an independent adversarial reviewer read a draft of sections 1-8 and tried to break it).

Verified by me (not only reported by an agent): the zero-failure floor and degenerate interval (`multiscale_ladder.py:90-96`), the same-event threshold (`:52-53`), the NaN-as-feature path (`multiscale_test.py:43-48`, ladder `:67-74`), the missing held-out exclusion in the large development run (`multiscale_gtrain_development.py:57-62` against `multiscale_ladder.py:189`), the class-dependent eligible fractions (results JSON), the EMV readout anomaly (section 1, item 7, recomputed from the npz files), the AMS rejection benchmarks (Vecchi p. 241; Kounine Fig. 17 caption; Hashmani section 2) read in the library PDFs.

## 1. Deliverable A — critical scientific assessment

### What is actually established (development evidence, ideal Geant4 only)

1. **Conventional observables separate electrons from protons almost perfectly** at 10-100 GeV: out-of-fold AUC 0.9999-1.0000 in the large development run (`results/multiscale_controls/multiscale_gtrain_development_*.json`; 0.99988 or more without total energy and occupancy), 0.9989-1.0000 in the earlier small run (`multiscale_ladder_development.json`), against interacting AND against all protons (truth selection is not what makes it easy). **No implementation error was found**: electrons and protons share one generation path and identical configurations (only the particle and the base seed differ, `geant4_backend.py:802-840`); no truth variable reaches a feature; seeds and grids are unique within each class and energy; cross-validation is over independent events with no fitted preprocessing (reviewer E). **It is a physically easy, unrealistic problem**: fixed and known beam energy (total deposit alone: univariate AUC 0.991 at 10 GeV per reviewer E; the adversarial reviewer found a cross-class KS of 0.998 for the total deposit, so the population behind 0.991 is restated in S0), normal incidence through one central cell, zero-threshold undigitised readout, truth-selected interacting protons, compact EM core (core fraction 0.87 against 0.54), an EM shower that always starts in layers 0-2.
2. **Every proton-rejection number reported so far is the zero-failure floor**, not a measurement: 9,696 [9,696, 9,696] means "no proton passed among 9,696". The valid one-sided 95% lower limit is about **3,240** (Clopper-Pearson, 0 of 9,696), about 6,680 for 0 of 20,000.
3. **At 10 GeV and on the readout at 10-50 GeV the multiscale-only rung measures missingness; elsewhere it carries information the conventional features already hold.** Eligibility (occupied cells >= 336 / 572) is 0-3% for electrons and 41-68% for interacting protons in those cells; the AUC of the eligibility indicator alone (exact from the class fractions) matches the observed multiscale AUC within 0.01 in 3 of the 8 interacting-proton cells and within 0.011 in a fourth. In deposition at 20-100 GeV the multiscale features carry large within-eligible information (AUC 0.96-0.999 against an indicator AUC of 0.57-0.59), but the gain over the conventional rung is about zero everywhere: redundant, not absent. Missingness is a deterministic threshold on `occupied_cells`, already a conventional feature.
4. **The development KS verdicts (79 "material" of 192 in the large run; 61 of 384 in `multiscale_development.json`) are not interpretable**: some rest on about 30 electrons from the extreme eligibility tail; no multiplicity control; the floor (median raw half-sample KS) and the verdict (lower bootstrap quantile of a stratified KS) are different statistics; the energy-preserving cell-shuffle null named in the pre-registration was never implemented (plan item A3).
5. **The registered "added information" rule is unmeetable by construction** in this regime: the largest possible gain 1 - AUC_conv is at most 1e-4 in the large run and 1.1e-3 in the small run, against a required 0.01.
6. **The usable scale window is small** (reviewer A): Moliere radius about 2 cm (about 2 cells) to the shower envelope (about 7 cells), about 0.5-1 decade laterally and at most about 1 decade longitudinally on the readout; below R_M, structure reflects transport, cuts and Poisson granularity. **The 18 x 72 grid is a stack of alternating one-dimensional x and y projections** (`projection.py`), so boxes spanning more than one superlayer mix the two measured axes: sides 3, 6, 9, 18 (in every registered window) do; side 2 aligned from layer 0 does not (layers 0-1, 2-3, ... share an axis; `multiscale.py:51-58`). The box counting treats the grid as a planar image.
7. **New anomaly, recomputed from the exposed files**: with `FTFP_BERT_EMV` the scintillator share of the deposited energy of 20 GeV electrons (median per event) is 4.13% against 6.53% (baseline), 6.64% (EMZ), 6.68% (0.1 mm cut), while the total deposit is identical (18.77 GeV). The deficit grows with depth (EMV / baseline per layer from 0.69 at layer 0 to 0.52 at layer 17; baseline and EMZ are flat at about 0.065), the configurations differ only in the physics list, and the hand-assembled standard-EM list reproduces FTFP_BERT exactly in the equivalence test (adversarial reviewer). That points to the EM option 1 constructor or its transport settings (multiple scattering, step limitation, fluctuations of low-energy electrons in 1 mm fibres), not to a tally bug. Whether option 1 can genuinely move the sampling fraction this much is an OPEN physics question (no source in hand). The EMV set is not used as a systematic until resolved. Checks: dump `G4EmParameters` and the process lists at initialisation for the three lists (no events); compare with the native `FTFP_BERT_EMV`; on approval, swap option-1 settings one at a time on 2,000 events. Documentation error found: the EMV YAML comment says the list comes from `G4PhysListFactory`, but the code assembles it by hand.

### Major risks

- Reading physics into estimator artefacts (missingness, occupancy, finite window, mixed x/y projections).
- Claiming performance from a regime that does not overlap AMS operation (ours: 10-100 GeV, normal incidence, fixed energy; AMS e/p work: 50 GeV-2 TeV in the ML studies, +/-20-25 degrees, energy measured by ECAL and tracker).
- Point-estimate rejections from zero-failure samples; threshold and evaluation on the same events.
- Two purposes competing for one sealed opening (DECIDE-2).
- Unlimited FastMC refinement justified by the need for large samples.

### Most credible hypotheses and their strongest competitors

| claim | most credible reading | strongest competitor |
|---|---|---|
| e/p "multiscale" difference | hadronic heterogeneity exists (pi0 sub-showers, secondary tracks, start depth) | it is occupancy, start depth, width and containment re-expressed (smooth profile + Poisson granularity null) |
| EM structure | mean-profile universality in t/t_max and r/R_M holds to about 10% (established for MEAN profiles) | no theory predicts collapse of higher-order statistics; EM may be nearly monofractal (Anchordoqui et al., D_q about equal, verified 2026-10-09) |
| readout survival | some e/p difference survives 9 mm, one-view-per-layer readout | what "survives" is conventional morphology, not multiscale structure |
| simulator fidelity | multiscale statistics can expose FastMC defects profiles miss (occupancy already does: AUC 1.000, 55% of occupied cells) | a classifier two-sample test or occupancy already exposes everything a multiscale statistic would |

### Literature-supported opportunities (not novelty claims)

- One main calorimeter fractal-dimension PID paper (Ruan et al., PRL 112, 012001, 2014; simulated binary-readout steel-RPC HCAL, q = 0 only; verified in full text 2026-10-09): **its PID cuts combine FD with the hit count**, so FD's independence from occupancy is not shown there. An occupancy-controlled test of whether D_q carries information beyond conventional morphology was **not found** in reviewer B's search (snippet and abstract depth; absence not established).
- Generalized-dimension (D_q) analyses of calorimeter EM showers: only Anchordoqui et al. (Geant4 aluminium, D_q about equal; counter-indicative) was found.
- Fast-simulation evaluation (CaloChallenge, arXiv:2410.21611, in the library): histograms, physics distances, classifier AUCs; whether multiscale statistics have been used as fidelity metrics was **not found** (snippet depth).
- Air-shower multifractal and lacunarity work exists (KASCADE multifractal moments, ARGO-YBJ lacunarity, Cherenkov-image D_q), mostly older, a different system.

### Weakest assumptions

1. Eligibility thresholds 336 / 572 transferred from equal-quanta synthetic controls (`multiscale_test.py:20-28`).
2. The 18 x 72 alternating-projection grid treated as a planar image by the box counting.
3. Zero-threshold readout (occupancy, eligibility and every multiscale statistic depend on the threshold).
4. KS 0.10 materiality taken by analogy from proton validation.
5. Truth-interacting protons as the background population (not deployable on real data).
6. Single fold seed and fixed classifier seed: training variance absent from every interval.

## 2. Deliverable B — hypothesis and experiment matrix (PROPOSED; none accepted)

Every hypothesis is tested at FINITE scale: the language is "finite-scale heterogeneity statistics"; "fractal" or "multifractal" only if a scaling window passes the estimator controls against a surrogate null (protocol sections 7-8).

| id | hypothesis | prior literature | observable | mechanism | null | main confounders | data | statistic / test | systematic controls | success / falsification | publication value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-A | EM universality of multiscale statistics: after conditioning on energy, depth and age, the distributions collapse | mean-profile universality (PDG 34.5, Grindhammer-Peters, Longo-Sestili); Anchordoqui (D_q equal) | depth- and age-resolved D_q (q >= 0), occupancy-scaling curves, energy-moment coarse-graining curves, entropy | branching cascade; X0 and Moliere scaling | the statistics depend on E beyond what t/t_max and r/R_M absorb | occupancy, leakage, hit statistics growing with E | fine voxels (stored) + readout, 10-100 GeV electrons; extended set for leakage | collapse residual against Poisson-resampled and smooth-profile surrogates; permutation over energy labels within age bins | cut, EMZ (EMV after its anomaly), threshold | success: residual within the surrogate band; falsified: an E trend beyond it | high either way; "monofractal / no window" is publishable |
| H-B | hadronic heterogeneity beyond conventional morphology | Ruan 2014 (e/h FD differs, not occupancy-controlled); intermittency tradition | lacunarity, D0-D2, intermittency moments, interacting protons vs electrons | pi0 sub-showers, tracks, nuclear fragments | no difference after matching on visible energy, occupancy, start depth, widths, containment | start depth, MIP segments, occupancy, energy fraction | Geant4 interacting protons (truth only as a stratum) | matched permutation test on occupancy- and morphology-matched pairs; FDR over statistics | FTFP_BERT / QGSP_BERT / QBBC, cut | success: survives matching and all lists with one sign; falsified: vanishes after matching | high |
| H-C | depth dependence is physical, not hit statistics | NKG age (mean profile) | layer- or age-resolved statistics | early core, late soft halo | the trend equals that of an occupancy-matched surrogate | occupancy, leakage | fine voxels | difference from occupancy-matched resampled surrogates per depth | threshold, cut | success: trend outside the surrogate band | medium |
| H-D | information survival under coarse-graining, measured continuously | not found (snippet depth) | paired change fine -> 9 mm 3D cells -> alternating-view readout; classifier log-loss information per level | sampling, granularity, one view per layer | (estimation) | readout threshold, sampling fluctuations | the same events at three levels | paired degradation curves, paired bootstrap clustered by run | threshold scan; noise once R-A/R-B exist | reported as a curve, never present/absent | high (core of Paper 1) |
| H-E | multiscale simulator fidelity: FastMC can agree on profiles and fail multiscale statistics | CaloChallenge metric families (snippet) | each statistic, generator vs Geant4, against the Geant4 split floor | smooth parameterisation replaces branching | generator inside the Geant4 sample-to-sample floor | occupancy (already failing) | FastMC + Geant4 calibration and hold-out | matched distance with permutation null | | success: a statistic flags a defect profiles miss, or confirms adequacy | medium-high |
| H-F | robustness under cuts, EM / hadronic lists, granularity, threshold, noise, entry, angle, material | Geant4 physics-list literature | every H-A..H-E statistic | implementation choices | a variant changes the statistic beyond its floor | | existing cut / EMZ / proton-list variants; angle and threshold variants TO GENERATE | sign and size change per variant | | a sign change under a variant = "systematics-dominated" | required for any claim |
| H-G1 | rejection at fixed electron efficiency improves with multiscale features | AMS: ECAL-only 10^2-10^3, ECAL+tracker > 10^4 at 90% | censored Clopper-Pearson rejection, all protons, realistic regime | | no improvement within uncertainty | energy knowledge, threshold, angle | realistic Geant4 (S7); >= 3e4 independent evaluation protons per claimed 1e4 | k of N at an independently calibrated threshold; paired comparison on the same events | detector response, physics list | success: the lower limit improves beyond the paired uncertainty | high if realistic |
| H-G2 | sample efficiency | | log-loss and censored rejection vs training size | inductive bias | curves coincide | | as G1 | repeated cross-fitting, seed variance | | | medium |
| H-G3 | model / compute efficiency | | performance at matched parameters and inference time | | | | | | | | medium |
| H-G4 | simulation diagnostics | = H-E | | | | | | | | | |
| H-G5 | leakage / energy-reconstruction correction | Kounine (leakage corrections from shower parameters) | energy-resolution change with multiscale inputs | late-shower structure | no gain beyond profile-based correction | | extended-depth electrons | paired resolution, bootstrap | | | low-medium |
| H-G6 | out-of-distribution / anomalous showers | | anomaly-score distributions | | | | needs defined anomalies | | | | low (speculative) |
| H-G7 | interpretability | | | | | | | | | | low (descriptive) |

## 3. Deliverable C — statistical validity audit

### Confirmed implementation problems (file:line)

| id | problem | location | correction (PROPOSED) | verification test |
|---|---|---|---|---|
| C1 | zero-failure floor 1/N gives rejection = N and a degenerate bootstrap interval | `multiscale_ladder.py:90-96` | report k of N; exact one-sided Clopper-Pearson upper bound on the FPR; rejection as a censored lower limit "> R_low (95% CL)"; Wilson / Jeffreys only as sensitivity | unit test: k = 0, N = 9696 gives R_low about 3,237; coverage simulation (1,000 replicates) |
| C2 | threshold set on the events used to count false positives; out-of-fold scores pool five models | `multiscale_ladder.py:52-53, 56-75` | four roles: train / model selection (none if settings are fixed) / operating-point calibration (electrons only) / final evaluation (fresh electrons for achieved efficiency, fresh protons for the FPR); one model per evaluation; threshold uncertainty from the order-statistic (Beta) law or a calibration-electron bootstrap | test that evaluation events never enter the threshold |
| C3 | NaN (ineligibility) used as a feature; eligibility is a threshold on `occupied_cells` | `multiscale_test.py:43-48`; ladder `:67-74` | explicit missing indicator; missingness-only control rung; eligible-in-both analysis; occupancy-matched pairs | missingness-only AUC beside every multiscale rung |
| C4 | KS on tiny eligible tails; strata with < 20 events skipped silently | `multiscale_test.py:33, 94, 107-115` | report n per stratum; minimum n fixed in advance; eligible-in-both | |
| C5 | floor and verdict are different statistics; code `low > floor` differs from the pre-registration wording; the cell-shuffle null is not implemented | `multiscale_test.py:118-128, 151`; plan A3 | label permutation within energy x occupancy strata; floor from matched-size splits; implement the shuffle null | calibrated type-I error on electron-vs-electron splits |
| C6 | KS bootstrap with replacement (ties), 100 draws | `multiscale_test.py:143-145`; `multiscale_gtrain_development.py:28` | permutation inference, >= 1,000 resamples | |
| C7 | intervals ignore training and fold variance | `multiscale_ladder.py:78-116`; `added_checks.py:69-76` | repeated cross-fitting over >= 5 seeds; report the seed spread | |
| C8 | the large development run includes the `event_index % 4 == 3` protons other modules hold out; the eligibility count used a different sample | `multiscale_gtrain_development.py:57-62` vs `multiscale_ladder.py:189` | one sample-definition module; a manifest | |
| C9 | "vs all protons" rows mix crossing protons into the multiscale question | `multiscale_gtrain_development.py:107-115` | crossing protons separate (as pre-registered); all protons only for the realistic-performance endpoint | |
| C10 | the AUC-gain rule cannot be met when AUC_conv >= 0.99 | `multiscale_ladder.py:115` | DECIDE-1 | |
| C11 | eligibility thresholds transferred from equal-quanta controls | `multiscale_test.py:20-28` | controls on realistic (unequal-amplitude) and 3D images (A4) | |
| C12 | the classifier two-sample row takes the first n rows without randomisation; the 99th percentile of 20 permutations is the maximum | `added_checks.py` (reviewer D) | randomised subsample; >= 200 permutations | to confirm against file ordering |
| C13 | 192 tests, no multiplicity control | all KS outputs | one pre-specified primary per hypothesis; Westfall-Young max-statistic or Benjamini-Hochberg across secondaries; family-wise control across endpoint families | |
| C14 | 31-bit per-event seeds (`generate_state >> 1`): 4,009 duplicated seed values in the exposed pool (about 4,000 deliberate paired variants, 9 cross-energy coincidences); **derived from the seed policy without reading any sealed file: 2 of the 21,000 sealed proton seeds equal seeds of exposed events** (sealed electrons: 0) | `geant4_backend.py` seed derivation (adversarial reviewer) | 63-bit seeds for every new set; before any opening, recompute the sealed seeds from the policy and intersect with the exposed pool; regenerate the sealed protons (about 1.5 h) or record the two collisions in the opening ledger (researcher's choice) | seed-intersection check logged before the opening |
| C15 | the EMV/EMZ variant YAML comments say the list comes from `G4PhysListFactory`; the code assembles it by hand | variant YAMLs in `../ams-ecal-qml-gen2/configs/variants/` | correct the comments | |

### Suspected methodological weaknesses (not confirmed)

Headroom only from matched hard subsets, not from dropping one feature; truth conditioning not deployable; 2-point windows have no fit uncertainty; D_q may be functions of occupancy and concentration and so redundant with the conventional rung; shared base seeds make the pilot variants paired, not independent (baseline, 0.1 mm cut, fixed entry, material share base seed 20260929); readout survival is untestable at 10-20 GeV (electron eligibility 0-0.1%); no logistic-regression or energy-only baseline.

### Zero-failure sample sizes (exact CP, one-sided 95%, independent protons, threshold fixed independently)

| rejection to claim | k = 0 | k = 1 | k = 2 | k = 5 |
|---|---|---|---|---|
| 1e3 | 2,995 | 4,742 | 6,294 | 10,511 |
| 1e4 | 29,956 | 47,437 | 62,956 | 105,128 |
| 1e5 | 299,572 | 474,385 | 629,578 | 1,051,301 |

A 10% relative precision on a MEASURED rejection of 1e4 needs about 100 false positives (1-sigma, Poisson), i.e. about 1e6 evaluated protons of the relevant population; a 10% half-width at 95% needs about 384 false positives, about 3.8e6 protons. Both ignore the threshold uncertainty from the calibration electrons (it adds variance; estimate it with a joint electron-proton bootstrap), assume independent protons at a fixed threshold, and hold only for the stated population (all protons and truth-interacting protons give different rejections).

### Claim template (estimand -> null -> statistic -> uncertainty -> assumptions -> acceptance -> failure)

| claim | estimand | null | statistic | uncertainty | assumptions | acceptance | failure interpretation |
|---|---|---|---|---|---|---|---|
| physics difference | distribution of one pre-specified D_q statistic, eligible-in-both, occupancy- and morphology-matched | same distribution given matching | matched KS or energy distance | label permutation within strata; max-statistic / FDR over secondaries | exchangeability within strata; eligibility explicit | the primary passes at alpha 0.05 after correction AND keeps its sign across physics variants | no difference beyond morphology, or confined to an under-populated tail |
| readout survival | paired change of the statistic and of the e/p information from fine to readout | (estimation) | paired difference; log-loss information per level | paired bootstrap clustered by run | enough eligible events per level | a curve with intervals | "not testable" where readout eligibility is near zero, never "destroyed" |
| simulator fidelity | distance generator vs Geant4 per statistic | generator inside the Geant4 split floor | matched KS or energy distance | permutation; floor from Geant4 splits | reference independent of calibration | within the floor and the physics-list band | the statistic locates a defect |
| realistic performance | rejection at fixed electron efficiency, all protons, realistic regime | (estimation) | k of N at an independently calibrated threshold | Clopper-Pearson plus threshold uncertainty | independent evaluation; no truth selection | a censored lower limit, never above N/3 with k = 0 | limited by evaluation statistics, stated as such |

Required before any registered test: an independent statistical review of the frozen analysis code (a separate reviewer, not the author).

## 4. Deliverable D — AMS realism gap assessment

### Published benchmarks

| result | value | regime | source (read by me unless marked) |
|---|---|---|---|
| ECAL-only classifiers (BDT, NN, likelihood, cuts) | proton rejection 10^2-10^3 at 90% signal efficiency | validated on test beam; above 180 GeV Monte Carlo only | Vecchi et al., SF2A 2012, arXiv:1210.0316, p. 241 |
| plus energy-momentum match | about 10^4; the TRD adds about one order | same | same |
| ECAL estimator + tracker | "exceeds 10^4" at 90% e efficiency, 3 GeV - 2 TeV/c, flight data, TRD-tagged protons | ISS | Kounine et al., NIM A 869 (2017) 110, Fig. 17 caption |
| DL on 18 x 72 images | MC 200 GeV - 2 TeV (about 11.5 M events in total), ISS 50-70 GeV; rejection at fixed efficiency; no uncertainties reported | MC and ISS | Hashmani et al., MLST 5 045008 (2024), section 2 |
| MC-trained models on ISS beaten by the AMS BDT by 4-9 times | | ISS | Hashmani M.S. thesis Fig. 5.22, p. 86 (**reported by reviewer C, not re-read by me**) |
| test-beam energy resolution | (10.4 +/- 0.2)%/sqrt(E) + (1.4 +/- 0.1)% | 6-250 GeV, normal incidence | Adloff et al., NIM A 714 (2013) Eq. 4 (**reported by C**) |

### Current capability against the minimum for an AMS-like claim

| item | now | minimum |
|---|---|---|
| energy range | 10-100 GeV, fixed energies | spectrum-weighted; overlap with a published regime (50 GeV - 2 TeV) |
| incidence | normal, one central cell | +/-20-25 degree acceptance, full face, fiducial 9 mm from edges |
| energy knowledge | true beam energy known | ECAL reconstructed energy; tracker rigidity with its resolution (not modelled; must not be invented and called validated) |
| response | true deposit and fibre energy, zero threshold | R-A sampling response; R-B thresholds, noise, attenuation (f = 0.17, 110 mm, 2605 mm, Adloff Eq. 1, reported by C), gain spread, saturation, dead cells |
| background | truth-interacting protons | all protons passing the same selection; difficult subsets defined by observables, not truth |
| statistics | 1e4 protons per energy | >= 3e4 per claimed 1e4 (k = 0); about 1e6 for a measured 1e4 at 10% precision |
| validation | Geant4 against Geant4 | hadronic model checked against test-beam data; flight calibration, dead-cell lists and selection cuts exist only inside the collaboration |

**What cannot be claimed without AMS data:** any statement of actual AMS-02 performance. Domain shift MC -> data is reported as large for DL models (thesis, via C). A validated AMS-like simulation cannot be built from public information alone (missing: per-cell noise, thresholds, ADC, Birks constant, dead-cell map, selection cuts). The honest ceiling is "AMS-like simulation with stated, sensitivity-scanned assumptions".

Recorded conflicts (convention-level, unresolved): superlayer orientation 5 x / 4 y (Adloff; repo) vs 4 x / 5 y (Li 2013; Hashmani 2024); density 6.8 g/cm^3 vs about 7.05 from the volume ratio (already noted in `geant4_transport.yaml`); acceptance 20 vs 25 degrees.

### Measured production rates (reviewer E, from `metadata.json`, 12-14 workers)

Electrons 2.5-30 CPU-s/event (10-100 GeV), 29-127 kB/event; protons 0.39-2.31 CPU-s/event, 11-34 kB/event. Estimates: sealed electrons 7 x 1000 about 22 CPU-h (1.6-1.9 h wall); sealed protons 7 x 3000 about 6.6 CPU-h (0.5-0.7 h); 1e5 protons about 33 CPU-h (2.5-3 h, 2.2 GB); 1e6 protons about 330 CPU-h (24-28 h, 22 GB with the fine arrays). Normal-incidence costs; TeV electrons and angles cost more.

## 5. Deliverable E — four-decision memo (ALL PENDING the researcher)

### DECIDE-1 — the registered information test

- **Alternatives:** (a) keep the 0.01 AUC-gain rule (unmeetable here: the result is fixed in advance); (b) replace it by a family of separate confirmatory endpoints; (c) declare the question unanswerable in the ideal regime and stop there.
- **Evidence:** C10; conventional AUC >= 0.9999 everywhere; headroom appears only in harder populations.
- **Recommendation (PROPOSED):** (b), as a dated amendment of the 2026-10-09 pre-registration before any opening, with four conditions from the adversarial pass: the P1 primary is NAMED in the amendment (statistic, window, representation, energies), not "one pre-specified statistic"; P2 gets a decision rule (for example a sign-consistent drop beyond the surrogate band at the readout level) or is reported as estimation only; the P1 estimand is stated as "a difference conditional on occupancy and morphology" (occupancy is itself a consequence of shower development, so a null after matching is uninterpretable without a positive control), and a planted-structure positive control at fixed occupancy must show recovery; family-wise error is controlled across the families. P1 is not testable where too few electrons are eligible (26 at 10 GeV deposition, 0-25 on the readout at 10-50 GeV); those cells are reported as "not testable". The endpoint families: four endpoint families, each with ONE pre-specified primary (section 3 template): P1 physics difference (eligible-in-both, matched, permutation + FDR); P2 readout survival as a continuous paired degradation; P3 simulator fidelity; P4 realistic performance (censored CP rejection on all protons with independent threshold calibration, plus cross-fitted log-loss gain on matched hard subsets). The original rule stays in the record and is reported as "unmeetable in the ideal regime" beside the new endpoints. Exploratory readings stay labelled exploratory.
- **Trade-offs:** more endpoints need multiplicity control and design work; the gain is a test whose outcome is not fixed in advance.
- **Depends on:** S1 (statistical tools), S2 (endpoint definitions), DECIDE-4 (the P2 levels).
- **Impact on registered analyses:** the 2026-10-09 numbers (336/572, KS 0.10, five strata, 0.01) are superseded only by an explicit amendment; 336/572 and KS 0.10 are flagged weak (section 1).

### DECIDE-2 — the sealed sets

- **Alternatives:** (a) new sealed sets per purpose; (b) one registered joint opening; (c) keep the existing sealed sets for their FIRST registered purpose (FastMC generator validation, DEC-014 Q11 and the added-checks registration) and generate the multiscale confirmatory sets only when that design is frozen.
- **Evidence:** generation is cheap (section 4: the existing design costs about 30 CPU-h, about 2.5 h wall); the redesigned confirmatory test needs a different population (realistic energies and angles, all protons, >= 3e4 evaluation protons per 1e4 rejection claim), which the existing sealed sets (normal incidence, 7 energies x 3000 protons) cannot provide anyway.
- **Recommendation (PROPOSED):** (c): independent sets per purpose in substance, without generating a confirmatory set for a design that is not frozen. A joint opening (b) couples two analyses' multiplicity and forking paths for no saving. The existing sealed sets cannot serve P4 anyway (3,000 protons per energy against >= 3e4 per 1e4 claim).
- **Weaknesses found by the adversarial pass, and the fixes:** (1) with FastMC paused (DECIDE-3) and its occupancy failure already known, spending the one opening on FastMC validation now would give little information, so the existing sets stay CLOSED and are opened for FastMC only when a frozen FastMC candidate exists; otherwise they remain an untouched reserve; (2) a FastMC target must not become a reason to refine until it passes (the DECIDE-3 stop rule says so explicitly); (3) an exit rule: if no frozen FastMC candidate exists by a date the researcher sets, the purpose is re-scoped in writing; (4) the 2 sealed-proton seed collisions (C14) are fixed or ledgered before any opening.
- **Trade-offs:** withdraws the 2026-10-09 agreement that the multiscale test could open the existing sets first (needs the researcher's explicit word).
- **Depends on:** the S8 design freeze; new 63-bit seeds, never reused.

### DECIDE-3 — FastMC electron tail and occupancy work

- **Alternatives:** override the 0.3-point rule for the faster tail; refine within the rule; pause refinement.
- **Evidence:** every lateral extension failed its pre-stated rule (E1, tail-sensitive, faster tail by 0.11 points, E2); the classifier separates every candidate from Geant4 by occupancy (AUC 1.000). Under the redesign FastMC has three possible roles: smooth null (adequate), development pipeline (adequate), large-sample generator (NOT adequate: no demonstrated fidelity in the difficult-background tails, and Geant4 is cheap enough: 1e6 protons about one day).
- **Recommendation (PROPOSED):** do not override (it relaxes a criterion after seeing the failure); keep Slice 2 as the candidate; **pause electron-generator refinement** with a stop rule (no further lateral extension until the H-E endpoint is defined; then at most one time-boxed attempt under a rule fixed in code); no FastMC sample in any performance or confirmatory claim; Geant4 for evaluation samples.
- **Depends on:** S6.

### DECIDE-4 — the fine-resolution representation

- **Alternatives:** a 2D fine image (projection); 3D sparse voxels / points as primary with 2D projections as controls; fibre level.
- **Evidence:** the exposed files already store 3D fine voxels (`fine_index`, `fine_edep_mev`; 3 x 3 x 4.625 mm, index space 216 x 216 x 36; about 3,900 non-empty voxels per 10 GeV electron) and fibre ids (reviewer E); the readout is a stack of alternating one-dimensional projections, not an image; the fine voxel is anisotropic; the readout cell is nearly isotropic (9 x 9 x 9.25 mm).
- **Weaknesses found by the adversarial pass:** the 336-cell eligibility rule is met trivially by about 3,900 occupied voxels, so 3D eligibility must be re-derived from 3D estimator controls (A4); sparse 3D box counts at small sides are dominated by granularity; the truth-level axis does not exist in data; and the readout is NOT a coarse-graining of the deposit: it is scintillator-only fibre energy (about 6.5% of the deposit) with sampling fluctuations, so a separate sampling step (R-A) sits between "9 mm cells in 3D" and "one view per layer".
- **Recommendation (PROPOSED, revised after the adversarial pass):** two objects with separate roles: (i) the view-aware readout analysis (x-layers and y-layers separately, never mixed in one box) as the detector-level object; (ii) 3D sparse voxels as the truth-level reference for the fine scale, with physical box sizes in mm, energy weighting (q-moments of the energy measure), an axis-free energy-weighted principal axis (the true entry point and direction only as a control), depth-resolved transverse analysis (per layer, per view), 3D eligibility from 3D controls, and controlled steps fine voxel -> 9 mm cells in 3D (granularity) -> fibre energy (sampling, R-A) -> one view per layer (the readout), each step measured separately. The 18 x 72 grid is analysed view-aware (x-layers and y-layers separately, never mixed in one box). 2D projections are complementary controls. Information loss stays measurable at each step (H-D).
- **Depends on:** S1 (estimators and nulls), A4 controls on the 3D geometry.

### Effect on the other decisions

DECIDE-5 (architectures, three efficiencies): lower urgency; waits for H-G evidence. DECIDE-6 (R-A, R-B): **higher urgency**, on the critical path of any realistic claim. DECIDE-7: the KS re-registration should adopt the permutation-based rules of S1 rather than a new fixed KS number; the gamma-inadequacy signs are unchanged. DECIDE-8 unchanged. DECIDE-9: incidence angle is now required (S5/S7), not optional. Accepted decisions are preserved: the detector-response program (R-A, R-B, 2026-10-02), DEC-014, the sealed-set opening discipline, the classifier as a diagnostic, the registered numbers until amended.

## 6. Revised slices

The authoritative version is `RESEARCH_PLAN.md` section 18.

## 7. Publication strategy (PROPOSED)

- **Paper 1** (working title "Multiscale geometry of calorimeter particle showers: finite-scale structure, hadronic heterogeneity, detector coarse-graining and simulation fidelity"). Simulation-based (Geant4 fine and AMS-like). Estimator validity and the available scale window; H-A to H-D, H-F; H-E as a section; the PID question in the ideal and the AMS-like regimes with censored rejections. Negative results that keep their value: no stable scaling window; differences that vanish under readout or under morphology matching; redundancy with conventional morphology; FastMC preserving profiles but not hierarchical fluctuations; no PID gain in realistic conditions.
- **Paper 2 (conditional):** realistic AMS-like PID performance, only if R-A/R-B and the realistic regime are built and checked against public test-beam numbers; it can never claim actual AMS performance without collaboration data.
- Simulator fidelity stays inside Paper 1 unless it becomes a generator-evaluation method of its own. QML stays downstream (Paper 3+). Split only where the evidence chains are genuinely separate.

## 8. Deliverable G — next-action shortlist (safe before the remaining decisions)

1. **S0 audit and housekeeping.**
2. **S1 statistical foundations.**
3. **S2 hypotheses and literature.**
4. **S5a detector-response evidence pass** (literature only).
5. **S3a existing Geant4 evidence** (after the S1 tools).

Details and reasons: `RESEARCH_PLAN.md` section 18.

## 9. Adversarial review (2026-10-11; independent adversarial reviewer, read-only, sealed files not opened)

| attack | verdict | what changed in this record |
|---|---|---|
| 1. conventional AUC is physically easy, not an implementation error | WEAKENED | AUC ranges tied to their files (large vs small run); the 0.991 / 0.998 population inconsistency flagged for S0; discriminating checks added to S0: a same-class null (electron set vs electron set, AUC about 0.5 expected), an energy-blind pooled test (energies pooled, deposit normalised), layer-block ablations |
| 2. the multiscale rung is "mostly missingness" | WEAKENED | item 3 rewritten: missingness at 10 GeV and on the readout; elsewhere large within-eligible information that is redundant with the conventional features; the indicator AUC is exact, not an approximation |
| 3. EMV is a configuration suspect | WEAKENED | item 7 rewritten: the depth-dependent deficit and the passing equivalence control point to the option-1 constructor or its transport settings; "not a credible EM effect" withdrawn as unsupported; initialisation-dump checks listed; YAML comment error recorded (C15) |
| 4. DECIDE-2 | WEAKENED + defect | 2 sealed proton seed collisions derived from the seed policy (C14); existing sets stay closed until a frozen FastMC candidate exists; exit rule; P4 insufficiency stated |
| 5. DECIDE-4 | box-side criticism SURVIVES for the registered windows (overstated for side 2); 3D-as-primary WEAKENED | item 6 corrected; recommendation revised to two objects (view-aware readout; 3D truth-level reference with an axis-free axis, 3D eligibility from 3D controls, the sampling step separate) |
| 6. endpoint family | WEAKENED | DECIDE-1 now requires a named primary, a P2 decision rule, an occupancy-conditional estimand with a planted positive control, family-wise control, and "not testable" cells |
| 7. sample-size table | SURVIVES (arithmetic) | 1-sigma vs 95% precision, threshold uncertainty, population and independence stated |
| 8. overclaims | partly | counts tied to their files (79 of 192 holds for the large run; 61 of 384 for `multiscale_development.json`); seeds unique per file only, paired variants share seeds by design; the different commits of the baseline (`48a4a78`) and the EM variants (`1984f4f`) recorded (they differ only in the list-selection code); an energy-only and a logistic baseline become an S0 gate, since total energy is the strongest separator |

The adversarial reviewer's bottom line: the central qualitative conclusions survive (the 0.01 rule is unmeetable here; the multiscale rung is largely redundant with conventional features; the sealed-set and FastMC decisions must be separated); several specific numbers and framings did not survive as first written and are corrected above.
