# RESEARCH PLAN — the single living plan of the AMS ECAL project

_Created 2026-10-10 at the researcher's request ("We only have one plan including everything"). Consolidates every plan, pre-registration, decision, experiment and hypothesis record in the repository as of commit `b0d7b13`. **This is the only plan.** Update it in place; do not create a new plan file per session._

## 0. How this file is used (the rules)

1. **One plan.** Status changes go in this file. Dated files in `research/plans/` are EVIDENCE RECORDS (notes, results, pre-registrations); they are cited here, never duplicated. A new idea is a new row here, not a new plan file.
2. **Read it every session, unprompted** (with `STATE.md`, `DECISIONS.md`, `research/graph/`, and `graphify query`), and state in one line where the work stands against it before doing anything else. Nothing is described as "new" or "untested" before this file and `research/graph/` have been checked.
3. **Status words:** DONE (tested, recorded) · DEV (development evidence only, not validated) · RUNNING · TODO (planned, never run) · BLOCKED (named blocker) · DECIDE (waits for the researcher) · CLOSED (tried, not adopted, kept as a result) · GAP (literature or fact not verified).
4. **Standing constraints:** sealed sets opened once, through `sealed_set.open_set`, with a registered harness; DEC-001 and the proton model are not changed by the electron work; nothing is called "validated" before the sealed verdict (use calibrated, candidate, development); the classifier AUC is a diagnostic, never a tuning target; rules and thresholds are fixed in code before a run and never relaxed after it; no literature PDF is downloaded without permission; commit only when asked, never push unasked; the researcher decides what becomes accepted knowledge.

## 1. Goal and scope

**RQ-001 (`questions/RQ-001_multiscale_shower_information.md`):** to what extent do e/p showers show discriminative multiscale structure, how much survives AMS-like readout, and what does it add beyond conventional observables and standard ML. Hypotheses H-002 (physical difference), H-003 (detector survival), H-004 (incremental information), H-005 (inductive bias / efficiency).

**Researcher's scope, 2026-10-10:** (1) does a fractal feature exist and survive the readout; (2) efficiency, all three meanings: sample efficiency, model/compute efficiency, electron selection at fixed rejection, together with information beyond conventional features; (3) fractal-inspired architectures for CNNs and ViTs (recorded in `CHAOS_FRACTALS_QML.md` section 6 B and C as candidates, gated by validated observables; NOT in the roadmap papers); (4) detector effects (an ACCEPTED program track: R-A sampling response, R-B instrumental effects, 2026-10-02). QML is Paper 2, downstream (roadmap).

**Evidence hierarchy (RQ-001):** real data where legitimately available > AMS-like projection of Geant4 > fine-grained Geant4 > FastMC as development / smooth control. The fractal-preservation gate compares THREE levels on the same events: fine Geant4, 18 x 72 projection, FastMC (cases A, B, C of RQ-001).

## 2. Status board (where we are)

| workstream | state |
|---|---|
| A. Estimators and controls | synthetic controls DONE (usable only from ~1000-3000 hits); two planned controls never implemented (A3, A4) |
| B. Geant4 data | exposed sets DONE; development-large sets DONE (electrons 10,000, protons 20,000 per energy); sealed sets generated, UNOPENED; electron systematics DONE 2026-10-11 (0.1 mm cut, EMZ, EMV at 20 and 100 GeV; analysis TODO) |
| C. Multiscale science | DEV readings only (existence strong in ideal Geant4; no headroom for added information in that regime); the registered test and the preservation gate are TODO |
| D. FastMC electron | Slices 1 and 2 DEV; four lateral extensions CLOSED; occupancy mismatch OPEN (AUC 1.000 vs Geant4) |
| E. FastMC proton | implemented, in-sample DEV; sealed validation TODO |
| F. Dataset | assembler and checks D1-D9 exist; no dataset assembled; thresholds pending |
| G. Detector response | R-A TODO; R-B BLOCKED on the evidence pass |
| H. Classical ML and architectures | HGB ladder DEV (rungs 1-3); raw/CNN/ViT baselines and fractal architectures TODO, need ~10^6 events |
| I. QML | gated, nothing started |
| J. Learning and ownership | tutor session open (Q1 unanswered); curriculum A-H planned |
| K. Process | STATE.md and research/graph refreshed 2026-10-10; all commits pushed 2026-10-10 (local = origin); full suite re-run started 2026-10-11 |

## 3. Workstream A — estimators and controls (EXP-008; `plans/2026-10-02_multiscale_estimator_*`)

| id | item | status | evidence / next |
|---|---|---|---|
| A1 | D0, D1, D2 estimators on 18 x 72 (box sides 1, 2, 3, 6, 9, 18) | DONE | `multiscale.py`, tests |
| A2 | Controls: smooth baseline, cascade, hit-count sweep; DEC-013 rules (10% false positives, 90% power, 15% recovery line) | DONE (synthetic) | NOT USABLE at <=300 hits; usable on 2 of 13 windows at 1000, 4 of 14 at 3000 hits; cells-equivalent: 336 and 572 |
| A3 | **Negative control: cells shuffled within each layer** (weekend plan section 2) | **TODO, never implemented** | add to `multiscale_controls.py`; also the registered "shuffle" baseline of the test |
| A4 | **Controls on a finer grid for comparison** (weekend plan section 2); needed for any fine-scale reading | **TODO, never implemented** | DECIDE first: how the 3D 3 x 3 x 4.625 mm voxels become a 2D image (alternating x/y views) |
| A5 | Eligibility count on real Geant4 events | DONE 2026-10-09 (exposed) | readout: 165-350 cells, below 336 for most electrons at 10-50 GeV; crossing protons ~20 cells |
| A6 | Ordering fraction 0.90; window rule stricter than saturation | confirmed "for now" 2026-10-09; the cell-count eligibility rule replaces the lenient saturation rule | review after first use |
| A7 | Finite-resolution bias characterisation (RQ-001 step 3) | partial (recovery table, 18 x 72 only) | extend with A4 |
| A8 | Literature gaps: Hentschel-Procaccia 1983, Bialas-Peschanski 1986 (not in library), q<0 snippets, Pattern Recognition 209 | GAP | `plans/2026-10-09_multiscale_literature_verification_record.md`; researcher adds papers |

## 4. Workstream B — Geant4 data

| id | item | status | note |
|---|---|---|---|
| B1 | Proton pilot, 10/20/50/100 GeV x 4000 + variants (QBBC, QGSP_BERT, fixed entry, extended, 0.1 mm cut at 10 and 100 GeV, material) (EXP-002) | DONE | held-out `event_index % 4 == 3` consumed by the crossing look (EXP-004) |
| B2 | Electron timing probe; electron sample 4 x 1000 (EXP-009) | DONE | exposed development data |
| B3 | Extended-depth electrons 4 x 1000, 270 layers | DONE | development hold-out for the FastMC |
| B4 | Sealed sets: protons 7 energies x 3000 (seed 20271001), electrons 7 x 1000 (20271002), hash manifests, opening guard | generated, **UNOPENED**, no opening ledger | one opening only; see DECIDE-2 |
| B5 | Development-large sets: electrons 4 x 10,000, protons 4 x 20,000 (seeds 20272001/2), clean commit `48a4a78` | DONE 2026-10-10 | files are in the worktree `../ams-ecal-qml-gen/data` (manifest and data-root resolver DONE 2026-10-11, S0: `research/data_manifest.json`, `validation/sample_definitions.py`; the files were NOT moved, they are found through the resolver). The name "G-train" is misleading (nothing is trained on it as a model); rename: DECIDE-9 |
| B6 | Electron systematics: 0.1 mm cut, `FTFP_BERT_EMZ` (EM option 4) and `FTFP_BERT_EMV` (option 1), 20 and 100 GeV | DONE 2026-10-11 00:40 (all rc=0; 2000 unique seeds per file, no NaN/inf) | cut set in `../ams-ecal-qml-gen/data` (commit `48a4a78`), EMZ/EMV in `../ams-ecal-qml-gen2/data` (commit `1984f4f`); analysis is B9; `FTFP_BERT_TRV` withdrawn (hadronic only) |
| B7 | EXP-007 alternative physics-list calibration (QGSP_BERT) | PARTIAL | `results/proton_model/physics_list_systematic.json` exists (a systematic, not a calibration); graph record says planned: reconcile |
| B8 | Incidence angle, backsplash, material systematic | DECIDE | "before or after the proton model" still open; normal incidence only today |
| B9 | Production-cut sensitivity before any sub-cell claim | partial data (proton 0.1 mm; electron running) | analysis TODO |
| B10 | Real / test-beam / ISS data | none available | only published ISS results (Hashmani et al. 2024: ISS 50-70 GeV, MC 0.2-2 TeV) |

## 5. Workstream C — the multiscale science (RQ-001)

| id | item | status | evidence / next |
|---|---|---|---|
| C1 | H-002 physical difference, e vs interacting p | DEV | matched KS 0.3-0.55 at 10-100 GeV (deposition), `research/RESEARCH_PLAN.md (Appendix E)` sections 11-13 |
| C2 | **Proton first look vs a matched smooth baseline** (same hit count, same layer-energy profile; fine voxels and 18 x 72) (weekend item 3, EXP-001 stage 1) | **TODO; item 3 of the approval table still "open"** | the development run compared classes, not each class against a smooth baseline |
| C3 | **Fractal-preservation gate: fine Geant4 vs 18 x 72 vs FastMC (cases A/B/C)** (EXP-001 stage 2) | **TODO (EXP-001 "planned", no approval recorded)** | needs A4, a fine-image definition, and a FastMC at fidelity |
| C4 | H-003 survival through readout | DEV | readout grid has too few cells for most electrons below 100 GeV; fine-scale part TODO |
| C5 | H-004 added information | DEV | no headroom in ideal Geant4 (conventional AUC 1.0000 even without energy and cells; multiscale gain +-0.001); registered rule unmeetable: DECIDE-1 |
| C6 | H-005 efficiency / inductive bias (sample, model/compute, selection-at-fixed-rejection) | TODO | needs the hard regime (detector level, realistic energies, ~10^6 events) |
| C7 | Scale-removal ablations (roadmap) | TODO | |
| C8 | Confounders of RQ-001 (null list) | partial | energy mismatch: by design (fixed energies) and matched; ordinary width/depth: ladder DEV; incidence angle: untested; containment/boundary: reported, untested as a confounder; thresholds/noise: needs G; preprocessing and track-centring (entry spot 0-9 mm): untested; simulator artefacts: cut and EM-list runs pending |
| C9 | The registered test: pre-registration `plans/2026-10-09_multiscale_test_preregistration.md` | numbers CONFIRMED "for now" 2026-10-09; amendment DECIDE-1; one opening of the sealed sets TODO | |
| C10 | Conventional-feature control and classical ladder rungs 4-5 (raw grid, raw + multiscale) | rungs 1-3 DEV; 4-5 TODO | `multiscale_ladder.py` supports them |

## 6. Workstream D — FastMC electron (`em_production`; DEC-014; notes `plans/2026-10-05..07_em_*`)

| id | item | status | note |
|---|---|---|---|
| D1 | DEC-001 smooth generator kept as `em_smooth_null` | DONE | control |
| D2 | Slice 1 longitudinal gamma family, calibrated origin/beta, Grindhammer-Peters widths, skew-normal ln T + copula | DEV | not at the Geant4 floor |
| D3 | Slice 2 Grindhammer-Peters lateral spots (R_M = 18 mm) | DEV | width KS 0.7, hit cells 55% of Geant4; AUC 1.000 vs Geant4 |
| D4 | E1, tail-sensitive calibration (two weights), faster tail, E2 | CLOSED (none adopted by the pre-stated rules) | faster tail missed one sub-criterion by 0.11 points: **DECIDE-3** |
| D5 | **Stage 4: layer-to-layer residual noise** (generator plan section 8.2) | **TODO, never started** | |
| D6 | Occupancy mismatch | OPEN | E2 (tail quantum 1.36 MeV) fixed occupancy on Slice 2 but regressed width, coupling, spot number; candidates (spot refit with E2 on; energy-dependent quantum) not started |
| D7 | Pre-stated signs of "gamma family inadequate" (before any sealed electron read) | **PENDING the researcher's confirmation** | `plans/2026-10-06_em_production_slice1_note.md` section 5 |
| D8 | Sealed electron opening (EM contract) | TODO | conflicts with the multiscale opening: DECIDE-2 |
| D9 | Unverified GP assumptions (E in GeV, Z = 82); R_M 18 mm vs 27.6 mm (Rossi) | GAP | recorded in the lateral note section 14 |
| D10 | Literature gaps: Geant4 manual, AtlFast3, NKG snippet-level | GAP | |
| D11 | Teaching notebook for `em_production` | TODO | notebooks only after the model is final (NAMES rule) |
| D12 | Open configuration questions: which statistic `T_bar` represents; whether the `deposition` regime should use the Grindhammer-Peters homogeneous constants so the regimes differ only by the sampling shift (STATE open questions 1-2, DECISIONS open list) | **DECIDE** | configuration-level, carried over from the retired state block |

## 7. Workstream E — FastMC proton (`proton_model`; EXP-003..007; DEC-004..011)

| id | item | status | note |
|---|---|---|---|
| E1 | Crossing branch, burst latent, lateral spill, bulk coupling (EXP-005) | DEV | in-sample only; frozen artifact `cee305f1` |
| E2 | Interacting model, lateral quanta, batch path (EXP-006) | DEV | in-sample only |
| E3 | Added-checks harness (layer correlations, classifier two-sample test, sparsity, ungated multiscale panel), mutation-tested | DONE | registration `plans/2026-10-02_added_validation_checks_preregistration.md` |
| E4 | Harness dry run | became a development pass (declared 2026-10-05) | stand-ins; not evidence |
| E5 | **Final validation on the sealed proton set** (step 10) | **TODO, sealed set unopened** | thresholds: KS 0.05 NOT confirmed as universal; **to be re-registered from a split-Geant4 floor before any opening: DECIDE-7** |
| E6 | QGSP_BERT calibration (EXP-007), claims about hadronic physics | TODO (see B7) | |
| E7 | Final report (plan section 44), teaching notebook (section 40), backsplash (section 26) | TODO | |

## 8. Workstream F — the e/p dataset (`validation/dataset.py`; plan `2026-10-04_dataset_level_checks_preregistration.md`)

| id | item | status | note |
|---|---|---|---|
| F1 | v0 perfect-event (`deposition`) dataset: log-uniform 10-100 GeV, normal incidence, 70/15/15, disjoint seeds | assembler and checks D1-D9 exist; **no dataset assembled** | needs Q9 answered, KS thresholds re-registered, and a validated electron generator |
| F2 | v1 detector-level dataset | BLOCKED on G | |
| F3 | The electron in the dataset is `em_production`; DEC-001 is a separate null, never mixed in | DECIDED 2026-10-05 | |
| F4 | Likely finding: electron/proton generator asymmetry lets a classifier separate classes (label leakage) | OPEN risk | D9 reported, not gated |

## 9. Workstream G — detector response (RQ-004, accepted split)

| id | item | status | note |
|---|---|---|---|
| G1 | **R-A sampling response** (true deposition -> fibre energy, calibrated on Geant4 and validated on sealed electrons) | **DEV 2026-10-15** (approved D6.1-D6.8, PA.1-PA.5) | electron class tables PASS the registered 3% closure at 10-100 GeV (`results/detector_response/r_a_closure.json`); electron table fails on protons and leave-one-energy-out (8-9%). Component-wise particle-agnostic version: tagged Geant4 run DONE (26 batches), registered criteria 1-4 NOT MET (design sections 11-12; EM ratio depends on shower energy scale); per-class tables stay the fallback. R-B module B1-B4 built (`readout_response.py`), sensitivity scans only. Open: v2 with a smooth energy-scale covariate (needs the researcher), manifest registration of the component samples, proton reproducibility caveat (6 of 300 events) |
| G2 | **R-B instrumental effects** (gain and PMT position effect, noise, thresholds, ADC range/saturation, dead cells, cross-talk) | **BLOCKED** | evidence pass incomplete: 2026-10-02 scout thin, no per-paper outcomes; per-effect "already inverted by AMS?" table missing |
| G3 | Open question 9: depth origin of AMS's fitted T0 (double counting) | queued | full Research OS literature task |

## 10. Workstream H — classical ML and architectures

| id | item | status | note |
|---|---|---|---|
| H1 | HGB ladder (conventional, multiscale, both), AUC and rejection at 90/99% | DEV | `multiscale_ladder.py` |
| H2 | Raw 18 x 72 baselines in the style of Hashmani et al. (MLP, CNN, ResNet, CvT) as controls | TODO | their MC training set was ~11.5 M events |
| H3 | Learnable multiscale representation (CHAOS section 6 B), geometry-aware branch with alternating X/Y (section 6 C) | PLANNED, gated by validated observables | not in the roadmap papers: **DECIDE-5** (promotion) |
| H4 | Fractal-inspired architectures (FractalNet etc. are in the library) | not yet defined | same promotion |
| H5 | Large samples (~10^6): FastMC arm at fidelity | TODO | the week's FastMC work serves this; fidelity gaps are D3/D6 |

## 11. Workstream I — QML (Paper 2): gated, nothing started

Candidates (roadmap, CHAOS section 7): compact validated multiscale inputs for VQCs, quantum kernels with matched classical kernels, QCNN as a multiscale bias, detector-topology-aware connectivity, Hamiltonian embeddings. Preconditions: Paper 1 establishes what multiscale information exists, survives readout and is useful; every quantum model faces a strong classical control with the same information and comparable budget. Out of scope: the separate chaos / recurrent-network program (CHAOS section 10).

## 12. Workstream J — learning and ownership

| id | item | status |
|---|---|---|
| J1 | Tutor session `01 Projects/AMS ECAL QML/Tutor Sessions/2026-10-09 Week Learn-Back Method and Decisions` (Research OS v0.8.5, HTML-native) | open, Q1 unanswered; curriculum blocks A detector and data, B shower physics, C calibration and adoption rules, D validation hygiene, E statistics, F multiscale, G ladder and headroom, H workflow |
| J2 | Defense (viva) on the whole work (Research OS defend-research) | TODO, after the curriculum |
| J3 | Agent-introduced unreviewed choices (tolerances, moment weights, spot structure, doubling constant, tail exponent, GP assumptions) | listed in `plans/2026-10-07_em_production_slice2_lateral_note.md` section 14; researcher review TODO |
| J4 | Older parked sessions: detector-response tutor session (Q1 posted), proton learn-back | parked; the Obsidian vault has no note for the 2026-10-02..10-10 branch |

## 13. Workstream K — process and housekeeping

| id | item | status |
|---|---|---|
| K1 | `STATE.md` "Next session - start here" (stale since 2026-09-29) rewritten | DONE 2026-10-10 |
| K2 | `research/graph/` statuses: EXP-001, 005, 006, 007, 009, H-002..H-005, RQ-003 updated with dated notes and attested approvals; graph validates with 0 errors (warnings: EXP-004, 005, 008 completed without result validation, which is true) | DONE 2026-10-10 |
| K3 | The five retired plan files carried over verbatim as Appendices A-E and deleted; every reference rewritten | DONE 2026-10-10 |
| K4 | Commits pushed 2026-10-10 at the researcher's request; local and origin equal at `9258ebe` (checked 2026-10-11) | DONE |
| K5 | Research OS v0.8.5 is installed but its repository `main` is at v0.8 (merge `e50692b` is on a Codex branch) | DECIDE-8 (sync) |
| K6 | Full test suite: 916 passed, 3 warnings, 1893 s (2026-10-11, `.venv/Scripts/python.exe -m pytest -q`) | DONE 2026-10-11 |
| K7 | `graphify update .` after code changes | TODO (hook-driven) |

## 14. Decision queue (the researcher's, in the order they block work)

_2026-10-11: revisions of DECIDE-1 to DECIDE-9 after the program review are in section 18.5 (memo: `research/reviews/2026-10-11_program_redesign_review.md`, section 5), **ACCEPTED 2026-10-11** (DECIDE-1 to 4 resolved as proposed; open sub-items listed in section 18). The entries below are kept as the history of the queue._

1. **DECIDE-1: amend the pre-registered information test.** Candidates: (i) rejection at fixed efficiency as primary with a ~10^6-event sample; (ii) matched design on events the conventional features cannot separate; (iii) detector-level, realistic-energy setting; (iv) report "unanswerable in the ideal regime"; recommended: (i) + (iii).
2. **DECIDE-2: the sealed sets.** They are registered as the one-opening validation of the FastMC generators (DEC-014 Q11; the added-checks registration); on 2026-10-09 the researcher agreed that the multiscale test could open them first. Both cannot hold. Options: new sealed sets for one purpose (about a day of Geant4), or one registered joint opening.
3. **DECIDE-3: the faster-tail candidate** (override the 0.3-point rule, refine, or keep Slice 2) and whether to continue the occupancy work (D6).
4. **DECIDE-4: the fine-image definition** (projection of 3D voxels, alternating x/y views) so A4, C2, C3 can start.
5. **DECIDE-5: promote the architectures and the three-efficiency study** into the roadmap (research question, hypothesis, paper boundary).
6. **DECIDE-6: start G1 (R-A) and the R-B evidence pass.**
7. **DECIDE-7: confirm the gamma-inadequacy signs (D7); re-register the KS thresholds from a split-Geant4 floor (E5); answer Q9 (F1).**
8. **DECIDE-8: sync the Research OS repository** (the push was done 2026-10-10).
9. **DECIDE-9: backsplash / incidence angle / material ordering (B8); rename "G-train" (B5).**

## 15. Critical path and time (estimate, not measurement)

generation finished (2026-10-11 00:40) -> DECIDE-1, 2, 4 -> A3, A4 (fine controls, ~1 day) -> C2 (proton smooth-baseline look, ~1 day) -> development analysis in the amended design (2-3 days) -> frozen commit -> the sealed opening (1 day). In parallel: G1 (R-A) and the R-B evidence pass; the FastMC arm; curriculum teaching (6-8 h). Architectures, efficiency curves and QML wait for the hard regime and a large sample.

## 16. Where the earlier plans went (nothing dropped)

The five retired plan files are carried over VERBATIM as Appendices A-E at the end of this file and have been deleted from `research/plans/` (retrievable from git, see each appendix header). Their items are tracked in sections 3-13 above. The remaining files in `research/plans/` are evidence records and binding pre-registrations (notes, results, literature records, the estimator, added-checks, dataset-level and multiscale-test pre-registrations); they are cited, not retired.

| appendix | content |
|---|---|
| A | the researcher's proton model specification (received 2026-09-29), steps 1-45 |
| B | tracks A-E, validated FastMC dataset and the detector response, Q1-Q7, progress record 5c |
| C | weekend estimator gate, proton first look, electrons, sealed sets, learning gates, approval table, timing probe |
| D | EM production generator: stages, F1-F5 decisions, calibration, slices 1-2 |
| E | multiscale test datasets, development readings 2026-10-09/10, scope, correction |

## 17. Corrections of earlier statements (kept for honesty)

- Fractal architectures: they ARE in the repository (CHAOS_FRACTALS_QML section 6 B, C) as gated candidates, though not in the roadmap papers.
- Detector effects: an accepted track, not something needing promotion.
- FastMC fidelity: not off the critical path once architectures and efficiency are in scope (needs ~10^6 events).
- Generation timings were underestimated (100 GeV electrons took ~6 h, not 4).
- (2026-10-11) "FastMC fidelity is on the critical path because ~10^6 events are needed": measured Geant4 rates make 10^6 normal-incidence protons about one day of wall time, so large samples do not require FastMC (section 18, DECIDE-3).
- (2026-10-11) Every proton-rejection number in the development readings is the zero-failure floor (rejection = number of protons), not a measurement; the valid 95% lower limit is about N/3 (section 18.3).

## 18. PROPOSED program redesign (2026-10-11) — every item PROPOSED or DECIDE, nothing accepted

> **Accepted 2026-10-11** (researcher: "Yes decisions are valid, and we can go for the slices in the plan 1 by one now"): DECIDE-1 to DECIDE-4 as proposed in 18.5, the consequences for DECIDE-5/6/7/9, the framework of 18.1 as the working framework, the gates of 18.3 and the slice order of 18.4, run one by one from S0. Still OPEN: the DECIDE-1 amendment text (from S2), the DECIDE-2 exit date, regenerate-or-ledger for the 2 sealed-proton seed collisions. The wording below is kept as proposed on 2026-10-11 (history); this note records its acceptance (`DECISIONS.md`, 2026-10-11 rows).

_Requested by the researcher on 2026-10-11 (scientific redesign, statistical audit, AMS realism). Evidence, reviewer reports, the adversarial pass and the decision memo are in `research/reviews/2026-10-11_program_redesign_review.md` ("the review"). This section changes no accepted decision, no registered number and no status above; it becomes part of the plan item by item as the researcher accepts it. Sealed sets were not read._

### 18.1 Broadened question (PROPOSED; a hypothesis framework, not a fact)

EM showers may show reproducible finite-scale geometric organisation governed by cascade development; hadronic showers may show additional spatial heterogeneity (hadronic interactions, secondary tracks, embedded EM sub-showers); these may be described by depth-dependent multiscale statistics, transformed by detector coarse-graining and sampling, and useful for physics interpretation, simulation validation and downstream inference. Competing explanations kept explicitly: EM nearly monofractal (Anchordoqui et al., D_q about equal); no robust scaling window (about 0.5-1 decade available on the readout); every difference reducible to occupancy, start depth, width and containment (smooth profile + Poisson granularity null). Language: "finite-scale heterogeneity statistics"; "fractal / multifractal" only after the estimator controls pass against a surrogate null. RQ-001 and H-001..H-005 stay as recorded; the new hypotheses H-A..H-G (the review, section 2) enter `research/graph/` only on acceptance.

### 18.2 Findings that drive the redesign (development evidence; the review, section 1)

1. Conventional AUC 0.9999-1.0000 is a physically easy, unrealistic problem (fixed known energy, normal incidence, zero-threshold readout); no reviewer found an implementation error.
2. Rejection numbers are zero-failure floors (C1); the threshold is set on the evaluated events (C2).
3. The multiscale rung is missingness at 10 GeV and on the readout; elsewhere it is redundant with the conventional features (C3).
4. The development KS verdicts are not interpretable (C4-C6, C13); the cell-shuffle null was never built (A3).
5. The registered 0.01 AUC-gain rule is unmeetable by construction (C10).
6. The readout grid is a stack of alternating one-dimensional views; the registered box sides mix x and y.
7. `FTFP_BERT_EMV` electrons have a 37% lower scintillator share at the same total deposit (open; not used as a systematic until explained).
8. Two sealed proton seeds equal seeds of exposed events (derived from the seed policy, no sealed read; C14).

### 18.3 Statistical validity gates (PROPOSED; no confirmatory work before they hold)

| gate | content |
|---|---|
| SV-1 | rejection reported as k of N with an exact one-sided Clopper-Pearson bound; a censored lower limit when k = 0; coverage simulated |
| SV-2 | four data roles: train / model selection / operating-point calibration (electrons) / final evaluation; threshold uncertainty propagated |
| SV-3 | repeated cross-fitting (>= 5 seeds); seed spread reported |
| SV-4 | missingness controls: indicator-only rung, eligible-in-both analysis, "not testable" cells declared |
| SV-5 | permutation nulls within energy x occupancy strata; the cell-shuffle null; floor from matched-size splits |
| SV-6 | multiplicity: one NAMED primary per endpoint family; Westfall-Young or Benjamini-Hochberg for secondaries; family-wise control across families |
| SV-7 | energy-only and logistic baselines beside every ladder; a same-class null (AUC about 0.5) |
| SV-8 | each claim written as estimand -> null -> statistic -> uncertainty -> assumptions -> acceptance -> failure interpretation (the review, section 3) |
| SV-9 | an independent statistical review of the frozen analysis code before any registered test |
| SV-10 | one sample-definition module and a data manifest; 63-bit seeds for new sets; seed intersection checked before any opening |

### 18.4 Revised slices (PROPOSED; each needs the researcher's go)

| slice | objective | prerequisite evidence | depends on | deliverables | validation tests | resources | stop / failure | approvals | learning (Research OS) |
|---|---|---|---|---|---|---|---|---|---|
| **S0 audit and housekeeping** | reconcile data and sample definitions; explain the easy AUC; diagnose EMV | the review, sections 1, 3 | nothing | data manifest and safe move/link of the exposed sets (B5); one sample-definition module (C8, C9); sanity audit on exposed data: same-class null, energy-only and logistic baselines, feature-family and layer-block ablations, label permutation, duplicate injection, indicator-only rung, repeated seeds, energy-blind pooled test; EMV initialisation dumps (no events); YAML comment fix (C15) | permutation and same-class AUC about 0.5; manifest hashes match | 2-20 CPU-min per check; no Geant4 events | stop if an implementation error appears: fix before anything else | go for S0 | Block D (validation hygiene), Block E (AUC, permutation) |
| **S1 statistical foundations** | SV-1..SV-8 as tested code | the review, section 3 | S0 sample module | modules: censored CP rejection, threshold calibration, cross-fitting, permutation nulls with FDR, cell-shuffle null (A3), coverage simulations; unit tests | coverage of the CP bound and of the permutation test on electron-vs-electron splits | CPU-hours | stop if coverage fails: no interval reported until fixed | go to change development code (registered numbers untouched) | Block E (binomial bounds, permutation, FDR) |
| **S2 hypotheses and literature** | formal H-A..H-G with named endpoints; full-text gap map | reviewer B's lead list (snippet depth); 2026-10-09 literature record | none (parallel) | full-text verification of the leads (Ruan 2014 and citing work, CaloChallenge metrics, air-shower multifractal work, resolution artefacts in intermittency); finite-scale interpretation rules; the DECIDE-1 amendment text with the NAMED primary, the P2 decision rule, the planted positive control | every implementation-affecting claim with a page locator read by the author; unreadable papers go to the researcher as links | hours | no novelty claim; only "not found" with the search scope | the researcher reviews hypotheses and amendment | Block F (D_q, scale windows), source discipline |
| **S5a detector-response evidence** | per-effect R-B table; R-A plan | reviewer C's table (part read in the PDFs) | none (parallel) | effect, documented value, source locator, "already corrected by AMS?", sensitivity range when undocumented | each value with a locator or "not found" | hours | no parameter without evidence or a declared sensitivity scan | DECIDE-6 to start R-A | Block A (detector), new detector-response block |
| **S3 existing Geant4 evidence** | source of conventional separability; stability; the variants | S0, S1 | S0, S1 | ablations; view-aware statistics on the readout; cut and EMZ variant comparison (EMV after its diagnosis); occupancy- and morphology-matched hard subsets; the proton smooth-baseline look (C2) | permutation nulls; sign stability across variants | CPU-hours, exposed data | a result that changes sign under a variant is "systematics-dominated" | go per analysis | Block G (ladder, headroom) |
| **S4 fine scale and readout preservation** | H-A, H-C, H-D on 3D voxels and the view-aware readout | S1, DECIDE-4 | S1, DECIDE-4, A4 | 3D estimator controls (A4) and 3D eligibility; depth-resolved statistics; paired degradation curves fine -> 9 mm 3D -> fibre energy (R-A) -> one view per layer | surrogate bands; planted-structure recovery | CPU-hours to a day (about 3,900 voxels per event) | no scaling window -> report "no window", no fractal language | DECIDE-4 | Block F |
| **S6 generator fidelity** | H-E; stop/go for FastMC | S1, S4 statistics | S1, S4, DECIDE-3 | multiscale fidelity tests of `em_production` and the proton model against the Geant4 split floor | permutation, floor | hours | DECIDE-3 stop rule: one time-boxed attempt at most | DECIDE-3 | Block C (generators) |
| **S5b / S7 realistic regime and difficult-background performance** | H-G1..H-G3 in an AMS-like regime | S1, S5a, R-A | DECIDE-6, DECIDE-9, S1, S5a | spectrum-weighted, angular, full-face Geant4 production with R-A/R-B (sensitivity-scanned); all-proton background; observable-defined hard subsets; censored CP rejection with independent calibration; conventional, cut-based, gradient-boosting and raw-image baselines with and without multiscale features | the SV gates | >= 3e4 evaluation protons per claimed 1e4 rejection (k = 0); about 1e6 for a measured 1e4 at 1-sigma 10%; about a day per 1e6 normal-incidence protons, more with TeV electrons and angles | never a statement of actual AMS performance; "AMS-like simulation with stated assumptions" only | go per production | Block A, detector-response block |
| **S8 confirmatory study** | the registered tests | everything above | DECIDE-1, DECIDE-2 | frozen hypotheses, estimands, controls and code in a tagged commit; NEW sealed sets per purpose (63-bit seeds); independent implementation and statistical review; opening ledger | SV-9 | about 30 CPU-h at the old sealed design, more for the realistic regime | the exit rule of DECIDE-2 | the researcher at freeze and at opening | defend-research viva before writing |

**Parallel now:** S0, S2, S5a (S1 once S0's sample module exists). **Then:** S3 (after S1), S4 (after S1 + DECIDE-4), S6 (after S4). **Last:** S5b/S7 (DECIDE-6, DECIDE-9, production), S8.

**S0 status, 2026-10-11 (branch `statistical-foundations`, uncommitted; record `research/reviews/2026-10-11_s0_sanity_audit.md`): DONE except one approval.** Manifest and resolver (`research/data_manifest.json`, `validation/sample_definitions.py`; nothing moved), the sanity audit (`multiscale_estimators/ladder_sanity_audit.py`) and the EMV initialisation dump are built and tested. **No implementation error found**: every control inside the tolerances fixed before the run (same-class nulls 0.483-0.515, permutations 0.488-0.513, fold-seed sd <= 8.4e-5; 0 duplicates and 0 shared seeds among the baseline pilot and development-large sets at each energy, not the pilot variants or sealed sets, so C14 stands), and the duplicate-injection positive control moves the AUC from 0.50 to 0.80/0.91, so the harness would show a leak. Findings: (1) the 0.991 vs 0.998 is readout vs deposition, not an inconsistency; (2) the problem is overdetermined: every feature family alone gives AUC >= 0.95 and the energy-blind pooled test (energy and occupancy dropped, four energies pooled) gives 0.99995, so "known beam energy" is only part of why it is easy; (3) the registered multiscale rung equals the eligibility indicator within 0.014 at 10 GeV and on the readout at 20-50 GeV (missingness), and exceeds it by 0.22-0.30 elsewhere (real but redundant information); (4) the held-out-quarter rule changes no AUC by more than 0.0038; (5) EMV differs from FTFP_BERT in at least seven physics settings (CORRECTED later the same day: the first dump read scalar getters only and reported two; Geant4's printed table shows also msc step limit `UseSafety` -> `Minimal`, e+- step function (0.2, 1 mm) -> (0.8, 1 mm), fluctuation model `Urban` -> `Universal`, combined TransportationWithMsc on, no Rayleigh scattering); the approved swap test reverting `ApplyCuts` and `MscRangeFactor` together restores only 41% of the scintillator share and FAILS its pre-stated closure test. **Researcher, 2026-10-11:** held-out defaults kept; EMV swap approved; go for S1.

**S1 status, 2026-10-11 (uncommitted; record `research/reviews/2026-10-11_s1_statistical_foundations.md`): tools BUILT and VALIDATED on simulations, not yet used.** Package `src/ams_ecal/inference/` (rejection limits with censored Clopper-Pearson, role-separated thresholds, repeated cross-fitting, missingness controls, stratified permutation nulls and the cell-shuffle null A3, Holm / Benjamini-Hochberg / Westfall-Young, the claim template), 56 tests; 27 coverage and type-I-error checks pass (lowest CP coverage 0.955; permutation type-I 0.030-0.060 on real electron halves; family-wise Holm/Westfall-Young 0.06-0.08 at 100 trials, consistent with control but to be repeated with at least 1,000 trials for S8). The old ladder and test modules are unchanged; they migrate to these tools in S3. **EMV swap result (approved test, 2,000 electrons at 20 GeV per list): the closure test FAILED**: `ApplyCuts` reverted restores 3% of the scintillator-share deficit, `MscRangeFactor` reverted 38%, both 41% (share 0.0512 against 0.0653); about 59% of the deficit is in settings not swapped (record: S0 section 7); further swaps (msc step limit, step function, fluctuation model, combined transportation, Rayleigh) are not approved and not run. **Next: S3** (re-read the existing Geant4 evidence with valid tools) once the researcher chooses the permutation count and the secondary-correction method (S1 record section 4); S5a (detector-response evidence) is DONE as an evidence record, see below.

**S3 waiting decisions, literature check 2026-10-15 (`plans/2026-10-15_s3_decisions_literature_record.md`; eight open-access papers read at the depths stated there, in the library `Stats/`):** permutation p-values must use (b+1)/(B+1); Holm needs B >= 20m - 1 (the author's derivation; m = 192 gives about 3,840 against the code default of 1,000); seed spread is not the AUC uncertainty (no unbiased variance of K-fold CV; nested or bootstrap variance needed); BH is proven under positive dependence and can exceed its rate under negative dependence by a bounded factor; the sign-structure check on the exposed electron/proton KS statistics (20 and 100 GeV) found them almost all non-negatively correlated under the permutation null (not a proof of PRDS). NOT found: a sourced permutation count or repeat count; the Westfall-Young validity dispute rests on unread primaries (Benjamini-Yekutieli 2001 and Westfall-Troendle 2008 still to be added to the library by the researcher). **DECIDED 2026-10-15 (researcher): B = 10,000; secondaries by Westfall-Young step-down where the joint permutation draws of the family's statistics exist (same B permutations, one permutation per row for all statistics, so no extra compute over Holm), by Holm otherwise; Holm always reported as the sensitivity, BH only as an exploratory view. Westfall-Young is used for a claim only after a family-wise-error check on same-class data for that family's statistics (Assumption C is not proven).** Checks run for the decision (record sections 6-8): on real feature dependence all procedures kept their claimed error rate (Holm and Westfall-Young FWER <= 0.05, BH false-discovery proportion 0.025 but FWER 0.10-0.22); Holm costs 3-5 points of power against Westfall-Young and about 8 against BH; one cross-fit takes about 2.4 s, so B = 10,000 is cheap for KS-type statistics (about 8 min) but 6.7 h per classifier statistic per energy (AUC-gain twice that), to be budgeted in the registration; the cell-shuffle test supports per-statistic claims through its excess estimand with event bootstrap, not through its shuffle p-values. Cross-fit seeds (5 kept as a project choice, uncertainty to come from coverage tests) still open. R-A (G1): the class-table error was traced (design sections 15-16) to the electromagnetic share of the deposit (sign-changing, 5-15%) and early interactions, not to the crossing/interacting split.

**S5a status, 2026-10-12 (uncommitted): evidence record delivered, `plans/2026-10-12_s5a_detector_response_evidence_record.md`.** Two independent scouts reached only snippets, but the decisive sources were already in the library and were read in full: Incagli PoS(TIPP2014)025 Sec. 6-7 states that "before clustering the hits" AMS applies fibre-attenuation correction (f = 0.17, 110 mm, 2605 mm), MIP equalization and HG/LG calibration (ratio about 33), with the 0.25 %/degC temperature slope corrected in offline reconstruction; Spinella et al. (IEEE TNS 2010) gives about 8 photoelectrons per MIP cell and pedestal subtraction with zero suppression. **Consequence for G3 / R-B:** gain-type effects are inverted at cell or layer level before shower shapes exist and must not be added as distortions; what can be added are the stochastic effects (Poisson photoelectron smearing at about 8 p.e./MIP is documented; the zero-suppression threshold, cross-talk and the AMS Birks value are NOT FOUND and stay sensitivity scans). **Updated later 2026-10-12 after NIM A 714 (2013) 147 and the two RDTM equalization papers were read:** the order attenuation -> equalization -> HG/LG with temperature offline is stated in two independent AMS papers; the 15% raw cell spread is corrected offline; the documented residual gain errors are 2.4% (proton MIP with Kalman filter) and 1.5% (helium); eight low-response cells exist since 2017; the energy sum gets a cell-edge/anode-efficiency correction (up to about 10%) and a rear-leakage correction; the PMT temperature slope is 0.25 %/degC in one paper and about 1e-3 per degC in another (unresolved, offline-corrected, not simulated). Still NOT FOUND: the readout zero-suppression threshold, a cross-talk value, the AMS Birks constant, the saturation level. The in-library JPCS 587 version of the in-space talk is unread.

**S2 status, 2026-10-12 (uncommitted): DRAFT delivered, awaits the researcher.** Independent discovery by six literature-scout agents in two rounds (neutral questions, no repository access, no downloads), then verification by the author in the PDFs: `plans/2026-10-12_s2_literature_verification_record.md` (11 sources verified in full text with locators, leads kept at their reading depth, per-paper outcomes, what was not found); `plans/2026-10-12_s2_hypotheses_and_amendment_DRAFT.md` (finite-scale interpretation rules; H-A..H-G as research-graph records H-008..H-014 with status `proposed`; the DECIDE-1 amendment text with five named primaries (P1-P4 plus **P5 efficiency**, added the same day at the researcher's correction that efficiency is accepted scope) written with the SV-8 claim template, a freezing rule, a planted positive control, the P2 decision rule, multiplicity and "not testable" cells). **Literature findings that change the picture:** the only calorimeter fractal-dimension paper (Ruan 2014) is simulation-only, support-based and its FD is tied to log N_hits; the closest precedent (HEGRA, Schaefer 2001) found multifractal parameters strongly correlated with conventional shape variables, a slight combined gain and pixels "much too coarse"; EM showers are near-monofractal in the one study found (Anchordoqui 2000); no multiscale fidelity metric exists in the fast-simulation literature; Geant4 lists differ from data and each other by 10-20% (CALICE, Geant4 9.6). **A null result is the literature-plausible outcome.** Two scout claims were found wrong when checked. **Waits for the researcher:** approval or changes of the named primaries and the freezing rule; INSPIRE citation trace of Ruan 2014 and the paywalled links (record section 6).

### 18.5 Decisions (PENDING; memo in the review, section 5)

| decision | proposal | what the researcher approves |
|---|---|---|
| DECIDE-1 | replace the single 0.01 AUC-gain rule by four endpoint families (P1 physics difference, P2 readout survival, P3 simulator fidelity, P4 realistic performance), each with a NAMED primary, through a dated amendment before any opening; the old rule stays in the record as "unmeetable in the ideal regime" | the amendment text (after S2) |
| DECIDE-2 | existing sealed sets stay closed; opened for FastMC validation only when a frozen FastMC candidate exists, otherwise an untouched reserve; NEW sealed sets for the multiscale confirmatory test once its design is frozen; the 2 seed collisions regenerated or ledgered; an exit date | withdrawing the 2026-10-09 "multiscale opens first" agreement; the exit date; regenerate or ledger |
| DECIDE-3 | no override of the 0.3-point rule; Slice 2 stays the candidate; electron-generator refinement PAUSED until H-E is defined, then one time-boxed attempt under a rule fixed in code; no FastMC sample in any performance or confirmatory claim | the pause and the stop rule |
| DECIDE-4 | two objects: view-aware readout analysis (detector level) and 3D sparse voxels as the truth-level fine reference (physical box sizes in mm, energy weighting, axis-free principal axis, 3D eligibility from 3D controls, the sampling step R-A separate from granularity) | the representation |
| DECIDE-5..9 | DECIDE-5 lower urgency (waits for H-G evidence); **DECIDE-6 higher urgency** (critical path of any realistic claim); DECIDE-7 re-registered with permutation-based rules instead of a new fixed KS; DECIDE-8 unchanged; DECIDE-9 incidence angle now required | as before |

### 18.6 Publication strategy (PROPOSED; the review, section 7)

Paper 1, simulation-based: finite-scale structure, hadronic heterogeneity, coarse-graining survival and simulator fidelity, with the PID question in the ideal and AMS-like regimes and censored rejections; negative outcomes (no window; differences that vanish under readout or matching; redundancy with conventional morphology; FastMC failing hierarchical fluctuations; no realistic PID gain) are results. Paper 2 conditional on R-A/R-B and the realistic regime; never a claim of actual AMS performance without collaboration data. QML stays downstream. `PUBLICATION_ROADMAP.md` is updated only on acceptance.

### 18.7 First actions, safe before the remaining decisions (each needs a go)

1. **S0** first: cheap, exposed data only, and it can expose an implementation error that would change everything else.
2. **S1** next: every later number depends on valid intervals and nulls; code only.
3. **S2** in parallel: the DECIDE-1 amendment needs named endpoints and verified literature before anything confirmatory.
4. **S5a** in parallel: literature only; it unblocks DECIDE-6, now on the critical path.
5. **S3** after S1: re-reads the existing Geant4 evidence with valid tools before any new production.

## 19. PROPOSED road to the machine-learning benchmarks (2026-10-15) — nothing here is accepted; every duration and cost is an estimate, not a measurement

_Requested by the researcher on 2026-10-15 ("plan ahead in detail, what have we left to do until we can start ML benchmarks"). New work is named by what it does (rule of 2026-10-02), not numbered._

### 19.1 What "ready to start" means (proposed definition)

- **Exploratory dry run (no claims):** the benchmark harness on the existing ideal-regime development-large data (electrons 10,000 and protons 20,000 per energy at 10-100 GeV). Purpose: debug the harness, measure learning-curve and compute scale, find leakage. Can start as soon as the harness exists (work package "benchmark protocol and harness"); it does not need S4 or new production.
- **Registered benchmarks (the Paper 1 numbers):** conventional-feature gradient boosting, raw-image models (MLP, CNN, ResNet, CvT-like, Hashmani et al. style) and the same with finite-scale features, at detector level in the AMS-like regime, evaluated for P4 (rejection at fixed electron efficiency) and P5 (efficiency: learning curves, model size and time). Entry conditions, all required: (1) the amendment signed and frozen; (2) the benchmark protocol registered before any evaluation event is looked at; (3) detector-level dataset v1 produced with passing checks; (4) R-A and R-B in place with sensitivity scans; (5) harness reviewed; (6) independent reviews done (on the researcher's command); (7) role-disjoint sets and electron-calibration thresholds fixed.

### 19.2 Work packages (dependencies in the second column)

| work package | depends on | what is done | researcher input | estimate |
|---|---|---|---|---|
| Housekeeping | nothing | push the local commits (the permission check blocked the agent's push; the researcher pushes or allows it); commit the run records; register the component and large-proton samples in `sample_definitions.py` and `data_manifest.json`; record the proton reproducibility caveat (6 of 300 events); unit test for `run_proton_split`; rerun the full suite (916 passed on 2026-10-11); refresh STATE and the research graph | push or allow | 0.5-1 day |
| Re-read of the existing evidence with valid tools (S3) | housekeeping, S1 tools | ablations, view-aware readout statistics, cut and EMZ variants (EMV after its diagnosis), occupancy- and morphology-matched hard subsets, proton smooth-baseline look, cell-shuffle null; registration of which statistics use B = 10,000 and which a smaller B; Westfall-Young only after the 1,000-trial family-wise check per family, otherwise Holm | list of secondary statistics | 3-5 days, CPU-hours |
| Fine-scale and degradation analysis (S4) | S3 tools, R-A decision | 3D estimator controls (A4) on the fine voxels (stored in `electron_train`, `proton_train`, the sample and pilot sets, so no new generation is needed), 3D eligibility, planted-structure recovery, depth-resolved statistics, paired degradation curves fine -> 9 mm 3D -> fibre energy (R-A) -> one view per layer | none | 4-6 days |
| Detector-response completion | R-A decision | R-A: either accept the unsplit class tables (closure 3-8% on the development set, 4-5.5% on the large tagged set) or register one more version; electrons beyond 200 GeV are an unvalidated extrapolation of the tables (fitted 10-100 GeV; a tagged 200 GeV sample exists); R-B: scan design (gain, photoelectrons, cross-talk, threshold, order B2) with provenance labels, no value without evidence | R-A choice; scan ranges | 2-3 days |
| Amendment, freeze and review | S2 draft, S4 | update the draft's multiplicity item (it still says BH for secondaries; now Westfall-Young with the guard, Holm otherwise, B = 10,000); planted-control criteria; the dated addendum after S4 (window, boxes, strata); statistics review of the frozen code (SV-9); tag and hash | approve P1-P5, freezing rule, headline null acceptable, P5 model/compute as a sixth family or a secondary, DECIDE-2 exit date and the two seed collisions, INSPIRE trace | 2-3 days + review latency |
| Realistic-regime design and generator | DECIDE-9, detector-response completion | spectra (electron and proton), angular distribution, full-face entry, ECAL-only, backsplash and material; today the generator has normal incidence and a 9 x 9 mm spot only, so angles and full-face entry are backend changes with a control run (reproduce the normal-incidence case) | DECIDE-9; spectra and angular range | 3-5 days |
| Production | generator + control | protons >= 1e6 (P4 needs >= 3e4 evaluation protons per claimed 1e4 rejection; about 1e6 for a measured 1e4 at 10%) and electrons >= 1e5-3e5; roles train / validation / calibration / evaluation with 63-bit seeds; new sealed sets for the confirmatory test (DECIDE-2). Cost under assumed spectra (proton E^-2.7 and electron E^-3 between 10 and 1000 GeV, from the measured 0.39-2.31 s per proton event and 0.25-0.30 s per GeV per electron): mean proton about 0.6 s, so 1e6 about 12 h; mean electron about 5.5 s, so 3e5 about 33 h, on the 14 workers of this machine | spectra, sample sizes | 2-4 days wall |
| Detector-level dataset v1 | production, R-A, R-B | apply the response (with the scans), roles and manifest, checks D1-D9, label-leakage check (F4), KS thresholds re-registered (DECIDE-7), question 9 | thresholds | 2-3 days |
| Benchmark protocol and harness | amendment text (for P4/P5); runs in parallel with production | registered protocol: classifier list and budgets, tuning discipline, energy-only, logistic and same-class baselines (SV-7), metrics (censored Clopper-Pearson rejection, log-loss learning curves, parameters, time), uncertainty procedure (nested or whole-pipeline bootstrap, with a coverage test on same-class data; seeds are not the uncertainty), leakage controls; dry run on the existing data. The machine has an RTX 4050 (6 GB) and 15 GB RAM; PyTorch is not installed in the project environment (a dependency decision); 1e6 events x 18 x 72 is 5.2 GB in float32, 2.6 GB in float16 (memory-mapped) | install PyTorch; classifier list | 4-6 days |
| Independent reviews | frozen code and protocol | statistics (SV-9), ML (leakage, fair comparison), simulation (production), physics (R-A/R-B), reproducibility audit | the command to release them | latency unknown |
| Learning and defence track | runs throughout | sampling calorimetry, the meaning of the ratios, fractals from zero (F0.5 onward), permutation and multiplicity ideas, one claim card per result, spaced retrieval | start T1 | in parallel |

### 19.3 Critical path

housekeeping -> S3 and detector-response completion (parallel) -> S4 -> amendment, freeze and review -> realistic-regime design -> production -> dataset v1 -> registered benchmarks. The protocol and harness run in parallel from the moment the amendment text is settled, and the dry run needs only existing data. If decisions arrive promptly the whole chain is about 4-6 weeks of working time, 3-6 days of which is compute. The estimate is dominated by the researcher's decisions and the review latency, not by compute.

### 19.4 What would change the plan

- A null result in S3 or S4 (the literature-plausible outcome): the benchmarks still run, because P5 and the efficiency curves are reportable whatever the finite-scale features do; the claims shrink, not the programme.
- R-A not meeting the 3% closure in the realistic regime: report the disclosed systematic in the degradation curves and the benchmarks; a new R-A version only by the researcher's go.
- Electron energies above 200 GeV: R-A is an extrapolation there; cap the electron spectrum at 200 GeV (the cheapest honest choice) or tag a higher-energy electron sample.
- Cost: cloud or fewer events change the schedule, not the design; the GPU is small, so ResNet and CvT-like models need small batches and float16.
- The reviews may invalidate a result; their findings are fixed before the registration, not after.

### 19.5 Decisions needed, in the order they block work

1. Push the local commits (or allow the push).
2. R-A: accept the unsplit class tables as the working version, or one more registered version.
3. Approve, change or reject P1-P5, the freezing rule and the planted-control criteria; is the headline null acceptable?; a sixth family for model/compute efficiency?; DECIDE-2 exit date and the two seed collisions.
4. DECIDE-9 (incidence angle is required for the realistic regime, backsplash, material) and the spectra and sizes.
5. Install PyTorch; the classifier list for the benchmarks.
6. Release the reviewers.

---

# APPENDICES — retired plans, verbatim

---

## Appendix A — Proton model implementation plan — researcher's specification, 2026-09-29

_Retired plan, carried over verbatim on 2026-10-10 (original file `research/plans/2026-09-29_proton_shower_model_implementation_plan.md`, last version in git at `e904286`; retrieve with `git show e904286:research/plans/2026-09-29_proton_shower_model_implementation_plan.md`). Content: the researcher's proton model specification (received 2026-09-29), steps 1-45. Headings are demoted one level; nothing else is changed except that references to retired files point to their appendix._


_Received in chat on 2026-09-29, after the Geant4 proton pilot. Recorded below in full so it
survives across sessions: the researcher's wording is kept, but many of the original's
one-item-per-line code blocks are flattened into prose lists; the pseudocode and the graph are
kept as blocks. The researcher's instruction: implement it **after the tutoring**
(the codebase-and-Geant4 probe in Obsidian,
`01 Projects/AMS ECAL QML/Tutor Sessions/2026-09-29 Codebase and Geant4 Probe.md`).
It is the researcher's decision on the proton model structure proposed in `research/STATE.md`._

### Researcher amendment, 2026-09-29 (during the tutoring)

**Two representations for protons, as for electrons.** the EM event generator offers a `deposition` regime
(true all-material deposit, the "perfect event") and a signal-level regime. The researcher
decided the proton model must likewise provide **true deposition** and **readout (fibre /
scintillator energy)** representations, so electron and proton FastMC events can always be
compared in one common representation. The pilot stores both per event (`deposit_grid_mev`,
`readout_grid_mev`), so the calibration builder produces both from the same events; choosing a
representation is explicit configuration, and provenance records it. Dependency analysis (step 0) must also settle
how this pairs with the EM event generator's two regimes (the EM event generator `sampling` regime moves the longitudinal
*shape* toward signal level but not the energy scale).

---

You are working on my AMS-02 ECAL research repository:
https://github.com/Chagatai404/ams-ecal-research
Your task is to implement the proton model — proton phenomenology using the results of the completed Geant4 proton pilot.
Do not reopen the EM event generator physics.
Do not repeat the broad hadronic literature search.
Do not expand this into the full RQ-001 Geant4 program.
This task is specifically:
Convert the measured thin-ECAL proton behavior from the Geant4 pilot into the simplest transparent, fast, configurable proton FastMC model that reproduces the detector-level distributions relevant to this project.

### 1. Read first

Before editing anything, inspect at minimum: README.md, RESEARCH_PROTOCOL.md, AGENTS.md,
research/STATE.md, research/PUBLICATION_ROADMAP.md,
research/questions/RQ-001_multiscale_shower_information.md, results/geant4_proton_pilot/,
notebooks/08_geant4_proton_pilot.ipynb. Then inspect: src/ams_ecal/, tests/, configs/, existing
FastMC code, the EM event generator implementation, geometry/readout code, ECALEvent, provenance/config
machinery. Use the actual repository state as authoritative. Do not assume filenames or APIs
from this prompt if the repo already contains a better existing abstraction.

### 2. Scientific status entering the proton model

EM event generator is complete. The Geant4 proton pilot is complete using Geant4 11.4.1 and the exact
AMS-like geometry. Approximately 32,000 proton events were generated across 10, 20, 50 and
100 GeV with: FTFP_BERT nominal, QBBC comparison, QGSP_BERT alternate high-energy model,
fixed-entry control, extended calorimeter control. The following pilot results now define the
scientific basis of proton model.

### 3. Interaction probability

The nominal AMS ECAL depth is 166.5 mm. The Geant4 proton samples give approximately no
inelastic interaction: 0.51–0.53, compared with exp(-0.6) ≈ 0.549. The fitted effective proton
interaction length is approximately lambda_eff ≈ 249–262 mm, with lambda_eff ≈ 255 mm as a
useful nominal calibration. The effective proton depth of the ECAL is therefore approximately
0.64–0.67 lambda rather than exactly 0.6 for these Geant4 proton samples. No significant energy
dependence of the interaction-depth distribution was resolved across the pilot energy range.

For the proton model v1: model the first inelastic interaction distance with one exponential
interaction-length parameter. Do not separately sample "interaction yes/no" and then
"interaction depth" unless the code architecture genuinely requires that representation.
Prefer one latent draw:

```
S_int ~ Exponential(lambda_eff)
if S_int >= traversed ECAL material depth:
    no inelastic interaction
else:
    first interaction occurs at S_int
```

This automatically makes interaction probability and interaction depth self-consistent.

### 4. Important provenance of lambda_eff

Do not present ~255 mm as a universal AMS proton interaction length. It is an effective proton
interaction length calibrated from the Geant4 11.4.1 pilot in the current AMS-like
material/geometry model over the tested energy range. Keep it configurable. Its provenance
should include at least: Geant4 version, physics list, geometry/config hash, energy range,
particle, incidence assumptions.

### 5. Current corrected interpretation of the depth-dominance hypothesis

The original proposition "first-interaction depth dominates proton event-to-event variation" is
only partly supported. The pilot found approximately: longitudinal centre S_D ~ 0.72;
hit/cell multiplicity S_D ~ 0.36–0.57; visible/deposited energy S_D ~ 0.24–0.50; lateral
width S_D ~ 0. Therefore the accepted interpretation is: first-interaction depth is the
dominant observed driver of longitudinal position, an important but incomplete driver of
visible energy and multiplicity, and not a meaningful direct driver of lateral width in the
tested normal-incidence thin-ECAL regime. The proton model factorization should reflect this.

### 6. Truncation result

The same-event thin-prefix vs extended-calorimeter experiment showed that correlations from
deep calorimeters do not transfer directly into the AMS ECAL. Interaction depth vs
deposited/visible energy was approximately -0.44 to -0.71 in the AMS-like thin ECAL and
+0.18 to +0.28 in the extended calorimeter; the correlation can reverse sign under truncation.
Consequences: do not import deep-calorimeter covariance matrices, whole-shower longitudinal
parameterizations, or generic hadronic correlation structures into proton model. Use the thin-geometry
pilot directly.

### 7. Detector-level mixture result

Truth non-interacting fraction ~51–53%; detector-level MIP-like fraction almost identical to
truth non-interacting; interacting events that remain MIP-like only ~1.4–3.6%, mostly very late
interactions, roughly within the final ~2 cm. The visible-energy distribution is broad,
continuous, with a heavy low-energy tail, not lognormal. There is no convincing evidence for a
separate low-visible interacting mode. Therefore: do not build an explicit three-class latent
model (non-interacting / low-visible interacting / showering). The low-visible region should
emerge continuously from interaction depth + remaining detector depth + stochastic interacting
response.

### 8. Models rejected by the Geant4 pilot

Do not implement as the proton model baseline: fixed visible-energy fraction; flat per-layer proton
shower; generic whole-hadronic-shower profile truncated at the ECAL edge; transferred
deep-calorimeter correlation matrix; independent low-visible interaction class; lognormal
visible-energy distribution; interaction-depth-controlled lateral width.

### 9. Important pre-implementation synthesis check

Before writing the final generator, use the existing Geant4 events to perform one small
dependency analysis. Do not generate another large production sample for this. The purpose is
to determine which conditional dependencies need to remain explicit. At minimum examine
relationships among: remaining depth R, visible/deposited energy E_vis, longitudinal centre,
longitudinal RMS, hit/cell multiplicity, lateral width, where R = L - S_int for interacting
events. Specifically determine whether, after conditioning on R and/or E_vis, there remains
substantial dependence between: lateral width and E_vis; hit multiplicity and longitudinal
morphology; longitudinal shape and E_vis. Use simple interpretable diagnostics (binned
conditional correlations, partial correlations where justified, conditional
medians/quantiles, simple 2D plots). Do not fit complex ML dependency models. The purpose is
merely to decide which arrows belong in the proton model generative graph. Record this result before
implementing the final factorization.

### 10. Proposed high-level proton model generative graph

```
primary energy E, track state / geometry
  -> interaction distance S_int
       -> S_int >= detector path -> crossing-track event
  -> remaining depth R
  -> visible interacting response E_vis
       -> longitudinal morphology
       -> hit/cell activity
       -> lateral development
```

Do not treat this diagram as immutable. Use the dependency analysis from §9 to finalize the
minimal conditioning structure.

### 11. Crossing-track / non-interacting component

Do NOT implement the pre-interaction component as "each layer receives an IID random MIP
deposit" without accounting for geometry. The fixed-entry Geant4 control showed the signal of a
crossing proton depends on where the track lies relative to the fibre pattern. Interacting
shower events were much less affected, but crossing tracks were sensitive to entry/fibre phase.
Therefore the baseline should conceptually be: known TrackState → exact geometry / fibre
crossing → expected sensitive path / cell placement → stochastic crossing-track deposition. The
track determines which cells are crossed, sensitive path length, fibre/view geometry. Then
stochastic fluctuations are applied around that geometrically defined response. Reuse the
existing geometry/readout machinery wherever possible. Do not introduce a second approximate
detector mapping.

### 12. Pre-interaction track before a shower

For an interacting event (S_int < L), generate the crossing-track component only up to the
interaction point. The event should contain pre-interaction MIP-like track + post-interaction
hadronic activity. Do not generate a full detector-crossing MIP track and then add a shower on
top (double-counting). Keep energy components conceptually clear.

### 13. Visible interacting energy

For interacting events R = detector path remaining after first interaction. Visible/deposited
energy depends substantially on primary energy E and remaining depth R and has broad stochastic
spread. Model conceptually E_vis ~ P(E_vis | E, R). Do not use a fixed fraction of E and do not
assume a named probability family unless the pilot data support it.

### 14. Tabulated versus fitted distributions — use a hybrid

Fit the physics that has a clear justified analytic form; use empirical conditional
distributions for the rest. Fit analytically: interaction distance exponential(lambda_eff).
Potentially fit simple smooth trends (conditional median / scale vs E and R) only if a simple
fit describes the Geant4 distributions well. For broad stochastic distributions such as visible
energy, prefer initially empirical conditional CDFs or conditional quantile tables, e.g.
u ~ Uniform(0,1), E_vis = Q_vis(u | E, R). This avoids imposing an unsupported lognormal /
gamma / Gaussian / mixture on a distribution already shown to be broad and non-lognormal.

### 15. Interpolation

Anchor energies 10, 20, 50, 100 GeV. Do not simply snap every event to the nearest energy table
if interpolation is straightforward. Likely: interpolation in log(E) for energy-dependent
quantities, and remaining-depth bins or simple interpolation in R. Keep interpolation
deterministic and testable. Do not extrapolate beyond the validated range unless explicitly
configured and visibly labelled. For proton model v1, restricting the validated domain to 10–100 GeV is
acceptable. Outside-domain use should raise, warn clearly, or require explicit extrapolation
configuration, following existing project conventions.

### 16. Longitudinal morphology

Strong dependence on first-interaction/remaining depth. Model the shower portion relative to
the interaction point rather than detector entry when appropriate: longitudinal_shape ~
P(shape | E, R, E_vis) depending on the dependency analysis. The profile should begin at the
interaction location, respect finite detector boundaries, preserve back leakage, and sum
consistently to the generated interacting visible energy. Do not use a whole-shower profile
whose natural maximum lies beyond the detector and merely truncate it.

### 17. Longitudinal representation

Choose the simplest representation that reproduces the Geant4 thin-ECAL layer distributions:
empirical normalized layer-fraction templates conditioned on R/E; low-dimensional basis of
normalized profiles; or simple shifted profile with stochastic width. Priority:
interpretability, energy conservation, few parameters, reproducibility, conditional agreement
with Geant4. If empirical conditional layer-fraction distributions work well enough, use them
as the v1 reference; a future simplified analytic fit can replace them if validation shows no
meaningful loss.

### 18. Hit/cell multiplicity

Partly controlled by interaction depth, strongly associated with visible energy. Do not draw it
independently unless the dependency analysis supports independence. Likely N_hit ~
P(N_hit | E_vis, R, ...) or allow multiplicity to emerge naturally from the spatial
energy-deposition model. Avoid a separate random variable if already determined by E_vis,
longitudinal morphology, lateral deposition, threshold definition. Prefer emergent behaviour.

### 19. Lateral width

Depth-associated variance ~ 0. Do not directly condition width on first-interaction depth.
Zero dependence on depth does not imply full independence: check whether width depends on E,
E_vis, hit multiplicity, longitudinal morphology; choose the minimal conditioning. A plausible
form is w_lat ~ P(w_lat | E, E_vis), not automatically accepted — use the §9 analysis.

### 20. Spatial deposition

The generated event must populate the canonical 18 × 72 representation, reusing TrackState,
geometry, readout orientation, cell mapping. The generator must not become only a list of
summary observables; it must generate physically plausible event-level cell energies suitable
for classical ML, multiscale analysis, later QML, FastMC-vs-Geant4 comparison, while remaining
phenomenological rather than microscopic.

### 21. Energy accounting

Preserve the EM event generator's explicit energy accounting. Distinguish conceptually: sensitive/readout
deposit; passive-material deposit if represented internally; longitudinal leakage; lateral
leakage; energy carried beyond the detector. Do not pretend FastMC can reconstruct the
microscopic nuclear invisible-energy decomposition unless explicitly modeled. Do not introduce
a constant "42% invisible-energy" factor.

### 22. Physics-list model dependence

Qualitative structure robust across FTFP_BERT, QBBC, QGSP_BERT, but the absolute visible-energy
scale showed model dependence (QGSP_BERT ~13–28% lower at ~20–100 GeV). This is epistemic
model uncertainty, not event-level variance. Do not average physics lists, randomly choose a
list per event, or inflate per-event variance. Implement separate calibration variants: nominal
FTFP_BERT-derived; systematic QGSP_BERT-derived; QBBC-derived comparison where useful (e.g.
`proton_model_calibration = "ftfp_bert" | "qgsp_bert"`). QBBC need not become a production
calibration if it adds no distinct high-energy structure, but preserve its pilot result.

### 23. Why QGSP_BERT remains

QGSP_BERT was added because QBBC and FTFP_BERT use closely related high-energy proton treatment
in the pilot regime. It provided a genuinely different high-energy model and exposed the
visible-energy-scale systematic. Keep the result.

### 24. Do not treat Geant4 as truth

Documentation and code should say "Geant4-calibrated" / "Geant4-derived phenomenology", not
"true proton distribution". Later AMS/test-beam evidence may alter the calibration.

### 25. Small remaining checks before freeze

25.1 Production-cut sensitivity: compact sample with a reasonable alternate production cut;
focus on hit/cell multiplicity, lateral width, small-deposit structure. If negligible relative
to stochastic/model uncertainty, record and stop. Do not optimize cuts.
25.2 Material-composition systematic: compact comparison for the 98 Pb + 1 Al foil
approximation and any relevant material-composition simplification; check lambda_eff,
visible-energy scale, key morphology. If negligible, keep as documented approximation; if
material, expose as calibration uncertainty or correct the geometry if the correct description
is established.
25.3 No broad incidence-angle campaign yet: architect the API so the model can later condition
on track angle and path length.

### 26. Backsplash

Extended geometry: 26–45 MeV returning into the ECAL region vs 11–13 MeV without. Do not use
extended-geometry ECAL lateral width as a proton model calibration source; use the AMS-only simulation
for width. Backsplash is not a proton model blocker; record as a caveat.

### 27. Event-level algorithm target

```
INPUT: primary energy E, TrackState, ECAL geometry, RNG, proton calibration config
COMPUTE: path through ECAL L
DRAW: S_int ~ Exponential(lambda_eff)
IF S_int >= L:
    crossing-track response using exact track/fibre geometry + stochastic per-crossing response
ELSE:
    crossing-track deposits only before S_int
    R = L - S_int
    E_vis ~ P(E_vis | E, R)
    longitudinal morphology conditioned on the validated minimal dependencies
    lateral morphology using validated minimal dependencies, no unjustified depth dependence
    project/distribute into canonical cells
    preserve finite detector leakage/accounting
RETURN: ECALEvent with provenance
```

The implementation should reuse existing abstractions and may not literally mirror this.

### 28. Correlation preservation

Do not independently sample every observable. Reproduce important Geant4 correlations; at
minimum validate depth↔visible energy, depth↔longitudinal centre, depth↔longitudinal RMS,
visible energy↔hit multiplicity, visible energy↔lateral width if present. Only preserve
correlations that are material. No generic covariance engine.

### 29. Empirical calibration artifacts

Do not hard-code large tables into Python. Store calibration data in an explicit versioned
form (configs/, data/calibration/, results/.../calibration/ per conventions) with provenance:
source Geant4 run, physics list, energy anchors, geometry hash, sample counts, generation date,
schema version. The runtime model loads a compact finalized artifact, not raw Geant4 events.

### 30. Calibration construction must be reproducible

Add a script/module that rebuilds the proton model calibration artifact from the accepted pilot outputs
(pilot events → calibration builder → versioned compact proton model calibration). Deterministic where
possible.

### 31. Train/calibration versus validation split

Deterministic split of the existing pilot data into calibration and validation subsets (event
IDs or another reproducible method). The validation subset stays untouched while choosing /
interpolating the model where practical. Not a full ML train/val/test system.

### 32. Proton model validation

Compare FastMC protons with held-out Geant4 AMS projections at matched energy, normal
incidence, entry conditions. At minimum: interaction (no-interaction fraction, depth
distribution); crossing tracks (total signal, layer distribution, entry/fibre-phase
dependence); energy (distribution, median, quantiles, tail); longitudinal (layer fractions,
centre, RMS, observed maximum, active layers); lateral (width, core concentration, cell
distribution); activity (hit multiplicity, maximum cell, sparsity/concentration); correlations
(depth↔energy, depth↔centre, depth↔RMS, energy↔multiplicity, relevant lateral). Do not
validate only means.

### 33. Statistical comparison

Quantile tables, ECDF comparisons, Wasserstein distance, KS where appropriate, correlation
difference, conditional distribution plots. No single arbitrary "proton model realism score"; aspects
pass or fail separately.

### 34. Acceptance philosophy

Acceptable when fast, reproducible, transparent, physically defensible, free of obvious
generator artifacts, reasonably faithful to the important thin-ECAL proton distributions, and
preserving structures relevant to downstream controlled studies. Do not endlessly tune
secondary statistics. Apply Research OS triage A (blocker) / B (testable uncertainty) /
C (reversible choice) / D (future refinement).

### 35. Important failure criterion

If the simplest conditional/tabulated model cannot reproduce the held-out joint distributions
without a large complicated latent architecture: STOP. Report what fails, why, which
observable/correlation cannot be represented, what minimal additional variable would fix it.
Do not silently grow proton model into a large generative ML model, normalizing flow, diffusion model or
high-dimensional copula.

### 36. Tests

Reproducibility (same seed → same event; different seed → variation). Interaction (S_int
exponential; no-interaction ⇔ S_int >= L; interaction point inside the detector). Geometry
(track deposits in valid cells; pre-interaction deposits stop appropriately; entry/fibre phase
affects crossing response as expected). Energy (nonnegative, finite, explicit accounting, no
accidental renormalization). Empirical distributions (sampling within calibration support;
monotonic quantile/CDF interpolation; boundary energies). Configuration (nominal vs systematic
selectable; provenance preserved). Event model (valid ECALEvent; 18×72). Ensemble sanity
within expected tolerances at representative energies; avoid brittle exact-moment tests.

### 37. Performance

Calibration loads once; sampling cheap; no per-event optimization or large data loading.
Benchmark enough for large downstream datasets. Do not micro-optimize prematurely.

### 38. Provenance

Every proton event identifies: FastMC version, the proton model model version, calibration version,
nominal/systematic physics-list calibration, geometry/config hash, seed, primary energy,
TrackState. If EventProvenance cannot represent the calibration version cleanly, extend it
minimally.

### 39. Documentation wording

Use: Geant4-calibrated proton phenomenology; effective interaction length; empirical
conditional distribution; physics-list systematic; validated over 10–100 GeV at normal
incidence. Avoid: true proton shower model; exact AMS proton response; physical interaction
length of AMS.

### 40. Teaching notebook

After code and validation: teach (1) why a thin hadronic calorimeter needs a different model
than a contained shower; (2) exponential interaction depth; (3) why one exponential draw
determines both status and position; (4) truth non-interacting vs detector MIP-like;
(5) remaining depth R; (6) the depth-dominance hypothesis variance-decomposition result; (7) why longitudinal and
lateral variables need different conditioning; (8) why empirical conditional CDFs instead of
forced analytic laws; (9) why physics-list differences are epistemic systematics, not event
noise; (10) how the generator maps these ideas into 18×72 events; (11) where the model is
validated and where uncertain. The notebook calls tested production code.

### 41. Research-state update

When proton model passes validation, update research/STATE.md. Accepted: exponential effective
interaction depth; geometry-aware crossing-track response; continuous interacting
visible-energy tail; remaining-depth-conditioned longitudinal model; no direct
interaction-depth control of lateral width unless new analysis supports it; FTFP_BERT nominal
calibration; QGSP_BERT energy-scale systematic. Rejected: fixed visible fraction; flat shower
profile; whole-shower truncation model; transferred deep-calorimeter correlation matrix;
explicit separate low-visible interacting class; lognormal visible-energy law. Still open:
angular dependence; real AMS backsplash/environment; production-cut sensitivity if not fully
resolved; material-composition systematic; exact Pb/Al construction approximation; external
validation against AMS/test-beam data. Keep Geant4-model dependence explicit.

### 42. Scope boundary

Not in this task: the detector response; full RQ-001 multifractal analysis; e-vs-p ML
benchmarks; QML; full angular proton calibration; Geant4 production redesign; learned
generative shower model.

### 43. Implementation slices

- Dependency analysis (step 0) — reconcile and dependency analysis: inspect pilot artifacts and FastMC
  architecture; compute remaining conditional dependencies; run the small production-cut /
  material checks; write the final proton model dependency graph. Do not implement the generator until
  this is clear.
- Interaction draw (step 1) — calibration builder: pilot events → deterministic builder → compact versioned
  artifact; nominal FTFP_BERT first, then QGSP_BERT systematic; tests.
- Crossing branch (step 2) — interaction + crossing-track component: S_int ~ Exp(lambda_eff);
  geometry-aware crossing-track deposits; non-interacting branch; pre-interaction track branch.
  Validate before adding shower deposition.
- Step 3 — interacting visible energy: P(E_vis | E, R) by empirical conditional
  sampling/interpolation; validate on held-out events.
- Step 4 — longitudinal morphology: simplest accepted conditional model; interaction-origin
  consistency, finite-depth behaviour, energy accounting; validate.
- Step 5 — lateral/activity model: only supported dependencies; width not driven by depth;
  multiplicity emergent where possible; validate.
- Final validation (step 6) — joint validation: matched FastMC vs held-out Geant4; marginals, conditionals,
  correlations, tails; resolve only material discrepancies.
- Alternative physics-list calibration (step 7) — systematic calibration: validate the QGSP_BERT path as an explicit epistemic
  scenario.
- Step 8 — notebook and documentation; reproduction commands and validation results.

### 44. Final report before declaring the proton model complete

Model (exact factorization); calibration (tabulated vs fitted); interaction (final lambda_eff,
uncertainty/provenance); crossing tracks (geometry dependence); visible energy
(sampling/interpolation); longitudinal (how D/R enters); lateral (dependencies and why);
correlations reproduced; systematics (FTFP_BERT vs QGSP_BERT); validation (held-out Geant4);
domain (energy range, angle range, geometry assumptions); performance; remaining material
limitations; files/commits; exact test results.

### 45. Success criteria

1. one exponential draw determines interaction status and depth; 2. crossing tracks use
detector geometry, not IID layer noise; 3. interacting visible energy follows the measured broad
conditional distribution; 4. low-visible events emerge without a forced class; 5. longitudinal
morphology depends strongly on interaction/remaining depth; 6. lateral width not driven by
depth; 7. important correlations reproduced; 8. empirical distributions built reproducibly;
9. calibration and validation separated; 10. FTFP_BERT nominal; 11. QGSP_BERT as a model
systematic, not event variance; 12. valid canonical 18×72 ECALEvents; 13. explicit
energy/leakage accounting; 14. fast enough for dataset production; 15. documented as
Geant4-calibrated phenomenology; 16. simple enough to replace/refine later.

Central design rule: use analytic physics where the pilot established it, and empirical
conditional distributions where the physics remains phenomenological. Do not invent a prettier
probability law than the evidence supports.

Begin with the dependency analysis (step 0).

---

## Appendix B — Validated FastMC dataset and the start of the detector response — PLAN

_Retired plan, carried over verbatim on 2026-10-10 (original file `research/plans/2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md`, last version in git at `e904286`; retrieve with `git show e904286:research/plans/2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md`). Content: tracks A-E, validated FastMC dataset and the detector response, Q1-Q7, progress record 5c. Headings are demoted one level; nothing else is changed except that references to retired files point to their appendix._


_Drafted 2026-10-02 on branch `fastmc-validated-dataset` (from `geant4-proton-pilot` at `ca448eb`), at the researcher's request: "plan and implement the remaining part of the stochastic event generation and start detector response. We need a validated FastMC dataset asap."_

Status of this document: **a plan.** Section 7 separates what the researcher's message of 2026-10-02 authorises from what still needs a decision. Nothing in sections 3-6 is an accepted conclusion.

Reads with: `research/DECISIONS.md` (DEC-003 to DEC-013), `research/STATE.md`, the proton dependency analysis (§3 factorization, §8 contract, §11 crossing result) and the weekend plan of the same date.

### 1. What "validated FastMC dataset" has to mean

A dataset is *validated* when every row below has been run **once, on Geant4 events no model choice ever saw**, with the verdict reported row by row (no single score, `RESEARCH_PROTOCOL.md` §5):

| layer | what is checked | status today |
|---|---|---|
| EM generator vs Geant4 electrons | contract rows (energy, longitudinal, lateral, activity, correlations) | **never done.** The generator is parameterised from literature and unit-tested; it has never been compared with transport. A 4000-electron Geant4 sample now exists (`data/geant4_electron_sample/`, unread). |
| Proton crossing branch | the §8 contract | run once, **failed** on event total, hits, max-cell, containment (§11 of the dependency analysis) |
| Proton interacting branch | the §8 contract + added checks | **not implemented** |
| Dataset level | energy/geometry matching between classes, seed isolation, split integrity, containment/leakage, no accidental label leakage | planned in `STATE.md`, not built |
| Detector response | per-effect evidence, then validation | **blocked** on the evidence pass (section 5) |

### 2. The shortest honest path, and why it does not wait for the detector response

DEC-006 fixed `deposition` as the common electron/proton representation. Both generators can emit it **without any detector response**. So the first validated dataset is a **perfect-event (true-deposition) dataset**, labelled as such; the detector response then adds a second, detector-level version. This is a recommendation for the researcher to accept or reject (decision Q1).

```text
Track A  proton generator         A1 crossing repair (burst + spill)  ->  A2 interacting model  ->  A3 batch path  ->  freeze
Track B  validation data          B1 register added checks (needs researcher's numbers)  ->  B2 sealed Geant4 sets (background)  ->  B3 one opening
Track C  electron side            C1 EM generator contract (pre-registered)  ->  C2 one opening on sealed electrons
Track D  detector response        D1 evidence pass (running)  ->  D2 sampling response R-A (Geant4-calibrated)  ->  D3 instrumental effects R-B (evidence-gated)
Track E  dataset assembly         E1 energy/angle sampling, seed policy, split, dataset-level checks  ->  v0 perfect-event dataset  ->  v1 detector-level
```

A, B and C run in parallel. The long wall-clock item is **A2** (the interacting model). Compute is not the constraint: the pilot cost 710 s for 4000 protons at 100 GeV on 12 workers, and the 4000-electron batch took about 51 minutes.

### 3. Track A — proton generator (steps 4-8 of the accepted sequence)

#### A1. Crossing repair: burst latent and lateral spill (DEC-008, DEC-009, EXP-005)

Design evidence, from **calibration crossing events only** (`event_index % 4 != 3`, about 1550-1590 per energy; the held-out events were not read):

1. **Bursts are cascades, not multipliers.** At 50-100 GeV the aligned deposition excess rises over one to three layers after onset and decays slowly; only 9-12% of a burst's in-ECAL energy is in its onset layer. At 10 GeV bursts are mostly one layer (about 89% in the onset layer). Late-onset bursts carry about half the in-ECAL energy of early-onset bursts (truncation by the back). Onset is roughly uniform over depth.
2. **Fibre and deposition bursts are the same bursts.** Spearman of the two excess sums within the largest 10% of deposition bursts: 0.51 / 0.76 / 0.84 / 0.88 at 10 / 20 / 50 / 100 GeV; fibre excess is about 6% of deposition excess (median 0.056-0.062). The amendment of DEC-006 (independent layer draws per representation) was based on the per-layer correlation of ordinary layers, 0.15-0.24; for bursts a shared latent is supported. **Design: one burst latent per seed shared by both representations**, with a Gaussian copula on the two amplitudes at the measured correlation. This is a refinement of "deposition and readout fluctuations may differ", not a reversal.
3. **Spill is a function of the layer's own excess, not an independent process.** P(layer has fibre energy outside the crossed cells > 0.1 MeV) = 2% for ordinary layers, 16-20% for 1.5-3x the median, about 50% for 3-8x, about 90% above 8x. In events without a deposition burst the layer-count clustering is close to binomial at 50-100 GeV (variance 0.96 vs 0.58). Spill is mostly one cell per layer (94.4% / 92% of layers have none; 5-6.5% have one), the energy of a spill cell is nearly independent of its distance (median 0.25-0.35 MeV out to far offsets) while the **rate** falls with distance. **Design: spill is a conditional split of the layer's total**, given its ratio to the chord-conditioned median, so layer and event totals are unchanged.
4. **The bulk table must lose the bursts.** The existing crossing table's tail already contains the burst layers; adding a burst on top would double count. The bulk table is rebuilt from burst-free layers and the burst is added on top.
5. *Interpretation, not a result:* these look like knock-on-electron cascades in the lead (about 9% of events with a > 100 MeV knock-on is the order of magnitude kinematics gives). It is recorded as a consistency check to run, not as the model. The model stays empirical.

Deliverables: calibration artifact **schema 2** (new directory; `ftfp_bert_v1` is kept untouched as the record of the first crossing validation), builder, generator, tests, and an **in-sample** diagnostic against the calibration events (explicitly not a validation).

Integrity: the repaired crossing branch is **not** re-validated on the 2026-09-29 held-out events a second time. It is validated once, on the sealed set (B2).

#### A2. Interacting model (DEC-005, EXP-006)

Amplitude x universal profile + albedo + lateral-scale latent, with the three amendments (back-edge factor; correlated residual in log-energy space with its normality tested first; shared albedo latent). It reuses the A1 machinery: the *spill* kernel is the same kernel (dependency analysis §3, "the same kernel the interacting events need"), and the *component abstraction* (onset layer, amplitude, profile in offset, lateral kernel) is shared between a knock-on burst and a hadronic interaction. Risks named in the dependency analysis and kept: cell-level sparsity from a smooth kernel (fall-back: explicit `N_hit` table), shape residual, albedo. The §35 stop rule stands.

**First A2 evidence (calibration events only, 2026-10-02): the residual is NOT log-normal.** DEC-005(b) required the distribution of the log-energy residual to be tested before it is called log-normal. For interacting events with the eight layers after the interaction layer inside the ECAL (n = 884 / 910 / 931 / 908 at 10 / 20 / 50 / 100 GeV; fibre energy; `ln e_k - ln A3 - mean profile`): skewness -1.77 / -0.38 / -1.49 / -1.51, excess kurtosis +26.9 / +4.5 / +12.5 / +9.7, Shapiro-Wilk p < 1e-38 at every energy, standardised 1% quantile -2.7 / -2.8 / -3.4 / -3.7 against -2.33 for a normal, while the quartiles sit at about +-0.45 (normal +-0.67): peaked with a heavy lower tail (occasional strongly suppressed layers). Adjacent-layer residual correlation +0.35 / +0.44 / +0.47 / +0.53 (the analysis had +0.3 to +0.4). Zero-energy layers are 0.4% at 10 GeV and fewer above. Consequence for the design: the family is an **empirical marginal quantile table** of the residual (energy-dependent scale) joined by a correlated Gaussian copula, the same hybrid as elsewhere; a log-normal residual would be wrong. The back-edge factor and the shared albedo latent are still to be measured. Script: not yet in the repository (scratch); it becomes part of the A2 builder.

#### A3. Batch path (step 7)

11-14 ms/event is too slow (10^6 events about 3 h). The cost is per-event Python (`cross` twice, tuple-of-tuples event with 1296-value validation). The A1 sampling code is written vectorised from the start; the per-event `ECALEvent` API stays for provenance, the dataset path returns arrays plus a provenance table.

### 4. Track B — validation data and registration

**B1. Register the added checks before any sealed event exists (DEC-007).** Draft: `research/plans/2026-10-02_added_validation_checks_preregistration.md`. Numbers are **proposed**; the researcher confirms them (Q2). Generation is held until then, as decided.

**B2. Sealed sets (approved item 6 of the weekend plan), proposed content:**

| set | species | energies (GeV) | events | purpose |
|---|---|---|---|---|
| proton sealed | p, FTFP_BERT, AMS-only, normal incidence, entry spot as the pilot | 10, 20, 50, 100 (3000 each) and **14, 30, 70 (3000 each)** | 21,000 | anchors plus **off-anchor** energies: the model interpolates in ln E, and a validation only at anchors would not test that |
| electron sealed | e-, same geometry and cut | 10, 20, 50, 100 and 14, 30, 70 (1000 each) | 7,000 | the EM contract; the existing 4000-electron sample stays the exploration/calibration set |

Seeds disjoint from every earlier batch; generated from a clean commit; per-file SHA-256 recorded; a code guard in the reader that refuses to open a sealed file without an explicit `--final` flag and appends to an **opening ledger** that is committed. Estimated wall time from the measured costs: about 1-1.5 h per species on 12 workers. Off-anchor energies are an addition to what you approved (item 6): decision Q3.

### 5. Tracks C and D — electrons and the detector response

**C. EM generator contract.** The EM generator is smooth by design (DEC-001: deterministic lateral profile, no cell-level fluctuation). The proton generator will not be smooth (bursts, spill, quanta, sparsity). **A classifier trained on the dataset could separate electrons from protons by that asymmetry alone.** That is the "accidental label leakage" the dataset check exists for, and the classifier two-sample test would flag it. DEC-001 excluded a stochastic lateral model from the EM generator, so this is not changed here: it is flagged as the most likely dataset-level finding, with the decision needed once measured against sealed electrons (Q4). Until then the dataset README must say which observables are not comparable between classes.

**D1. Evidence pass (started 2026-10-02).** A fresh-context discovery scout ran with the neutral question and no model details. **Its result is thin and unverified**: abstract/snippet depth only, it did not find the two AMS papers already verified in the vault (Zhang 2016, Li 2013), and its per-effect "corrected before the fit" table is inference, not evidence. It is recorded as a discovery pass that did **not** answer the sub-question. Next for D1: a source-verification pass on the primary AMS ECAL references (full text), then the adversarial pass, then the per-effect inversion table. Nothing from the scout is promoted.

**D2. A split of the detector response that the evidence map did not make (proposal, Q5).**
- **R-A, sampling response** (true deposition -> fibre energy: depth-dependent fibre fraction, sampling fluctuation). Geant4 already provides both `deposit_grid_mev` and `readout_grid_mev` for protons, and the electron sample provides them for electrons. This is exactly the "electron fibre-energy scale from a small Geant4 electron calibration" that DEC-006 said was missing, and it can be **calibrated, then validated on sealed electrons**. It is built against the `deposition` regime, as the 2026-09-28 amendment requires, so it does not re-apply the sampling shift. It is also the experiment that converts the double-counting inference (open question 5 of the evidence map) into a measurement: fit gamma profiles to both deposition and readout in Geant4 electrons and compare.
- **R-B, instrumental effects** (gain and PMT position effect, noise, thresholds, ADC range and saturation, dead cells, cross-talk). No Geant4 equivalent; AMS-specific; stays blocked on the evidence pass and its per-effect inversion table.

### 5b. Research OS v0.7.0 (checked 2026-10-02, after the researcher's update)

v0.7.0 was already installed in `~/.claude` (all 15 skills and the 10 agent definitions match the repository HEAD apart from line endings), so no reinstall was needed. What changes for this work:

- **Literature sessions** are complete only when every selected useful paper has an outcome: saved PDF and metadata, an existing-library match, an unavailable PDF with its citation, or a reviewed rejection (`skills/research-session`). **Applied retroactively to D1:** the evidence pass above is **incomplete** under this rule. The scout's candidate papers have **no recorded outcome**; none was reviewed, none was ingested (downloading a PDF needs explicit permission), and several are snippet-level only. They stay "candidate, unreviewed" until a verification pass assigns each an outcome.
- **The tutor** is concept-first with separate conceptual, transfer and application evidence (`skills/tutor`). It applies to the learn-back after the interacting model (section 6), not to the code work.
- Visual selection is intent-based (`skills/visualize`); no new visual is needed for this slice.

### 5c. Progress, 2026-10-02 (branch `fastmc-validated-dataset`, uncommitted)

**A1, crossing repair: implemented and tested; in-sample fit checked; NOT validated.**

| item | state |
|---|---|
| `src/ams_ecal/proton_model/proton_structure.py` (new) | burst latent with onset, amplitude, **downstream extent** and fibre share; lateral spill as a conditional split; bulk coupling; builder; vectorised samplers |
| `src/ams_ecal/proton_model/proton_calibration.py` | artifact schema 2 (loads schema 1 too); `build_calibration(structure=True)`; bulk table rebuilt from burst-free events |
| `src/ams_ecal/detector/crossing.py` | `place_layer_energies` (exact energy conservation per layer) |
| `src/ams_ecal/proton_model/proton.py` | structured crossing path, fixed draw order, burst latent in the provenance; schema-1 path kept |
| artifact | `data/calibration/proton_model/ftfp_bert_v2` (development build from an unclean tree, `tracked_changes: true`; the **frozen** artifact is rebuilt from a clean commit before the sealed set is generated). `ftfp_bert_v1` is untouched. |
| tests | 529 passing (484 before), ruff clean; three mutation checks confirmed the new tests catch a broken bulk selection, a spill that ignores the layer ratio, and an untruncated burst |
| in-sample check | `uv run python -m ams_ecal.proton_model.proton_structure_check` (calibration events only) -> `results/proton_model/crossing_structure_in_sample.json` |

**Beyond the two accepted repairs, for the researcher (Q6 and Q7):**

1. *Shared burst latent* across readout and deposition (Q6), supported by the calibration correlation 0.5-0.9 between the two excess sums.
2. *Downstream extent* as a latent. This is the fourth element of the structure you specified (probability, onset, energy, extent). It was needed because a fixed-extent burst under-dispersed the event total.
3. **Bulk coupling (Q7), NOT one of the accepted repairs.** In burst-free events the two readout layers of one superlayer correlate (fibre lag-1 within a superlayer 0.13-0.16, across superlayers about 0.02; deposition 0.15-0.19 and 0.11-0.12). Independent bulk draws miss it. It is a three-parameter Gaussian copula on the bulk layer draws (within-superlayer, neighbouring-superlayer, lag-2), calibrated on burst-free events. It is the smallest change that reproduces a measured layer-correlation row, but it is a third element and DEC-005's "no generic covariance engine" spirit asks you to decide it. Proposal: keep it; it is one row of the added checks.
4. The bulk per-layer table is now built from burst-free **events**, not from all layers: pre-onset layers of burst events are measurably heavier than the layers of burst-free events, and pooling them made burst-free events 2% too energetic.

**Findings from the calibration events that are not yet explained** (recorded, not acted on): the extreme burst tail rests on about three events per energy. In `results/proton_model/crossing_structure_in_sample.json` (4000 generated events against about 1550 calibration events per energy; KS floor about 0.03-0.04) the readout rows are at the floor for the event total (KS 0.016-0.031) and close for hit cells (0.047-0.058), maximum cell (0.035-0.055) and containment (0.060-0.077, inflated by a spike at 1); lag-1 layer correlation matches within 0.03. What is still off: the event-total variance ratio is 19-24% low in deposition at 50-100 GeV (4.42 vs 5.80, 5.45 vs 6.69) and 6-10% low in readout, and lag-5 correlation is 10-25% low in deposition at 100 GeV (0.10 vs 0.13). Generation costs 10-18 ms per event, unchanged from before (Python-bound; the batch path is A3).

### 6. Order of work in this session and the gates

1. Done: branch; baseline suite (484 passed, 3m21s); evidence pass launched (thin, section 5); calibration-only structure study; **A1 implemented and tested** (section 5c).
2. Next in code: A2, the interacting model, then A3, the batch path. Notebooks only after the model is final (DEC, 2026-10-02).
3. Blocked on the researcher: Q2 (numbers) -> B2 generation. Q1, Q3-Q7 shape tracks A, C-E.
4. Before the sealed set: commit, rebuild the frozen artifact from a clean tree, register the added checks (B1).

### 7. What is authorised, what is not

| item | authority |
|---|---|
| New branch `fastmc-validated-dataset` | researcher, chat, 2026-10-02 |
| Proton steps 4-8 as in the weekend plan (burst + spill, interacting model, batch path, freeze) | **item 7 of the weekend plan was "approval to start Monday"; the 2026-10-02 message ("plan and implement the remaining part of the stochastic event generation") is treated as that approval.** Please correct me if it was not. |
| Start the detector response | the evidence pass only (D1). No detector-response code. |
| Sealed proton set generated early | approved 2026-10-02 (item 6), **after** B1 is registered |
| Burst latent structural, lateral spill | accepted 2026-09-30 (DEC-008, DEC-009) |
| Shared burst latent across both representations | **new, proposed here**, supported by the calibration evidence in A1; needs the researcher's nod (Q6) because it refines the DEC-006 amendment |
| Off-anchor sealed energies; electron sealed set; using the electron sample for EM contract; R-A | **proposed, not authorised** |

### 8. Decisions needed, in the order they block work

| # | question | recommendation |
|---|---|---|
| Q2 | Confirm or amend the numbers in the added-checks pre-registration | confirm as drafted; this unblocks sealed generation |
| Q1 | First validated dataset = perfect-event (`deposition`) version, detector-level version second | yes |
| Q6 | Shared burst latent across readout and deposition | yes, per the A1 evidence |
| Q3 | Add off-anchor energies (14, 30, 70 GeV) to the sealed proton set; add an electron sealed set | yes |
| Q4 | If the EM generator separates from Geant4 electrons at cell level, extend it with a calibrated cell-level fluctuation (reversing DEC-001's exclusion) | decide after the electron contract, not now |
| Q5 | Split the detector response into R-A (Geant4-calibrated sampling response, may start after the EM contract) and R-B (evidence-gated) | yes |
| Q7 | Keep the three-parameter bulk coupling (section 5c) as a third element of the crossing repair | yes; it is the smallest change that reproduces a measured layer-correlation row |

### 9. Risks

- The interacting model may hit the §35 stop rule (cell-level sparsity from a smooth kernel). Then the report states what failed and the minimal added variable.
- Everything proton-side is Geant4 11.4.1 with one physics list; the dataset inherits that limitation (QGSP_BERT calibration, EXP-007, is still required before any claim about hadronic physics).
- Normal incidence only. A dataset with an angular distribution needs an angular calibration that does not exist; the first dataset is normal-incidence and says so.
- The detector-level version depends on R-B evidence that may be partly non-public (AMS internal notes).

---

## Appendix C — Weekend multiscale check, proton model completion and the road to the detector response — PROPOSED PLAN

_Retired plan, carried over verbatim on 2026-10-10 (original file `research/plans/2026-10-02_weekend_multiscale_check_and_proton_model_completion_plan.md`, last version in git at `e904286`; retrieve with `git show e904286:research/plans/2026-10-02_weekend_multiscale_check_and_proton_model_completion_plan.md`). Content: weekend estimator gate, proton first look, electrons, sealed sets, learning gates, approval table, timing probe. Headings are demoted one level; nothing else is changed except that references to retired files point to their appendix._


_Drafted 2026-10-02 at the researcher's request: finish the proton model, then move to the detector response after learning again, and plan the first multiscale check for the weekend. **Status when drafted: proposed. Nothing in this file was approved or executed.** Approval is per item, in the table at the end, which records each approval with its date._

Role of this document: it is a plan in the sense of `RESEARCH_PROTOCOL.md`. No simulation, build or experiment starts
until the researcher approves the specific item. It follows the proton model sequence already accepted on 2026-09-30
(`research/STATE.md`, steps 4-10) and adds the multiscale check (EXP-001 stage 1, gated by the estimator-validity gate / EXP-008).

### 0. What is known, and what is not

Measured or recorded facts this plan relies on:

- The proton pilot batch at 100 GeV (4000 events, FTFP_BERT, AMS-only geometry) took **710 s on 12 workers**
  (`data/geant4_proton_pilot/baseline/E100GeV/metadata.json`), about 0.18 s per event wall time. The machine has 16 cores.
  The proton data on disk is 1.1 GB across all samples.
- Proton events are cheap because the thin ECAL absorbs only about 150 MeV per proton. **An electron deposits about
  95 GeV per event**, so electron cost per event is expected to be much larger. It has **not been measured**. This is
  the main schedule risk and is why a short timing probe comes first.
- The Geant4 backend takes the primary particle as a configuration string (`particle` in
  `configs/geant4_proton_pilot.yaml`), so an electron sample needs a configuration, not new transport code.
  `e-` thresholds are already recorded per material. This is an inference from the code, to be confirmed by the probe.
- The pilot records fine fibre and 3 x 3 x 4.625 mm voxel deposits, and the readout grid, per event.
- On the 18 x 72 grid, square boxes divide it evenly only at sizes **1, 2, 3, 6, 9, 18**: six scales and no more.
  This is a hard limit on any scaling claim at AMS granularity.
- Researcher's knowledge state for the estimators (Research OS knowledge graph, 2026-10-02): box counting is
  defined and the prediction landed; *what a slope means* has **not** landed (Q10 partial, Q11 wrong);
  marginal versus joint landed same-day only; Jensen is `fragile`. A delayed check is scheduled for 2026-10-04 09:00.

Not known: electron cost, the best handling of the alternating X/Y views, and whether any multiscale estimator is
usable on 18 x 72 at all. The last is exactly what the check is for.

### 1. Order of work, with the reason for each

```text
Fri 2-Oct  learn the math just enough  ->  pre-register  ->  electron timing probe
Sat 3-Oct  estimator code + synthetic controls on 18 x 72  (estimator-validity gate)
Sun 4-Oct  first look at existing proton Geant4 data, exploratory;  electron production if affordable
Mon 5-Oct onward   the proton model steps 4-8, then sealed test set, final validation; learn again; the detector response plan
```

Principle: do the **gate first** (the estimator-validity gate: synthetic controls). A multiscale number computed before the estimators are
validated cannot be interpreted, and the estimator-validity gate is accepted as the gate for every multiscale or fractal claim. The first
check on real Geant4 events is therefore *after* the controls on the same grid, on the same day at the earliest.

### 2. Estimator controls: the estimator-validity gate on the 18 x 72 grid (EXP-008)

**Before any code (learn, define, pre-register).** The research workflow says: learn and verify the mathematics,
define estimators, validate on controlled cases. So:

1. **Learn, about 30 min.** A tutoring block (Obsidian, with a verified visual) on the partition sum
   Z_q(s) = sum p_i^q, and on D_q as the slope of its log against log(1/s), starting from the q = 2 toy already
   used on 2026-09-20. Box counting is Z_0. No more is needed for the first check. Multifractal spectra stay out of scope.
2. **Verify the definitions** against a standard source (bounded source-verifier pass; the definitions are
   textbook mathematics, not hypothesis-sensitive). Record the page and equation in the pre-registration.
3. **Pre-register**, in a new `research/plans/` file before any estimator output is read: the estimators, the six box
   sizes, the fit rule and scale range, the controls, and the acceptance numbers below.

**Estimators (first check).** On a non-negative deposit image: occupied-box count N(s), D_0; and Z_q(s) for q = 1 and 2
(D_1 via the entropy form, D_2). Local slopes between adjacent scales are reported alongside any fitted slope, so the
scale dependence shown in the 2026-10-02 tutoring is visible, not averaged away. Lacunarity and entropy-across-scale
are listed as candidate extras, not part of the first check.

**Controls, all on 18 x 72 and on a finer grid for comparison:**

| Control | Purpose | Known answer |
|---|---|---|
| Smooth source sampled by the same number of hits (Poisson quanta) | Negative: the counting-limit baseline | slope set by hits and scale range, not structure |
| Same image with cells shuffled within each layer | Negative: kills spatial structure, keeps layer energies | baseline-like |
| Multiplicative binomial cascade on 2^k grid, then **projected** to 18 x 72 | Positive: structure with known D_q, passed through the lossy readout | known D_q on the fine grid |
| Hit-count sweep (30 to 3000 quanta) | Finite-sample bias of each estimator | none; the bias curve is the output |

**Proposed acceptance (to be fixed by you before output is read; these numbers are a proposal):** an estimator is
"usable at AMS granularity" for a given q only if (a) on the cascade control it recovers the known D_q within 15%
after projection, and (b) on the negative controls its value stays within the baseline's own spread, over the same
hit counts and scale range. Otherwise the result is "not usable here", which is a valid, reportable outcome.

**Output:** `src/ams_ecal/multiscale_estimators/multiscale.py` plus tests; a results folder with the control tables; a verified teaching
visual of the controls. Cost: one day of build, minutes of compute. No Geant4.

### 3. Proton first look — first look at existing proton Geant4 events (EXP-001, stage 1, exploratory)

Only after the estimator controls pass for the estimators it will use, and only if the researcher confirms.

- **Data:** the existing FTFP_BERT baseline proton events at 10/20/50/100 GeV. **Calibration events only**
  (`event_index % 4 != 3`); the held-out events stay untouched, as already decided.
- **Do:** compute the validated estimators on (i) the fine voxel deposits and (ii) the same events projected to
  18 x 72, and compare each with a **matched smooth baseline** (same hit count, same layer-energy profile).
- **What it can say:** whether proton showers differ from the smooth baseline at fine scales, and whether any
  difference survives projection to 18 x 72. This is part of H-002 and H-003 for protons only.
- **What it cannot say, and the write-up must state:** anything about electron versus proton (no electrons yet),
  anything about particle ID, anything about fractals as a physical claim, or anything about AMS data. It is Geant4
  11.4.1 with one physics list. The production cut was only checked at readout level, and the pilot's own gate says
  no sub-cell claim without a cut-sensitivity study; this check is therefore labelled exploratory and the cut study
  is listed as a follow-up.
- **Pre-registered:** the comparison and the decision rule, written before the first number is read.

### 4. Electron data — electrons, so the real e/p comparison exists (EXP-001, stage 2)

1. **Timing probe, about 20 events at 10 and 100 GeV**, same geometry, cut, entry spot and seed policy as the proton
   pilot. This is a measurement, not a result, and is the only electron Geant4 run proposed before approval of more.
2. **Go/no-go** on a full electron sample using the probe: events per energy, energies, expected wall time, disk size.
   If the cost is hours, run it in the background while Tracks A and B proceed (the machine has 16 cores).
   If the cost is days, reduce energies or events, or subsample, and say so. The decision is the researcher's.
3. **Do not compare** electrons with the existing proton set until both are projected and processed by the same code.

### 5. Proton model completion (the accepted sequence, steps 4-10)

| Step | Work | Notes |
|---|---|---|
| 4 | Burst latent and lateral spill on the crossing branch (EXP-005) | structural burst (probability or type, onset, energy, downstream extent), as specified |
| 5 | Second-look diagnostic | |
| 6 | Interacting model under the interacting-proton factorization with its three amendments (EXP-006) | residual tested for normality before being called log-normal |
| 7 | Speed: 11-14 ms per event is too slow for large datasets | batch path |
| 8 | Freeze; register the added validation checks rows | registered before the test set is generated |
| 9 | Fresh Geant4 proton test set | **proposed change below** |
| 10 | Final the final validation (step 6) validation on that set | pre-registered the validation contract and the added validation checks contract |
| 7' | Alternative physics-list calibration (step 7): QGSP_BERT calibration (EXP-007) | needed only before claims about hadronic physics |

**Proposed speed-up, needs explicit approval because it reorders the accepted sequence.** Step 9 does not need a
frozen model, only a frozen, registered contract: the fresh proton set can be generated **early and sealed** (never
opened, file hashes recorded, disjoint seeds), then opened once at step 10. Integrity is preserved because the set has
not been seen by anyone, and the generation runs in the background while the build proceeds. Added validation checks must be registered
**before** generation, as already decided, which is a half-day task. If the original order is preferred, step 9 stays
after the freeze.

### 6. Learning gates

- **Weekend:** the partition-function block above, before estimator definitions.
- **2026-10-04 09:00:** the scheduled delayed check (histograms versus joint, Jensen zero condition and conserved sum,
  the slope-and-hit-count idea). It already covers the idea the weekend's controls depend on.
- **After step 10:** the post-build teach-back, a teach-back on the interacting model (amplitude x profile, albedo, burst), before any
  Detector response work.
- **Detector response** stays blocked until its evidence and adversarial pass runs: for each candidate detector effect, state
  whether AMS already inverted it out of the deposits its shower parameters were fitted to (the double-counting
  question in `research/STATE.md`). That pass is independent literature work and starts after step 10, then a plan,
  then explicit approval.

### 7. Risks, stated plainly

- Electron cost may make the weekend electron sample impossible. Then the weekend delivers the validated estimator
  controls and a proton-only first look, and the e/p comparison follows later. That is still a real result.
- Six scales at AMS granularity may be too few for any estimator. The controls will show it, and the answer
  "not usable here" is acceptable and publishable under the roadmap's outcome categories.
- The weekend is tight if the learning block and the delayed check both run. If time slips, cut the proton first look, not the estimator controls.
- Everything here is Geant4 with one physics list and no AMS data.

### 8. Approval table (items approved on 2026-10-02 are marked with the date; blank means still open)

| # | Item | Needs | Approved? |
|---|---|---|---|
| 1 | Estimator controls: learn block, pre-registration, estimator code and synthetic controls (EXP-008) | your approval | **2026-10-02** |
| 2 | Acceptance rules for the controls (95% interval, 10% false positives, 90% power, 15% recovery reference) | your numbers | **2026-10-02**, confirmed as written after a literature check (record DEC-013); the ordering fraction 0.90 still awaits confirmation |
| 3 | Proton first look on calibration events, exploratory (EXP-001 stage 1) | your approval, after the estimator controls | open |
| 4 | Electron timing probe: 20 events at 10 and 100 GeV | your approval | **2026-10-02** |
| 5 | Full electron sample (10, 20, 50, 100 GeV x 1000) | decision after the probe | **2026-10-02** ("go"); finished the same day |
| 6 | Early sealed fresh proton test set (changes the accepted step order) | explicit approval | **2026-10-02** (the added validation checks must be registered before generation) |
| 7 | Proton model steps 4-8 as in section 5 | your approval to start Monday | open |
| 8 | Handling of the alternating X/Y views | your decision | **2026-10-02**: combined 18 x 72 primary, per-view secondary |

### 9. Result of the electron timing probe (run 2026-10-02, approved item 4)

Config `configs/geant4_electron_probe.yaml`; data in `data/geant4_electron_probe/` (not for analysis). A cost measurement
only: 20 events per energy, FTFP_BERT, AMS-only geometry, 8 workers.

| energy | events | wall time (8 workers) | size on disk |
|---|---|---|---|
| 10 GeV | 20 | 54.5 s | 0.6 MB |
| 100 GeV | 20 | 65 s | 2.5 MB |

**Reading, with its limits.** Both wall times include worker start-up and geometry construction, which is unknown here, so
the per-event cost cannot be read directly. The 10.5 s difference between the two energies times 8 workers gives an
estimate of about 8 CPU-seconds per 100 GeV event if start-up is near 45 s, and the hard upper bound (no start-up at all)
is 26 CPU-seconds. Either way electron events are **far cheaper than feared**: a 1000-event batch at 100 GeV is roughly
10 to 35 minutes on 12 workers, and about 125 kB per event on disk. The first full batch will give the real number.
Protons for comparison: 4000 events at 100 GeV took 710 s on 12 workers.

**Recommendation for item 5 (decision still the researcher's):** electrons at 10, 20, 50 and 100 GeV, 1000 events each,
matched to the proton pilot (same geometry, cut, entry spot, FTFP_BERT), disjoint seeds. Estimated total wall time under
two hours and under 0.5 GB. Nothing from the probe is analysed for shower structure.

---

## Appendix D — EM production generator — PLAN (no implementation before approval)

_Retired plan, carried over verbatim on 2026-10-10 (original file `research/plans/2026-10-05_em_production_generator_plan.md`, last version in git at `e904286`; retrieve with `git show e904286:research/plans/2026-10-05_em_production_generator_plan.md`). Content: EM production generator: stages, F1-F5 decisions, calibration, slices 1-2. Headings are demoted one level; nothing else is changed except that references to retired files point to their appendix._


_Drafted 2026-10-05 on branch `fastmc-validated-dataset`, after the first out-of-sample comparison of the DEC-001 EM generator with Geant4 electrons. Direction accepted by the researcher on 2026-10-05 (`research/DECISIONS.md`); the model's formulation is NOT yet approved, and by `CLAUDE.md` nothing is implemented until it is._

### 1. Why

`results/em_generator/exploration_comparison.json` (module `ams_ecal.electron_studies.em_exploration_check`, unsealed exploration electrons, `deposition` representation): KS 0.57-0.75 on event energy, 0.90-0.94 on longitudinal RMS, 0.64-0.73 on core fraction, 0.76-0.80 on containment, 0.72-1.0 on hit cells, 0.29-0.68 on width; classifier AUC 1.00 at every energy; layer-correlation row material at every energy. The generator was never fitted to these events, so this is an honest out-of-sample falsification of its adequacy as a PRODUCTION generator. It is not a defect in its other role.

### 2. Two models with two roles

| name | what | role |
|---|---|---|
| `em_smooth_null` | the DEC-001 generator, unchanged (`ams_ecal.electron_model.stochastic`) | permanent smooth analytic reference: control for the multiscale estimators, interpretable baseline. Never mixed into the e/p dataset. |
| `em_production` | NEW, calibrated to Geant4 electrons | the electron half of datasets. Supersedes DEC-001 for dataset generation only. |

The AMS-inspired mean longitudinal profile may stay inside `em_production` as a prior (physics constraint). Scientific upside: smooth null -> calibrated FastMC -> full Geant4 is a controlled hierarchy for the multiscale project; a statistic that moves monotonically along it is measuring shower structure that the simple parameterisation erased.

### 3. What the comparison localises (targets for the hierarchy)

1. Visible energy and leakage: model energy 3-5% high at 100 GeV with a missing low tail (Geant4 5% quantile 78.0 GeV against the model's 88.1 GeV): rear leakage and fluctuation of the contained fraction.
2. Longitudinal: mean layer-energy ratio model/Geant4 is 0.02-0.07 in layer 0 and 0.22-0.41 in layer 1, 1.15 in the middle, 0.76-0.89 at the end; the longitudinal RMS has almost no event-to-event spread (30.9-31.8 mm against 31.4-36.9 mm).
3. Lateral: width 14.9 against 13.9 mm, core fraction 0.89 against 0.87, 387 against 334 hit cells at 20 GeV and 621 against 483 at 100 GeV: too wide and too smooth.
4. Correlations: layer-to-layer and longitudinal-lateral, flagged by the layer-correlation row.

### 4. Calibration order (hierarchical; each stage frozen before the next)

Stage 1 visible energy and leakage. Stage 2 longitudinal fluctuations and shower maximum. Stage 3 lateral width, tails and hit sparsity. Stage 4 layer-to-layer and longitudinal-lateral correlations. Never optimise classifier AUC or any single discriminator: AUC stays an omnibus diagnostic, otherwise the generator is trained to fool one classifier. No stage adds parameters merely to remove a KS value; each parameter needs a named physical or statistical meaning and a measured need.

### 5. Procedure required by CLAUDE.md for stochastic generation (status)

| step | status |
|---|---|
| probe the researcher's current understanding | not done: part of the planned tutoring session |
| missing prerequisites identified | not done |
| independent external review of shower-fluctuation literature | DONE as discovery, thin (2026-10-05, abstract depth, air-shower papers wrongly used and struck); see `plans/2026-10-05_em_literature_verification_record.md` |
| independent verification of implementation-affecting claims | DONE for what exists: Grindhammer-Peters verified from the full text; AMS-02 ECAL geometry verified; no published fluctuation numbers for THIS detector (the Geant4 electrons supply them: `results/em_generator/longitudinal_fluctuations.json`); adversarial pass done (its headline refuted by a control, see the record, section 4b); a fuller counterevidence search is still open |
| reconcile external evidence with the deterministic generator | partly: the comparison above is the data side of it |
| formulate the stochastic physical model | not done |
| researcher approval | not given |
| implementation | not started |

### 6. Data and protocol

- Calibration and development data: `data/geant4_electron_sample` (4 energies x 1000 electrons, exploration). All of it has now been looked at, so its events can never provide validation evidence (amendment of 2026-10-05 to the registrations).
- **The sealed electron set (`data/geant4_electron_sealed`, 7 energies x 1000, base seed 20271002) was not exposed and remains closed.** A "fresh untouched validation sample" may be unnecessary: the existing sealed set can serve, unless the researcher wants a new one with different seeds. Decision needed.
- Tolerances: from a split-Geant4 self-comparison noise floor plus physics-motivated tolerance per observable class (energy/leakage, longitudinal, lateral, sparsity, correlations); registered BEFORE any sealed opening.
- Geant4 EM sensitivity: a different EM option (for example `FTFP_BERT_EMZ`; availability in the installed 11.4.1 build to be verified) on electrons. It needs its own approval under the Geant4 workflow. It is not the QGSP_BERT comparison, which changes only hadronic physics.

### 7. Questions for the researcher (answered 2026-10-05: Q10 yes; Q11 use the existing sealed electron set; Q12 literature first, tutoring after)

- **Q10.** Approve the two-role naming and keeping the DEC-001 code untouched as `em_smooth_null`?
- **Q11.** Use the existing sealed electron set as the untouched validation sample, or generate a new one?
- **Q12.** When should the tutoring on the generator's physics (probe, prerequisites) take place relative to the literature verification?

### 8. Reconciliation and PROPOSED formulation (2026-10-05, for the researcher's approval; nothing implemented)

#### 8.1 What the DEC-001 code does and what the evidence says (reconciliation)

Read in `src/ams_ecal/electron_model/stochastic.py` (lines 169-263) and `longitudinal.py` (lines 311-340): one standard normal per event gives `T0 = exp(mu(E) + s(E) z)` with `s(E) = 1/(intercept + slope ln(E/E_c))` (the width law has the Grindhammer-Peters form; the shower-maximum width therefore already agrees with the measured sigma(ln T), `results/em_generator/longitudinal_fluctuations.json`), then `alpha = 1 + beta T0` with `beta` FIXED, a gamma profile integrated over the 18 layers, and a deterministic lateral grid (`lateral.track_centered_cell_fractions`).

| element | DEC-001 | measured in Geant4 electrons (exploration) | literature (verified) |
|---|---|---|---|
| sigma(ln T) | G-P width law | agrees with the formula within about 11% | G-P App. A.2.2 |
| second longitudinal variable | none: alpha locked to T0 through fixed beta (rho = 1) | sigma(ln alpha) 0.16-0.21, rho(ln T, ln alpha) 0.5-0.7 | G-P: bivariate Gaussian, rho 0.57-0.62 |
| mean profile | gamma, fixed beta | layer 0 holds 2-7% and layer 1 22-41% of Geant4's energy; middle layers +15%, last layers -11 to -24% | G-P has no thin-calorimeter regime |
| leakage / contained fraction | follows from the profile | median 0.951 / 0.936 / 0.915 / 0.890 at 10 / 20 / 50 / 100 GeV; model 3-5% high at 100 GeV with no low tail | not found for this detector |
| lateral | deterministic grid | width, hit cells, core fraction, containment correlate with ln T (Spearman up to 0.7-0.8 at 100 GeV); too wide and too smooth in the model | G-P: lateral parameters deterministic in tau, fluctuations from spot generation coupled through tau_i |
| layer residuals around the profile | none | relative rms about 4-10% in the middle layers at 100 GeV, up to 25% or more at the ends | not found |

#### 8.2 Proposed model `em_production` (hierarchical; each stage frozen before the next; nothing here is fitted to classifier AUC)

1. **Mean profile and leakage (stage 1).** Mean `<ln T>(E)` and `<ln alpha>(E)` as smooth functions of `ln y` with free coefficients fitted to the Geant4 profile fits, replacing the fixed beta; Grindhammer-Peters forms as the starting structure. First test, before any model code: how much of the first-layer deficit and of the leakage misfit disappears when beta is free; if a per-layer mean correction is still needed it is added as a named, interpolated table (a stated number of parameters), not hidden.
2. **Longitudinal fluctuations (stage 2).** `(ln T, ln alpha)` bivariate with `sigma(ln T)`, `sigma(ln alpha)` and `rho` as functions of `ln y`, starting from the Grindhammer-Peters sampling forms, coefficients calibrated. If the bivariate distribution is measurably non-Gaussian (skewness of ln alpha 0.3-0.6 was reported, unchecked), use a Gaussian copula with empirical marginals, as in the proton model. Plus a layer-level multiplicative residual around the profile, with its measured scale per layer.
3. **Lateral structure (stage 3).** One event-level lateral latent coupled to the longitudinal variables with the measured coupling (not assumed), a depth-dependent core-plus-halo radial law, and cell-level granularity through the same quanta mechanism as the proton lateral model (`proton_lateral.py`), with its own calibration for electrons. Sharing the mechanism is a choice with a consequence: the two classes then differ by calibrated parameters, not by one being smooth and the other not.
4. **Correlations (stage 4).** Layer-to-layer correlation of the residuals through the same Gaussian copula with a common factor and an AR chain used for the protons; checked against the layer-correlation row.

Representation: `deposition` only. The fibre-energy scale is the separate detector-response step R-A.

#### 8.3 Stop rule and discipline

A stage is accepted when its contract rows reach the registered tolerances on the calibration data; it is then frozen. A stage that cannot reach them with a smooth model is recorded as a residual, not tuned further. No parameter is added without a named meaning and a measured need. The sealed electron set is opened once, at the end, through `sealed_set.open_set`.

#### 8.4 Decisions needed to proceed

- **F1.** Approve the structural prior (Grindhammer-Peters forms with coefficients calibrated to Geant4, copula if the marginals are non-Gaussian) for stages 1-2.
- **F2.** Reuse the proton quanta mechanism for the electron lateral structure (stage 3), or implement the Grindhammer-Peters spot model.
- **F3.** `deposition` only for `em_production` (the detector response later), as proposed.
- **F4.** Whether a per-layer mean correction table (named parameters) is acceptable if the free-beta gamma profile does not remove the first-layer deficit.
- **F5.** Approve the first implementation step as ANALYSIS only: the stage-1 test (free beta against the first-layer deficit and the leakage), with no generator code.

### 9. Resolution of the section 8.4 decisions (2026-10-05)

The researcher's answers to F1-F5 are recorded one row each in `DECISIONS.md` (with DEC-014) and are not repeated here. In short: the Grindhammer-Peters
forms are the structural prior with the AMS mean constraint kept as a separate question (F1); the GP spot model is the lateral route (F2); `deposition` only (F3);
a per-layer mean correction only if conditions are met (F4); and the next step is analysis only (F5). The analysis is
`research/plans/2026-10-05_em_longitudinal_structure_analysis_note.md`.

### 10. Depth-origin and mean-beta calibration (2026-10-06; analysis only, nothing implemented)

Approved scope (`DECISIONS.md`, 2026-10-06 rows): one common depth origin z0 and one mean beta calibrated on the exposed Geant4 deposition profile at 10, 20, 50, 100 GeV;
the origin is a calibrated coordinate convention, not a physical shower start; the sampling-set widths are approved in principle and frozen only after this calibration;
the copula and any first-layer floor are gated; the sealed electron set stays unopened as the final fresh validation set.

Result, in one paragraph (details and numbers in `research/plans/2026-10-06_em_depth_origin_calibration_note.md`): z0 = -0.98 X0 and beta = 0.571 for the primary form, with a spread of
about 1 X0 in z0 across model forms; beta = 0.65 is disfavoured on the deposition profile and marginal on the readout profile; the covariant sampling-set widths and the
T-alpha correlation survive at the calibrated coordinate; ln T stays right-skewed at every origin; the first-layer floor is rejected as not applicable; the mean profile is much
improved but not adequate (chi-square per dof 36, 100 GeV worst). **The implementation gate is not met** (conditions 1, 3 and 6).

Revisions of section 8 that follow from it: stage 1 (mean profile) must carry a named origin with its model form, and an energy-dependence test (beta or origin) comes before
freezing; stage 2 (widths) keeps the covariant sampling-set widths; the original bivariate Gaussian is not assumed for ln T; no first-layer floor. Open before the first implementation
slice: the energy dependence and the depth law (why the 100 GeV profile is not reached), and whether the skewed ln T marginal changes the leakage tail and the contained-fraction
distribution.

### 11. Profile shape study (2026-10-06; analysis only, nothing implemented)

The researcher asked for the three analyses proposed at the end of section 10 before any implementation, with the profile details checked in the literature. Result, in one paragraph
(numbers and locators in `research/plans/2026-10-06_em_profile_shape_study_note.md`): the single gamma is a good bulk description and a poor description of the two ends. A slowly decaying
tail component with the decay length of lead taken from Leroy and Rancoita (3.3-3.9 X0) cuts the mean-profile chi-square per dof from 35.6 to 16.4 with held-out gains of +27% to +52%;
energy-dependent beta or origin and a free depth law do not. The ln T skewness is real and matters for the leakage distribution only together with the tail; the AMS 0.65 is an origin
convention that becomes compatible with the core once the tail is present. The tail rate, shape and weight are not identified by the 18-layer prefix.

Revisions of section 8 that follow: stage 1 (mean profile) is a gamma core plus a tail component with a named origin; stage 2 (widths) keeps the covariant sampling-set widths; the ln T marginal is
skewed (skew-normal or empirical) and is chosen jointly with the tail; no first-layer floor. Open before the first implementation slice: the tail's rate, shape and weight (an extended-depth electron
sample would measure it), whether the tail weight fluctuates event by event, and the entrance energy (layer 0 = 34.6 MeV + 0.416 MeV/GeV x E).

### 12. Extended-depth electrons (2026-10-06; analysis only, nothing implemented)

The researcher approved a development sample of electrons with the extended depth to measure the tail the 18-layer prefix could not constrain
(`configs/geant4_electron_extended.yaml`, 1000 events at each of 10, 20, 50, 100 GeV; exposed development data, no sealed set). Result (numbers and locators in
`research/plans/2026-10-06_em_extended_tail_note.md`): the tail is an energy-independent exponential of rate 0.27-0.28 per X0 beyond about layer 20, in agreement with the
literature value for lead; 5.1-11.3% of E is behind the prefix, nearly all in layers 18-29. The tail components fitted on the prefix extrapolate wrongly and are only an empirical in-prefix shape
correction; no additive gamma-based family fits prefix and tail together; the event-level energy behind the prefix is mostly set by the prefix shape.

Revision of section 8: the far tail is not a stage of the generator. Stage 1 (mean profile) is a gamma core with a named empirical in-prefix tail correction (energy-weighted, calibrated on the prefix and its containment);
any quantity behind the prefix is calibrated directly from the extended sample. Open: the choice between implementing now with named limitations and first looking for a better functional family.

### 13. Implementation Slice 1 (2026-10-06): the gamma-family longitudinal generator

The researcher approved the first implementation slice and locked the gamma density as the baseline physics family. Implemented in `src/ams_ecal/electron_model/em_production.py` (generator, parameter
artifact), `em_production_calibration.py` (calibration) and checked by `src/ams_ecal/electron_studies/em_production_development_check.py`; the note is
`research/plans/2026-10-06_em_production_slice1_note.md`. It realises stages 1 and 2 of section 8.2 in the form the evidence supports: a gamma core at a fixed front-face origin, an empirical in-prefix tail correction, covariant
Grindhammer-Peters widths, a skew-normal ln T with a Gaussian copula, no first-layer floor. Stage 3 (lateral) and stage 4 (layer-to-layer residuals) are NOT part of the slice; DEC-001's lateral grid is used unchanged.
Development result: far closer to Geant4 than DEC-001 on all metrics but the centre of gravity, not at the sample-to-sample floor; named residuals in the note. The sealed electron set is opened once, after the remaining stages, through
`sealed_set.open_set`.

---

## Appendix E — Datasets for the multiscale test — PLAN (for the researcher's approval; nothing here is built)

_Retired plan, carried over verbatim on 2026-10-10 (original file `research/plans/2026-10-09_multiscale_test_dataset_plan.md`, last version in git at `e904286`; retrieve with `git show e904286:research/plans/2026-10-09_multiscale_test_dataset_plan.md`). Content: multiscale test datasets, development readings 2026-10-09/10, scope, correction. Headings are demoted one level; nothing else is changed except that references to retired files point to their appendix._


_Drafted 2026-10-09 at the researcher's request: "make a complete plan, verified by the literature and Geant4 simulations we made to produce the final datasets to start testing our multiscale hypothesis on; we have been spending a week on this, we should speed up."_

Status: **a plan.** Section 8 lists the decisions that block it. Reads with `research/PUBLICATION_ROADMAP.md` (Paper 1), `research/STATE.md` (RQ-001, H-001),
`2026-10-02_multiscale_estimator_validity_preregistration.md`, `2026-10-02_multiscale_estimator_controls_results.md`,
`RESEARCH_PLAN.md (Appendix B)`, `research/sealed/README.md`.

### 1. The finding that sets the order: FastMC fidelity is not on the critical path of the first test

The registered core evidence path of Paper 1 is **fine-grained Geant4 -> AMS-like projection -> does the signal survive -> incremental information -> classical ML**.
The roadmap states that FastMC is "a smooth phenomenological control and development simulator" that "cannot by itself establish or falsify microscopic
fractal/multifractal cascade structure". The week of lateral and longitudinal electron work was FastMC fidelity. It is needed for one arm (does FastMC preserve the
multiscale observables) and for cheap large samples, **not** for the first test of H-001.

Geant4 is cheap here: 4000 electrons took about 51 minutes and 4000 protons at 100 GeV 710 s on 12 workers (plan of 2026-10-02, section 2). Every Geant4 run stores the
deposition grid, the readout grid and, for the sealed configs, the fine prefix deposits (`store_fine_prefix_deposits: true`). **The fastest honest route is to produce
the test datasets with Geant4 and to run the FastMC arm in parallel.** I have not independently re-derived the timings this session; they are the recorded values.

### 2. What exists (verified by listing the repository today)

| set | content | status |
|---|---|---|
| `geant4_electron_sample` | e-, 10/20/50/100 GeV x 1000, base seed 20270000 | exposed: development |
| `geant4_electron_extended` | e-, same energies x 1000, 270 layers (prefix = hold-out for FastMC) | exposed: development |
| `geant4_proton_pilot` | p, 10/20/50/100 GeV x ~4000, normal incidence; events with `event_index % 4 == 3` are held out | calibration and a held-out quarter |
| `geant4_electron_sealed` | e-, 7 energies (10, 14, 20, 30, 50, 70, 100) x 1000, base seed 20271002, fine prefix stored | sealed, never opened |
| `geant4_proton_sealed` | p, same 7 energies x 3000, base seed 20271001, fine prefix stored | sealed, never opened |

Fixed by the existing configs: FTFP_BERT, AMS-only geometry, 0.7 mm production cut, entry spot over one 9 mm cell, **normal incidence only**.

### 3. What the literature and the controls already support, and what they do not

| claim the test depends on | support | gap |
|---|---|---|
| Showers are cascades with scale-dependent structure worth testing | the repository's cascade controls (EXP-008) recover the exact cascade value | EXP-008 is synthetic; **no claim about real showers exists** |
| The estimators work at AMS granularity | usable only with about 1000-3000 hits (2 of 13 and 4 of 14 windows); coarse pair (9, 18) from 100 hits, fine pair (1, 3) needs 3000 | **the eligibility count (occupied cells per event) of real Geant4 events was never run** |
| Multifractal / intermittency analysis of showers in the literature | the estimator pre-registration cites it | its snippet-level sources are **still unverified** (open item in DECISIONS); not read in full text |
| EM lateral/longitudinal reference (Grindhammer-Peters, Kounine, AMS papers, PDG) | read in full text, with locators (2026-10-06/07 notes) | concerns FastMC, not the estimators |
| The Geant4 simulation is a trustworthy truth | one physics list, one cut, normal incidence; `cut_0p1mm` and `QGSP_BERT` variants exist for protons; the electron EM-option variant (EXP-007) was **never run** | systematics before any sub-cell claim |

Consequence: the plan has an explicit **literature verification gate** (phase 1) and a **systematics gate** (phase 3). Under Research OS v0.8, a source is labelled by what
was read (full text, abstract, snippet); a snippet is a gap, and nothing downloads without your permission (I give links).

### 4. The datasets to produce

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

### 5. Phases, gates and time

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

### 6. Controls that keep the result honest

1. **Conventional-feature control.** Every multiscale effect is reported conditional on width, depth, total energy, containment and hit count (the ladder step "conventional + multiscale"). Differences that reduce to width or depth are an outcome of the paper, not a failure.
2. **Estimator validity gate (DEC-010).** No multiscale value is read on events or windows the controls rule unusable; the eligible fraction is part of every result.
3. **Fine vs projected.** The same events analysed at the fine scale and after the 18 x 72 projection, so "present in transport, destroyed by segmentation" is measurable.
4. **Smooth null.** `em_smooth_null` (DEC-001) and the smooth baseline of EXP-008 are run through the same pipeline.
5. **Systematics.** The G-syst variants, per observable, before any sub-cell statement.
6. **No classifier tuning.** The classifier two-sample test stays an omnibus diagnostic; no generator or observable is tuned to it.
7. **Opening discipline.** G-final is opened once; every threshold and tolerance is fixed in code before; none relaxed afterwards.

### 7. Stated limitations that travel with the datasets

Normal incidence only (no angular calibration exists). One physics list and one cut in the main sets. Perfect-event (deposition) data first; the detector-level version
needs the instrumental effects (R-B), which are still blocked on the evidence pass and are **not** part of this plan. FastMC electrons are distinguishable from Geant4
(classifier AUC 1.000, measured today); they are a smooth control, so that is a documented property, not a defect of the first test.

### 8. Decisions I need from you (in the order they block work)

1. **Approve the order**: Geant4-first for the multiscale test; FastMC refinement moves to a parallel, non-blocking track. (If you disagree because the paper must use FastMC samples, the plan changes: say which claim needs them.)
2. **Approve generating G-train** (10,000 per class per energy; Geant4 compute) and **G-syst** (needs your approval to run Geant4).
3. **Confirm the sealed-set allocation**: the sealed e/p sets serve as G-final for the multiscale test AND as the validation of FastMC. They can be opened once only. Proposal: the multiscale test opens them first; FastMC validation then uses the opening's fine-grained comparison, or you authorise new sealed electron/proton sets for FastMC (cheap, about a day of compute).
4. **Confirm the pre-registration numbers** in phase 2 when drafted (orders q = 0, 1, 2; box sides 1, 2, 3, 6, 9, 18; the 0.90 ordering fraction is still an open interpretation).
5. **Literature**: permission to run the verification pass of the multiscale sources; I give you links for papers I cannot read, and add nothing to the library myself.

### 9. What I will not do without your word

Open any sealed set; build anything in phases 1-5 before approval; change DEC-001, the proton model, or the sealed-set rules; call anything validated; download literature.

### 10. Phase 1 result: the eligibility count (2026-10-09, exposed events only, no multiscale value computed)

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

### 11. First development reading of the registered statistics (2026-10-09; exposed events; NOT the registered test)

Code `multiscale_test.py` (statistics) and `multiscale_development.py` (driver), result `results/multiscale_controls/multiscale_development.json` (60 bootstraps; electrons: the 1000 baseline events per energy; protons: truth-interacting calibration events, 1408-1453 per energy). Sealed sets unread.
- **Deposition grid, electrons against interacting protons, matched on occupied cells (five strata):** at 20, 50 and 100 GeV every one of the 28 registered statistic-window rows at the two largest windows is above the registered rule (matched KS 0.27-0.63 against a split-half floor of 0.05); for example D0 - D2 over all six scales: 0.37 / 0.51 / 0.63 at 20 / 50 / 100 GeV. At 10 GeV only 3% of electrons are eligible, so nothing is compared.
- **Readout grid:** nothing at 10 and 20 GeV (0% of electrons reach 336 cells); 50 GeV compares only the 3% of electrons that are eligible; 100 GeV (72% eligible) gives matched KS 0.31-0.57. This is the plan's expected loss of resolving power at the AMS readout, now measured.
- **Matching on the deposited energy is not usable** (no common support: electrons deposit nearly all of the beam energy, interacting protons are spread; matched KS is NaN at 20 GeV and 0.78-0.96 at 100 GeV where only the thin overlap remains), so the occupied-cell matching carries the comparison.
- **What this does NOT show:** (1) the difference may reduce to conventional shape (width, depth, hit count) that the cells matching does not remove; that is the ladder's job (Q-D) and has not been run; (2) five cell strata can leave residual confounding (D0 depends on the cell count by construction); (3) electrons are 1000 events, protons are interacting-truth only, a selection that real data cannot make; (4) exposed development data, used for design, so it is not evidence for H-001; (5) the electron samples are Geant4 FTFP_BERT, one list, one cut, normal incidence.
- Next: the conventional-feature ladder and the proton-rejection metric on G-dev, then on G-train when it exists; the fine-scale estimator controls.

### 12. First development reading of the information ladder (2026-10-09; exposed events; NOT the registered test)

Code `multiscale_ladder.py`, result `results/multiscale_controls/multiscale_ladder_development.json` (100 bootstraps; the two raw-grid rungs not fitted in this run; electrons 1000, interacting protons 1408-1453 or all calibration protons 3000 per energy).
- **The conventional features already separate the classes perfectly:** AUC 1.000 (0.999 at the lowest) at every energy, both grids, both proton sets. Proton rejection at 90% and at 99% electron efficiency equals the number of protons available (no proton passes), so the rejection is only bounded from below (greater than 1400-3000).
- **Consequently the registered added-information rule cannot be met:** the gain of conventional + multiscale over conventional is 0.0000 (interval within +-0.003) everywhere, "adds information" is False everywhere, and the 0.01 AUC criterion has no headroom. This does NOT show the multiscale features add nothing: it shows this setting cannot ask the question.
- **Multiscale features alone** reach AUC 0.83 / 0.99 / 1.00 / 1.00 (deposition, interacting protons, 10-100 GeV) and 0.70 / 0.77 / 0.82 / 0.95 (readout); at low energies on the readout much of that is the eligibility indicator itself (NaN versus not NaN follows the cell count), not a dimension.
- **Why it is easy here (hypothesis, not tested):** the samples have fixed beam energies, so the total deposited energy and the layer profile separate electrons (deposit nearly all of the beam energy) from protons (spread, often leaving early) almost completely; AMS does not know the energy this way (it has the tracker rigidity, which would enter an E/p cut).
- **Consequence for the pre-registration (an amendment to decide before the final opening; NOT made):** the primary quantity should be the proton rejection at fixed electron efficiency, which needs far more protons than 3000 to resolve (a rejection of 1e3-1e5), and/or a harder setting for the conventional rung (for example without the total deposited energy, or within narrow energy windows, or with the energy treated as unknown). G-train (10,000 electrons and 20,000 protons per energy) lifts the resolvable rejection to about 2e4 at best; reaching AMS-relevant rejections needs a large simulated sample, which is the purpose of the FastMC arm. The 0.01 AUC rule is kept as registered until the researcher decides.

### 13. Large-sample development reading (2026-10-10; exposed Geant4, 10 / 20 / 50 GeV; NOT the registered test; sealed sets unread)

Code `multiscale_gtrain_development.py`, result `results/multiscale_controls/multiscale_gtrain_development_10_20_50GeV.json`. Electrons: 10,000 per energy; protons: 20,000 per energy, of which 9,647-9,709 truth-interacting are compared (the others are crossing and ineligible). 100 bootstraps. The 100 GeV electrons were still generating.
- **The registered statistics, matched on occupied cells (deposition grid):** D0, D1, D2 and D0 - D2 over all six scales differ between electrons and interacting protons by matched KS 0.33-0.55 at 10, 20 and 50 GeV (intervals within about +-0.05; split-half floors 0.02-0.10), every one material by the registered rule. The eligible fraction of electrons is 0.03 at 10 GeV (so the 10 GeV value rests on 3% of the electrons), 0.91 at 20 and 1.00 at 50 GeV; of interacting protons 0.68-0.80.
- **Readout grid:** nothing eligible at 10 GeV (0% of electrons) and at 20 GeV (0%); at 50 GeV only 3% of electrons are eligible but the matched KS of those is 0.27-0.56, material by the rule.
- **The ladder has no headroom, even with the harder rungs:** on the deposition grid the conventional rung gives AUC 1.0000 and no proton passes at 99% electron efficiency (rejection above 9,600, bounded by the sample); dropping the total energy gives AUC 0.9999-1.0000 and a rejection of 4,800-9,700; dropping the energy and the occupied-cell count too gives AUC 0.9999-1.0000 and a 99% rejection of 1,900-4,800 (a false-positive rate of about 2-5 x 10^-4, i.e. 2-4 protons in 9,700). The paired AUC gain from adding the multiscale features is +0.0000 (intervals within +-0.0003) in all three, at all energies, on both grids. The multiscale features alone reach AUC 0.84 / 0.99 / 1.00 (10 / 20 / 50 GeV, deposition).
- **The sample-efficiency curves are flat at the ceiling:** with only 100 events per class the harder rungs already reach AUC 0.996-0.999, and 0.9998-0.9999 from 1,000; adding the multiscale features changes nothing at any size.
- **What this means (measured, then interpretation):** in this simulation (normal incidence, perfect events, fixed beam energy, no noise, no backsplash) the conventional shower-shape features separate electrons from truth-interacting protons almost completely, so the question "do the multiscale features add information" cannot be answered by electron-versus-proton classification in this setting; the matched statistics say the dimensions differ strongly between the classes, but a perfectly separating conventional classifier shows that this difference is not independent evidence of added information. The pre-registered added-information design needs an amendment (researcher's decision) before the final opening: candidate designs are (i) the high-rejection regime, which needs about 10^6 protons (a FastMC-scale sample), (ii) a matched design that restricts to events the conventional features cannot separate, (iii) a detector-level setting (noise, angles, unknown energy), (iv) reporting the question as unanswerable here, which is a publishable negative outcome of the roadmap. None is adopted.
- **Disclosed:** the two harder rungs were added after seeing AUC 1.000 and are exploratory; all variants are reported, none selected.
- **100 GeV (added 2026-10-10 after the 100 GeV electrons finished; 10,000 electrons, 9,766 interacting-proton comparison events; `multiscale_gtrain_development_100GeV.json`):** the same pattern. Matched KS of D0, D1, D2 and D0 - D2 over six scales: 0.53 / 0.52 / 0.54 / 0.55 on the deposition grid (eligible 1.00 / 0.83) and 0.51 / 0.44 / 0.37 / 0.41 on the readout grid (eligible 0.75 / 0.71), split-half floor 0.02. The ladder: AUC 1.0000 for the conventional rung and for both harder rungs on the deposition grid (99% rejection above 3,200 even without energy and cells), 0.9999-1.0000 on the readout grid; the multiscale gain is -0.0003 to +0.0001 (intervals within +-0.001), never a registered gain. The four-energy picture (10, 20, 50, 100 GeV) is therefore consistent: strong matched differences in the dimensions, and no headroom for added information in this simulated setting.

### 14. Scope clarified by the researcher (2026-10-10); recorded, no change to the registered test

The researcher stated that the study is not only whether the multiscale (fractal) features improve the AUC. The goals are: (1) whether a fractal feature exists and survives the readout; (2) efficiency, together with information beyond the conventional features (the meaning of "efficiency" is being clarified: sample efficiency, model/compute efficiency, or electron selection efficiency at fixed rejection); (3) possible fractal-inspired architectures for CNNs and vision transformers; (4) detector effects, which will be added.
Against the roadmap: (1) and the sample-efficiency and scale-removal questions are Paper 1; architectures and detector-response robustness are NOT committed (only "dedicated architecture studies", "finite-resolution representation learning" and "detector-response robustness" appear as possible later papers) and need promotion through the research workflow; Paper 2 is the quantum study.
Consequences recorded: (a) the development finding of section 13 (no headroom) applies to ideal Geant4 events at a known beam energy and does not bound a detector-level, realistic-energy setting, where the real-data difficulty reported by Hashmani et al. 2024 (ISS 50-70 GeV) suggests headroom exists; (b) the detector response (readout-level sampling response exists; instrumental effects are blocked on the evidence pass) is on the critical path; (c) architecture comparisons need about 10^6 events (Hashmani et al. trained on about 11.5 million MC events), far beyond the 10^4 per class available, so the FastMC arm and its fidelity to Geant4 become necessary, not optional (this reverses the statement of section 1 that FastMC is off the critical path); (d) the amendment options of section 13 (i) a large sample and (iii) a detector-level setting match this scope. No amendment is made; the decision and the promotion of the new goals are the researcher's.

**Correction, same day (the researcher's challenge, verified against `research/DECISIONS.md`):** the detector response IS an accepted part of the program and does not need promotion. Accepted 2026-10-02 ("Yes to all"): the first validated dataset is the perfect-event (`deposition`) version and the detector-level version is second; the detector response is split into a Geant4-calibrated sampling response (R-A) and the evidence-gated instrumental effects (R-B). R-B stays blocked on the evidence pass (open item: the 2026-10-02 discovery pass was thin and no per-paper outcomes are recorded), and "detector response after proton model step 10" is recorded as the plan. What the roadmap does not contain is detector-response robustness as a separate scientific CLAIM or paper; the dataset layer itself is approved. Only the fractal CNN / vision-transformer architectures are genuinely outside the roadmap and would need promotion. The statement above that "detector effects need promotion" is withdrawn. The three efficiencies (sample, model/compute, electron selection at fixed rejection) are all to be reported (researcher, 2026-10-10: "We have no reason to not check all").
