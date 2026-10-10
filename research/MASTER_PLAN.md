# MASTER PLAN — the single living plan of the AMS ECAL project

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
| B. Geant4 data | exposed sets DONE; development-large sets DONE (electrons 10,000, protons 20,000 per energy); sealed sets generated, UNOPENED; electron systematics RUNNING |
| C. Multiscale science | DEV readings only (existence strong in ideal Geant4; no headroom for added information in that regime); the registered test and the preservation gate are TODO |
| D. FastMC electron | Slices 1 and 2 DEV; four lateral extensions CLOSED; occupancy mismatch OPEN (AUC 1.000 vs Geant4) |
| E. FastMC proton | implemented, in-sample DEV; sealed validation TODO |
| F. Dataset | assembler and checks D1-D9 exist; no dataset assembled; thresholds pending |
| G. Detector response | R-A TODO; R-B BLOCKED on the evidence pass |
| H. Classical ML and architectures | HGB ladder DEV (rungs 1-3); raw/CNN/ViT baselines and fractal architectures TODO, need ~10^6 events |
| I. QML | gated, nothing started |
| J. Learning and ownership | tutor session open (Q1 unanswered); curriculum A-H planned |
| K. Process | STATE.md "start here" stale since 2026-09-29; research/graph statuses stale; 15 local commits unpushed |

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
| B5 | Development-large sets: electrons 4 x 10,000, protons 4 x 20,000 (seeds 20272001/2), clean commit `48a4a78` | DONE 2026-10-10 | files are in the worktree `../ams-ecal-qml-gen/data`; TODO housekeeping: move or link into `data/`, write a manifest. The name "G-train" is misleading (nothing is trained on it as a model); rename: DECIDE-9 |
| B6 | Electron systematics: 0.1 mm cut (20 GeV done, 100 GeV running), `FTFP_BERT_EMZ` (EM option 4) and `FTFP_BERT_EMV` (option 1) | RUNNING / queued | `FTFP_BERT_TRV` withdrawn (hadronic only) |
| B7 | EXP-007 alternative physics-list calibration (QGSP_BERT) | PARTIAL | `results/proton_model/physics_list_systematic.json` exists (a systematic, not a calibration); graph record says planned: reconcile |
| B8 | Incidence angle, backsplash, material systematic | DECIDE | "before or after the proton model" still open; normal incidence only today |
| B9 | Production-cut sensitivity before any sub-cell claim | partial data (proton 0.1 mm; electron running) | analysis TODO |
| B10 | Real / test-beam / ISS data | none available | only published ISS results (Hashmani et al. 2024: ISS 50-70 GeV, MC 0.2-2 TeV) |

## 5. Workstream C — the multiscale science (RQ-001)

| id | item | status | evidence / next |
|---|---|---|---|
| C1 | H-002 physical difference, e vs interacting p | DEV | matched KS 0.3-0.55 at 10-100 GeV (deposition), `plans/2026-10-09_multiscale_test_dataset_plan.md` sections 11-13 |
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
| G1 | **R-A sampling response** (true deposition -> fibre energy, calibrated on Geant4 and validated on sealed electrons) | **TODO (accepted 2026-10-02)** | the Geant4 files hold both `deposit_grid_mev` and `readout_grid_mev` |
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
| K1 | `STATE.md` "Next session - start here" (dated 2026-09-29) replaced by a pointer to this file | TODO |
| K2 | `research/graph/` statuses: EXP-005/006 (implemented), EXP-009 (finished), H-002..H-005 (DEV readings), EXP-001 (partial), RQ-003/004 | TODO (update in place) |
| K3 | Banners on superseded plan files pointing here (nothing moved or deleted) | TODO |
| K4 | 15 local commits ahead of origin, none pushed | DECIDE-8 |
| K5 | Research OS v0.8.5 is installed but its repository `main` is at v0.8 (merge `e50692b` is on a Codex branch) | DECIDE-8 (sync) |
| K6 | Full test suite last run: 896 passed (2026-10-09); later modules tested individually | TODO (re-run when the CPU is free) |
| K7 | `graphify update .` after code changes | TODO (hook-driven) |

## 14. Decision queue (the researcher's, in the order they block work)

1. **DECIDE-1: amend the pre-registered information test.** Candidates: (i) rejection at fixed efficiency as primary with a ~10^6-event sample; (ii) matched design on events the conventional features cannot separate; (iii) detector-level, realistic-energy setting; (iv) report "unanswerable in the ideal regime"; recommended: (i) + (iii).
2. **DECIDE-2: the sealed sets.** They are registered as the one-opening validation of the FastMC generators (DEC-014 Q11; the added-checks registration); on 2026-10-09 the researcher agreed that the multiscale test could open them first. Both cannot hold. Options: new sealed sets for one purpose (about a day of Geant4), or one registered joint opening.
3. **DECIDE-3: the faster-tail candidate** (override the 0.3-point rule, refine, or keep Slice 2) and whether to continue the occupancy work (D6).
4. **DECIDE-4: the fine-image definition** (projection of 3D voxels, alternating x/y views) so A4, C2, C3 can start.
5. **DECIDE-5: promote the architectures and the three-efficiency study** into the roadmap (research question, hypothesis, paper boundary).
6. **DECIDE-6: start G1 (R-A) and the R-B evidence pass.**
7. **DECIDE-7: confirm the gamma-inadequacy signs (D7); re-register the KS thresholds from a split-Geant4 floor (E5); answer Q9 (F1).**
8. **DECIDE-8: push the 15 commits; sync the Research OS repository.**
9. **DECIDE-9: backsplash / incidence angle / material ordering (B8); rename "G-train" (B5).**

## 15. Critical path and time (estimate, not measurement)

generation finishes (~4-6 h) -> DECIDE-1, 2, 4 -> A3, A4 (fine controls, ~1 day) -> C2 (proton smooth-baseline look, ~1 day) -> development analysis in the amended design (2-3 days) -> frozen commit -> the sealed opening (1 day). In parallel: G1 (R-A) and the R-B evidence pass; the FastMC arm; curriculum teaching (6-8 h). Architectures, efficiency curves and QML wait for the hard regime and a large sample.

## 16. Where the earlier plans went (nothing dropped)

| earlier file | contribution | status of its items |
|---|---|---|
| `2026-09-29_proton_shower_model_implementation_plan.md`, `..._proton_dependency_analysis.md`, `..._proton_model_literature_and_adversarial_reconciliation.md` | proton model structure, factorization, contract (section 8), 45 sections | E1-E7; open: E5-E7 |
| `2026-10-02_weekend_multiscale_check_and_proton_model_completion_plan.md` | estimator gate, proton first look, electrons, sealed sets, learning gates | A1-A5 DONE; A3, A4, C2 never run; detector-response order -> G |
| `2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md` | tracks A-E, Q1-Q7, sealed sets, R-A / R-B | B4, E, F, G |
| `2026-10-02_multiscale_estimator_*` | pre-registration and controls | A |
| `2026-10-02_added_validation_checks_preregistration.md`, `2026-10-04_dataset_level_checks_preregistration.md` | binding rows and thresholds | E3, E5, F1 (pending thresholds) |
| `2026-10-05_em_production_generator_plan.md` and the 2026-10-05..07 EM notes | EM stages 1-4, F1-F5 decisions | D1-D11 (stage 4 TODO) |
| `2026-10-09_multiscale_test_dataset_plan.md`, `..._preregistration.md`, `..._literature_verification_record.md` | the test design, development readings, literature | C, A8; its section 1 claim that FastMC is off the critical path was reversed in its sections 13-14 |
| `CHAOS_FRACTALS_QML.md`, `PUBLICATION_ROADMAP.md` | framing, papers 1-2 | sections 1, 10, 11 |

## 17. Corrections of earlier statements (kept for honesty)

- Fractal architectures: they ARE in the repository (CHAOS_FRACTALS_QML section 6 B, C) as gated candidates, though not in the roadmap papers.
- Detector effects: an accepted track, not something needing promotion.
- FastMC fidelity: not off the critical path once architectures and efficiency are in scope (needs ~10^6 events).
- Generation timings were underestimated (100 GeV electrons took ~6 h, not 4).
