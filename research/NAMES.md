# NAMES — what the old labels meant

New work uses descriptive names only. This file translates the **old labels** that still appear in dated records, vault notes and
file or directory names, so the chronology stays readable. It is a lookup table, not a naming scheme: do not use the old labels in new work.

Chronology is kept by (1) dates in file names and records, (2) the notebook numbers `00` to `08`, (3) numbered steps inside a plan,
and (4) the research records: question, hypothesis, experiment, decision (RQ-, H-, EXP-, DEC-).

## Development order (formerly "Blocks")

| Notebook | Old label | Name now | Approximate date |
|---|---|---|---|
| `00_ecal_calorimetry_and_geometry` | Block 0 | ECAL geometry | before 2026-09-20 |
| `01_tracker_state_and_projection` | Block 1 | tracker projection | before 2026-09-20 |
| `02_readout_orientation_and_cell_mapping` | Block 2 | cell mapping | before 2026-09-20 |
| `03_canonical_event_model` | Block 3 | event model | before 2026-09-20 |
| `04_ecal_geometry_fidelity` | (geometry check) | geometry fidelity | before 2026-09-20 |
| `05_longitudinal_em_shower` | Block 4 | longitudinal profile | before 2026-09-20 |
| `06_lateral_em_shower` | Block 5 | lateral profile | before 2026-09-20 |
| `07_stochastic_em_events` | Block 6A | EM event generator | 2026-09-21 to 2026-09-29 |
| `08_geant4_proton_pilot` | (pilot) | Geant4 proton pilot | 2026-09-29 |
| (not a notebook yet) | Block 6B | proton model | from 2026-09-28 |
| (not started) | Block 7 | detector response | blocked since 2026-09-21 |
| (not started) | Block 8 | dataset generation | planned |
| README roadmap | Blocks 9-14 | Geant4/C++ foundation, ECAL geometry, physics-list selection, primary generation, sensitive detector and export, FastMC-Geant4 validation | planned |

A notebook for the multiscale estimator controls was removed on 2026-10-02: notebooks explain a finished model, so the controls are run by
`python -m ams_ecal.multiscale_estimators.multiscale_controls` and recorded in `research/plans/2026-10-02_multiscale_estimator_controls_results.md`.
(That work had been mislabelled "Block 8a".)

## Steps of the proton model plans (formerly "Slices")

Steps keep their numbers. The pilot plan (2026-09-29) numbered its own steps 1-14; the proton model plan (2026-09-29) numbered steps 0-8.

| Old label | Name now |
|---|---|
| Slice 0 (proton model) | dependency analysis |
| Slice 1 | interaction draw |
| Slice 2 | crossing branch |
| Slices 3-5 | interacting-event model |
| Slice 6 | final validation |
| Slice 7 | alternative physics-list calibration |

## Decisions of 2026-09-29 to 2026-09-30 (formerly "D1" to "D8")

| Old label | Name now | Record |
|---|---|---|
| D1 | interacting-proton factorization | DEC-005 |
| D2 | shared interaction draw | DEC-006 |
| D3 | order of the open checks | (in the dependency analysis) |
| D4 | validation contract | DEC-007 |
| D5 | burst latent | DEC-008 |
| D6 | lateral spill | DEC-009 |
| D7 | added validation checks | DEC-007 |
| D8 | estimator-validity gate | DEC-010 |

## Acceptance rules of 2026-10-02 (formerly "C1" to "C5")

| Old label | Name now |
|---|---|
| C1 | false-positive limit |
| C2 | detection power, and the ordering D0 >= D1 >= D2 |
| C3 | saturation-window rule |
| C4 | recovery reference |
| C5 | verdict |

## Other old labels

| Old label | Name now |
|---|---|
| P7 | depth-dominance hypothesis (first-interaction depth explains most of the variation) |
| Track A / B / C (weekend plan) | estimator controls / proton first look / electron data |
| Stage 18 | post-build teach-back (a Research OS learning step) |
| Stage I / II / III (README) | kept: they have descriptive titles in the README |

## Names kept on purpose (renaming needs approval)

| Kept | Why |
|---|---|
| Record IDs `RQ-`, `H-`, `EXP-`, `DEC-` | required by the research-record format; always read next to a title |
| `MODEL_NAME = "block6b-proton"`, manifest `kind: block6b_proton_calibration`, `data/calibration/proton_6b/`, `results/block6b/` | written into artifacts and provenance; renaming means regenerating the calibration artifact from a clean tree, which was frozen deliberately |
| Comments inside `configs/*.yaml` (including the two electron configs) | the files are hashed into provenance digests that the generated data records; editing a comment changes the digest. They still say "Block 6B", "Slice 0" or "Track C" |
| Vault note titles that contain "Block" | they live in the vault; the repo refers to them by their existing titles |
| The researcher's own verbatim quotes | kept word for word |

## Electron generator slices of 2026-10-05 and 2026-10-06 (formerly "F1" to "F6")

The researcher's decision rows F1-F5 stay in `DECISIONS.md` as record IDs next to a title; files, results and tests no longer carry the labels.

| Old label | Name now | Where |
|---|---|---|
| F1 | structural prior for the electron longitudinal model | `DECISIONS.md` |
| F2 | spot model for the electron lateral structure | `DECISIONS.md` |
| F3 | `deposition` only for the production electron generator | `DECISIONS.md` |
| F4 | conditional per-layer mean correction | `DECISIONS.md` |
| F5 | longitudinal-structure analysis of Geant4 electrons | `src/ams_ecal/electron_studies/em_longitudinal_structure_analysis.py`, `research/plans/2026-10-05_em_longitudinal_structure_analysis_note.md`, `results/em_generator/longitudinal_structure*` (was `f5_*`, `f5/`) |
| F6 | depth-origin and mean-beta calibration | `src/ams_ecal/electron_studies/em_depth_origin_calibration.py`, `research/plans/2026-10-06_em_depth_origin_calibration_note.md`, `results/em_generator/depth_origin_calibration*` (was `f6_*`, `f6/`) |

## Module folders (2026-10-06)

`src/ams_ecal/` was flat; it is now grouped by role (`detector`, `electron_model`, `electron_studies`, `proton_model`, `multiscale_estimators`,
`validation`, `geant4_simulation`) and `tests/` mirrors it. An old `ams_ecal.<name>` is now `ams_ecal.<folder>.<name>`; `MODULE_MAP.md` lists every module.
