# Research records

This directory contains the durable scientific state of the project.

It is not a replacement for source code, notebooks, results, or the
researcher's Obsidian knowledge vault.

## Core records

`STATE.md`

: The current human-approved project state. Read this before substantial
scientific work.

`MODULE_MAP.md`

: What each module in `src/ams_ecal/` is for in the model, which test checks it, and whether it is a generator, a calibration,
an analysis, a control or a validation. Start here when a file name is unclear.

`PUBLICATION_ROADMAP.md`

: The accepted multi-paper program. The first paper studies multiscale shower
information with classical ML; QML is deliberately downstream.

`questions/RQ-001_multiscale_shower_information.md`

: Formal statement of the active first-paper research question, hypotheses,
confounders, and evidence requirements.

`CHAOS_FRACTALS_QML.md`

: Broader conceptual framing connecting stochastic particle cascades,
fractal/multifractal ideas, chaos theory and its limits, multiscale ML, and
future QML hypotheses.

## Where information belongs

```text
Obsidian
├── learning sessions
├── concepts
├── derivations
├── literature notes
├── quizbook
└── personal synthesis

Git repository
├── research/STATE.md
├── research/PUBLICATION_ROADMAP.md
├── research/CHAOS_FRACTALS_QML.md
├── research/questions/
├── research/hypotheses/
├── research/experiments/
├── research/decisions/
├── notebooks/
├── src/
├── tests/
└── configs/
```

The project should create research records only when they serve a real
scientific purpose.

---

# Daily research workflow

## 0. Synchronize

Before substantial work:

1. read `RESEARCH_PROTOCOL.md`;
2. read `research/STATE.md`;
3. read `research/PUBLICATION_ROADMAP.md` for publication-level work;
4. read `research/CHAOS_FRACTALS_QML.md` for fractal, multiscale, chaos, QCNN,
   Hamiltonian-embedding, or quantum-reservoir work;
5. inspect recent Git changes;
6. correct `STATE.md` if code or accepted research decisions have moved ahead
   of it.

## 1. Choose the working mode

Pick one dominant mode:

- learning;
- literature;
- derivation;
- experiment design;
- coding;
- analysis;
- review.

Avoid mixing every mode into one task.

## 2. Learning mode

For concepts that affect scientific decisions:

1. open a Tutor Session in Obsidian;
2. probe prerequisite understanding;
3. build a dependency map;
4. teach from legitimate sources;
5. use retrieval/testing;
6. promote durable understanding only after the researcher can reconstruct it.

## 3. Scientific gate before code

Before implementing a new scientific method, establish:

- purpose;
- assumptions;
- mathematical definition;
- physical/statistical meaning;
- units/domain;
- limiting behavior;
- source provenance;
- validation plan.

Do not reopen already accepted decisions without new evidence.

## 4. Independent evidence workflow

For implementation-defining or hypothesis-sensitive questions:

```text
neutral question
→ independent literature discovery
→ source verification
→ counterevidence / adversarial review
→ repository reconciliation
→ human decision
```

Distinguish:

- AMS-specific evidence;
- external result in the relevant regime;
- transferred approximation;
- project phenomenological assumption;
- unresolved question.

## 5. Experiment design

Before seeing results, record:

- research question;
- hypotheses / nulls;
- variables;
- controls;
- confounders;
- metrics;
- uncertainty/statistical plan;
- seeds/configuration;
- reproduction command;
- invalidation criteria.

## 6. Implementation

Reusable logic belongs in `src/ams_ecal/` with tests in `tests/`.

Notebooks should call tested code and serve as:

- scientific narratives;
- derivation records;
- visualization environments;
- validation records.

## 7. Attack the result

Use independent review when useful:

- physics review;
- statistics review;
- source verification;
- reproducibility audit;
- adversarial review.

Do not use an independent reviewer merely to obtain agreement.

## 8. Close the session

Record:

```text
I learned:
...

The evidence says:
...

I still do not know:
...

Decisions made:
...

Next action:
...
```

Only human-accepted conclusions update `research/STATE.md`.

---

# Current project cadence

```text
FastMC engineering             Multiscale research preparation
------------------             -------------------------------
Stochastic generation stochastic EM          scaling laws
Proton model model          fractal dimensions
Detector response      generalized dimensions
Dataset generation datasets               finite-resolution bias
Geant4 reference               estimator validation
             \                 /
              \               /
        RQ-001 preservation study
                    ↓
      Paper 1: information + classical ML
                    ↓
        Paper 2+: justified QML
```

The broader chaos-theory roadmap remains a parallel learning track and must not
be forced into the shower-physics interpretation.

## Guiding rule

**Understanding lives in Obsidian. Evidence and reproducibility live in Git.
The human researcher decides what becomes accepted knowledge or conclusion.**

---

## Record index: the validated-FastMC-dataset branch (2026-10-02 to 2026-10-05)

Branch `fastmc-validated-dataset`. Read in this order for a tutoring session on the processes. "Decision" IDs are in `DECISIONS.md`; commits are in `git log`. The Obsidian vault has no note for this branch yet (its newest research session is 2026-09-29, its newest tutor session 2026-10-02); the repository is the only record of it.

| process | what was done | where it is recorded | commits |
|---|---|---|---|
| plan and approvals | the plan (tracks A-E), the researcher's "Yes to all" on Q1-Q7 | `plans/2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md`, `DECISIONS.md` | |
| pre-registration | the added validation checks (layer correlations, classifier two-sample test, sparsity, ungated multiscale panel), numbers confirmed before any sealed event existed; amended 2026-10-05 | `plans/2026-10-02_added_validation_checks_preregistration.md` | |
| crossing repair | burst latent, lateral spill, bulk coupling; in-sample check | `STATE.md` (branch sections), `results/proton_model/crossing_structure_in_sample.json`; DEC-008, DEC-009 | e822e04, f4b79ea (superseded) |
| sealed Geant4 sets | configs, hash manifests, guard, opening ledger (no opening yet), generation logs | `sealed/README.md`, `sealed/*_manifest.json`, `sealed/generation_log_part*.txt`, `src/ams_ecal/validation/sealed_set.py` | 7f75f73, 467117f, 35bd01f |
| interacting model | per-offset tables, copula, back-edge, upstream albedo, amplitude mapping; lateral quanta; batch path | `STATE.md`, `results/proton_model/interacting_in_sample.json`; DEC-005 (formulation accepted 2026-10-05) | 501b875, 958f6dd |
| harness and dataset | added-checks harness, dataset assembler, checks D1-D9, mutation-tested decision rules; dataset-level registration (amended 2026-10-05) | `plans/2026-10-04_dataset_level_checks_preregistration.md`, `src/ams_ecal/validation/added_checks.py`, `src/ams_ecal/validation/dataset.py` | 1688c2b |
| dry run and repairs | the dry run exposed a deposition lateral shape far from Geant4 (my in-sample check had compared lateral shape for readout only); repairs; artifact frozen `cee305f1` | `results/added_checks/dry_run.json`, `STATE.md` (2026-10-04 later), amendment in the 2026-10-02 registration | 8c8a5b9, 4701165 |
| reviewer agents | physics, simulation, statistics, ML, adversarial, reproducibility, literature scout; my assessment of each finding | `reviews/2026-10-05_reviewer_agents_round1.md` | bfe964b |
| EM out-of-sample check | the DEC-001 EM generator against exploration Geant4 electrons: strong disagreement | `results/em_generator/exploration_comparison.json`, `STATE.md` (2026-10-05) | bfe964b (tag `fastmc-pre-em-repair-2026-10-05`) |
| decisions of 2026-10-05 | DEC-005 accepted; EM: smooth null kept, new production generator; thresholds; protocol wording; sealed proton set stays closed | `DECISIONS.md` (rows of 2026-10-05) | |
| EM production plan | gated plan, no implementation | `plans/2026-10-05_em_production_generator_plan.md` | |
| physics-list systematic | the proton model against QGSP_BERT and QBBC pilot samples (a systematic, not validation) | `results/proton_model/physics_list_systematic.json` | |

Other places that hold records: `RESEARCH_PROTOCOL.md` and `CLAUDE.md` (the rules), `NAMES.md` (translation of old labels), `results/` (every numeric result, each with a status line), the git history (about 80 commits, one per step), the assistant's file memory and claude-mem (what happened, never a substitute for these), and the Obsidian vault `01 Projects/AMS ECAL QML` (evidence maps, research sessions, tutor sessions: the learning record, to be extended for this branch when the tutoring starts).
