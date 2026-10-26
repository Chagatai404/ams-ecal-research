# Research records

This directory contains the durable scientific state of the project.

It is not a replacement for source code, notebooks, results, or the
researcher's Obsidian knowledge vault.

New to the project? Read the repository `README.md` first (plain-language
overview, status, results, next steps, glossary), then the two core records below.

## Core records

`COMPLETED_WORK.md`

: What has been done, tested and decided: results by area with evidence labels, the registered tests, the accepted decisions, corrections of earlier statements, and where things live. Read it first, every session.

`OPEN_DECISIONS.md`

: What is waiting for the researcher (each decision with its options and evidence), and the plan to the machine-learning benchmarks. These two files replace the earlier state, decisions and plan files and all dated plans and reviews; the originals are in the git tag `docs-before-consolidation-2026-10-21` and the Obsidian folder `Research Records`.

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

`NAMES.md`

: Translates old labels (Block N, Slice N, S0 to S8) that remain in dated records.

`NOTEBOOK_COVERAGE.md`

: Generated list of modules that no teaching notebook covers yet. Regenerate it; do not edit it by hand.

`graph/`

: Machine-readable research records (RQ-, H-, EXP-, DEC-) written by the Research OS.
It lags the prose records: it stops at DEC-013 and EXP-009, while `COMPLETED_WORK.md`
cites DEC-014 and EXP-010 to EXP-014. Treat the two prose files as current and the
graph as needing a sync (listed in `OPEN_DECISIONS.md`).

`sealed/`

: Manifests, generation logs and the opening rules for the two sealed Geant4 validation sets (unopened).

`data_manifest.json`

: Where every exposed Geant4 sample lives (nothing is copied; see `validation/sample_definitions.py`).

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
├── research/COMPLETED_WORK.md
├── research/OPEN_DECISIONS.md
├── research/PUBLICATION_ROADMAP.md
├── research/CHAOS_FRACTALS_QML.md
├── research/questions/
├── research/graph/            (hypotheses, experiments and decisions as records)
├── research/sealed/
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
2. read `research/COMPLETED_WORK.md` and `research/OPEN_DECISIONS.md`;
3. read `research/PUBLICATION_ROADMAP.md` for publication-level work;
4. read `research/CHAOS_FRACTALS_QML.md` for fractal, multiscale, chaos, QCNN,
   Hamiltonian-embedding, or quantum-reservoir work;
5. inspect recent Git changes;
6. correct the two files if code or accepted research decisions have moved ahead
   of them.

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

Only human-accepted conclusions update `research/COMPLETED_WORK.md`.

---

# Current project cadence

As of 2026-10-22 the work runs in this order (details in `OPEN_DECISIONS.md`):

```text
done:    detector foundation, FastMC (paused), proton model (development),
         Geant4 reference and pilots, estimators and controls (2D, 3D),
         statistical tools, R-A closed, high-energy pilot
waiting: the 3D control-design decision (blocks the P1 production)
next:    amendment and freeze  ->  production  ->  detector-level dataset v1
         ->  independent reviews  ->  registered classical benchmarks
         ->  sealed validation (once)  ->  Paper 1
later:   Paper 2+, QML on validated structure (nothing started)
```

The broader chaos-theory roadmap remains a parallel learning track and must not
be forced into the shower-physics interpretation.

## Guiding rule

**Understanding lives in Obsidian. Evidence and reproducibility live in Git.
The human researcher decides what becomes accepted knowledge or conclusion.**

---
