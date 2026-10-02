# Research records

This directory contains the durable scientific state of the project.

It is not a replacement for source code, notebooks, results, or the
researcher's Obsidian knowledge vault.

## Core records

`STATE.md`

: The current human-approved project state. Read this before substantial
scientific work.

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
