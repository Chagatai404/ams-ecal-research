# CLAUDE.md

This repository is a human-led scientific research project.

## Start here

Before substantial work:

1. Read `RESEARCH_PROTOCOL.md`.
2. Read `research/STATE.md`.
3. Read `research/DECISIONS.md` (the master list of accepted decisions).
4. Inspect the relevant source, tests, notebook, and configuration files.
5. Use installed Scientific Research OS skills when appropriate.

## Authority

The human researcher is the principal investigator and learner.

Do not silently convert:

- an analogy into a mechanism,
- a hypothesis into a fact,
- an AI explanation into the user's permanent understanding,
- an exploratory result into an accepted conclusion.

## Role assignment

Do not assume that Claude has a fixed role in this project.

The human researcher assigns the task role for each session.

Claude may be asked to perform any compatible research task, including:

- tutoring,
- physics or mathematics explanation,
- derivation,
- literature discovery,
- source verification,
- hypothesis formation,
- experiment design,
- scientific coding,
- test writing,
- numerical experiments,
- statistical review,
- physics review,
- reproducibility review,
- adversarial review.

Do not infer that another provider, including Codex, has a complementary or predefined role.

Provider identity does not determine scientific responsibility.

## Agent handoff model

For substantial work, prefer one primary agent at a time.

A typical workflow is:

```text
human defines the task
        ↓
one primary agent performs it
        ↓
human review
        ↓
second agent independently verifies/reproduces/reviews when useful
        ↓
human decision
```

When Claude is the primary worker, complete a coherent task before handing off.

When Claude is acting as a reviewer, reconstruct the scientific, statistical, or implementation justification independently where practical rather than inheriting the first agent's reasoning.

Cross-validation means independent reconstruction, verification, or reproduction. It does not mean agreement checking.

## Independent literature discovery

For hypothesis-sensitive, model-selection, or implementation-defining questions, separate:

```text
independent discovery
→ source verification
→ adversarial counterevidence
→ repository reconciliation
```

During the independent discovery pass:

- formulate the scientific question neutrally;
- use only project context required to define the detector/system/regime;
- do not inspect or rely on implementation-specific equations, parameterizations, current source choices, or desired conclusions unless they are genuinely needed to scope the search;
- search for alternative models, contradictory findings, null results, and validity limitations;
- do not use the names of our current implementation choices merely because they already exist in the repository.

If repository files were already read before the literature task began, explicitly isolate the discovery task with a fresh subagent/context and provide only the neutral question and minimum scope.

After the external evidence landscape is established, perform a separate reconciliation pass against the current repository.

A disagreement with the codebase is evidence to investigate, not something to smooth over.

For implementation-affecting assumptions, classify the support as:

- AMS-specific evidence,
- externally established in the relevant regime,
- transferred approximation,
- project phenomenological assumption,
- unresolved.

## Live learning / Obsidian

When the Scientific Research OS Obsidian tooling is configured:

- use the active tutor/research session note as the live learning record;
- log substantial explanations, source sets, dependency maps, quizzes, and unresolved questions;
- do not automatically promote AI-written text into permanent concept notes;
- let the user restate/rewrite durable understanding.

## Branch / worktree policy

Do not perform substantial writes in the same worktree currently being used by another agent.

Separate worktrees are optional for sequential work, but recommended for independent implementations or preserved alternatives.

If used:

```text
ams-ecal-qml/             # human integration worktree
ams-ecal-qml-agent-a/     # optional agent worktree
ams-ecal-qml-agent-b/     # optional independent-review worktree
```

Do not merge your own branch into the human integration branch unless the human explicitly asks you to do so.

Before handoff, report:

- commits/files changed,
- tests/checks run,
- scientific assumptions introduced,
- sources relied on,
- unresolved questions.

## Scientific source rules

Use `RESEARCH_PROTOCOL.md`.

For important scientific claims:

- favor primary research and authoritative sources;
- record exact pages/sections/equations/figures when possible;
- verify claims that materially affect the implementation;
- conduct independent discovery before reconciling with current repository assumptions.

For AMS-specific detector facts, prefer official AMS/collaboration sources.

When an important source was found because it already existed in the repository, label it as an inherited project citation rather than presenting it as independently discovered evidence.

## Research workflow

Do not assume a fixed sequence solely from provider identity.

Follow the task-specific workflow chosen by the human researcher and the relevant Scientific Research OS skill.

When learning is part of the task:

1. probe the researcher's current understanding;
2. identify missing prerequisites;
3. teach from the deepest missing dependency;
4. verify understanding through retrieval/reconstruction;
5. return to the research task only after required gaps are closed.

When literature determines a scientific implementation:

1. formulate a neutral question;
2. perform independent external discovery;
3. independently verify implementation-affecting claims;
4. seek counterevidence and competing models;
5. reconcile the evidence with the repository;
6. formulate the scientific model;
7. obtain human approval;
8. only then implement.

For the current multifractal direction:

1. learn/verify the mathematics;
2. define estimators;
3. validate estimators on controlled cases;
4. characterize electron/proton showers;
5. control energy and geometry;
6. quantify uncertainty;
7. compare with conventional observables;
8. only then design specialized ML/QML architectures.

For the stochastic generation stochastic FastMC work:

1. probe the researcher's current understanding;
2. identify missing prerequisites;
3. independently review the external shower-fluctuation literature;
4. independently verify implementation-affecting claims;
5. reconcile external evidence with the deterministic FastMC;
6. formulate the stochastic physical model;
7. obtain human approval;
8. only then begin implementation.

## Secrets

Do not inspect, reveal, summarize, or commit credentials, tokens, private keys, or unrelated environment secrets.
