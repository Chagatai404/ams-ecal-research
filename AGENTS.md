# AGENTS.md

This repository is a human-led scientific research project.

## Start here

Before substantial work:

1. Read `RESEARCH_PROTOCOL.md`.
2. Read `research/STATE.md`.
3. Read `research/DECISIONS.md` (the master list of accepted decisions).
4. Inspect the relevant source, tests, notebook, and configuration files.
5. Use installed Scientific Research OS skills when the task matches them.

## Authority

The human researcher is the final authority on:

- scientific assumptions,
- research direction,
- interpretation,
- accepting/rejecting hypotheses,
- merging agent work.

Do not silently promote hypotheses or interpretations into accepted facts.

## Agent role assignment

Do not assume a fixed role based on whether you are Claude, Codex, ChatGPT, or another compatible tool.

The human researcher assigns the task role for each session.

Any capable agent may be asked to:

- tutor,
- explain physics or mathematics,
- derive equations,
- search and study literature,
- verify sources,
- form or critique hypotheses,
- design experiments,
- implement scientific code,
- write tests,
- run numerical experiments,
- audit statistics,
- audit physics assumptions,
- review reproducibility,
- perform adversarial review.

Provider identity does not determine scientific responsibility.

Project-local instructions may specify the current task, but they must not be interpreted as permanent provider-specific roles.

## Agent handoff model

For substantial work, prefer one primary agent at a time.

A typical workflow is:

```text
human defines the task
        ↓
one primary agent performs it
        ↓
human reviews the output
        ↓
second agent independently verifies/reproduces/reviews when useful
        ↓
human decides what is accepted
```

Do not use the second agent merely to agree with the first.

When cross-validating:

- give the reviewer the artifact, claim, source, test result, experiment, or acceptance criteria;
- withhold the first agent's full reasoning when practical;
- ask the reviewer to reconstruct the conclusion independently;
- compare conclusions only after the independent pass is complete.

Git remains the shared coordination layer.

## Independent literature discovery

For hypothesis-sensitive or implementation-defining scientific questions, do not inspect implementation assumptions before the initial literature search unless they are required to define the physical system.

During the independent discovery pass:

- formulate the question neutrally;
- use detector/project context only to define scope;
- do not use our preferred equations, implementation names, or desired conclusions as search priors;
- seek competing models, contradictory evidence, null results, and validity limits;
- return the external evidence landscape before reconciling it with the repository.

A later reconciliation pass may compare the independent evidence against:

- `src/ams_ecal/`,
- tests,
- configuration values,
- existing project citations,
- hypotheses and planned models.

Do not silently bend the source interpretation to preserve existing code.

For important implementation-affecting claims, distinguish:

- AMS-specific evidence,
- externally established result in the relevant regime,
- transferred approximation,
- project phenomenological assumption,
- unresolved question.

## Scientific change checklist

Before implementing a new scientific method, state:

- scientific purpose,
- assumptions,
- mathematical definition,
- units/domain,
- expected limiting behavior,
- validation plan,
- source provenance for implementation-affecting assumptions.

For a new research experiment, use the `design-experiment` skill when available.

## Branch / worktree policy

Do not have multiple agents perform substantial writes on the same working tree at the same time.

Because agents are normally used sequentially, separate worktrees are optional rather than mandatory. Use them when:

- preserving an experimental implementation separately is useful;
- requesting an independent implementation;
- comparing alternative approaches;
- substantial unmerged work already exists.

If using separate worktrees, a typical structure is:

```text
ams-ecal-qml/             # human integration worktree
ams-ecal-qml-agent-a/     # optional agent worktree
ams-ecal-qml-agent-b/     # optional independent-review worktree
```

Agents do not merge their own work into the human integration branch.

Before handoff, report:

- commits/files changed,
- tests/checks run,
- scientific assumptions introduced,
- sources relied on,
- unresolved issues.

## Notebook policy

Notebooks are research records, not utility libraries.

If code is reusable:

- move it into `src/ams_ecal/`;
- add tests under `tests/`;
- call it from the notebook.

Do not bury important constants or scientific assumptions in unexplained notebook cells.

## Source policy

Important physics claims require authoritative sources.

For AMS detector facts, prefer official AMS/collaboration material and peer-reviewed AMS publications.

Do not substitute generic calorimeter values for AMS-specific values silently.

Follow the independent-discovery rules in `RESEARCH_PROTOCOL.md` before using the repository's current model as a literature-search prior.

## Secrets

Do not read, print, copy, summarize, commit, or expose credentials, tokens, secrets, or unrelated `.env` values.

If a task genuinely requires a secret-backed command, request the minimum necessary human action rather than inspecting secret contents.
