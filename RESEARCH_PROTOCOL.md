# AMS ECAL QML — Research Protocol

This repository follows the [Scientific Research OS](https://github.com/Chagatai404/scientific-research-os) workflow.

The human researcher is the final authority on:
- scientific assumptions,
- interpretation of evidence,
- research direction,
- promotion of claims into accepted conclusions.

AI agents may explain, search, propose, implement, test, reproduce, and critique. They must not silently convert a hypothesis or interpretation into an accepted fact.

## 1. Scientific scope

The project studies AMS-02 ECAL particle-shower representations for electron/positron versus proton classification, including:

- physically faithful ECAL simulation and preprocessing,
- shower morphology and longitudinal development,
- multiscale / multifractal shower observables,
- classical ML baselines,
- geometry-aware models,
- quantum ML and QCNN-style multiscale models.

Current research priority:

> Determine whether electromagnetic and proton showers exhibit measurably different multiscale or multifractal structure in AMS ECAL after controlling for energy and geometry.

QML remains a computational research branch. It is not assumed to provide an advantage.

For the complete conceptual framing of chaos, fractals, multifractals, and QML in this project, read:

```text
research/CHAOS_FRACTALS_QML.md
```

## 2. Source policy

For detector facts and geometry:

1. Official AMS documentation and AMS collaboration material are the default authority.
2. Peer-reviewed AMS publications supersede informal summaries when they explicitly specify a value or convention.
3. Generic calorimeter literature may motivate methods, but it must not silently replace AMS-specific detector values.

For physics and mathematical claims:

1. Primary peer-reviewed literature.
2. Academic textbooks / monographs.
3. High-quality review articles.
4. Official scientific or institutional material.
5. University course material.
6. Reputable technical websites.
7. Videos only as teaching supplements unless independently verified.

Record exact page, section, equation, figure, DOI, or stable URL when practical.

### Independent literature discovery and bias control

For questions that may determine a physics model, stochastic parameterization, estimator, architecture, or scientific conclusion, do not let the current codebase become the default search prior.

Use the following sequence:

```text
neutral scientific question
        ↓
independent external discovery
        ↓
independent source verification
        ↓
adversarial counterevidence search
        ↓
reconciliation with repository assumptions/code
        ↓
human decision
```

During the independent discovery pass:

- use only AMS/project context necessary to define the detector, material, geometry, energy range, particle type, or observable;
- do not expose implementation details, preferred equations, existing parameterizations, current project citations, or desired conclusions unless they are genuinely required to scope the search;
- search for competing models, contradictory evidence, null results, and validity limits as well as supportive work;
- do not use search queries that merely restate the names of our existing implementation choices when those choices are under evaluation.

For important claims, source verification should be independent of the original scout's interpretation when practical. Give the verifier the claim, strongest source(s), and minimum regime context rather than the first agent's reasoning.

Only after independent discovery and verification should the evidence be compared with the current FastMC, multifractal hypothesis, or proposed QML architecture.

A mismatch between external evidence and the codebase is a research finding. Do not reinterpret the source to preserve the implementation.

For implementation-affecting assumptions, label the provenance explicitly as one of:

- AMS-specific evidence,
- externally established in the relevant regime,
- transferred approximation from another detector/material/energy regime,
- project phenomenological assumption,
- unresolved.

Cross-validation means independent reconstruction, verification, or reproduction. It does not mean asking a second agent whether it agrees with the first.

## 3. Epistemic labels

Keep these distinct:

- **Established fact** — supported by reliable evidence in the relevant regime.
- **Model assumption** — chosen simplification or parameterization.
- **Hypothesis** — falsifiable claim under test.
- **Interpretation** — inference from evidence.
- **Open question** — unresolved.
- **Speculation** — interesting idea without adequate evidence.

Never promote a claim between categories silently.

## 4. Evidence ladder

Internal project maturity labels:

- **E0** — idea / speculation.
- **E1** — theoretical argument.
- **E2** — toy simulation.
- **E3** — controlled simulation.
- **E4** — realistic detector simulation.
- **E5** — real data.
- **E6** — independent reproduction.

These labels track how directly this project has tested a claim; they are not a universal ranking of scientific truth.

## 5. Experiment contract

Before running an experiment, record:

1. Research question.
2. Hypothesis and null/alternative where meaningful.
3. Prediction if H1 is true.
4. Prediction if H0/alternative is true.
5. Independent, dependent, and controlled variables.
6. Confounders.
7. Dataset/simulation population and sampling.
8. Metrics chosen before seeing the result.
9. Statistical analysis plan.
10. Random seed policy.
11. Sanity checks.
12. Reproduction command.
13. Git commit and configuration.

Negative and inconclusive results are retained.

## 6. Physics checks

Before accepting a physics result, check:

- units and dimensional consistency,
- coordinate conventions,
- detector geometry,
- energy and angular regime,
- conservation/limiting behavior where applicable,
- parameterization validity range,
- stochastic versus deterministic interpretation,
- whether simulation simplifications are labeled assumptions.

Do not call stochastic shower development "chaos" without a mathematically justified dynamical-systems definition.

## 7. Chaos / fractal / multifractal boundary

For this project:

- fractal and multifractal shower structure is an active hypothesis;
- deterministic chaos is not an established property of AMS ECAL shower development;
- visual irregularity, branching, or fractal-looking geometry is not sufficient evidence of chaos;
- Lyapunov exponents or attractor language require an explicitly defined dynamical system;
- the term "effective multifractal" should be preferred over ideal-fractal language when discussing finite detector resolution;
- finite-size / finite-resolution bias must be quantified before interpreting dimensions physically.

Chaos theory may later provide mathematical tools or motivate separate neural-dynamics / quantum-chaos research, but it must not be inserted into AMS claims without a precise model.

## 8. Multifractal research rules

For shower multifractality:

- distinguish finite-resolution effective scaling from an ideal mathematical fractal;
- quantify the available scale range imposed by AMS ECAL granularity;
- condition or stratify by primary energy and incidence geometry;
- test sensitivity to thresholds, noise, cell granularity, and patch definition;
- compare against conventional shower-shape observables;
- report estimator bias and uncertainty;
- treat \(D_q\), \(\tau(q)\), \(f(\alpha)\), and lacunarity as observables to validate, not assumed truths.

The first core hypothesis should be tested before constructing a specialized neural architecture around it.

## 9. ML / QML comparison rules

- Classical and quantum models must receive comparable information.
- Report strong classical baselines.
- Do not claim quantum advantage from a single benchmark win.
- Control parameter count / model capacity where feasible.
- Separate representation benefit from classifier benefit.
- Report variance across seeds where training stochasticity matters.
- Avoid tuning on the test set.
- For QML, record qubit count, encoding, circuit depth, shots, optimizer, and simulator/hardware details.
- Any claimed advantage must identify the resource or inductive-bias mechanism being tested.

Specific candidate QML ideas currently allowed as hypotheses include:

1. compact multifractal observables as a physically motivated low-dimensional quantum input;
2. QCNN-style hierarchical pooling as a possible multiscale inductive bias;
3. scale-structured Hamiltonian / coupling maps based on detector topology.

These are hypotheses to compare against matched classical controls, not assumed benefits.

Quantum chaos / quantum reservoir computing is a separate later-stage research branch and is not required for the first AMS multifractal study.

## 10. Code / notebook boundary

Reusable scientific methods belong in:

```text
src/ams_ecal/
```

with tests in:

```text
tests/
```

Jupyter notebooks should:

- state the scientific question,
- call tested reusable code,
- document the experiment,
- visualize results,
- record interpretation.

A notebook should not be the only implementation of a reusable numerical method.

## 11. AI collaboration rules

Claude and Codex should:

- read `research/COMPLETED_WORK.md` and `research/OPEN_DECISIONS.md` before substantial work;
- read this protocol before proposing scientific conclusions;
- read `research/CHAOS_FRACTALS_QML.md` when working on multiscale, fractal, chaos, neural-architecture, or QML questions;
- use installed Scientific Research OS skills when appropriate;
- state assumptions before implementation;
- leave final interpretation to the human researcher;
- report uncertainty and conflicting evidence;
- avoid reading credentials or unrelated secrets.

Claude and Codex are used interchangeably as primary workers rather than as simultaneous co-authors by default. One agent should own a substantial task at a time. The other may later perform an independent verification, implementation review, source check, or reproduction.

For independent review:

- prefer a separate subagent/context from the authoring agent;
- do not preload the reviewer with the author's full reasoning when independence is important;
- give the reviewer the artifact, claim, source, experiment, or acceptance criteria needed to reconstruct the issue independently;
- compare independent conclusions only after both are complete.

## 12. Research state updates

`research/COMPLETED_WORK.md` is the shared record of what has been done, tested and decided, and `research/OPEN_DECISIONS.md` the shared record of what is waiting for the researcher.

Agents may propose updates, but only human-accepted conclusions should enter:

- Established facts
- Accepted decisions
- Current hypothesis status

Speculative ideas belong under Open Questions / Candidate Ideas until accepted.
