---
research_schema: 1
type: research-question
id: RQ-001
project: ams-ecal-qml
status: active
created: 2026-09-20
artifacts: ["research/PUBLICATION_ROADMAP.md", "research/CHAOS_FRACTALS_QML.md", "research/STATE.md"]
learning_dependencies: ["multiscale-analysis.scaling-and-box-counting", "multiscale-analysis.estimator-controls-and-finite-scale-bias", "multiscale-analysis.fastmc-versus-geant4-control-logic", "statistics.marginal-versus-joint-distributions"]
---
# RQ-001 — Multiscale information in AMS-02 ECAL-like particle showers

_Status: active research question_

_Last human review: 2026-09-21_

## Research question

> To what extent do electromagnetic and hadronic particle showers exhibit
> discriminative multiscale spatial structure, how much of this structure
> survives AMS-02 ECAL-like segmentation, and what information does it
> contribute to electron/positron-versus-proton classification beyond
> conventional calorimetric observables and standard ML representations?

---

# Why this question exists

Particle showers are stochastic branching cascades.

That makes scale-dependent spatial organization physically plausible, and
highly granular calorimeter studies motivate the use of fractal-sensitive and
multiscale observables.

However, this does **not** establish that:

- AMS-02 ECAL showers are ideal mathematical fractals;
- a stable multifractal spectrum exists;
- AMS segmentation preserves enough scales to estimate one;
- a fractal variable adds information beyond ordinary shower morphology.

Those are empirical questions.

---

# Hypotheses

## H1 — physical multiscale difference

Detailed electromagnetic and proton-induced showers exhibit measurably
different scale-dependent spatial statistics over at least part of the
physically accessible scale range.

## H2 — detector survival

A nontrivial part of those differences survives projection onto an AMS-02
ECAL-like readout.

## H3 — incremental information

Validated multiscale descriptors provide particle-ID information beyond
conventional observables such as:

- shower maximum;
- longitudinal width;
- lateral width;
- containment;
- leakage;
- total energy;
- concentration / sparsity.

## H4 — useful inductive bias

Explicit multiscale information may improve sample efficiency or low-data
performance relative to comparable models that must infer the same structure
implicitly.

---

# Null / competing explanations

Apparent multiscale differences may instead be caused by:

- primary-energy mismatch;
- incidence-angle mismatch;
- containment or detector-boundary effects;
- ordinary shower width/depth;
- finite segmentation;
- thresholds/noise;
- preprocessing;
- track-centering choices;
- unstable finite-resolution estimators;
- simulator-specific artifacts.

---

# Evidence hierarchy

The decisive physical test must not rely on smooth FastMC alone.

Preferred hierarchy:

```text
real test-beam / detector data where legitimately available
                ↑
AMS-like projection of detailed Geant4 transport
                ↑
fine-grained Geant4 transport
                ↑
FastMC as development / smooth-control simulation
```

Geant4 is still a model rather than direct reality.

Real detector/test-beam information remains valuable when available.

---

# Fractal-preservation gate

Analyze the same detailed showers at three levels:

1. fine-grained Geant4 deposition/hit structure;
2. the same events projected to the canonical AMS-like `18 × 72` readout;
3. FastMC events in that canonical representation.

Interpretation:

### Case A

```text
fine Geant4        signal
AMS projection     no signal
FastMC             no signal
```

The relevant scale information is destroyed by detector segmentation.

### Case B

```text
fine Geant4        signal
AMS projection     signal
FastMC             no signal
```

FastMC masks physically accessible multiscale structure.

### Case C

```text
fine Geant4        signal
AMS projection     signal
FastMC             comparable signal
```

FastMC preserves enough of the tested multiscale structure for that purpose.

Ordinary agreement in mean profile, containment, or energy resolution is not
sufficient to pass this gate.

---

# Candidate observables

Only after estimator validation and finite-scale stability checks:

- box-counting-style dimension;
- generalized dimensions `D_q`;
- partition functions `Z_q(epsilon)`;
- scaling exponent `tau(q)`;
- multifractal-spectrum summaries where justified;
- lacunarity;
- entropy/concentration across scales;
- occupancy under controlled coarse-graining;
- energy moments under controlled coarse-graining.

No observable is accepted simply because it has "fractal" in its name.

---

# Planned analysis logic

1. learn and verify the mathematics;
2. validate estimators on controlled synthetic structures;
3. characterize finite-resolution bias;
4. generate matched detailed electron/proton Geant4 events;
5. analyze fine-grained transport;
6. project the same events into AMS-like readout;
7. quantify which observables survive segmentation;
8. compare against conventional calorimeter features;
9. test incremental information statistically and predictively;
10. benchmark classical ML with and without explicit multiscale features;
11. study low-data/sample-efficiency behavior;
12. only then promote a representation into the later QML program.

---

# Relationship to FastMC

FastMC remains important infrastructure for:

- pipeline development;
- scalable baseline generation;
- controlled ablations;
- debugging preprocessing;
- debugging training code;
- a smooth phenomenological control.

FastMC is **not** evidence for or against the existence of microscopic cascade
self-similarity because its parameterized longitudinal/lateral profiles do not
explicitly model the branching hierarchy that could generate that structure.

This is not a failure of FastMC. It is part of the experimental design.

---

# Relationship to QML

RQ-001 is deliberately answered before QML architecture selection.

A later quantum study should start from a validated statement about the data,
for example:

> useful particle-ID information is distributed hierarchically across spatial
> scales and survives AMS-like detector readout.

Only then should QCNNs, quantum kernels, VQCs, or scale-structured Hamiltonian
embeddings be tested as candidate inductive biases.

QML must not be used as evidence that RQ-001 is true.
