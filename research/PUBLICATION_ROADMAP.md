# Publication roadmap

_Last human review: 2026-09-21. Status note added 2026-10-26 (not a human review of the plan itself)._

**Where Paper 1 stands (2026-10-22).** Done: the simulation and estimator
infrastructure, the statistical tools, a critical program redesign (slices S0 to
S8, validity gates SV-1 to SV-10), several negative or exploratory readings in
the ideal Geant4 regime (electron/proton separation is already near perfect, so
the registered 0.01 AUC-gain rule cannot be met), and a high-energy pilot.
Waiting: the researcher's choice of 3D control design, then the DECIDE-1
amendment (five named endpoints P1 to P5), production, detector-level dataset v1,
independent reviews and the registered benchmarks. Nothing for Paper 2 has
started. See `OPEN_DECISIONS.md` for the plan and `COMPLETED_WORK.md` for the
evidence.

This repository is the shared research codebase for a multi-paper AMS-02 ECAL
research program.

Papers share detector geometry, event schemas, simulation, preprocessing,
validation, and analysis infrastructure. The default convention is therefore
one evolving repository with publication-specific frozen releases rather than
one repository per paper.

---

# Paper 1 — Multiscale information in particle showers

## Primary objective

> Determine whether electromagnetic and hadronic particle showers exhibit
> discriminative multiscale spatial structure, determine how much of that
> structure survives AMS-02 ECAL-like segmentation, and quantify what
> information it contributes to particle classification beyond conventional
> calorimetric observables and standard raw-data ML representations.

This paper is intentionally **pre-QML**.

It must not assume in advance that AMS showers are mathematical fractals or
multifractals.

Fractal dimension, generalized dimensions, partition-function scaling,
multifractal spectra, lacunarity, entropy, and related quantities are candidate
finite-scale descriptors whose validity must first be established.

## Core evidence path

```text
detailed cascade physics
        ↓
fine-grained Geant4 energy deposition
        ↓
multiscale/fractal-sensitive analysis
        ↓
projection to AMS-like 18 × 72 readout
        ↓
does the signal survive segmentation?
        ↓
incremental-information study
        ↓
classical ML benchmarks
```

FastMC is a smooth phenomenological control and development simulator.

It cannot by itself establish or falsify microscopic fractal/multifractal
cascade structure because its parameterizations intentionally replace
microscopic branching with smooth probability profiles.

## Required simulation comparison

At minimum compare:

1. fine-grained Geant4;
2. the same Geant4 events after AMS-like readout projection;
3. FastMC in the same canonical representation.

This distinguishes:

- structure present in detailed transport but destroyed by detector
  segmentation;
- structure surviving detector segmentation but absent from FastMC;
- relevant multiscale observables preserved sufficiently by FastMC.

## Classical information / ML ladder

Planned comparison:

1. conventional calorimeter physics features;
2. validated multiscale features only;
3. conventional + multiscale features;
4. raw AMS-like representation with compact classical baselines;
5. raw representation + explicit multiscale features.

Questions:

- Do multiscale variables add information conditional on conventional shower
  observables?
- Can raw ML infer the same information implicitly?
- Does explicit multiscale structure improve sample efficiency?
- Are gains strongest in low-data regimes?
- How does performance change as scale information is deliberately removed?

The goal is not simply the highest classification score. The paper is about
the **information content of shower structure**.

## Desired outcome categories

A strong result can be positive or negative.

Examples:

- multiscale structure survives AMS readout and adds independent PID
  information;
- structure survives but conventional/raw ML already captures nearly all of it;
- structure exists in fine transport but AMS segmentation destroys it;
- proposed fractal estimators are unstable at AMS granularity;
- apparent differences reduce to conventional width/depth variables.

Any of these can be publishable if established rigorously.

---

# Paper 2 — QML on validated multiscale structure

Only after Paper 1 establishes what multiscale information exists, survives
readout, and is useful should quantum architectures be tested against that
structure.

Candidate directions:

- compact validated multiscale inputs for VQCs;
- quantum kernels with matched classical kernels;
- QCNN/hierarchical circuits as multiscale inductive biases;
- detector-topology-aware quantum connectivity;
- scale-structured Hamiltonian embeddings.

Every quantum model must be compared with strong classical controls receiving
the same information and comparable tuning/resource budgets.

A detector-specific quantum architecture is a conditional research outcome,
not a predetermined deliverable.

---

# Possible later papers

Potential later studies may include:

- FastMC–Geant4 domain transfer;
- detector-response robustness;
- finite-resolution representation learning;
- dedicated quantum architecture studies;
- quantum-hardware execution if scientifically justified.

These are not committed publications until promoted through the research
workflow.

---

# Repository and reproducibility convention

Keep one evolving research-program repository while papers share substantial
infrastructure.

For each paper:

1. create a dedicated paper/experiment directory once the study becomes
   concrete;
2. freeze configurations;
3. record seeds and dataset provenance;
4. preserve figure-generation code;
5. preserve exact experiment manifests;
6. create a Git tag/release for the submitted/published version;
7. archive the tagged release with a persistent DOI service such as Zenodo;
8. cite the exact version in the manuscript.

Create a separate repository only if a later study becomes genuinely
independent in scientific scope, codebase, collaborators, or release lifecycle.
