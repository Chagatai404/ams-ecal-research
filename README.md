# AMS-02 ECAL Research

A human-led research program on particle showers in the electromagnetic
calorimeter (ECAL) of the AMS-02 experiment. It asks whether the *multiscale
spatial structure* of a shower carries information that helps tell electrons
from protons, and whether that information survives the detector's coarse
readout. Classical machine learning comes first; quantum machine learning (QML)
is deliberately last.

> **Status as of 2026-10-22 (the date of the latest records).** Everything
> below is simulation (Geant4 11.4.1 or a fast parametric Monte Carlo), never
> AMS data. No result in this repository is called *validated*: the sealed test
> sets are unopened. The honest headline so far is mostly negative or
> inconclusive (section 5). The next step waits on a decision by the researcher
> (section 6).

If you read only one more file, read
[`research/COMPLETED_WORK.md`](research/COMPLETED_WORK.md) (what was done) and
[`research/OPEN_DECISIONS.md`](research/OPEN_DECISIONS.md) (what is waiting and
what comes next). This README is the guide to them.

## Contents

1. [The project in plain language](#1-the-project-in-plain-language)
2. [The detector and the data representation](#2-the-detector-and-the-data-representation)
3. [How the research works](#3-how-the-research-works)
4. [The simulation stack](#4-the-simulation-stack)
5. [What has been done](#5-what-has-been-done)
6. [What is planned next](#6-what-is-planned-next)
7. [Papers](#7-papers)
8. [Repository map](#8-repository-map)
9. [Setup and commands](#9-setup-and-commands)
10. [Where to read next](#10-where-to-read-next)
11. [Glossary](#11-glossary)
12. [Independence and limitations](#12-independence-and-limitations)

---

## 1. The project in plain language

**The detector.** AMS-02 is a particle detector on the International Space
Station. Its ECAL is a stack of lead and scintillating fibres, about 17
radiation lengths deep. A particle entering it starts a *shower*: a cascade of
secondary particles whose energy is deposited in the fibres and read out as 1,296
numbers (18 layers by 72 cells).

**The job.** Cosmic-ray electrons and positrons are rare; protons are far more
abundant (by orders of magnitude). To measure electrons, the analysis must reject
protons at a level of order 10^3 to 10^4 while keeping most electrons (published
ECAL-only figures are in section 5). Electron and
proton showers look different: electron showers are compact and
electromagnetic, proton showers are broader and often start deep or not at all.

**The research question (RQ-001).**

> To what extent do electron/positron and proton showers show discriminative
> multiscale spatial structure, how much of it survives AMS-02 ECAL-like
> readout, and what does it add beyond conventional calorimetric observables
> and standard machine learning?

"Multiscale structure" means: how the deposited energy is concentrated or
spread when you look at the shower through boxes of different sizes. Fractal
and multifractal dimensions, generalized dimensions D_q and lacunarity are
candidate descriptions. The project does **not** assume in advance that showers
are mathematical fractals, that a stable multifractal spectrum can be measured
at AMS granularity, that QML will beat classical methods, or that the fast
Monte Carlo reproduces microscopic cascade structure. Negative results count.

**The program order (accepted).**

```text
detector and event foundation  ->  simulation (fast MC + Geant4)
   ->  does multiscale structure exist?  ->  does it survive AMS-like readout?
   ->  what information does it add?     ->  classical ML benchmarks
   ->  QML on whatever structure is validated
```

**Four strands of scope** (researcher, 2026-10-10): (1) does a fractal-type
feature exist and survive the readout; (2) efficiency, in three senses (sample
efficiency, model and compute efficiency, electron selection at fixed proton
rejection), together with information beyond conventional features; (3)
fractal-inspired CNN and vision-transformer architectures, as gated candidates;
(4) detector effects as an accepted track.

## 2. The detector and the data representation

The classification target is **e± versus p**, not e+ versus e-: charge sign
comes from the tracker, not from the ECAL.

The canonical event is **18 longitudinal samplings x 72 transverse cells**.
The 18 x 72 image is a stack of alternating one-dimensional views (successive
layers measure x and y alternately), not a 72 x 72 plane. All code uses the same
`ECALEvent` record whether the event comes from the fast Monte Carlo or from
Geant4.

| Quantity | Nominal value |
|---|---:|
| Active readout area | 648 x 648 mm² |
| Active depth | 166.5 mm |
| Superlayers / readout samplings | 9 / 18 |
| Cells per sampling / total cells | 72 / 1296 |
| Cell pitch | 9 mm |
| Electromagnetic depth | 17 X₀ |
| Hadronic depth | about 0.6 λᵢ |
| Effective density / critical energy | 6.8 g/cm³ / 7.6 MeV |

Coordinates: origin at the centre of the front face, +z into the ECAL. The
geometry is configuration (`configs/geometry.yaml`), not hard-coded, and is
kept separate from detector response. Invariants such as 18 x 72 = 1296 and
9 x 18.5 mm = 166.5 mm are checked in code.

Two representations are used (decision DECIDE-4): the **view-aware readout** at
detector level, and **3D sparse voxels** (216 x 216 x 36 mesh) as the
truth-level fine reference.

## 3. How the research works

This is a human-led project. The researcher is the principal investigator and
decides what becomes accepted knowledge. AI agents do tasks the researcher
assigns and never silently turn an analogy into a mechanism, a hypothesis into a
fact, or an exploratory result into a conclusion. The full rules are in
[`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md), [`CLAUDE.md`](CLAUDE.md) and
[`AGENTS.md`](AGENTS.md).

**Evidence labels.** Every result in the records carries one:

| label | meaning |
|---|---|
| ESTABLISHED | checked against a primary source, or passed a test registered before it was run |
| DEVELOPMENT | measured on exposed (development) data; not validated; never a final claim |
| EXPLORATORY | not powered or registered; no claim follows |
| NEGATIVE | a registered test that gave no usable result |
| CLOSED | tried under a pre-stated rule and not adopted; kept as a result |
| GAP | a literature fact or number not verified |

**Source labels** for implementation-affecting assumptions: AMS-specific
evidence, externally established in the relevant regime, transferred
approximation, project phenomenological assumption, unresolved.

**Pre-registration.** Tests, thresholds and adoption rules are written down and
hashed *before* the numbers they govern are produced, and are never relaxed
afterwards. [`COMPLETED_WORK.md`](research/COMPLETED_WORK.md) section 5 lists
each registration with its SHA-256 prefix.

**Sealed sets.** Two Geant4 sets (7 energies; 3,000 protons and 1,000
electrons per energy) were generated and sealed on 2026-10-04. They can be
opened once, through `ams_ecal.validation.sealed_set.open_set`, with a stated
purpose and a registered harness. Neither has been opened. See
[`research/sealed/README.md`](research/sealed/README.md).

**Other standing rules.** The classifier AUC is a diagnostic, never a tuning
target. Held-out events (`event_index % 4 == 3`) are not used for fitting.
Independent literature discovery precedes reconciliation with the repository.
Names describe the work. Notebooks are written only after a model is final.
Commits are made only when the researcher asks.

**Where knowledge lives.** The repository is the source of truth for the
implementation and the evidence. The researcher's Obsidian vault holds
understanding, learning sessions and curated notes. They are not duplicated into
each other.

## 4. The simulation stack

### Detector and event foundation (complete)

Geometry, tracker projection, alternating-readout cell mapping and the canonical
`ECALEvent` (event ID, particle truth, energy, `TrackState`, validated geometry,
non-negative 18 x 72 cell energies, provenance with backend, version,
configuration hash and seed, schema version). Notebooks `00` to `04` teach these.

### Fast Monte Carlo (FastMC)

A transparent, cheap, parametric generator. Its role: pipeline development,
ablations, comparison with Geant4, and a **smooth control** for multiscale
statistics. It cannot establish or falsify microscopic cascade structure,
because it replaces branching by smooth profiles.

- **Mean model.** Gamma longitudinal profile `f(t) = β (βt)^(α-1) exp(-βt)/Γ(α)`
  with `α = 1 + βT`, integrated over each readout layer; lateral density from
  the AMS test-beam form `ρ(r) = 3R²/[π(r+R)⁴]`; leakage retained.
- **DEC-001 stochastic electron generator** (`electron_model/stochastic.py`,
  notebook `07`). One lognormal depth variable `T0` per event reshapes the
  profile coherently; `β = 0.65` fixed as in AMS; a `regime` switch
  (`deposition` or `sampling`) pairs mean depth and fluctuation width; the
  true-deposition event is recoverable from the same seed. Kept permanently as
  the smooth null (`em_smooth_null`). AMS publishes no mean-depth formula, so
  the offsets are external choices.
- **`em_production`** (DEC-014) supersedes DEC-001 for production: calibrated
  longitudinal generator (slice 1) and Grindhammer-Peters lateral spot model
  (slice 2). It is a development candidate that does not match Geant4 on cell
  occupancy. Under DECIDE-3 FastMC refinement is paused and no FastMC sample is
  used in any performance claim.
- **Proton model** (`proton_model/`): analytic interaction depth plus empirical
  conditional distributions; a crossing branch (protons that do not interact)
  and an interacting branch (per-offset tables with a Gaussian copula). A
  calibrated phenomenological surrogate, not hadronic physics. The sealed
  validation has not run.

### Geant4 reference simulation (complete and in use)

Geant4 11.4.1 through `geant4-pybind` (`geant4_simulation/geant4_backend.py`):
the AMS-like ECAL image (scintillating fibres only, 43,155 fibres, no overlaps,
mass 475.4153 kg), step-based truth, bit-for-bit reproducible events, fine 3D
voxel output, optional component tags (seven channels of where a deposit came
from), and seeds that are a pure function of (base seed, energy). FTFP_BERT is
the default physics list; QBBC and QGSP_BERT are used as systematics.

### Detector response

- **R-A (deposit to fibre energy)**: class tables of the response ratio.
- **R-B (instrumental effects)**: photoelectrons, gain, cross-talk, threshold;
  sensitivity scans only, because AMS corrects gain-type effects before
  clustering and several parameters (zero-suppression threshold, cross-talk,
  Birks constant, saturation) were not found in the literature.

### Multiscale estimators and statistics

2D and 3D estimators of D0, D1, D2 with synthetic controls and registered
acceptance rules (`multiscale_estimators/`); statistical foundations in
`inference/` (exact Clopper-Pearson rejection limits, role-separated
cross-fitting, stratified permutation nulls, cell-shuffle null,
Holm/Westfall-Young); a role-separated classifier benchmark harness
(`benchmarks/`).

## 5. What has been done

Chronology, from the git history.

| Dates | Work |
|---|---|
| to 2026-09-20 | detector and event foundation; mean longitudinal and lateral profiles; DEC-001 stochastic generator; Geant4 proton pilot |
| 09-21 to 10-05 | program accepted (Paper 1 is the multiscale study); proton model and its decisions (DEC-003 to DEC-009); estimator-validity gate (DEC-010, DEC-013); first FastMC-versus-Geant4 comparison (disagreement) |
| 10-05 to 10-09 | `em_production` slices 1 and 2; lateral extensions tried and closed; sealed sets generated (10-04); registered multiscale test; first ladder reading |
| 10-09 to 10-11 | large development sets; EMZ/EMV systematics; ladder found to have no headroom; critical program review and redesign (slices S0 to S8, validity gates SV-1 to SV-10); S0 sanity audit; S1 statistical tools |
| 10-11 to 10-17 | R-A sampling response, R-B module, component-tagged proton sets; S3 re-read of 192 development verdicts, cell-shuffle excess, matched pairs; S4 part 1 (3D estimator, negative); benchmark harness dry run; notebooks regenerated |
| 10-21 | research documents consolidated into `COMPLETED_WORK.md` and `OPEN_DECISIONS.md` (originals kept in git tag `docs-before-consolidation-2026-10-21`) |
| 10-22 | R-A noise floor and capped round; EXP-010 3D control sensitivity check; EXP-011 high-energy pilot |

### Results in one page (details and caveats in `COMPLETED_WORK.md`)

- **Electron against proton is trivially separable in ideal Geant4.** With fixed
  known energy, normal incidence and zero-threshold readout, AUC is 0.9999 to
  1.0000, and at least 0.95 from any single feature family. The S0 audit found
  no implementation error (null AUCs 0.48 to 0.52, duplicate-injection control
  works).
- **Multiscale features add nothing detectable beyond conventional features
  there.** The registered 0.01 AUC-gain rule cannot be met by construction
  (largest possible gain at most 1e-4); an amendment is drafted, not yet
  binding.
- **Proton-rejection numbers so far are zero-failure floors, not
  measurements.** The valid 95% lower limit is about N/3.
- **Finite-scale statistics** separate the classes at the 2D-testable level, but
  after matching on occupied cells and deposited energy the difference is not
  detectable with 100 to 136 pairs (exploratory). Only 64 of 192 development
  tests were testable at all.
- **The 3D estimator is not usable on any development event** under the
  registered rules (negative). The 2026-10-22 sensitivity check (EXP-010,
  exploratory) found no window usable under all three baselines; the negative
  verdict stands as the primary result.
- **High-energy pilot (EXP-011).** Electrons at 400 GeV reach the 3D
  eligibility occupancy (23,721 voxels) in 98% of events (41.7% at 200 GeV);
  interacting protons at 0.5 to 2 TeV overlap the occupancy of 100 to 400 GeV
  electrons; cost and memory gates passed; the energy behind the 18-layer
  prefix is a median 20.9% at 1 TeV. The threshold itself is not established.
- **Detector response R-A.** Electron tables pass the registered 3% closure at
  10 to 100 GeV. Every proton variant failed (eight registered versions; the
  search is closed), and the noise floor shows 3% is structurally unmeetable
  for interacting protons. The unsplit proton tables are the working R-A with
  their disclosed error (about 3.5 to 7%).
- **Proton physics.** Geant4 11.4.1: no inelastic interaction in the prefix with
  probability 0.51 to 0.53 (against exp(-0.6) = 0.549); the visible-energy scale
  depends on the physics list (QGSP_BERT 13 to 28% lower).
- **Open anomaly.** `FTFP_BERT_EMV` electrons have a 37% lower scintillator
  share; at least seven settings differ; the swap test restored only 41%.
- **Literature.** A null or small gain is plausible (HEGRA precedent, near-
  monofractal EM showers); published ECAL-only proton rejection is 10^2 to 10^3
  at 90% electron efficiency, about 10^4 with energy-momentum match.

Tests: 1,088 collected on 2026-10-26 (`pytest --collect-only`). The last
*recorded full pass* was 916 tests on 2026-10-11; a full rerun is pending.

## 6. What is planned next

The road to the classical benchmarks, in dependency order (full table with
estimates in [`OPEN_DECISIONS.md`](research/OPEN_DECISIONS.md) section 2):

1. **Decide the 3D control design** (the immediate blocker): register a new
   control design reported beside the original verdict, accept the negative
   result, or leave the finite-scale question to the readout-level statistics.
2. **Amendment and freeze (DECIDE-1).** Replace the single AUC-gain rule with five
   endpoint families, each with one named primary: P1 physics difference, P2
   readout survival, P3 simulator fidelity (inactive), P4 realistic
   performance (inactive), P5 efficiency. Freeze windows, box sizes, strata and
   thresholds in a dated addendum before any confirmatory look.
3. **Production for P1** (EXP-013) at energies and sizes to be chosen (a tenfold
   pilot is about 1e6 CPU-seconds, one day on 14 workers).
4. **Detector-response completion**: R-B scan design; whether to restate the 3%
   R-A criterion in noise terms; validation of electron R-A tables above 200 GeV.
5. **Realistic regime (DECIDE-9)**: incidence angles, full-face entry, spectra,
   backsplash and material behind the ECAL (today only normal incidence and a
   9 x 9 mm entry spot).
6. **Detector-level dataset v1**, then fix the harness defects and register the
   benchmark protocol (classifier list, budgets, uncertainty procedure).
7. **Independent reviews** (statistics, ML, simulation, physics, reproducibility)
   of the frozen code and protocol.
8. **Registered classical benchmarks**, then the sealed validation sets, once.

If decisions arrive promptly the estimate is four to six weeks of working time,
3 to 6 days of it compute. A null result shrinks the claims, not the program:
P5 and the efficiency curves stay reportable.

Planned but not started: layer-to-layer residual noise in the electron
generator, proton-model sealed validation, a fractal-preservation gate across
fine Geant4, the 18 x 72 projection and FastMC, scale-removal ablations,
fractal-inspired architectures, and QML.

## 7. Papers

- **Paper 1: multiscale information in particle showers.** A physics and
  classical-ML study, not a QML benchmark. Classical ladder: conventional
  features; multiscale features only; both; raw 18 x 72 with compact baselines;
  raw plus multiscale. The target is incremental information, not the best AUC.
  A null result is reportable. Three-level comparison: fine Geant4, the same
  events projected to 18 x 72, and FastMC.
- **Paper 2+: QML on validated structure**, only if Paper 1 establishes
  something to exploit, with strong classical controls and matched budgets. No
  advantage is assumed. Nothing has started.

Details: [`research/PUBLICATION_ROADMAP.md`](research/PUBLICATION_ROADMAP.md).
Each paper gets a frozen configuration, recorded seeds, a tagged release and a
DOI archive (for example Zenodo); one repository serves the whole program.

## 8. Repository map

```text
.
├── README.md                  this file
├── RESEARCH_PROTOCOL.md       scientific rules: sources, evidence, review
├── CLAUDE.md, AGENTS.md       rules for AI agents working in the repository
├── research/                  durable scientific state (section 10)
│   ├── COMPLETED_WORK.md      what was done, registered and decided
│   ├── OPEN_DECISIONS.md      what is waiting, and the plan to the benchmarks
│   ├── MODULE_MAP.md          what each module is for and which test checks it
│   ├── NAMES.md               translations of old labels (Block N, Slice N, ...)
│   ├── PUBLICATION_ROADMAP.md the paper plan
│   ├── CHAOS_FRACTALS_QML.md  conceptual framing of fractals, chaos, QML
│   ├── NOTEBOOK_COVERAGE.md   which modules have a teaching notebook
│   ├── questions/             RQ-001, the first-paper question
│   ├── graph/                 machine-readable questions, hypotheses, decisions, experiments
│   ├── sealed/                sealed-set manifests and generation logs
│   └── data_manifest.json     where every exposed sample lives
├── src/ams_ecal/              all reusable code, grouped by role
│   ├── detector/              geometry, event, projection, R-A and R-B response
│   ├── electron_model/        electron generators (DEC-001, em_production)
│   ├── electron_studies/      evidence about the electron generator
│   ├── proton_model/          proton generator, calibration, validation
│   ├── multiscale_estimators/ estimators, controls, development ladder, S3 and S4 analyses
│   ├── inference/             statistical tools (limits, nulls, multiplicity)
│   ├── validation/            dataset checks, sealed-set guard, sample definitions
│   ├── benchmarks/            classifier benchmark harness
│   └── geant4_simulation/     Geant4 backend, truth, pilot analysis, seed checks
├── tests/                     mirrors src/ams_ecal/ folder by folder
├── configs/                   geometry, FastMC and Geant4 sample configs, run scripts
├── notebooks/                 00 to 08, teaching notebooks for completed work
├── results/                   JSON, CSV and figures written by the analyses
├── data/                      Geant4 outputs and calibration artifacts (large)
└── graphify-out/              generated code-relationship graph
```

Import as `ams_ecal.<folder>.<module>`; run as
`python -m ams_ecal.<folder>.<module>`. `ECALEvent`, `load_geometry` and the
configuration classes are exported from `ams_ecal` itself.

Notebooks: `00` ECAL and geometry, `01` tracker projection, `02` readout and
cell mapping, `03` event model, `04` geometry fidelity, `05` longitudinal
shower, `06` lateral shower, `07` stochastic EM events, `08` Geant4 proton
pilot. Later work (estimators, detector response, statistics, the 3D studies)
has no notebook yet by design; see `research/NOTEBOOK_COVERAGE.md`.

## 9. Setup and commands

The project targets GIL-enabled CPython 3.14 and uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Chagatai404/ams-ecal-research.git
cd ams-ecal-research
uv python install 3.14
uv sync
```

Dependency groups (`pyproject.toml`): `dev` (pytest, ruff, jupyter, matplotlib),
`geant4` (`geant4-pybind`), `validation` (scikit-learn), `ml` (PyTorch; the CUDA
build is installed from the PyTorch index as noted in `pyproject.toml`). For
example `uv sync --group geant4 --group validation`.

```bash
uv run ruff check .
uv run pytest -q
uv run jupyter lab
```

Geant4 production runs go through the backend, for example one batch:

```bash
python -m ams_ecal.geant4_simulation.geant4_backend run --sample pilot --energy 100 --workers 8
```

and the resumable drivers `configs/run_*.sh`. Before launching any new
production, check seeds with
`python -m ams_ecal.geant4_simulation.seed_collision_check`. Geant4 data are large and are located through
`research/data_manifest.json` and `validation/sample_definitions.py`; sealed sets
are never read directly.

## 10. Where to read next

| If you want to... | read |
|---|---|
| know what has been established and what has not | `research/COMPLETED_WORK.md` |
| know what is blocked, what is waiting, and what is next | `research/OPEN_DECISIONS.md` |
| understand the scientific rules | `RESEARCH_PROTOCOL.md` |
| find the code for something | `research/MODULE_MAP.md`, then `research/NAMES.md` for old labels |
| understand the fractal and chaos framing | `research/CHAOS_FRACTALS_QML.md` |
| read the formal research question and hypotheses | `research/questions/RQ-001_multiscale_shower_information.md` |
| learn the physics step by step | notebooks `00` to `08` |
| see the paper plan | `research/PUBLICATION_ROADMAP.md` |
| work in the repository as an agent | `CLAUDE.md` or `AGENTS.md` |

Detailed design and evidence records (literature verification, the R-A design,
the program-redesign review) are not in the working tree: they are in the git tag
`docs-before-consolidation-2026-10-21` and in the researcher's Obsidian vault.

## 11. Glossary

- **ECAL**: the AMS-02 electromagnetic calorimeter, lead plus scintillating fibres.
- **X₀, λᵢ**: radiation length (electromagnetic shower scale) and nuclear
  interaction length (hadronic scale).
- **Shower**: the cascade of secondaries started by a particle in the calorimeter.
- **Crossing / interacting proton**: a proton that passes the ECAL without an
  inelastic interaction (about half) versus one that starts a hadronic shower.
- **FastMC**: the parametric fast Monte Carlo. **Geant4**: the detailed transport
  simulation used as reference.
- **Multiscale / box counting**: describing energy concentration by covering the
  image with boxes of different sizes. **D0, D1, D2 (D_q)**: generalized
  dimensions of order q = 0, 1, 2.
- **Eligibility**: whether an event has enough occupied cells (or voxels) for an
  estimator to be trusted under its controls.
- **AUC**: area under the ROC curve; **rejection at x% electron efficiency**: how
  many protons are rejected per accepted proton while keeping x% of electrons.
- **Clopper-Pearson limit**: exact binomial confidence limit; with zero failures
  the rejection is only bounded below (about N/3), not measured.
- **KS**: Kolmogorov-Smirnov statistic. **Westfall-Young, Holm**: corrections for
  testing many statistics at once.
- **R-A, R-B**: detector-response tracks: deposit to fibre energy, and
  instrumental effects.
- **Sealed set**: a held-back Geant4 set that may be opened once for the final
  validation.
- **DEC-, EXP-, H-, RQ-**: identifiers of decisions, experiments, hypotheses and
  research questions in `research/graph/`.
- **P1 to P5**: the five named primary endpoints of the drafted amendment.

## 12. Independence and limitations

This is an independent academic research project. It is not an official AMS
Collaboration package and is not endorsed by AMS-02, CERN, NASA or the
International Space Station program.

FastMC events are research approximations; Geant4 is a simulation, not detector
reality; physics lists differ from data and from each other by 10 to 20%.
Conclusions are limited by simulation fidelity, detector-response knowledge,
finite granularity, data availability, preprocessing, sample size, estimator
validity, classical-control strength and quantum simulation scale. Claims about
physical shower structure should be checked against test-beam or detector data
where possible.

Core references: AMS-02 ECAL detector and reconstruction publications, AMS
test-beam studies, Particle Data Group reviews, Grindhammer and Peters
(arXiv:hep-ex/0001020), Geant4 and GFlash documentation. Implementation-
affecting sources, with exact locators, are kept in the research records and in
code and configuration provenance.

Licensed under the terms in [`LICENSE`](LICENSE).
