# AMS-02 ECAL Research

A physics-informed research program for AMS-02 Electromagnetic Calorimeter
(ECAL) shower simulation, multiscale shower analysis, classical machine
learning, and later quantum machine learning.

The repository is intentionally broader than a single QML study. It provides
the shared detector, simulation, preprocessing, validation, and analysis
infrastructure for a sequence of related research papers.

> **Current status:** detector/event foundations and deterministic FastMC
> Blocks 0–5 are complete, and **Block 6A stochastic electromagnetic event
> generation is implemented** (`src/ams_ecal/stochastic.py`,
> `notebooks/07_stochastic_em_events.ipynb`). Block 7 detector response is
> **blocked** pending an adversarial pass on the open question of whether a
> response model would double-count detector behaviour already absorbed into
> AMS shower parameters fitted to observed deposits. Block 6B proton
> phenomenology has no accepted model. The first intended publication is a
> multiscale shower-information study using detailed transport and AMS-like
> readout, with QML deliberately downstream.

---

## Research program

The project is organized around a simple scientific sequence:

```text
detector and event foundation
        ↓
FastMC + Geant4 simulation
        ↓
does multiscale shower structure exist?
        ↓
does it survive AMS-like readout?
        ↓
what information does it add?
        ↓
classical ML benchmarks
        ↓
QML on validated structure
```

The project does **not** assume in advance that:

- AMS showers are mathematical fractals;
- a stable multifractal spectrum is measurable at AMS granularity;
- QML will outperform classical methods;
- FastMC reproduces microscopic cascade structure.

Negative results are scientifically useful.

### Central research direction

> What physically meaningful multiscale information in electromagnetic and
> hadronic showers survives AMS-02 ECAL-like readout, how useful is that
> information for particle identification beyond conventional observables, and
> can later quantum models exploit a validated multiscale representation under
> fair resource constraints?

### First-publication research question

> To what extent do electromagnetic and hadronic particle showers exhibit
> discriminative multiscale spatial structure, how much of that structure
> survives AMS-02 ECAL-like segmentation, and what information does it add to
> electron/positron-versus-proton classification beyond conventional
> calorimetric observables and standard ML representations?

See:

- `research/STATE.md`
- `research/PUBLICATION_ROADMAP.md`
- `research/questions/RQ-001_multiscale_shower_information.md`
- `research/CHAOS_FRACTALS_QML.md`

---

# Physics scope

The AMS-02 ECAL is a lead–scintillating-fiber sampling calorimeter.

The detector-level classification target in this project is:

```text
e± versus p
```

rather than `e+ versus e-`, because charge-sign information belongs to the
tracker rather than the ECAL alone.

## Canonical ECAL representation

The canonical calorimeter event representation is:

```text
18 longitudinal samplings × 72 transverse cells
```

Each longitudinal sampling measures one transverse coordinate according to the
fiber orientation of its parent superlayer.

Important nominal quantities used by the current ideal geometry include:

| Quantity | Nominal value |
|---|---:|
| Active readout area | 648 × 648 mm² |
| Active depth | 166.5 mm |
| Superlayers | 9 |
| Longitudinal readout samplings | 18 |
| Cells per sampling | 72 |
| Total readout cells | 1296 |
| Cell pitch | 9 mm |
| Electromagnetic depth | 17 X₀ |
| Hadronic depth | approximately 0.6 λᵢ |
| Effective bulk density | 6.8 g/cm³ |
| Effective critical energy | 7.6 MeV |

The code independently checks detector invariants such as:

```text
18 × 72 = 324 × 4 = 1296
9 × 18.5 mm = 166.5 mm
```

---

# Current simulation stack

## Stage I — detector and event foundation

Complete:

- Block 0 — ECAL geometry
- Block 1 — tracker state and projection
- Block 2 — alternating readout and cell mapping
- Block 3 — canonical `ECALEvent`

The canonical event contains:

- event ID;
- particle truth;
- primary energy;
- reconstructed `TrackState`;
- validated `ECALGeometry`;
- nonnegative finite `18 × 72` cell energies;
- simulation provenance;
- schema version.

`EventProvenance` records:

- backend (`fastmc` or `geant4`);
- simulation version;
- configuration hash;
- random seed.

## Geometry Fidelity Pass

Complete.

The geometry model explicitly separates:

- active volume;
- readout geometry;
- sampling structure;
- material properties;
- material depth;
- coordinate system.

The detector description remains configurable rather than hard-coding
scientific constants into implementation logic.

---

# Stage II — physics-informed FastMC

FastMC is intentionally transparent and computationally inexpensive.

Its role is to support:

- learning and controlled physics development;
- pipeline development;
- scalable baseline datasets;
- ablations;
- classical/QML infrastructure;
- comparison with a detailed Geant4 reference.

FastMC is **not** intended to replace detailed transport.

## Block 4 — longitudinal electromagnetic profile

Complete.

The deterministic mean longitudinal energy-deposition profile is:

```text
f(t) = β (βt)^(α - 1) exp(-βt) / Γ(α)
```

with:

```text
T = t_max = (α - 1) / β
```

and therefore:

```text
α = 1 + βT
```

The current AMS-specific model uses:

```text
β = 0.65
T(E) = ln(E / E_c) - 0.5
```

with `E_c = 7.6 MeV` supplied by the geometry/material configuration.

The continuous profile is integrated over each finite readout interval rather
than evaluated only at layer centers. Longitudinal leakage beyond the finite
17 X₀ detector is retained rather than renormalized away.

## Block 5 — lateral electromagnetic profile

Complete.

The mean transverse density uses the AMS test-beam parameterization:

```text
rho(r) = 3 R² / [pi (r + R)⁴]
```

with an energy- and layer-dependent lateral scale.

The profile is projected into the alternating ECAL readout and integrated over
finite cells. Lateral leakage is retained explicitly.

## Block 6A — stochastic electromagnetic generation

**Implemented.** `src/ams_ecal/stochastic.py`, `tests/test_stochastic.py`,
`notebooks/07_stochastic_em_events.ipynb`.

The first stochastic model deliberately remains simple: one random variable per
event, the depth of shower maximum `T0`, re-shapes the whole longitudinal
profile coherently. Independent per-layer jitter would destroy the correlation a
real shower has, where a late-starting shower is deeper in every layer at once.

```text
T_bar(E) = ln(E / E_c) + offset(regime) + sampling_correction(regime)

s(E)     = 1 / (intercept(regime) + slope(regime) * ln(E / E_c))

mu(E)    = ln(T_bar(E)) - 0.5 * s(E)^2

ln(T0)   ~ Normal(mu(E), s(E)^2)

alpha    = 1 + 0.65 * T0
```

The `-0.5 s^2` term is the centring that makes `E[T0] = T_bar(E)` exactly, so
the stochastic model reduces on average to the validated mean model instead of
biasing it deep by `exp(s^2/2)`.

Then:

1. integrate the gamma profile over the 18 finite longitudinal intervals;
2. obtain stochastic layer energies;
3. distribute each layer energy with the existing deterministic lateral
   fractions around the projected track;
4. preserve longitudinal and lateral leakage;
5. produce a reproducible `ECALEvent` carrying its own seed and a SHA-256
   digest of the configuration that produced it.

### Two regimes, and the perfect event

The mean depth and the fluctuation width are a **matched pair**. A top-level
`regime` in `configs/fastmc.yaml` selects both together:

| regime | describes | offset | `s(E)` at 100 GeV | `T_bar` at 100 GeV |
|---|---|---|---|---|
| `deposition` | true deposition in the composite — the **perfect event** | PDG `-0.5` | 0.0948 | 8.985 `X_0` |
| `sampling` (default) | signal-level longitudinal shape | G&P `-0.812` plus a geometry shift of `-0.353` | 0.1069 | 8.319 `X_0` |

The sampling regime peaks shallower, because `e/mip` falls with depth as the
cascade softens, and fluctuates more, because sampling adds longitudinal shape
fluctuation. Its depth shift is computed from `configs/geometry.yaml`
(`F_S = 4.897`, `e/mip = 0.651`) rather than hard-coded.

**The perfect event is recoverable from the seed.** The generator draws one
standard normal variate from the seed *first* and applies the regime
transformation afterwards, so a single seed names a corresponding pair of
events:

```python
model.true_deposition().generate_event(..., random_seed=seed)
```

returns the true-deposition event behind the sampled event that `seed`
produced.

### Provenance, which is not uniform

`beta = 0.65` is **AMS-specific evidence**, held fixed for all showers and all
energies exactly as AMS does. Both width laws and the sampling depth shift are
**transferred approximations** from Grindhammer & Peters
(arXiv:hep-ex/0001020, appendices A.1.2, A.2.2, A.2.3). AMS publishes **no**
closed-form mean-depth formula at all — it fits `T0` per shower to observed
cell deposits — so the offsets are external choices, not AMS values. Because
the published AMS form is `alpha = 1 + b*T0` for any `T0`, choosing an offset
does not break agreement with it.

Explicitly excluded from Block 6A:

- fluctuating beta — AMS holds `b` fixed, so this matches AMS rather than
  simplifying away from it;
- an explicit shower-start variable;
- independent random jitter of all 18 layers;
- a two-variable correlated `(T, alpha)` model;
- a new lateral fluctuation model;
- microscopic particle transport.

These simplifications will later be judged against Geant4 rather than expanded
pre-emptively.

### Boundary with Block 7

Under `regime: sampling` the longitudinal **shape** has already been moved to
signal level, so a Block 7 response model must not re-apply the depth shift or
the extra shape fluctuation. Under `regime: deposition` nothing
detector-related has been applied and Block 7 owns all of it, which makes it
the cleaner base to build Block 7 against.

## Block 6B — proton phenomenology

Planned separately.

A phenomenological proton generator must never be presented as equivalent to
full hadronic transport.

## Block 7 — detector response

Planned.

Candidate effects include:

- visible-energy / sampling response where justified;
- noise;
- thresholds;
- gain variation;
- saturation;
- dead/noisy channels;
- calibration effects.

Potential double counting with parameters fitted from observed AMS shower
depositions must be monitored rather than assumed away.

## Block 8 — validated FastMC datasets

Planned.

Datasets will preserve:

- exact simulator/configuration provenance;
- seeds;
- generation conditions;
- particle, energy, and geometry metadata;
- split isolation.

---

# Stage III — Geant4 reference simulation

Geant4 is the detailed-transport reference for both FastMC validation and the
first publication's multiscale-structure question.

Planned blocks:

- Block 9 — Geant4/C++ foundation
- Block 10 — ECAL geometry
- Block 11 — physics-list selection
- Block 12 — primary generation
- Block 13 — sensitive detector and export
- Block 14 — FastMC–Geant4 validation

Geant4 should provide both:

1. a fine-grained transport/deposition representation;
2. projection into the same canonical AMS-like `18 × 72` event representation.

## Multiscale-preservation gate

RQ-001 must be evaluated at three levels:

```text
fine-grained Geant4
        ↓
same events projected to AMS-like 18 × 72
        ↓
FastMC in the same canonical representation
```

This separates three different outcomes:

1. multiscale structure exists in detailed transport but is destroyed by AMS
   segmentation;
2. structure survives AMS segmentation but is absent from smooth FastMC;
3. FastMC preserves the multiscale observables relevant to the tested task.

Agreement only in mean profiles, containment, or energy resolution is not
enough to answer this question.

---

# Publication 1 — multiscale information in particle showers

The first intended paper is a **physics/representation + classical ML study**,
not a QML benchmark.

The paper asks:

1. Do detailed electromagnetic and hadronic showers show robust
   scale-dependent spatial structure over the accessible finite scale range?
2. Which signatures survive AMS-like detector segmentation?
3. Are those signatures independent of ordinary differences in shower energy,
   depth, width, containment, and incidence geometry?
4. Do multiscale descriptors add particle-ID information beyond conventional
   calorimeter observables?
5. Can standard raw-data ML learn the same information implicitly?
6. Does explicitly exposing multiscale structure improve low-data or
   sample-efficient learning?

Candidate observables include:

- box-counting-style dimensions;
- generalized dimensions `D_q`;
- partition functions `Z_q(epsilon)`;
- scaling exponents `tau(q)`;
- multifractal-spectrum summaries only when the accessible scale range
  supports them;
- lacunarity;
- entropy/concentration across scales;
- occupancy and energy moments under controlled coarse-graining.

The paper must not presuppose that particle showers are ideal mathematical
fractals or multifractals.

## Classical comparison ladder

Planned comparisons:

1. conventional calorimeter physics features;
2. validated multiscale features only;
3. conventional + multiscale features;
4. raw AMS-like representation with compact classical baselines;
5. raw representation + explicit multiscale features.

The primary target is **incremental information**, not merely the best AUC.

Low-data and energy-binned analyses are first-class experiments.

---

# Publication 2+ — QML on validated multiscale structure

QML starts only after the first paper establishes what multiscale structure
exists, survives readout, and is useful.

Candidate directions include:

- quantum kernels with matched classical kernels;
- variational classifiers on compact validated multiscale representations;
- QCNN/hierarchical circuits as candidate multiscale inductive biases;
- detector-topology-aware or scale-structured quantum connectivity.

Any quantum model must be compared with strong classical controls that receive
the same information and comparable tuning/resource budgets.

No quantum advantage is assumed.

---

# Research methodology

The project follows a human-led evidence workflow:

```text
question
→ independent literature discovery
→ source verification
→ adversarial review
→ human decision
→ implementation
→ experiment
→ independent audit
→ accepted / rejected conclusion
```

Important implementation-affecting claims are distinguished as:

- AMS-specific evidence;
- externally established result;
- transferred approximation;
- project phenomenological assumption;
- unresolved question.

Reusable scientific code belongs under `src/ams_ecal/`, with tests under
`tests/`. Notebooks are for teaching, derivation, visualization, and scientific
validation rather than being a second implementation.

See `RESEARCH_PROTOCOL.md` and `research/README.md`.

---

# Repository layout

Current and planned top-level structure:

```text
.
├── configs/
├── notebooks/
├── research/
│   ├── STATE.md
│   ├── PUBLICATION_ROADMAP.md
│   ├── CHAOS_FRACTALS_QML.md
│   └── questions/
├── src/
│   └── ams_ecal/
├── tests/
├── AGENTS.md
├── CLAUDE.md
├── RESEARCH_PROTOCOL.md
├── pyproject.toml
└── README.md
```

The repository is the shared research-program codebase. Individual papers should
eventually receive frozen experiment/manuscript directories and tagged releases
rather than separate repositories by default.

---

# Environment setup

The project targets ordinary GIL-enabled CPython 3.14.

```bash
git clone https://github.com/Chagatai404/ams-ecal-research.git
cd ams-ecal-research

uv python install 3.14
uv sync
uv run python --version
```

Run checks with:

```bash
uv run ruff check .
uv run pytest -q
```

Open notebooks with:

```bash
uv run jupyter lab
```

---

# Reproducibility and publication convention

For every publication:

1. freeze the exact experiment configuration;
2. record seeds and dataset provenance;
3. preserve figure-generation code;
4. tag the exact submitted/published repository state;
5. archive that tagged release with a persistent DOI service such as Zenodo;
6. cite the exact release in the manuscript.

A separate repository should be created only if a later study becomes genuinely
independent in scientific scope, codebase, collaborators, or release lifecycle.

---

# References

Primary detector information and simulation decisions should be tied to
authoritative detector or physics sources.

Core references currently include:

- AMS-02 ECAL detector documentation
- AMS-02 ECAL reconstruction publications
- AMS-02 ECAL performance/test-beam studies
- Particle Data Group reviews
- Grindhammer & Peters electromagnetic-shower parameterization
- Geant4/GFlash physics documentation

Implementation-affecting sources and exact locators belong in the corresponding
research records and code/configuration provenance.

---

# Independence and limitations

This is an independent academic research project.

It is not an official AMS Collaboration package and is not endorsed by AMS-02,
CERN, NASA, or the International Space Station program.

FastMC events are research approximations. Geant4 remains a simulation rather
than direct detector reality. Claims about physical shower structure should,
where possible, be checked against suitable test-beam or detector data.

Conclusions are limited by:

- simulation fidelity;
- detector-response knowledge;
- finite granularity;
- data availability;
- preprocessing;
- sample size;
- estimator validity;
- classical-control strength;
- quantum simulation/hardware scale.
