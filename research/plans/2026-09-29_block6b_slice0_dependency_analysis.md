# Block 6B Slice 0 — dependency analysis and proposed factorization

_Written 2026-09-29 by Claude as primary worker. **Agent proposal, not accepted.** Evidence
level E3 (controlled simulation): every number is Geant4 11.4.1 in this project's material model,
not detector truth. The researcher decides D1–D4 in §9._

Plan sections implemented: §9 (dependency analysis), §25.1 (production cut), §25.2 (material),
§31 (calibration / validation split), §32–33 (validation contract, written here *before* any
generator exists). Slice 0 forbids implementing the generator until this is clear.

Reproduce every number below:

```bash
uv run python -m ams_ecal.proton_dependency      # results/block6b/slice0/summary.json     (13 s)
uv run python -m ams_ecal.proton_checks          # results/block6b/slice0/variant_checks.json
```

Code: `src/ams_ecal/proton_calibration.py` (split), `proton_dependency.py`, `proton_checks.py`;
tests `tests/test_proton_{calibration,dependency,checks}.py` (32 tests). Variant Geant4 configs:
`configs/variants/`.

---

## 1. Integrity of the analysis

**Split (fixed before any 6B choice).** Within every batch, an event is *validation* when
`event_index % 4 == 3`, otherwise *calibration*. Deterministic, interleaved, identical for every
sample; a `fixed_entry` event shares the fate of its baseline twin (same seed). Calibration is
3000 events/energy (1408 / 1449 / 1453 / 1432 interacting at 10 / 20 / 50 / 100 GeV); validation
is 1000/energy and **has not been read by any number in this document**.

**Caveat, stated plainly.** The pilot report of 2026-09-29 was run on *all* events before this
split existed and shaped the plan. The split protects every table and choice made *from now on*;
it does not make the held-out events unseen in aggregate.

**Conventions and when they were adopted (Research OS: no silent goalposts).**

| convention | value | adopted |
|---|---|---|
| dependence "explicit" | abs(partial rho) ≥ 0.20 at ≥ 2 energies, one sign | 2026-09-29, **after** the first partial-correlation table was read |
| dependence "weak" | 0.10 ≤ abs(partial rho) < 0.20 | same |
| variant shift "material" | > 5% **and** > 2 bootstrap SE | 2026-09-29, **before** either variant was analysed |

The partial-correlation estimator is Pearson on within-cell-centred ranks (cells: 10 equal-count
bins of `R`, or a 5 × 5 grid of `R` × visible energy). An exploratory first pass used Spearman of
centred values and differed by up to 0.1 for the weakest pairs (visible energy ~ centre of
gravity); signs and the ordering of pairs agree. Interval half-width ≈ 0.055 (2σ) at n ≈ 1400.

Notation: `D` = first-inelastic depth (`truth_z_mm`, = `S_int`), `R = 166.5 − D`, `l_D` = readout
layer containing `D` (9.25 mm layers), `k` = layer offset from `l_D`, `A3` = fibre energy in
layers `l_D+1 … l_D+3` (model-free amplitude), `j` = upstream distance.

---

## 2. Observations (Geant4, calibration events)

### F1. The post-interaction amplitude is (nearly) independent of depth, with an energy-independent shape

| E (GeV) | median A3 (MeV) | Spearman(A3, D) [95%] | KS p (early vs late D) | A3/median at 5/16/50/84/95% |
|---|---|---|---|---|
| 10 | 79 | −0.064 [−0.12, −0.01] | 0.37 | 0.11 / 0.40 / 1 / 1.58 / 2.03 |
| 20 | 122 | −0.076 [−0.13, −0.02] | 0.12 | 0.09 / 0.43 / 1 / 1.59 / 1.91 |
| 50 | 192 | +0.029 [−0.03, +0.09] | 0.16 | 0.12 / 0.44 / 1 / 1.61 / 1.97 |
| 100 | 275 | −0.085 [−0.14, −0.03] | 0.008 | 0.08 / 0.42 / 1 / 1.61 / 2.04 |

- The **scale** grows as ≈ E^0.54 (log-log slope over 10 → 100 GeV; pairwise 0.62, 0.51, 0.52); the
  **shape** of `A3/median` is the same at all four energies to within the quantile noise.
  Distribution is broad, left-tailed (log-sd ≈ 0.9, skew ≈ −2): the heavy low-visible tail of the
  pilot.
- Dependence on `D` is *weak*: −0.06 to −0.09 at three energies, consistent with a back-edge
  effect (late interactions lose returning particles from material that is not there): the last
  depth tercile is ≈ 10% lower at 100 GeV. **Approximately independent, not exactly.**
- Exactly-zero `A3`: 0–2 events in ≈ 1250 per energy; smallest non-zero 0.3 MeV. The low tail is a
  continuum, not a separate mode.

### F2. The post-interaction layer profile is a universal function of the offset from `l_D`

Median layer energy at offset `k` agrees across `l_D` groups 0–3, 4–7, 8–11 to a maximum relative
spread of 0.16 / 0.16 / 0.24 / 0.15 (10 / 20 / 50 / 100 GeV; the 0.24 is 50 GeV at `k = 0`, the
mixed interaction layer). The first principal component of the log profile over `k = 1 … 8`
carries **79 / 87 / 89 / 91%** of the variance (n = 820 / 839 / 860 / 839): one amplitude plus a
small shape residual (PC2 5–9%).

### F3. Upstream of the interaction there is real shower activity, tied to the amplitude

Median excess over the crossing-proton layer energy (0.63 – 0.73 MeV) in the layers *before* `l_D`
(`j = 1 … 6`), MeV:

| E | j=1 | 2 | 3 | 4 | 5 | 6 | rho(upstream, A3) |
|---|---|---|---|---|---|---|---|
| 10 | 10.0 | 5.3 | 3.7 | 2.9 | 2.3 | 2.0 | 0.50–0.54 |
| 100 | 18.4 | 11.6 | 9.0 | 6.8 | 5.4 | 4.2 | 0.64–0.71 |

This is **not** a clean MIP track. The six layers upstream of `l_D` carry an excess of 26 MeV
(10 GeV) and 55 MeV (100 GeV) over a MIP track — about 9% and 4% of the median interacting fibre
energy (≈ 290 and ≈ 1330 MeV, pilot summary). The plan's §12 model — "MIP track until `S_int`, then
a shower" — would miss it, and it is correlated with the shower's size. *Interpretation:* backward
going secondaries (evaporation neutrons, π0 photons, recoil protons in the polystyrene).
Readout / deposit ratio upstream ≈ 0.125, against ≈ 0.09 in the post-interaction layers and
≈ 0.07 for a crossing proton: **the fibre share depends on the type of energy**, which matters
for the two-representation requirement (§7).

### F4. `D` acts on an event only through geometry

Given F1–F3, `D` sets (i) where the profile starts, (ii) how many layers exist after it
(truncation), (iii) the upstream/downstream split. Nothing else is needed: the pilot's negative
`r(D, E_vis)` in the thin ECAL (−0.44 to −0.71) then follows from *more layers ⇒ more energy* and
should **reverse** when the calorimeter is extended, which is what the extended-geometry pilot
sample showed (+0.18 to +0.28). This is the pilot's "truncation sign flip" recovered as a
consequence rather than fitted. *Interpretation.*

### F5. Partial rank correlations, given `R` (and given `R` × visible energy)

| pair | 10 GeV | 20 | 50 | 100 |
|---|---|---|---|---|
| log E_vis ~ width, given R | **+0.28** | +0.18 | +0.06 | −0.06 |
| log E_vis ~ log hits, given R | +0.90 | +0.90 | +0.86 | +0.84 |
| log E_vis ~ cog, given R | +0.02 | +0.14 | **+0.23** | **+0.27** |
| log E_vis ~ long. RMS, given R | **−0.22** | −0.15 | −0.18 | **−0.23** |
| log hits ~ width, given R and E_vis | +0.39 | +0.53 | +0.61 | +0.63 |
| long. RMS ~ width, given R and E_vis | +0.25 | +0.28 | +0.37 | +0.45 |
| cog ~ width, given R and E_vis | −0.18 | −0.18 | −0.27 | −0.32 |
| log hits ~ RMS, given R and E_vis | +0.20 | +0.19 | +0.34 | +0.37 |
| log hits ~ cog, given R and E_vis | −0.08 | −0.24 | −0.34 | −0.34 |
| log E_vis ~ log E_deposit, given R | 0.94 | 0.96 | 0.98 | 0.99 |

Reading against the F1–F3 structure (**interpretation**, not tested causally):

- `E_vis ~ width` is present **only at 10–20 GeV** and vanishes above; the plan's
  `w_lat ~ P(w | E, E_vis)` is therefore not needed as a general arrow.
- `hits` follow `E_vis` (0.84–0.90) and, at fixed `R` and `E_vis`, follow lateral **width**
  (0.4–0.6): multiplicity is a *consequence* of energy and lateral spread, as the plan preferred.
- At fixed `E_vis`, `hits ~ RMS` (+) and `hits ~ cog` (−) are what one expects when energy spread
  over more layers lights more cells — a per-layer effect, not a new latent.
- `RMS ~ width` and `cog ~ width` at fixed `R`, `E_vis` share one sign pattern (RMS↑, cog↓, width↑)
  that is the signature of **more upstream (albedo) energy**, which is both longitudinally
  broad and laterally wide. This is a *candidate* single latent (albedo strength); F3 supplies it.
  Not yet tested.

### F6. Lateral structure

Fraction of layer energy at |cell offset| ≤ 1 ("core") and ≥ 5 ("halo"):

| E | core k=1 / 3 / 8 | halo k=1 / 3 / 8 | PC1 share of log layer RMS |
|---|---|---|---|
| 10 | 0.69 / 0.55 / 0.39 | 0.12 / 0.18 / 0.27 | 0.67 |
| 100 | 0.71 / 0.75 / 0.71 | 0.13 / 0.10 / 0.10 | 0.72 |

A stable core plus a wide halo (10–27% beyond 4 cells), broadening with `k` at low energy only.
At 10 GeV the per-layer core fraction is nearly independent of the layer's energy (0.53–0.57 across
the upper four energy quintiles; exploratory script, not in the module). One shared **event-level
lateral scale** carries 67–75% of the variance of the log layer RMS. Depth `D` does not appear
(S_D ≈ 0, pilot).

### F7. Crossing protons (non-interacting), fibre by fibre

- **The exact-geometry mapping is validated against Geant4.** Energy in the fibres actually
  crossed by the straight track is 0.1947 / 0.1977 / 0.2023 / 0.2035 MeV per mm of chord at
  10 / 20 / 50 / 100 GeV; polystyrene MIP ≈ 0.205. The lattice gives 66.2–67.8 crossed fibres and a
  chord path of 52.1–52.8 mm per event.
- **But geometry explains only 3–5% of the per-event variance** (corr(total fibre energy, crossed
  chord sum) = 0.13–0.24). Fibre energy *outside* the crossed fibres is 17–24% of the total with a
  heavy tail (sd 2.1 → 5.5 MeV, 10 → 100 GeV; total sd 1.85 → 6.7 MeV) — δ-rays and small EM
  cascades born in the lead.
- **Multiple scattering at low energy.** Of the ≈ 66 geometrically crossed fibres, 54.9 / 60.6 /
  65.3 / 64.8 carry a deposit at 10 / 20 / 50 / 100 GeV. A back-of-envelope Highland estimate
  (θ0 ≈ 6 mrad at 10 GeV over 17 X0, ≈ 0.6 mm lateral shift at the back; **not computed in the
  repository**) is the same size as the fibre radius, so the straight-line chord is less
  reliable at 10 GeV than at 100 GeV. *Interpretation.*
- Consequence: the phase / chord dependence is real but small per event; the per-layer response is
  a table conditioned on the exact chord path, with the heavy tail carried by the table, not a
  deterministic geometric term.

---

## 3. Proposed factorization (replaces the plan's §10 sketch; **for decision D1**)

```text
E, TrackState (normal incidence in v1; API keeps angle and path length)
 ├─ S_int ~ Exponential(lambda_eff)                                   [analytic; plan §3]
 │     S_int >= L  ──►  crossing event: per-layer response | exact chord path, E   (F7)
 │     S_int <  L  ──►  D = S_int, l_D = floor(D / t), R = L − D
 │
 ├─ A  =  s(E) · xi        s(E): smooth power law (≈ E^0.54); xi: ONE pooled empirical
 │                          distribution of A3/median (F1). A is independent of D.
 │
 ├─ post-interaction layers  l_D + k :  e_k = A · g_E(k) · (shape residual)        (F2)
 │     g_E(k): empirical profile, log E interpolated; layers beyond 17 do not exist
 │     (truncation is geometry, not a modelled dependence; back-edge factor if validation asks)
 │
 ├─ upstream layers  l_D − j :  MIP-like crossing response + albedo_j        (F3)
 │     albedo_j ∝ A · h_E(j) with its own fluctuation, rank-correlated 0.5–0.7 with A
 │
 ├─ lateral: per-layer kernel (core + halo) x ONE event-level lateral scale w    (F6)
 │     w independent of D; weak link to A only at 10–20 GeV; albedo may need its own kernel
 │
 ├─ cell structure: energy placed in cells as discrete quanta so that sparsity and
 │     hit multiplicity EMERGE from energy and lateral spread (F5)
 │
 └─ representation ∈ {readout (fibre), deposition (all material)}: same latents, one table
       set per representation, per-component fibre share (crossing 0.07 / shower 0.09 /
       albedo 0.125)
```

### What this changes relative to the plan's literal graph

| plan §10 / §14 / §17 | Slice 0 evidence | proposal |
|---|---|---|
| `E_vis ~ P(E_vis \| E, R)` as a 2-D conditional table | `A` is independent of `D`; `E_vis(R)` follows from profile truncation | draw `A` once (1-D pooled shape × smooth scale); `E_vis(E, R)` **emerges** |
| "MIP track before `S_int`" (§12) | 4–9% of the median visible energy sits in six upstream layers as albedo, correlated with `A` | add an explicit upstream component |
| `w_lat ~ P(w \| E, E_vis)` (§19) | `E_vis`~width only at 10–20 GeV | one lateral-scale latent; no general `E_vis` arrow |
| layer-fraction **templates** conditioned on `R` (§17) | profile is universal in `k`; PC1 79–91% | one profile `g_E(k)` × amplitude × small residual: **fewer tables, and no replay of individual Geant4 patterns** |
| hit multiplicity "emergent where possible" (§18) | hits follow energy and lateral spread | keep emergent; explicit `N_hit` table is the fallback |

**Why not template resampling.** Replaying whole Geant4 layer/cell patterns would make FastMC a
bootstrap of the pilot: EXP-001's "FastMC as smooth control" comparison would be circular, and
downstream classifiers would see repeated patterns. The proposal keeps every random variable
either analytic (`S_int`) or a low-dimensional empirical distribution, and builds cell patterns
from a kernel.

### Riskiest pieces (each is a Slice with its own held-out test; §35 stop rule applies)

1. **Cell-level sparsity and multiplicity** from a smooth kernel + quanta (F5/F6). If this cannot
   reproduce hits / max-cell / participation, fall back to an explicit conditional `N_hit`.
2. **Shape residual** after removing `A` (PC2–4: 9–21% of log-profile variance): small, but it sets
   the spread of centre / RMS / max-layer.
3. **Albedo** as a separate fluctuating component (needs its own amplitude coupling).

---

## 4. Dependency decisions (convention of §1; validation is the final arbiter)

| arrow | evidence | decision |
|---|---|---|
| `D → start of profile, truncation, upstream/downstream split` | F2, F4 | explicit (geometry) |
| `D → A` | rho −0.09…+0.03 | **absent** (weak edge effect recorded) |
| `E → A`: scale | ≈ E^0.54 | explicit, smooth |
| `E → A`: shape | identical at 4 energies | **absent** (pool) |
| `A → post-interaction layers` | PC1 79–91% | explicit |
| `A → upstream albedo` | rho 0.5–0.7 | explicit |
| `E_vis → lateral width` | +0.28 / +0.18 / +0.06 / −0.06 | **weak, energy-dependent**; validate at 10–20 GeV |
| `D → lateral width` | S_D ≈ 0 (pilot) | **absent** |
| lateral scale latent | PC1 67–75% | explicit, one per event |
| hits ← `E_vis`, width | 0.84–0.90; 0.4–0.6 | **emergent**, not drawn |
| longitudinal centre / RMS ← `E_vis` | +0.02…+0.27; −0.15…−0.23 | **emergent** from the component mixture |
| readout ↔ deposit | rho 0.94–0.99 | shared latents; comonotone coupling |

---

## 5. Production cut (plan §25.1) — negligible at readout level

Cut 0.7 → 0.1 mm, paired (same seeds), 750 calibration events / energy; the interacting subsets
are 364 (10 GeV) and 353 (100 GeV). Material flags: **none** for interacting events.

| quantity (variant / baseline) | 10 GeV | 100 GeV |
|---|---|---|
| lambda_eff | 249 / 250 mm | 261 / 261 mm |
| mean hit cells | 0.981 [0.933, 1.032] | 1.004 [0.958, 1.056] |
| cells > 0.1 MIP (small-deposit structure) | 0.976 [0.926, 1.029] | 0.999 [0.953, 1.052] |
| lateral width, median | 1.012 [0.961, 1.061] | 0.980 [0.933, 1.019] |
| max-cell fraction, mean | 1.067 [0.970, 1.157] | 1.009 [0.938, 1.080] |
| KS p (fibre energy / hits / width) | 0.23 / 0.41 / 0.98 | 0.91 / 0.86 / 0.44 |

Two honest notes. (i) The compact sample resolves shifts of ≈ 5–10% only. (ii) The single flag —
crossing-proton fibre-energy **sd** at 100 GeV, ×0.66 [0.48, 1.00] — is a tail statistic dominated
by a handful of large δ-ray events; the mean (×0.997) and median (×1.031) do not move. Crossing KS
p = 0.013 / 0.029 suggests a small shape difference. **Triage C** (reversible; documented). This
check says nothing about *fibre-level* structure, which remains an RQ-001 question.

## 6. Material composition (plan §25.2) — known density effect; one borderline interacting shift

`relative_volume` vs `average_density`: composite 6.80 → 7.21 g/cm³; 0.618 → 0.644 lambda_I;
16.68 → 17.86 X0.

- **Interaction.** P(no inelastic) 0.513 → 0.496 (10 GeV), 0.529 → 0.516 (100 GeV); lambda_eff
  250 → 238 and 261 → 251 mm (each ±5%). The predicted change from the depth alone is
  exp(−0.644)/exp(−0.618) = −2.6%; observed −3.3% and −2.5%. A predictable effect, comparable to
  the current calibration uncertainty.
- **Readout vs deposition.** Crossing-proton *fibre* energy: ×1.000 / ×0.994 (unchanged).
  Crossing-proton *total deposit*: ×1.067 [1.047, 1.086] at 10 GeV (flagged), ×1.048 at 100 GeV.
  The **readout representation is nearly insensitive to the material description; the deposition
  representation is not** (a denser matrix absorbs more of a MIP).
- **Interacting.** 10 GeV: nothing flagged. 100 GeV: `A3` median ×1.144 [1.006, 1.277], mean hit
  cells ×1.060, cells > 0.1 MIP ×1.071 flagged. Across both variants ≈ 76 quantities were compared
  and 5 were flagged (one of them the physical crossing-deposit effect above); about 3–4 flags are
  expected by chance under this convention. The interacting effect appears at one energy only and
  its interval touches 1. **Triage B** (testable, not blocking): keep the density-matched default;
  carry ≈ 5–15% as a material systematic on the interacting amplitude, comparable to (smaller
  than) the FTFP/QGSP spread. The two published AMS descriptions are mutually inconsistent, so
  there is no basis to "correct" the geometry.

---

## 7. Two representations for protons, and how they pair with Block 6A (**proposal, decision D2**)

The researcher's amendment requires protons in **true deposition** and **readout (fibre)**
representations, like 6A electrons. Slice 0 findings:

1. The two Geant4 energies are strongly rank-correlated given `R` (0.94, 0.96, 0.98, 0.99) but
   not identically: their ratio varies by **component** (crossing ≈ 0.07, shower ≈ 0.09,
   albedo ≈ 0.125) and with energy. So the two representations are *not* related by one constant.
2. **Proposal.** One set of latent draws per seed (S_int, `A`'s quantile, the lateral scale, …)
   drives both, so **one seed names a corresponding pair**, exactly the 6A property
   (`as_representation("deposition")` ≙ `true_deposition()`). Each representation has its own
   calibrated tables; the pairing is comonotone in the shared quantiles, exact in each
   marginal, approximate in the joint (rank correlation 0.94–0.99 ⇒ documented limitation).
3. **Pairing with 6A regimes.** Both 6A regimes emit energy in *deposited-energy units*
   (fractions of the primary energy, ≈ 94% contained at 100 GeV); the `sampling` regime moves the
   longitudinal **shape** to signal level but not the **energy scale**. Therefore:
   - `proton deposition` ↔ `electron deposition`: same physical quantity; directly comparable now.
   - `proton readout` ↔ `electron sampling`: **not** comparable in energy scale — 6A `sampling`
     is not fibre MeV. A readout-level e/p comparison needs an electron fibre-energy scale from
     Block 7 (sampling fraction) or a small Geant4 electron calibration through the same backend
     (which would also test Block 6A itself). **Until then, the common e/p representation is
     `deposition`.** Not resolved by 6B; flagged for the researcher.

---

## 8. Validation contract (pre-registered, plan §32–33; nothing below has been run)

Population: held-out validation events only (`event_index % 4 == 3`; 1000 per baseline energy,
≈ 475 interacting), matched energy, normal incidence, entry spot uniform over the illuminated
cell, generated with the same entry-point rule. FastMC: ≥ 20,000 events per energy so its own
sampling noise is negligible. Each row passes or fails **separately**; there is no single score.

| aspect | statistics compared | material discrepancy iff |
|---|---|---|
| interaction | no-inelastic fraction (Wilson interval); depth KS | held-out interval excludes model value, or KS ≥ 0.10 with p < 0.01 |
| crossing | total fibre energy, per-layer mean, 5/16/50/84/95% quantiles; energy by entry-phase quartile | KS ≥ 0.10 with p < 0.01, or a quantile off by > 10% and outside the held-out interval |
| energy | `E_vis`, `E_dep` for interacting events, marginal and within `R` terciles; 5% tail | same |
| longitudinal | layer means by `l_D` group; centre, RMS, max layer, active layers | same |
| lateral | width, core fraction, containment, cell distribution vs abs(offset) | same |
| activity | hits, max-cell fraction, participation | same |
| correlations | D~E_vis, D~centre, D~RMS, E_vis~hits, E_vis~width, per energy | difference ≥ 0.15 **and** outside the 95% bootstrap interval of the held-out value |

**Noise floor.** Each comparison is also reported for *calibration events vs held-out events*: the
KS distance an ideal model could not beat. A model that is as close to the held-out sample as the
calibration sample is called "at the floor".

Sanity checks before any comparison: seeds reproduce events bit-for-bit; energies non-negative
and finite; energy accounting closes; ECALEvent valid; no held-out event was read while fitting
(asserted in the builder). Seed policy: `SeedSequence.spawn`, as in Block 6A. Reproduction
command and commit: to be recorded with the results.

**Failure criterion (plan §35).** If the tables-plus-kernel model cannot reproduce the held-out
joint distributions without a large latent architecture, the work **stops** and reports what
fails, why, and the minimal additional variable that would fix it.

---

## 9. Decisions for the researcher

| # | decision | recommendation | triage |
|---|---|---|---|
| **D1** | Adopt the §3 factorization (amplitude × universal profile + upstream albedo + lateral scale) in place of the plan's `P(E_vis \| E, R)` table plus `R`-conditioned templates? | **Yes** — it follows from the dependency analysis §9 delegated, has fewer tables, and predicts the truncation sign flip | A: shapes Slices 1, 3–5 |
| **D2** | Coupling of the two representations (§7) and using `deposition` as the e/p common representation until an electron fibre-energy scale exists | Accept as proposed | B |
| **D3** | Order of the open checks: production cut (done, negligible), material (done, B), backsplash and incidence angle | After 6B v1; API keeps angle / path length | C |
| **D4** | Validation contract §8 | Accept before Slice 6 | B |

Slices 1–2 that do **not** depend on D1 — the `S_int` draw, the crossing-track table with exact
chords, `lambda_eff` provenance, the calibration artifact format and the proton configuration —
can proceed without waiting.

## 10. Not examined in Slice 0

Incidence angle; QGSP_BERT dependency structure (Slice 7); energies between anchors; backsplash
(extended geometry only); fibre-level structure; whether the amplitude scale is `E^0.54` outside
10–100 GeV (not claimed).

Files: `src/ams_ecal/proton_{calibration,dependency,checks}.py`, 3 test files,
`configs/variants/*.yaml`, `results/block6b/slice0/`. Geant4 variant batches (ignored by git):
`data/geant4_proton_pilot/{cut_0p1mm,material_relative_volume}/`, run from commit `5b89df5`, tree
clean of tracked changes, configuration hashes in each `metadata.json`.

---

## 11. Addendum, later the same day — what Slices 1–2 found while building the crossing branch

Written after the crossing branch was implemented and **before** its held-out validation was read.
Three findings from calibration events change or qualify §3 and §7.

**(a) The fibre and all-material energies of a crossing proton are only moderately correlated**
(calibration events; Spearman of the event totals 0.52 / 0.49 / 0.45 / 0.47 at 10 / 20 / 50 / 100
GeV, mean per-layer 0.15–0.24). The §7 proposal of a shared comonotone quantile therefore does
**not** hold for crossing tracks (it is fine for interacting events, 0.94–0.99). Implemented instead:
one seed shares the *interaction draw*; the layer fluctuations of the two representations are
drawn independently (real correlation 0.2, modelled 0). D2 is amended accordingly: **the
deposition and readout representations of one seed are corresponding only in interaction status
and depth, not in their crossing-layer fluctuations.**

**(b) Independent layers under-disperse the event total.** var(event total) / sum var(layers) in
calibration events is 1.45 / 2.79 / 4.42 / 5.52 at 10 / 20 / 50 / 100 GeV (independent layers
would give 1). The excess lives in rare bursts that span layers: at 10 GeV a single layer holding
10–15 MeV (a hard delta-electron or recoil); at 100 GeV an extended cascade (top event 129 MeV
over ≥ 12 layers). The top 10% of crossing events by total carry on average 1.4 (10 GeV) to 3.7
(100 GeV) layers above 3× the median layer energy, against 0.2–0.3 in the other 90%. The plan's
§35 asks for the minimal additional variable when a table cannot reproduce a joint distribution:
here it is **one event-level "burst" latent** (probability, onset layer, energy, extent). It is
**not** added: the bulk model is validated first, and the burst is proposed as decision **D5**
below rather than grown silently.

**(c) Layer energy depends on the exact chord path**, through a monotone rise of the median
0.47 → 0.60 MeV (10 GeV) and 0.37 → 0.68 MeV (100 GeV) from the lowest to the highest chord bin,
with the all-material energy falling with chord (9.2 → 7.5 MeV): the chord conditioning is
worthwhile for the fibre image and only weakly relevant to the deposit. Layer position matters
little at 10 GeV and raises the mean by ≈ 25% from layer 0 to layer 12 at 100 GeV (tail-driven;
medians flat). No layer-position term is included.

### Implemented (crossing branch only, all tested)

| piece | file | tests |
|---|---|---|
| split, crossing quantile table, versioned artifact, deterministic builder | `src/ams_ecal/proton_calibration.py` | `tests/test_proton_calibration.py` (33) |
| exact chord / cell geometry (reuses `FibreLayout`) | `src/ams_ecal/crossing.py` | `tests/test_crossing.py` (18) |
| proton configuration | `configs/fastmc_proton.yaml`, `src/ams_ecal/proton_config.py` | `tests/test_proton_config.py` (21) |
| interaction draw + crossing generator | `src/ams_ecal/proton.py` | `tests/test_proton.py` (29) |
| provenance: optional `EventProvenance.model_details` | `src/ams_ecal/event.py` | `tests/test_event.py` (+9) |
| held-out comparison | `src/ams_ecal/proton_validation.py` | `tests/test_proton_validation.py` (11) |

Whole suite: 442 passing, ruff clean. Calibration artifact
`data/calibration/proton_6b/ftfp_bert_v1/` (20 kB): pooled effective interaction length
**255.7 ± 3.4 mm** (per energy 262 / 253 / 251 / 257 ± 7 mm, no energy trend resolved). Its manifest
records `tracked_changes: true` because it was built with uncommitted edits; **rebuild it from a
clean commit before relying on it** (`uv run python -m ams_ecal.proton_calibration build`).

### D5 (new, for the researcher)

Add the event-level burst latent to the crossing branch if the held-out validation shows the
event-total tail failing, as §11(b) predicts. Recommendation: yes if and only if the validation
flags the tail; it is the smallest variable that repairs a measured failure and does not change
any other row. Triage B.

### Validation result — crossing branch, run once on the held-out events

20,000 FastMC crossing events per energy and representation against the held-out Geant4 crossing
events (528 / 496 / 519 / 538 at 10 / 20 / 50 / 100 GeV), under the §8 contract, which was not
changed. Data: `results/block6b/slice2/crossing_validation.json` and `.png`; reproduce with
`uv run python -m ams_ecal.proton_validation` (36 minutes). The KS distance is given; **F** = material
discrepancy under the contract; "floor" = calibration-vs-held-out distance.

| row (readout unless stated) | 10 GeV | 20 | 50 | 100 |
|---|---|---|---|---|
| interaction: crossing fraction inside the held-out interval; depth KS p | pass (0.70) | pass (0.82) | pass (0.98) | pass (0.50) |
| layer energy, pooled over layers | pass, at floor (0.011) | pass, at floor (0.017) | pass, at floor (0.010) | pass, at floor (0.023) |
| **event total, readout** | pass, at floor (0.046) | not material (0.095) | **F 0.120** | **F 0.129** |
| **event total, deposition** | **F 0.103** | **F 0.174** | **F 0.172** | **F 0.258** |
| **hit cells** | **F 0.212** | **F 0.270** | **F 0.369** | **F 0.359** |
| **max-cell fraction** | **F 0.128** | **F 0.194** | **F 0.261** | **F 0.284** |
| **containment (±2 cells)** | **F 0.159** | **F 0.169** | **F 0.187** | **F 0.223** |
| energy by chord quartile (model vs held-out mean) | within ≈ 1.5σ | within ≈ 1σ | within ≈ 2σ | trend steeper than Geant4: +25% vs +13% (±5%) |

**What passes.** The interaction law and the per-layer response conditioned on the exact chord — the
two pieces that are analytic or a table of the right distribution — reproduce the held-out
events to the noise floor. The exact-geometry dependence on entry phase is right at 10–50 GeV.

**What fails, and why.** These are large effects (KS 0.10–0.37, p < 0.001), not borderline flags,
so the ≈ 5–8% per-row false-alarm rate does not explain them. Each was predicted by the
calibration-side analysis above; a confirmed prediction is not a pass.

1. *Event-total tail and median (readout at 50–100 GeV; deposition at all energies).* The model
   median is 5–8% high (readout 12.2 vs 11.6 MeV at 50 GeV, 12.6 vs 11.9 at 100 GeV; deposition
   180 vs 164 MeV at 100 GeV) and its upper tail is thin (readout 95% quantile 17.1 vs 18.7 MeV at
   50 GeV). Independent layers spread the burst energy evenly over all events. The event-total
   variance over the summed layer variance is 1.01–1.15 in the model against 1.44–2.24 (readout,
   10–20 GeV), 3.6–4.1 (50–100 GeV) and 2.0–5.1 (deposition) in Geant4.
2. *Mean layer profile at high energy (not a contract row, visible in the figure).* At 100 GeV the
   Geant4 mean layer energy rises from 0.60 to 0.83 MeV with depth; the model is flat at 0.73 MeV.
   A burst that starts at a random layer and develops downstream reproduces this; a
   layer-position term on the median does not, because the medians are flat.
3. *Cell-level activity (all energies).* The model puts a layer's energy only in the cells of the
   fibres the track crosses: at most 18–19 hit cells, containment exactly 1. Geant4 has 17–22 hit
   cells (95% quantile 20 → 22), 4–7% of the energy outside the track cell, and about 5% of events
   with energy beyond ±2 cells. The containment KS is inflated by a spike at exactly 1 (median 1
   in both); its physical size is small (5th percentile 0.95–0.96). Hits and max-cell fraction are
   real.

**Minimal additional variables (plan §35: report, do not grow silently).**

- *D5 — an event-level burst latent* (probability, onset layer, energy, downstream extent) repairs
  items 1 and 2 and does not touch the rows that pass.
- *D6 — a per-layer lateral spill*: a small fraction of a layer's energy placed at cell offsets
  drawn from an empirical kernel (core plus halo, Slice 0 F6). It repairs item 3 and is the same
  kernel the interacting events need.

The condition I attached to D5 — "yes if the validation flags the tail" — is met.

**Integrity note.** The direction of both fixes was identified from calibration events before this
run (the variance ratios, the burst structure, the 17–24% of fibre energy outside the crossed
fibres). This run is the first and only look at the held-out crossing events. A repaired model
validated on the **same** events is a second look: report it as such, and prefer to spend the
remaining held-out events (interacting protons) on the interacting slices.

**Performance (plan §37).** 11–14 ms per event (≈ 75–90 events/s), including canonical-event
construction, which validates 1296 cell values in Python. That is too slow for large datasets
(10^6 events ≈ 3 h). The cost is in per-event Python (`FibreCrossingGeometry.cross` twice, the
tuple-of-tuples event, its validation), not in the physics; a batch path returning arrays is the
fix. Not optimized yet, per the plan.

**Two properties of the contract, stated for the next slice.** The per-row false-alarm rate of a
95% interval is ≈ 5–8% under a correct model (measured 8% on the interaction row over 200
replications), so with ≈ 30 rows a couple of marginal flags are expected by chance and must be read
collectively. And a KS threshold flags a discrete distribution with a spike (containment) more
readily than its physical size warrants; each flag above was checked against the quantiles.
