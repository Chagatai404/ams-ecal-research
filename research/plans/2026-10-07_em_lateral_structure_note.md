# Lateral structure of electron showers: literature check, measurement and proposed spot-model formulation (analysis only; not implemented)

Date 2026-10-07, branch `fastmc-validated-dataset`. Decisions: DEC-014 (F2: the electron lateral mechanism is the Grindhammer-Peters spot model, not the proton quanta). Code:
`src/ams_ecal/electron_studies/em_lateral_structure_study.py`; test `tests/electron_studies/test_em_lateral_structure_study.py`; results `results/em_generator/lateral_structure_study.json` and `lateral_structure_study_layers.csv`.
Command: `python -m ams_ecal.electron_studies.em_lateral_structure_study`.

**Status.** Analysis only, on the exposed baseline Geant4 electrons (10, 20, 50, 100 GeV, 1000 events each). The sealed sets were not read, no generator and no DEC-001 behaviour changed. Nothing here is validated. The formulation in section 5 needs the researcher's approval before any implementation.

## 1. What the literature says (read in the library PDFs, not from subagent reports)

| claim | source and location | support class |
|---|---|---|
| radial law `f(r) = p 2rR_C^2/(r^2+R_C^2)^2 + (1-p) 2rR_T^2/(r^2+R_T^2)^2`; R_C linear in tau = t/T; R_T smallest near tau = 1; p largest near the shower maximum | Grindhammer and Peters, hep-ex/0001020, Sec. 3.4 Eqs. 22-26, Appendix A.1.3 | externally established (homogeneous media; their sampling set differs by small corrections, Eqs. 35-37) |
| the same law with R_C, R_T, p functions of E_0 and tau = t/T_0, integrated over the fibre, applied to AMS data | Kounine et al., NIM A 869 (2017) 110, Eqs. 1-3, Fig. 8 | **AMS-specific** (reconstruction and test beam) |
| the same law as the average lateral distribution in the AMS ECAL, parameters varying with energy and depth, applied to side leakage | Chin. Phys. C 40 (2016) 096204, Eq. 2 and text | **AMS-specific** |
| spots: `N_Spot = 93 ln(Z) E^0.876` (homogeneous), `N_Spot = (10.3/c) E^0.959` with sigma/E = c/sqrt(E) (sampling); spot depth density a gamma with `T_Spot = T(0.698+0.00212Z)`, `alpha_Spot = alpha(0.639+0.00334Z)` (hom.) or `T(0.831+0.0019Z)`, `alpha(0.844+0.0026Z)` (sam.); per-interval variance `sigma^2(eps)/(eps(1-eps)) = 1/N_S`, constant in depth | Grindhammer and Peters Eqs. 29, 31-33, 38-40 (read from the PDF pages) | externally established, transferred approximation (E units assumed GeV) |
| longitudinal-lateral correlation: replace tau by `tau_i = (t/<t>_i) exp<ln a>/(exp<ln a> - 1)` with `<t>_i = a_i/b_i`; "only the simulations using tau_i are able to predict the fluctuations observed with GEANT correctly" | Grindhammer and Peters Eq. 34 and the text after it | externally established |
| AMS cell fluctuations: Poisson in a number of quanta `N = eps/c`, with c a function of E_0, tau and the distance R | Kounine et al. Eq. 5, Fig. 11 (the readout amplitude: includes sampling and light statistics) | AMS-specific, a different observable from the deposition truth |
| transverse scale: `R_M = X0 Es/Ec`; on average 10% of the energy outside R_M, about 99% inside 3.5 R_M | PDG Review 2025, Sec. 34.5, Eq. 34.37 and the text after it | externally established |
| a general parameterisation trades accuracy for generality; the parameters should be extracted from detailed simulation of the same detector | Geant4 Physics Reference Manual, GFlash chapter (web) | supports calibrating, not transferring |
| competing approach: binned empirical radial histograms per layer and PCA bin, no closed-form law | AtlFast3, arXiv:2109.02551 (in the library; not read in detail here) | competing model, not adopted |

**Reading depth of each source (added 2026-10-08 under the Research OS v0.8 literature-reliability rule: reading depth is not independent claim verification, and a snippet or abstract is a gap, not support).**

| source | what was actually read | status of the claims taken from it |
|---|---|---|
| Grindhammer and Peters, hep-ex/0001020 | full-text extraction of Sec. 3.4-3.5 and Appendix A (Eqs. 22-26, 29, 31-41, A.1.3, A.2.4-A.2.5), and pages 9-10 read as images for Eqs. 31-40 | full-text read by the coordinator (not a separate verifier); exact locators recorded; the E-in-GeV unit and Z = 82 for the composite are ASSUMPTIONS, not read |
| Kounine et al., NIM A 869 | pages 113-114 read as images (Eqs. 1-3, Fig. 8) and text extraction of Eq. 5 and Fig. 11 | full-text read by the coordinator; the parameters R_C, R_T, Q_C are not given in the paper (they are cited from its ref. [3]) |
| Chin. Phys. C 40 (2016) 096204 | text extraction of the lateral-distribution paragraph (Eq. 2) | full-text read by the coordinator |
| PDG Review 2025, Sec. 34.5 | text extraction of the transverse-development paragraph (Eq. 34.37, 10% and 99% statements) | full-text read by the coordinator |
| Geant4 Physics Reference Manual, GFlash chapter | **a web-search result snippet only; the page was not fetched or read** | CANDIDATE support only: "supports calibrating, not transferring" is qualified as snippet-level, not verified |
| AtlFast3, arXiv:2109.02551 | **title and a search snippet only; the library PDF was not read** | named as a competing approach, not characterised beyond that |
| Nishimura-Kamata-Greisen | search snippets only | dismissed as out of regime (air showers); not verified |
| Longo and Sestili (1975), Bock et al. (1981), Peters' thesis | not retrieved | unknown |

No literature manifest (the opt-in v0.8 format) was assembled for this slice, and none of these claims received an independent source verifier; the Grindhammer-Peters and AMS equations are used as a transferred structure that the Geant4 comparison then tests (the radial law's adequacy is an empirical result of this project, not a literature fact).

Not usable: the Nishimura-Kamata-Greisen function (air showers). Not read: Longo and Sestili (1975), Bock et al. (1981) and Peters' thesis (cited by Grindhammer and Peters for the radial fits), which I could not retrieve. If you want the radial law's origin cross-checked, those are the sources to add to the library.

**Reconciliation with the repository.** DEC-001's lateral law is a *single-component* `3R^2/(pi(r+R)^4)` with `R = A(E) layer^2 + B`. It is not the law the AMS papers use. Its provenance (the "AMS test-beam parametrization" of the module docstring) was not found in the papers read here, and it was not checked against them.

## 2. What the Geant4 electrons show

Observable: the layer-normalised mean cell profile of the all-material deposition (`deposit_grid_mev`), averaged over the entry spot's sub-cell position; shape error = total variation, in points of layer energy. The shower axis is the entry point (the energy centroid is within 0.2-0.65 mm sd of it).

| law | layers 3-17 | layers 0-2 |
|---|---|---|
| two-component fit | **0.2-2.1** (0.2-1.3 at 100 GeV) | 0.4-3.7 |
| one-component fit | 3.8-9.0 | 5-11 |
| DEC-001 | 2.6-5.4 | 5.2-43.7 (grows with energy) |
| Grindhammer-Peters published constants | 3.8-11.1 | not summarised (the table is in the CSV) |

- The two-component law is the adequate family: the one-component law is 4-9 points off everywhere, so the second scale is required, not decorative. DEC-001's central three cells hold 0.98 of the layer energy in layers 0-1 against 0.54-0.91 in Geant4: its entrance width is far too small.
- Fitted parameters (strip fits; the core is bounded at 0.5 mm because a 9 mm cell does not resolve a core below about 1 mm): R_C 0.5 to 3.4 mm rising with depth; R_T 40 mm at the front, 13-14 mm near tau = 1, rising to 15-20 mm at the back; p = 0.36-0.7 at layer 0, 0.88-0.90 at tau about 0.4-0.5, falling to 0.14-0.46 at the back. The within-energy slope of p with tau (-0.4 per unit tau) matches the across-energy slope at fixed layer, which is what a depth-scaled law needs; a proper collapse test is part of the calibration, not done here.
- The published constants (homogeneous lead, R_M = 18 mm, E in GeV) are qualitatively right (p near 0.94 at tau 0.56 against the measured 0.88-0.90) but 4-11 points off at depth: a transferred approximation to be calibrated, as the Geant4 manual advises.
- Containment: the fitted law leaves 0.146-0.160 of the energy outside 18 mm and 0.019-0.021 outside 3.5 R_M, against the PDG's 0.10 and 0.01. The cell grid measures one coordinate, so this assumes radial symmetry; and the composite R_M is itself inconsistent in the repository (nominal 18 mm, where the Rossi formula with the configured X0 = 9.98 mm and E_c = 7.6 MeV gives 27.6 mm). The comparison is therefore indicative, not a test.
- Fluctuations: the inverse relative variance of the central-three-cell fraction, N_S, is **5-6 times larger after conditioning on the event's own centre of gravity at 100 GeV** (mid layers: raw about 150-450, conditioned 800-850), i.e. most of the raw event-to-event spread of the lateral profile is the depth coordinate, as Grindhammer and Peters state for tau_i. The explained variance is 0.44-0.74 at layer 11 and 0.18-0.61 at layer 17 (10-100 GeV). The conditioned N_S is nearly constant over layers 5-12 at 50 and 100 GeV (the constant contribution of Eq. 29) and grows with energy: mid-layer medians 135 / 232 / 501 / 829 at 10 / 20 / 50 / 100 GeV, a layer-count exponent of about 0.74. That is lower than the 0.876 of the total spot number, partly because the per-layer count also depends on how the spots spread over depth (the exponents are not strictly comparable). The first layers have N_S = 12-17 (layer 0) and 60-70 (layer 2): few contributing particles.

## 3. Cautions (what this does not show)

1. The core radius is not identified below about 1 mm and the fits hit the 0.5 mm bound at layers 0-1; this does not matter for the 9 mm readout but means R_C there is not a measurement.
2. The 18 x 72 grid is one coordinate per layer; the 2D radial structure is assumed symmetric.
3. The exposed Geant4 deposition is one physics list (FTFP_BERT) at one production cut; the AMS Poisson law of the readout adds sampling and light statistics that the deposition does not have.
4. The N_S conditioning uses a quadratic in a model-free centre of gravity, an approximation to the tau_i mechanism; the true test is whether a generator using tau_i reproduces the measured lateral-longitudinal coupling.
5. The radial parameters are fitted on the same exposed sample they are shown against; there is no hold-out here yet (the extended-depth sample has the same prefix and can serve as one).

## 4. What is not claimed

No validation, no statement that the spot model reproduces the Geant4 cell-level distributions (not yet built), no claim about hadronic showers.

## 5. Proposed formulation for the lateral slice (for approval; nothing implemented)

1. **Radial law**: the two-component law with `R_C(tau_i, E)`, `R_T(tau_i, E)`, `p(tau_i, E)` in the Grindhammer-Peters functional forms (their Appendix A.1.3 structure, 12 numbers) with the constants calibrated to the exposed Geant4 deposition (a transferred structure, a calibrated value), strip fractions from the closed form, no numerical integral in the generator.
2. **Coupling**: `tau_i` from Eq. 34 using the event's own (T_i, alpha_i) from Slice 1, so the measured lateral-longitudinal coupling follows from the model and is not hard-coded.
3. **Fluctuations**: `N_S` spots per layer of equal energy, drawn as a multinomial over the cell probabilities (distributionally identical to drawing radii and azimuths, and vectorised); the total spot number and its depth density calibrated to the conditioned N_S (two parameters: amplitude and energy exponent), the gamma depth-density shape of Eqs. 32-33 or 39-40.
4. **Conserved**: layer energy from Slice 1, never renormalised; the lateral leakage is the mass outside the 72 cells.
5. **Scope**: 10-100 GeV, deposition representation, normal incidence at the entry spot, DEC-001 lateral grid kept as the control `em_smooth_null`.
6. **Pre-stated checks before any sealed look**: layer-profile shape error below 2 points for layers 3-17 (the fit already reaches it, so this tests the smooth parameterisation), the lateral-longitudinal coupling (Spearman of lateral width with ln T) within the Geant4 sample-to-sample floor, and the cell-fraction variance against the conditioned N_S. These are proposals; the researcher sets the thresholds.

## 6. Questions for the researcher

1. Is the transferred Grindhammer-Peters functional structure (12 radial numbers, calibrated) acceptable, or should the radial law be a free per-layer table (the AtlFast3 style) at the cost of more parameters?
2. Which composite Moliere radius is the unit: the geometry's 18 mm (the AMS statement that a cell is 0.5 R_M) or the Rossi value 27.6 mm? It changes the radial constants' units, not the physics.
3. Should the hold-out be the extended-depth sample's prefix (same seeds family, different run) before the sealed set?
4. Add Longo and Sestili (1975), Bock et al. (1981) and Peters' thesis to the library to cross-check the origin of the radial form, or accept the AMS papers as sufficient AMS-specific support?
