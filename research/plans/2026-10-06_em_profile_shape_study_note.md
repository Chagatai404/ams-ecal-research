# Profile shape study: why a single gamma cannot reach the electron profile (analysis only)

Date 2026-10-06, branch `fastmc-validated-dataset`. Decisions: `DECISIONS.md` (DEC-014 and the 2026-10-06 rows). Code:
`src/ams_ecal/electron_studies/em_profile_shape_study.py` (+ `_plots.py`); the mean-profile model extensions are in
`em_depth_origin_calibration.py`; tests `tests/electron_studies/test_em_profile_shape_study.py`; results
`results/em_generator/profile_shape_study.json`, `profile_shape_study_extensions.csv`, `profile_shape_study_residuals.csv` and eight plots in
`results/em_generator/profile_shape_study/`. Command: `python -m ams_ecal.electron_studies.em_profile_shape_study`.

**Status.** Analysis only. No generator was written or changed (DEC-001 and the proton model are untouched), no sealed data was read, nothing is
validated. Numbers are on the EXPOSED Geant4 electrons (1000 events at each of 10, 20, 50, 100 GeV): development / calibration data. Every extension
below is a **candidate** structure scored on held-out resamples of those same development data, not a validation.

## 1. Answers, in the order approved

| question | answer |
|---|---|
| 1. Why can the gamma form not reach the 100 GeV profile? | **Mostly because the profile has a slowly decaying tail that a single gamma cannot hold**, and (second) because the entrance layers carry energy that grows with E. Per-event gamma fits leave a fixed structure: layers 0-1 above the fit at 100 GeV (+26% in layer 0), layers 1-3 and the middle below, and the fit **over-contains by 1.7-1.8 points at every energy** (0.894 against 0.877 at 100 GeV; 0.9615 against 0.9433 at 10 GeV). A tail component whose decay length is taken from the literature (lead 3.3-3.9 X0, central 3.6) with no tuning cuts the mean-profile chi-square per dof from 35.6 to 16.4 and the layers 0-3 error from 11 / 11 / 15 / 22% to 6.0 / 3.7 / 5.4 / 9.3%. Held out: +27% (leave one energy out), +52% (split half), +27% (100 GeV left out). Letting the tail weight depend on ln E: chi-square per dof 9.8, held out +59% / +69% / +62%. |
| What does not help | Energy-dependent beta (+15% held out) or origin (+25%), both together (+22%): they move layers 0-3 at 100 GeV from 22.1% to 20.6-21.9% only. A free depth-law slope and offset: -11% held out. The composite radiation length (9.98 mm, so 16.68 X0 against the 17 used for the layer bounds): using the Geant4 depth makes the fit worse (chi-square 2776 against 2494). |
| Caveat on the tail | Freeing the tail rate gives chi-square per dof 8.2 (held out +73%) **but the rate runs to its lower bound (0.15 per X0, an attenuation length of 6.7 X0)** with a smaller weight: the data want an even flatter tail than lead's photon attenuation, and the tail beyond layer 17 is constrained only through the leakage. The tail rate and weight are therefore **not identified** from the 18-layer prefix; the residual still oscillates (positive in layers 2-5 and 11-16 for the gamma-plus-tail fits), so a fixed-shape tail is not the whole story. |
| Entrance | Layer-0 energy of the Geant4 electrons is 34.6 MeV + 0.416 MeV/GeV x E (10 to 100 GeV; the first half-layer voxel holds 12.6, 14.9, 21.2, 28.5 MeV). The gamma form puts almost no energy there (layer-0 residual +26% at 100 GeV). The scintillator fraction is flat with depth (0.061-0.068, layer 1 about 0.061 at every energy), so no transition effect is visible in the visible/all-material ratio and the entrance excess is not a sampling artefact. With the tail the layer-0 residual falls to -1% at 100 GeV but stays +7 to +10% at 10-50 GeV: the tail trades the entrance, it does not remove it. |
| 2. Does the ln T skewness survive a profile family that can hold the tail? | **Yes.** ln T skewness is +0.80, +0.89, +1.00, +0.82 for gamma-only fits at the common origin and +0.82, +0.91, +1.12, +1.07 for gamma-plus-tail fits. A planted-null control (symmetric ln T, a tail, borrowed residuals, refitted both ways) returns skewness -0.10 to +0.07 for both fits against a planted -0.10 to +0.07: **neither fit manufactures the skewness**. |
| Does a skewed marginal matter? | **Yes, but only together with a tail-capable mean profile.** With the gamma-plus-tail ensembles a Gaussian ln T gives a leakage standard deviation 22-26% too small (|log ratio| 0.22 / 0.22 / 0.22 / 0.26) and a 99th leakage percentile 0.023-0.066 too low; a skew-normal marginal (same mean, sd, skewness, Gaussian copula on the rank correlation) reduces the leakage-sd mismatch to 0.055 / 0.078 / 0.085 / 0.018 and the percentile error to 0.000 / +0.017 / +0.021 / -0.001, against ensemble noise of 0.002-0.02. With the **gamma-only** family it goes the other way (the Gaussian leakage sd is already within 0.02-0.06 and the skew-normal makes it 0.11-0.21 too wide), because that family over-contains. The marginal decision cannot be separated from the tail decision. |
| Cost | The ensembles with a fixed tail weight degrade the distribution of the longitudinal width (KS 0.15-0.24 against 0.06-0.10 gamma-only). The per-event tail weight is **not identified** (per-event w mean 0.16-0.21, sd 0.13-0.15; ln alpha sd 1.5-1.9 times the Grindhammer-Peters value), so the tail is a population-level feature here, not an event-level variable. |
| 3. The AMS statement (b = 0.65, constant) | **Reproducible as a convention.** The event-level median beta of a single-gamma fit equals 0.65 at an origin of -1.6, -1.6, -1.9, -1.9 X0 (deposition) and -1.2, -1.2, -1.4, -1.4 X0 (readout, amplitude free) at 10, 20, 50, 100 GeV. At the front face it is 0.52-0.53, flat in energy (slope +0.003 per ln E); at upstream origins it falls slowly with energy (-0.006 per ln E at -1.0, -0.014 at -2.0, about 2-5% over a decade): constant within a few percent. In mean-profile fits, beta = 0.65 with a gamma-only profile costs a scaled delta chi-square of 59 (deposition) and 22 (readout); **with the tail it costs 6.3 (deposition) and 2.7 (readout)**: not rejected on the readout, marginal on deposition. With a tail the free core beta is 0.64-0.71, bracketing 0.65. |
| 4. Literature against the electrons | At the front-face origin the Geant4 electrons follow the PDG electron formula (t_max = ln y - 0.5, b about 0.5): median depth of the maximum 6.49 / 7.25 / 8.22 / 8.98 X0 against 6.68 / 7.38 / 8.29 / 8.98, median beta 0.52-0.53 against 0.5, and the mean-profile front-face free beta is 0.475. Grindhammer-Peters homogeneous lead gives 6.32 / 7.02 / 7.93 / 8.63 (0.2-0.35 X0 shallower) and beta 0.467-0.481. The simulated setup is therefore standard; the AMS 0.65 is a different convention, not a different shower. |

## 2. What the literature says, and how far I could verify it

**Read by me directly in the researcher's library (pdftotext and rendered pages), with locators:**

- **Grindhammer and Peters, hep-ex/0001020** (`0001020v1.pdf`): gamma mean profile, Eq. 2-4 (p. 3); T_hom = ln y + t1, alpha_hom = a1 + (a2 + a3/Z) ln y, Eq. 7-8 (p. 3), coefficients **T_hom = ln y - 0.858, alpha_hom = 0.21 + (0.492 + 2.38/Z) ln y** (App. A.1.1, p. 13); ln T and ln alpha "approximately normal" and the gamma description of **individual** profiles "becomes a worse approximation with decreasing shower energy", fluctuations underestimated at low energy (p. 4, below Eq. 11); the **transition effect** in sampling calorimeters (e/mip falls with depth, signal maximum earlier) and its correction T_sam = T_hom + t1/F_S + t2(1 - e), alpha_sam = alpha_hom + a1/F_S, Eq. 16-21 (pp. 5-6), coefficients in App. A.2.2-A.2.3 (p. 15). It acts on the visible signal; our deposition target is all-material.
- **PDG review, section 34.5, Eq. 34.35-34.36**: gamma profile, t_max = (a - 1)/b = ln y + C_j, **C_e = -0.5** (electron), +0.5 (photon), fits from carbon to uranium at 1-100 GeV, b about 0.5 or from Fig. 34.21.
- **Leroy and Rancoita, Rep. Prog. Phys. 63 (2000) 505, section 2.2.1, pp. 513-514**: b about 0.5, but "more precise values for b as function of A and E" (Fig. 1, EGS4 up to 100 GeV: b varies with material and energy, about 0.45-0.5 for the heavy absorbers); **measured longitudinal attenuation length of electron-induced cascades, Table 2: lead 21.3-24.7 g/cm2 = 3.3-3.9 X0, very little energy dependence** (0.6-6 GeV), while the gamma form implies 1/b of about 2 X0; the parametrisation "proves to be not fully adequate ... in particular for low values of E_c"; the fall-off beyond the maximum "displays a two-component structure, each component approximately exponential with a logarithmic dependence on the incident energy" (Si/W calorimeter).
- **Zhang et al., Chin. Phys. C 40 (2016) 096204, Eq. 1 (p. 2)**: gamma profile of the deposited energy per layer, a and b fitted to 180 GeV test-beam data, "the fit agrees well with the data". **No value of b and no energy dependence is stated**; the "4.2%" is the fraction of particles crossing the dead cells, not a leakage fraction.
- **Vecchi et al., arXiv:1210.0316**: five-page proceedings: detector description and performance; **no profile function, no scale parameter**.
- **Aguilar et al., Phys. Rep. 894 (2021), section 1.7.1** (read in an earlier session): "the scale parameter b is constant (b = 0.65)", reconstruction source Kounine et al., NIM A 869 (2017) 110. **Not in hand:** the origin convention and the fitting target of that 0.65.

**Not verified, and not relied on.** The independent discovery pass, the full-text verifier and the library reconciler returned reports whose specific claims I could not
reproduce: the verifier read no full text; one source it mapped (arXiv:1503.03037) is a cosmic-ray air-shower paper (Ter-Antonyan), so the "negligible correlations
of alpha, T_max, N_max" claim does not describe calorimeters; a "beta about 0.5" attributed to the AMS paper is not in it; the reconciler's "beta proportional to E^-0.5 for
Grindhammer and Peters", "transition effect about 10%", "first conversion shifts the origin within 1-2 X0" and the "mean leakage 4.2%" of the AMS dead-cell paper are not in the
pages or text I checked. None of these is used.

**What the literature does not give.** No published quantitative residual of the single-gamma form in the first radiation lengths for a lead-scintillating-fibre calorimeter at
10-100 GeV; no published origin convention for the AMS fit; no published event-to-event distribution of the shower-maximum position beyond "approximately normal in the logarithm".
Papers I could not read and that would help are listed in section 5.

## 3. Reading of the results

- The gamma form is a good **bulk** description (layers 4-17 within 2-4%) and a poor description of the **ends**: the front (energy that grows with E) and the tail (longer than 1/b).
  Both are documented qualitatively in the sources (Leroy and Rancoita on the tail; Grindhammer and Peters on low energy), neither quantitatively for this detector.
- A single gamma forced to carry the tail has its core pulled out of shape: its beta comes out at 0.57 and its origin at -0.98. With the tail the core beta is 0.64-0.71 and the origin
  -1.0 to -1.5, so the AMS 0.65 is compatible with the **core**; the front-face 0.475-0.52 is what a single gamma gives when the origin is the calorimeter face.
- The skewed ln T is physical in these events and matters once the leakage is modelled by a tail; whether the skew is the mark of the shower-start (first-conversion) fluctuation is **not tested** here.

## 4. Implementation gate (updated)

| condition | verdict | evidence |
|---|---|---|
| 1. a defensible common origin | **still a convention** | -0.98 (gamma only) to -1.5 (with the literature tail); the origin moves with the profile family; z0 spread about 1 X0 across forms |
| 2. beta identifiable | **yes, family-dependent** | 0.57 gamma only, 0.64-0.71 with a tail; AMS 0.65 admissible with a tail |
| 3. mean profile substantially improved and adequate | **improved, not yet adequate** | chi-square per dof 36 to 16 (literature tail) to 8-10; layers 0-3 error 3-9%; the fixed-shape tail leaves an oscillating residual; tail rate and weight not identified from 18 layers |
| 4. calibrated-coordinate GP widths | **yes** | covariant widths 0.95-1.10 (gamma-only fits) |
| 5. T-alpha structure defensible | **yes with a deficit** | rho 0.35-0.44 against 0.57-0.62 for gamma-only fits |
| 6. ln T marginal settled | **settled in direction, to be decided jointly with the tail** | skewness +0.8 to +1.1 is real; a skew-normal or empirical marginal is needed when a tail is present; Gaussian under-predicts leakage sd by 22-26% |
| 7. floor rejected or justified | **rejected** | unchanged; the tail and the entrance trade-off replace it |

**Recommendation: still do not start the generator.** Conditions 1 and 3 are not met. The structure is now much clearer: gamma core, a tail component, covariant sampling-set widths,
a skewed ln T marginal, no floor. What blocks a justified first implementation is the **tail**: its rate, shape and weight are not identified by an 18-layer prefix, and its
event-level behaviour is unknown. Proposed next steps, all analysis or data generation, none of them generator physics:

1. **Extended-depth electrons** (needs your decision and compute): the Geant4 backend can score behind the prefix (the research-instrument extension, same events). A development sample of
   electrons with the extension, at 10, 20, 50, 100 GeV, measures the tail beyond 17 X0 directly and fixes the tail rate and weight. It is exposed development data; no sealed set is involved.
2. **Event-level tail**: with the extended sample, test whether the tail weight fluctuates from event to event or is a population feature, and whether the ln T skewness tracks it.
3. **Entrance**: the layer-0 energy (34.6 MeV + 0.416 MeV/GeV x E) is not explained by the tail; test the first-conversion distribution against the fine (half-layer) profile.

## 5. Papers to add to the library (I could not read them; you can fetch them)

- **Kotwal and Hays, "Electromagnetic shower properties in a lead-scintillator sampling calorimeter", NIM A 729 (2013) 25** (CDF central calorimeter; arXiv:1308.2025, https://arxiv.org/abs/1308.2025): only the abstract was readable to me. It reports back-leakage and the scintillator-sampled fraction with Geant4 for a lead-scintillator calorimeter of similar depth.
- **Kounine, Weng, Xu, Zhang, NIM A 869 (2017) 110** (AMS ECAL shower reconstruction; no identifier I can verify from here): the primary source of b = 0.65, its origin and fitting target.
- **Longo and Sestili, NIM 128 (1975) 283** and **Bock et al., NIM 186 (1981) 533** (the gamma parametrisation and the EGS-era fits of b): old and usually behind a paywall.
- **Grindhammer, Rudowicz, Peters, NIM A 290 (1990) 469** is already in the library (`1-s2.0-016890029090566O-main.pdf`); I did not re-read it for this note.

## 6. Limits

- Exposed development data only; chi-square errors rescale by chi-square per dof (8-36), so intervals are model-conditional.
- The tail components share one gamma-density shape; the oscillating residual shows it is crude. Tail rate, shape and weight are degenerate with the core beta and the origin.
- The skewness comparison uses 5 ensemble seeds of 4000 events; the planted-null control borrows residuals from other events.
- The first six radiation lengths were examined at half-layer resolution only descriptively.
- Per-event tail-aware fits are poorly conditioned and are used only to show that, not as an event-level model.
