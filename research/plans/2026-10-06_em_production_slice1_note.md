# em_production Slice 1: the gamma-family electron generator (candidate, development; not validated)

Date 2026-10-06, branch `fastmc-validated-dataset`. Decisions: `DECISIONS.md` (DEC-014 and the 2026-10-06 rows); plan sections 8-13. Code:
`src/ams_ecal/electron_model/em_production.py` (the generator and the parameter artifact), `em_production_calibration.py` (the calibration);
`src/ams_ecal/electron_studies/em_production_development_check.py` (+ `_plots.py`); tests `tests/electron_model/test_em_production*.py`,
`tests/electron_studies/test_em_production_development_check.py`; artifact `data/calibration/em_production/gamma_baseline_v1/parameters.json`;
results `results/em_generator/em_production_calibration.json`, `em_production_development_check.json` and four plots in
`results/em_generator/em_production_development_check/`.
Commands: `python -m ams_ecal.electron_model.em_production_calibration`, `python -m ams_ecal.electron_studies.em_production_development_check`.

**Status.** Implementation Slice 1, approved by the researcher on 2026-10-06 ("Proceed with implementation Slice 1. Do not search for an alternative longitudinal functional
family. Lock the gamma distribution as the baseline physics family unless later validation provides strong evidence that it is inadequate"). The model is a **calibrated candidate
for development**, checked on exposed Geant4 electrons only: the sealed electron set was **not opened**, DEC-001 (`stochastic.py`) and the proton model are **unchanged**, and nothing here is
called validated.

## 1. What the generator is

An event's longitudinal profile is a **gamma density** (locked family) in the depth `u = x - z0` from a depth origin `z0`, drawn event by event:

| ingredient | form | source and status |
|---|---|---|
| shower maximum | `x_max = ln(E/E_c) - 0.5`, `T_o = x_max - z0` | PDG review Eq. 34.36, electron: externally established |
| ln T | skew-normal, median `ln T_o`, sd `s_T x_max / T_o`, skewness `gamma_T` | width: Grindhammer-Peters sampling set App. A.2.3 with the covariant factor, transferred approximation; skewness: calibrated |
| ln alpha | `ln(1 + beta T_o) + s_a (rho z_t + sqrt(1 - rho^2) z_perp)` | width: Grindhammer-Peters, transferred; `beta`, `rho(ln y)` calibrated |
| copula | Gaussian: `z_t` drives ln T through the normal CDF | the correlation is the Gaussian-copula correlation |
| profile | `(1 - w(E)) core + w(E) tail`, tail a gamma density of shape `a_2` and rate `1/3.6` per X0, `w = w_0 + w_1 ln(E/30 GeV)` | tail rate: Leroy and Rancoita Table 2 and the extended-depth electrons (0.27-0.28); `w_0, w_1, a_2`: an **empirical Geant4 calibration correction** inside the prefix, not a physical tail |
| lateral | DEC-001's deterministic lateral grid, unchanged | not part of this slice; known too wide and too smooth |
| leakage | the missing part of the 18 fractions, never renormalised | |

Refusals: outside 10-100 GeV the model raises unless `allow_extrapolation` is set; the width-law domain, a maximum in front of the origin and an altered artifact (sha256 of the
parameters) are all refused. Seeds follow DEC-001's convention (`default_rng(seed)`, two consecutive standard normals per event; a batch equals the same draws one at a time).

## 2. The calibrated parameters (exposed baseline electrons, 1000 events at each of 10, 20, 50, 100 GeV)

| parameter | value | note |
|---|---|---|
| origin `z0` | **0 (the front face), a convention** | see below |
| `beta` | 0.5253 (scaled sd 0.005) | the PDG gives b about 0.5; the data's event median is 0.52-0.53 |
| tail weight `w_0`, slope `w_1` per ln E, shape `a_2` | 0.0435, -0.0203, 1.93 (sd 0.007, 0.004, 0.08) | empirical in-prefix correction; not at a bound |
| `rho = r_0 + r_1 ln y` | 0.9107 - 0.0396 ln y (0.62 at 10 GeV, 0.53 at 100 GeV) | measured with per-event gamma fits at the front face |
| ln T skewness | 0.739 | mean of the four energies (0.61-0.88) |
| fixed by the sources | `delta = -0.5`, tail rate `1/3.6`, energy reference 30 GeV | |

**The origin is fixed, and why.** With the origin free the calibration has no fixed point: the correlation and skewness measured at an origin near +0.08 send the central fit to z0 = -1.65
(beta 0.71, tail weight 0.12, chi-square 1402), and those measured at -1.65 send it back to +0.08 (beta 0.53, weight 0.05, chi-square 964), a two-cycle that repeated for all 8 iterations. This is the origin-correlation
degeneracy of the earlier studies. The origin is a coordinate convention, so it is fixed at the front face, where the measured beta and rho agree with the PDG review and Grindhammer-Peters and where the problem has one basin
(`fixed_origin=None` reproduces the cycle).

**The fit.** Mean layer fractions: chi-square 1284 for 68 degrees of freedom (18.9; the errors are the Geant4 standard errors, so this measures the family's shape error, not noise); layers 0-3 error 7.7 / 4.3 / 7.0 / 3.1% and layers 4-17
6.9 / 3.8 / 3.3 / 2.3% at 10 / 20 / 50 / 100 GeV. **The contained-fraction constraint is not met**: the pre-stated tolerance was 0.3 points (below the project's 0.5-point materiality floor) and the fit stays 0.42 / 0.71 / 0.48 / 0.89 points too high.
No parameter was added to chase it: this is the limit of a gamma core plus one gamma tail correction, recorded as a residual.

## 3. Development check against Geant4 and DEC-001

Distances from Geant4 (same metrics as the earlier comparison; hold-out = prefix of the extended-depth electrons, other seeds; floor = the two Geant4 samples against each other):

| metric | Slice 1 (calibration / hold-out) | DEC-001 | floor |
|---|---|---|---|
| layer-0 mean, abs log ratio | 0.02-0.16 / 0.01-0.13 | 2.3-4.0 | 0.005-0.03 |
| layers 0-3 profile error | 0.022-0.056 / 0.029-0.053 | 0.42-0.62 | 0.014-0.023 |
| layers 4-17 profile error | 0.015-0.058 / 0.015-0.061 | 0.08-0.12 | 0.008-0.012 |
| mean leakage difference (points of E) | 0.4-0.85 / 0.2-0.5 | 3.4-5.0 | 0.01-0.6 |
| leakage sd, abs log ratio | 0.01-0.25 / 0.04-0.19 | 0.40-1.06 | 0.01-0.22 |
| contained-fraction KS | 0.14-0.25 / 0.12-0.22 | 0.51-0.74 | 0.05-0.09 |
| longitudinal-width KS | 0.13-0.20 / 0.11-0.20 | 0.86-0.95 | 0.03-0.05 |
| lag-correlation difference | 0.21-0.30 / 0.19-0.28 | 0.29-0.44 | 0.016-0.023 |
| centre-of-gravity KS | 0.047-0.077 / 0.037-0.106 | 0.029-0.048 | 0.031-0.044 |

- **Better than DEC-001 on everything except the centre of gravity**, by one to two orders of magnitude on the profile (layers 0-3 error 2-5% against 42-62%) and on the width and leakage distributions; **not at the floor on any metric** (the sample-to-sample distance).
  The hold-out agrees with the calibration set (no sign of overfitting): layers 0-3 error 0.051 / 0.032 / 0.056 / 0.022 on the calibration set against 0.053 / 0.043 / 0.036 / 0.029 on the hold-out.
- Per-layer mean error, Slice 1 against the calibration set: layer 0 -14.9 / +2.2 / +8.4 / -5.9%, layer 1 +0.8 / +8.1 / +9.9 / +0.6%, layer 17 +14% at 10 GeV; elsewhere within about 10%.
- Per-event gamma-fit moments of the generated events against Geant4 (sd ln T 0.155 / 0.138 / 0.116 / 0.105 against 0.154 / 0.149 / 0.124 / 0.119; sd ln alpha 0.196 / 0.178 / 0.156 / 0.147 against 0.209 / 0.205 / 0.168 / 0.160;
  rho 0.62 / 0.57 / 0.55 / 0.53 against 0.62 / 0.60 / 0.60 / 0.51; skewness 0.80 / 0.72 / 0.61 / 0.75 against 0.61 / 0.75 / 0.88 / 0.72) agree to 5-15%; the median beta of the generated events is 0.48-0.51 against 0.52-0.53 (4-8% low).
- Leakage: mean 0.0527 / 0.0674 / 0.0911 / 0.1134 against 0.0567 / 0.0734 / 0.0954 / 0.1230 (low by 0.4-0.9 points), standard deviation 10-28% too wide at 10-50 GeV and right at 100 GeV, skewness 2.2 / 2.1 / 1.7 / 1.6 against 3.5 / 3.4 / 3.0 / 2.4 (too symmetric), 99th percentile 0.196 / 0.220 / 0.250 / 0.282 against 0.168 / 0.222 / 0.243 / 0.315.

## 4. Named limitations (not hidden, not chased)

1. **Containment** 0.4-0.9 points too high with the leakage tail too light (constraint not met): the limit of the locked family with this correction.
2. **Layer-to-layer residual noise is not modelled** (plan stage 4): the lag-correlation distance (0.19-0.30) and the width KS (0.11-0.20) stay several times the floor.
3. **Centre-of-gravity distribution** slightly worse than DEC-001 (0.04-0.11 against 0.03-0.05).
4. **Layer 0-1 and layer 17 at 10-50 GeV** 8-15% off in the mean.
5. **Lateral structure** is DEC-001's and unchanged; a later slice (the Grindhammer-Peters spot model with longitudinal-lateral coupling that survives the confounder checks).
6. **The origin is a convention** fixed at the front face; the in-prefix tail correction is empirical and its parameters are not physical; the gamma family is locked, so any family-level residual is reported, not removed.
7. **Calibrated range 10-100 GeV** only; the hold-out is not a fresh validation (earlier analyses looked at it descriptively); the **sealed electron set stays unopened** as the final validation, to be opened once, through `sealed_set.open_set`.

## 5. What would count as strong evidence against the gamma family (stated now, not applied)

The researcher locked the family unless later validation gives strong evidence of inadequacy. The sealed-set validation should therefore be read against these pre-stated signs, not tuned afterwards: layer-mean errors beyond 15% in more than two layers at any energy; a mean leakage error beyond 1.5 points; contained-fraction or width KS distances growing, not shrinking, with energy; or
a failure of the hold-out that the sealed data repeat. The development check does not reach them; the closest approach is layer 0 at 10 GeV (-14.9%) and layer 17 (+14%) on the calibration set, which is just inside the first sign and is therefore the first place to look at the sealed set.

## 6. Next steps (not started)

Lateral Slice (the spot model), the layer-to-layer residual noise (stage 4), then one opening of the sealed electron set; a detector response (R-A) is separate. Each needs the researcher's approval.
