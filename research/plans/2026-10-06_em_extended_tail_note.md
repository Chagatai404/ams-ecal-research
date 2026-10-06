# Extended-depth electrons: the longitudinal tail measured (analysis only)

Date 2026-10-06, branch `fastmc-validated-dataset`. Decisions: `DECISIONS.md` (DEC-014 and the 2026-10-06 rows). Data: `configs/geant4_electron_extended.yaml`
(1000 electrons at each of 10, 20, 50, 100 GeV followed through 270 layers: the 18-layer prefix plus 126 superlayers; generated in about 66 minutes on 12 workers,
base seed 20271010; `data/geant4_electron_extended`, git-ignored, regenerate from the config). Code: `src/ams_ecal/electron_studies/em_extended_tail_study.py`
(+ `_plots.py`; the exponential-onset tail is a new option of `em_depth_origin_calibration.py`); tests `tests/electron_studies/test_em_extended_tail_study.py`; results
`results/em_generator/extended_tail_study.json`, `extended_tail_study_prediction.csv`, `extended_tail_study_decay_length.csv` and five plots in
`results/em_generator/extended_tail_study/`. Command: `python -m ams_ecal.electron_studies.em_extended_tail_study`.

**Status.** Analysis only. The sample is EXPOSED development data (like the 4000-electron sample); it is not a sealed set and never enters a validation. The sealed electron set
was not touched; no generator was written or changed; DEC-001 and the proton model are untouched. Nothing is validated.

## 1. Answers

| question | answer |
|---|---|
| Is the new sample the same physics as the baseline? | **Yes.** Prefix layers against the baseline sample (other seeds): rms z 1.00, 0.86, 1.31, 1.35; largest |z| 2.2, 2.5, 3.8, 2.9; contained-fraction KS 0.09, 0.05, 0.06, 0.06; layer-0 energy 36.3 / 43.1 / 59.8 / 74.5 MeV against 37.3 / 42.7 / 58.3 / 74.9. The extension holds 99.50 / 99.58 / 99.61 / 99.63% of the primary energy; the missing 0.37-0.50% is not deposited in the 270 layers (not identified further). |
| How much energy is behind the prefix, and where? | 5.10 / 6.73 / 9.17 / 11.31% of E at 10 / 20 / 50 / 100 GeV (event sd 3.6-4.4% of E, skewness 5.4 down to 1.9). **Nearly all of it is in layers 18-29** (4.87 / 6.44 / 8.76 / 10.79%); layers 30-39 hold 0.21-0.47%, layers beyond 40 only 0.02-0.05%, beyond 80 essentially nothing. |
| What is the tail? | **An exponential with an energy-independent rate.** The local decay rate of the mean layer energy is 0.27-0.28 per X0 in layers 20-40 at every energy (an attenuation length of 3.6 X0), and the model-free rate over layers 22-60 is 0.247 +/- 0.025, 0.262 +/- 0.011, 0.246 +/- 0.007, 0.236 +/- 0.015 per X0 with a slope of -0.006 per ln E (flat). Leroy and Rancoita measured 3.3-3.9 X0 for lead with little energy dependence; the tail rate I took from them (1/3.6 = 0.278) was right, and the gamma form's own decay (1/beta of about 1.8-2.1 X0) is too fast. The amplitude behind the maximum follows the shower maximum (ln E). |
| Did the prefix-only fits predict the tail? | **No.** Energy behind layer 17, predicted / measured at 10 / 20 / 50 / 100 GeV: gamma only 0.83 / 0.83 / 0.86 / 0.89 (it under-predicts by 11-17%); gamma tail at the literature rate 1.35 / 1.16 / 1.03 / 0.97; with an energy-dependent weight 1.18 / 1.13 / 1.10 / 1.08; free-rate gamma tail 1.33 / 1.17 / 1.06 / 1.03; exponential-onset tail 0.82-0.91. **In layers 30 and beyond the gamma-shaped tail components predict 1.6 to 5.8 times the measured energy** (literature rate 1.6-3.3, energy weight 2.1-2.8, free rate 2.7-5.8), and gamma only 0.39-0.50 times; only the exponential-onset tail at its free rate lands near it (0.99-1.33) while it under-predicts layers 18-29 (0.86-0.91). The free-rate gamma fit that ran to its bound (rate 0.15) was wrong, not merely unconstrained: the measured rate is 0.27. |
| Then what are the tail components fitted on the prefix? | **An empirical in-prefix shape correction, not a physical tail.** The energy-weighted literature-rate variant reproduces the prefix containment to within 0.5 points at every energy (predicted - measured -0.5, -0.5, -0.5, -0.5; gamma only +1.3, +1.6, +1.7, +1.6) and the energy in layers 18-29 to 2-10% (1.10 / 1.06 / 1.04 / 1.02), but it puts the wrong energy deeper than layer 30. Its parameters (core beta 0.72, origin -1.55) are calibration values, not shower physics. |
| Does one additive component fit the prefix and the tail together? | **No.** Over the first 80 layers (normalisation free) the chi-square per dof is 25.7 (gamma only), 11.7 (gamma tail at the literature rate), 9.6 (free-rate gamma tail), 17.4 / 10.4 (exponential tail at the literature / free rate). The families that fit layers 18-39 best (5-8% rms) degrade layers 0-3 (11-15%) and vice versa. A gamma core with one extra component of these forms is structurally short of the profile. |
| Event level: does the tail fluctuate independently? | **Mostly not.** A gamma fitted to the prefix of each event explains 80 / 85 / 90 / 91% of the event-to-event variance of the energy behind the prefix (Spearman 0.79-0.94). It has the wrong scale: its mean is 0.8-1.0 points low and its standard deviation 23-32% too large (0.0475 against 0.0361 at 10 GeV). The remainder (sd 1.7-2.2 points of E) correlates with the shape parameters (with ln alpha +0.41, +0.35, +0.28, +0.19; with ln T -0.21, -0.28, -0.45, -0.45) and not with the entry position (|rho| below 0.1) or layer 0 (|rho| below 0.17). There is no sign of an independent event-level tail weight. |
| ln T skewness | Prefix fits give +0.90, +1.02, +0.87, +0.51 in this sample (the baseline sample: +0.75, +0.85, +0.96, +0.79): the skewness is positive in both samples and its estimate varies by up to 0.3 between samples of 1000. |

## 2. What this changes

- The earlier worry (an 18-layer prefix cannot identify a physical tail) is confirmed, and **the tail is now measured**: an energy-independent exponential of rate 0.27-0.28 per X0 beyond about layer 20.
- **The generator does not need the far tail.** It outputs 18 layers; the energy behind them is `1 - contained - 0.4%`. What matters is the prefix shape and containment, and the energy-weighted
  tail correction reproduces the containment to 0.5 points and layers 0-3 to 3-8% (profile shape study). It must be labelled an **empirical Geant4 calibration correction** (name: in-prefix tail correction),
  never "the tail".
- If a later slice needs the deposited energy behind the prefix (containment or a leakage observable beyond layer 17), it can be **calibrated directly** from this sample (mean per energy, event distribution,
  exponential rate 0.27-0.28, strongly tied to the prefix shape) instead of extrapolated from a gamma.
- The gamma-based families used so far cannot hold both ends of the profile at once. A better functional family (for example a rise and fall with a different asymptotic rate, or two components with
  energy-dependent weights) would need to be proposed and tested; this note does not do that.

## 3. Implementation gate (updated)

| condition | verdict | evidence |
|---|---|---|
| 1. a defensible common origin | **still a convention** | -0.98 gamma only, -1.5 with the in-prefix tail correction; family-dependent |
| 2. beta identifiable | **family-dependent** | 0.57 gamma only, 0.64-0.72 with the correction |
| 3. mean profile substantially improved and adequate | **improved, not adequate; the tail question is closed** | prefix chi-square per dof 36 to 14-18, layers 0-3 error 3-9%, containment within 0.5 points; no additive gamma family fits prefix and tail together |
| 4. GP widths | **yes** | covariant ratios 0.95-1.10 |
| 5. T-alpha structure | **yes with a deficit** | rho 0.35-0.44 against 0.57-0.62 |
| 6. ln T marginal | **skewed, chosen jointly with the correction** | skewness +0.5 to +1.1 in two samples; planted-null control (profile shape study) |
| 7. floor | **rejected** | unchanged |

**Recommendation.** The blocker I named is removed: nothing further is needed from the far tail for an 18-layer generator, and no more data is required for that. Conditions 1 and 3 remain not fully met
because the gamma-based mean profile is not statistically adequate (chi-square per dof 14-18) and the origin is a convention. Two honest options for you:

1. **Start the first implementation slice as a candidate development generator with named limitations**: gamma core with the in-prefix tail correction (energy-weighted, labelled empirical), covariant
   sampling-set widths, a skewed ln T marginal (skew-normal with a Gaussian copula) chosen jointly with the correction, no floor, `deposition` only, DEC-001 kept as the control, calibrated on the exposed
   electrons, the sealed electron set opened once at the end. Expected residuals (to be reported, not hidden): layers 0-3 3-9%, layer 1 at 100 GeV about -14%, an oscillating mid-profile residual,
   rho 0.15-0.22 low, the longitudinal-width distribution degraded by a fixed correction weight (KS 0.15-0.24).
2. **Look for a better functional family first** (the profile ends carry about half of the structure left), judged on the same held-out scores, before implementing anything.

I recommend option 1 only if you accept those named limitations as a development stage; option 2 if you want the mean profile adequate before any generator code.

## 4. Limits

- Exposed development data only; chi-square errors rescale by chi-square per dof (10-39), so intervals are model-conditional.
- Layers are 17/18 X0 thick, the convention of every earlier fit; in Geant4's own radiation length they are 0.927 X0, which scales a decay rate by 2%.
- The decay length window (layers 22-60) mixes the clean exponential (to about layer 40) with a noisy far end; the 20-40 windows are the cleaner estimate.
- The free-rate and free-weight fits sit at parameter bounds in several families: reported as such, not as measurements of those parameters.
- 0.4-0.5% of the energy is not deposited in the 270 layers (backscatter or lateral escape); it is not modelled here.
