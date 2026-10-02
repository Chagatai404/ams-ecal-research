# Multiscale estimator controls: results (synthetic controls only)

_Result record of experiment EXP-008, run 2026-10-02. The rules it was judged by are in
`research/plans/2026-10-02_multiscale_estimator_validity_preregistration.md`. Nothing here concerns showers, electron versus proton,
or particle ID._

## What was run

```text
uv run python -m ams_ecal.multiscale_controls          # seed 20261002, 1000 replicates per case, about 35 s
```

Code: `src/ams_ecal/multiscale.py` (estimators) and `src/ams_ecal/multiscale_controls.py` (controls and evaluation); tests in
`tests/test_multiscale.py` and `tests/test_multiscale_controls.py`. Output: `results/multiscale_controls/` (`controls.json`,
`recovery.json`, `windows.json`, `controls.png`). The smooth baseline is an illustrative map, not a calibrated FastMC; the
cascade is a deterministic hierarchy on 18 x 72 with exact answers at the aligned box sizes. "Hits" are equal quanta drawn
multinomially.

## Results

1. **The estimators are exact where the answer is known.** A filled square, a line and the Sierpinski triangle give 2, 1 and 1.585
   at every order q; on the cascade population the estimator equals the closed-form value at every aligned pair of box sizes.
2. **The false-positive limit is met everywhere** (0% to 8% false positives, limit 10%). This is close to a calibration check of the
   percentile procedure, because the fresh set comes from the same generator as the baseline. It rules out a broken procedure; it
   is not strong evidence on its own.
3. **The saturation-window rule alone is too lenient.** Wide windows count as interpretable (x <= 0 uses the window centre) even
   though they include saturated fine scales. Those windows can show high detection power with an ordering rate near 0, meaning
   the wrong direction: D0 - D2 exceeded the baseline only because the baseline was more negative. The ordering requirement is what
   catches them. This is a finding about the transferred rule on a six-scale grid.
4. **Verdict, with detection power and ordering both required:**

   | Hits | Verdict |
   |---|---|
   | 30, 100, 300 | NOT USABLE: no interpretable window passes |
   | 1000 | USABLE on 2 of 13 windows: all six scales (1..18) and (2, 3, 6, 9) |
   | 3000 | USABLE on 4 of 14 windows: (1, 2, 3, 6), (1, 2, 3, 6, 9), (2, 3), (2, 3, 6) |

   The verdict does not depend on the ordering fraction: with 0.5, 0.9 or 1.0 no window passes at 300 hits or fewer.
5. **Recovery of the exact cascade value (reference line 15%).** Coarse pair (9, 18): within 15% from 100 hits. Pair (3, 9): from 300
   hits for q = 1, 2 and from 1000 for q = 0. Finest pair (1, 3): needs 3000 hits, and q = 0 is still 17% off at 3000.

## A deviation, kept on record

The pre-registered rule for detecting the cascade has two parts: detection power, and the ordering D0 >= D1 >= D2. The **first
run applied only the power part** and merely reported the ordering. Its verdict was "USABLE" at 300, 1000 and 3000 hits on 6, 6 and 5
windows, and "NOT USABLE" at 30 and 100. Found after seeing output; the code now requires both parts, as the pre-registration
says. The fraction of replicates that must keep the ordering is not stated there; **0.90 is an interpretation, awaiting the
researcher** (see the addendum to the pre-registration).

## Limits that travel with this result

Equal-quanta hits are not real deposits (real cells have continuous amplitudes), so the number that matters for real events is not
yet known. One smooth baseline and one cascade; orders 0, 1, 2 only; A = 1 and D_f = 2 in the saturation rule are project
assumptions. No independent result validation is recorded.

## What follows

The estimators may be applied only to images with enough occupied cells for a non-saturated window, and the eligible fraction of real
events is unknown. The next step proposed is an **eligibility count** (occupied cells per event), not a structure measurement. It
needs the researcher's approval.
