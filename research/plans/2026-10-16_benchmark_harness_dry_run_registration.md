# Benchmark harness and exploratory dry run: registration (2026-10-16, written BEFORE any benchmark number was computed)

Status: harness development and a dry run on EXISTING ideal-regime development data. No claim, no sealed set, no confirmatory use. The researcher approved the harness and PyTorch on 2026-10-16 ("Start those and install pytorch too"). The registered Paper 1 benchmarks (roadmap section 19, entry conditions) need the frozen amendment, the realistic-regime dataset v1 and the reviews; this dry run only exercises the machinery and measures its cost, so its defects are found before the protocol is registered.

## 1. Data and roles (fixed here)

- Electrons (`electron_development_large`) and **all** protons (`proton_development_large`, crossing and interacting; the P4 population) at 20 and 100 GeV, representations `deposit_grid_mev` and `readout_grid_mev`.
- Roles, disjoint (`rejection_limits.assert_disjoint_roles`, checked before any fit): 2,000 calibration electrons (threshold only), 2,000 evaluation electrons and 4,000 evaluation protons (fixed per block), and training pools of the remaining electrons and protons. Training sets of 100, 300, 1,000 and 3,000 events **per class** are drawn from the pools for each of 5 repeats (the registered protocol uses at least 10); the same training subsets are used for every model of a repeat, so differences between models are paired.

## 2. Rungs and models (fixed here, nothing is tuned)

- Feature rungs: `energy_only` (log total deposit), `conv` (the conventional features of `multiscale_test.conventional_features`), `conv_noE` (without total energy), `conv_noE_nocells` (also without the occupied-cell count), each alone and with the 24 two-dimensional multiscale statistics (`+ms`; NaN for ineligible events, handled natively by the boosted trees). The two harder rungs are the exploratory ones of the development run (the registered conventional rung is already at AUC 1.000 in this regime).
- Classifiers: gradient-boosted trees with the registered `CLASSIFIER_SETTINGS`; a standardized logistic regression on `conv` (the SV-7 baseline); a small convolutional network on the raw 18 x 72 image (input log(1 + E / 0.5596 MeV), convolutions 1->16->32 with ReLU, 2 x 2 max pooling, a 64-unit dense layer, one logit; Adam at 1e-3, batch 128, 30 epochs, fixed seed; PyTorch on the GPU). Fixed architecture and epochs, no tuning, no early stopping.

## 3. Metrics (fixed here)

Class-balanced log-loss on the evaluation set (primary for the efficiency question, P5, because the AUC saturates); AUC (a diagnostic, never a target); proton rejection at 90% and 99% electron efficiency from the calibration-electron threshold, as k passing of 4,000 protons with the exact Clopper-Pearson limit (censored when k = 0); wall-clock per fit; GPU memory of the network. Paired difference "rung with multiscale minus rung without" per repeat, reported as mean and standard deviation over the repeats (the spread over repeats is **not** a confidence interval: the dry run draws no inference).

## 4. Controls (SV-7)

Electron-against-electron halves for every model class (the network included): AUC must be 0.5 within 0.02 or the pipeline is not sound; the energy-only and logistic baselines are reported beside every ladder.

## 5. Reading rules written in advance

- Nothing from this dry run is a result about multiscale features or about efficiency; it is a statement about the harness (it runs, it is cheap or expensive, its controls hold, its paired differences have a spread of a stated size).
- Any harness defect found (a leak between roles, an unfair comparison, a control that fails) is fixed and recorded before the registered protocol is written.
- The cost per fit and the memory of the network are recorded because they set the budget of the registered benchmarks (roadmap section 19).

## 6. What this does not do

The realistic regime, the observable-defined hard subset of P5 (frozen in the amendment addendum), the vision transformer or ResNet baselines, quantum models, architecture search, any statement about AMS performance.
