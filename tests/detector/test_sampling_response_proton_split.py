import json

import numpy as np

from ams_ecal.detector import sampling_response_validation as svm
from ams_ecal.validation import sample_definitions as sd


def write_protons(root, energy=10.0, n=400):
    directory = root / sd.SAMPLES["proton_development_large"].directory / f"E{energy:g}GeV"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    interacting = (np.arange(n) // 2) % 2 == 0  # independent of the event_index parity that splits fit and validation
    deposit = rng.exponential(2.0, (n, 18, 72)) * (rng.random((n, 18, 72)) < 0.4)
    ratio = np.where(interacting, 0.12, 0.07)[:, None, None]  # a different response per truth class
    np.savez(
        directory / "events.npz",
        event_index=np.arange(n),
        seed=np.arange(n) + 1000,
        truth_occurred=interacting,
        deposit_grid_mev=deposit.astype(np.float32),
        readout_grid_mev=(deposit * ratio).astype(np.float32),
        edep_total_mev=deposit.sum(axis=(1, 2)),
        edep_scintillator_mev=deposit.sum(axis=(1, 2)) * 0.1,
    )


def test_each_validation_event_is_predicted_with_the_table_of_its_own_truth_class(tmp_path, monkeypatch):
    write_protons(tmp_path)
    monkeypatch.setenv(sd.DATA_ROOTS_ENV, str(tmp_path))
    monkeypatch.setattr(svm, "RESULT_PATH", tmp_path / "out" / "r_a_closure.json")
    (tmp_path / "out").mkdir()
    result = svm.run_proton_split("proton_development_large")
    row = result["split_proton_validation"]["10"]
    assert row["classes"]["crossing"]["n_validation_events"] == 100
    assert row["classes"]["interacting"]["n_validation_events"] == 100
    assert row["interacting_fraction"] == 0.5
    # the two classes have different true responses (0.12 against 0.07): a single pooled table could not close both,
    # the split tables must (the synthetic response is exactly proportional, so the closure is limited by sampling only)
    assert row["combined_max_abs_layer_closure"] < 0.05
    assert (tmp_path / "out" / "r_a_proton_split_closure.json").is_file()
    assert (tmp_path / "out" / "r_a_proton_crossing_model.json").is_file()
    assert json.loads((tmp_path / "out" / "r_a_proton_split_closure.json").read_text())["tolerance"] == 0.03
