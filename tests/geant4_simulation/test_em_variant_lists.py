"""The EM-variant physics lists of the backend: the rebuilt list must reproduce FTFP_BERT exactly when given its own EM constructor."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from ams_ecal.geant4_simulation import geant4_backend as backend

CONFIG = """schema_version: 2
particle: e-
energies_gev: [10.0]
incidence: {{theta: 0.0, phi: 0.0}}
entry_spot: {{x_min: 0.0, x_max: 9.0, y_min: 0.0, y_max: 9.0}}
gun_z: -1.0
physics:
  lists: [FTFP_BERT, FTFP_BERT_STDEM, FTFP_BERT_EMZ]
  production_cut: 0.7
samples:
  ref: {{geometry: ams, physics_list: FTFP_BERT, events_per_energy: 6, base_seed: 20272999}}
  std: {{geometry: ams, physics_list: FTFP_BERT_STDEM, events_per_energy: 6, base_seed: 20272999}}
  emz: {{geometry: ams, physics_list: FTFP_BERT_EMZ, events_per_energy: 6, base_seed: 20272999}}
store_fine_prefix_deposits: false
output_dir: {out}
"""


def run(config: Path, sample: str, out: Path) -> dict[str, np.ndarray]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ams_ecal.geant4_simulation.geant4_backend",
            "run",
            "--sample",
            sample,
            "--energy",
            "10",
            "--workers",
            "2",
            "--out",
            str(out),
            "--pilot-config",
            str(config),
        ],
        check=True,
        capture_output=True,
    )
    return dict(np.load(out / "events.npz"))


def test_the_em_variants_are_named_and_the_equivalence_check_uses_the_standard_constructor() -> None:
    assert backend.EM_VARIANT_LISTS["FTFP_BERT_EMZ"] == "G4EmStandardPhysics_option4"
    assert backend.EM_VARIANT_LISTS["FTFP_BERT_EMV"] == "G4EmStandardPhysics_option1"
    assert backend.EM_VARIANT_LISTS["FTFP_BERT_STDEM"] == "G4EmStandardPhysics"


def test_the_rebuilt_list_reproduces_ftfp_bert_event_for_event_and_emz_differs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.format(out=(tmp_path / "data").as_posix()), encoding="utf-8")

    ref, std, emz = (run(config, name, tmp_path / name) for name in ("ref", "std", "emz"))

    assert (ref["seed"] == std["seed"]).all()
    assert np.array_equal(ref["deposit_grid_mev"], std["deposit_grid_mev"])
    assert not np.array_equal(ref["deposit_grid_mev"], emz["deposit_grid_mev"])
