from pathlib import Path

import pytest

from ams_ecal.geant4_simulation.geant4_backend import (
    COMPONENT_NAMES,
    COMPONENT_OUTPUT_NAMES,
    load_pilot_config,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_existing_configs_keep_tagging_off() -> None:
    assert (
        load_pilot_config(CONFIGS / "geant4_electron_train.yaml").component_tagging
        is False
    )
    assert (
        load_pilot_config(CONFIGS / "geant4_proton_train.yaml").component_tagging
        is False
    )


@pytest.mark.parametrize("name", ["electron", "proton", "pion", "muon", "proton_qbbc"])
def test_component_configs_enable_tagging_and_use_new_seeds(name: str) -> None:
    config = load_pilot_config(CONFIGS / f"geant4_component_{name}.yaml")
    assert config.component_tagging is True
    assert 20273001 <= config.samples["train"].base_seed <= 20273005
    assert config.store_fine_prefix_deposits is False


def test_component_channels_end_with_the_derived_other_channel() -> None:
    assert COMPONENT_OUTPUT_NAMES == (*COMPONENT_NAMES, "other")
    assert len(set(COMPONENT_OUTPUT_NAMES)) == len(COMPONENT_OUTPUT_NAMES)
