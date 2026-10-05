from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ams_ecal.detector.geometry import (
    ActiveVolume,
    CoordinateSystem,
    ECALGeometry,
    GeometryConfigError,
    MaterialDepth,
    MaterialProperties,
    ReadoutGeometry,
    SamplingStructure,
    load_geometry,
)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "geometry.yaml"

SUPERLAYER_FIBER_AXES = (
    "x",
    "y",
    "x",
    "y",
    "x",
    "y",
    "x",
    "y",
    "x",
)

LAYER_FIBER_AXES = (
    "x",
    "x",
    "y",
    "y",
    "x",
    "x",
    "y",
    "y",
    "x",
    "x",
    "y",
    "y",
    "x",
    "x",
    "y",
    "y",
    "x",
    "x",
)


def make_geometry() -> ECALGeometry:
    """Return the documented ideal geometry without reading YAML.

    Direct construction lets unit tests isolate dataclass validation from the
    configuration parser.
    """

    return ECALGeometry(
        active_volume=ActiveVolume(
            width_x_mm=648.0,
            width_y_mm=648.0,
            depth_z_mm=166.5,
        ),
        readout=ReadoutGeometry(
            number_of_superlayers=9,
            number_of_layers=18,
            cells_per_layer=72,
            cell_pitch_mm=9.0,
            photomultipliers=324,
            anodes_per_photomultiplier=4,
            approximate_fibers_per_cell=35,
            superlayer_fiber_axes=SUPERLAYER_FIBER_AXES,
        ),
        sampling_structure=SamplingStructure(
            superlayer_thickness_mm=18.5,
            absorber_foils_per_superlayer=11,
            absorber_foil_thickness_mm=1.0,
            fiber_planes_per_superlayer=10,
            fiber_diameter_mm=1.0,
            fiber_horizontal_pitch_mm=1.35,
            fiber_row_spacing_mm=1.73,
            adjacent_row_stagger_fraction=0.5,
            standard_absorber_material="lead",
            terminal_foil_material="aluminum",
        ),
        material_properties=MaterialProperties(
            average_density_g_cm3=6.8,
            relative_volume_lead=1.0,
            relative_volume_scintillating_fiber=0.57,
            relative_volume_optical_glue=0.15,
            effective_critical_energy_mev=7.6,
        ),
        material_depth=MaterialDepth(
            total_depth_x0=17.0,
            total_depth_lambda_i=0.6,
        ),
        coordinate_system=CoordinateSystem(
            origin="front_face_center",
            positive_z_direction="into_ecal",
            theta_reference_axis="positive_z",
        ),
    )


def write_modified_config(tmp_path: Path, change) -> Path:
    """Write a temporary schema-v2 config after applying one mutation."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    change(config)
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_loads_documented_geometry() -> None:
    """Load the schema-v2 detector facts used by the ideal FastMC."""

    geometry = load_geometry(CONFIG_PATH)

    assert geometry.active_volume.width_x_mm == pytest.approx(648.0)
    assert geometry.active_volume.width_y_mm == pytest.approx(648.0)
    assert geometry.active_volume.depth_z_mm == pytest.approx(166.5)

    assert geometry.readout.number_of_superlayers == 9
    assert geometry.readout.number_of_layers == 18
    assert geometry.readout.cells_per_layer == 72
    assert geometry.readout.cell_pitch_mm == pytest.approx(9.0)
    assert geometry.readout.photomultipliers == 324
    assert geometry.readout.anodes_per_photomultiplier == 4
    assert geometry.readout.approximate_fibers_per_cell == 35
    assert geometry.readout.superlayer_fiber_axes == SUPERLAYER_FIBER_AXES

    assert (
        geometry.sampling_structure.superlayer_thickness_mm
        == pytest.approx(18.5)
    )
    assert geometry.sampling_structure.absorber_foils_per_superlayer == 11
    assert geometry.sampling_structure.absorber_foil_thickness_mm == pytest.approx(
        1.0
    )
    assert geometry.sampling_structure.fiber_planes_per_superlayer == 10
    assert geometry.sampling_structure.fiber_diameter_mm == pytest.approx(1.0)
    assert (
        geometry.sampling_structure.fiber_horizontal_pitch_mm
        == pytest.approx(1.35)
    )
    assert geometry.sampling_structure.fiber_row_spacing_mm == pytest.approx(
        1.73
    )
    assert (
        geometry.sampling_structure.adjacent_row_stagger_fraction
        == pytest.approx(0.5)
    )

    assert geometry.material_properties.average_density_g_cm3 == pytest.approx(
        6.8
    )
    assert (
        geometry.material_properties.effective_critical_energy_mev
        == pytest.approx(7.6)
    )
    assert geometry.material_depth.total_depth_x0 == pytest.approx(17.0)
    assert geometry.material_depth.total_depth_lambda_i == pytest.approx(0.6)


def test_preserves_stage_i_flat_geometry_api() -> None:
    """Keep the original geometry, tracking, cell-mapping and event-model code working while new code uses structured components."""

    geometry = load_geometry(CONFIG_PATH)

    assert geometry.width_x_mm == geometry.active_volume.width_x_mm
    assert geometry.width_y_mm == geometry.active_volume.width_y_mm
    assert geometry.depth_z_mm == geometry.active_volume.depth_z_mm
    assert geometry.number_of_superlayers == geometry.readout.number_of_superlayers
    assert geometry.number_of_layers == geometry.readout.number_of_layers
    assert geometry.cells_per_layer == geometry.readout.cells_per_layer
    assert geometry.cell_pitch_mm == geometry.readout.cell_pitch_mm
    assert geometry.total_depth_x0 == geometry.material_depth.total_depth_x0
    assert (
        geometry.total_depth_lambda_i
        == geometry.material_depth.total_depth_lambda_i
    )


def test_derives_detector_cross_checks() -> None:
    """Verify independent descriptions of the detector agree."""

    geometry = load_geometry(CONFIG_PATH)

    assert geometry.total_cells == 1296
    assert geometry.total_readout_anodes == 1296

    assert geometry.total_absorber_foil_positions == 99
    assert geometry.number_of_lead_foils == 98
    assert geometry.number_of_aluminum_foils == 1

    assert geometry.mean_readout_slice_thickness_mm == pytest.approx(9.25)
    assert geometry.mean_readout_slice_thickness_x0 == pytest.approx(17 / 18)
    assert geometry.nominal_moliere_radius_mm == pytest.approx(18.0)

    assert len(geometry.uniform_layer_centers_z_mm) == 18
    assert geometry.uniform_layer_centers_z_mm[0] == pytest.approx(4.625)
    assert geometry.uniform_layer_centers_z_mm[-1] == pytest.approx(161.875)

    assert len(geometry.uniform_layer_bounds_x0) == 18
    assert geometry.uniform_layer_bounds_x0[0] == pytest.approx((0.0, 17 / 18))
    assert geometry.uniform_layer_bounds_x0[-1] == pytest.approx(
        (17 * 17 / 18, 17.0)
    )

    assert geometry.x_bounds_mm == pytest.approx((-324.0, 324.0))
    assert geometry.y_bounds_mm == pytest.approx((-324.0, 324.0))
    assert geometry.z_bounds_mm == pytest.approx((0.0, 166.5))
    assert geometry.layer_fiber_axes == LAYER_FIBER_AXES


def test_normalizes_relative_material_volumes() -> None:
    """Convert the published 1:0.57:0.15 ratio only when needed."""

    geometry = load_geometry(CONFIG_PATH)
    fractions = geometry.material_properties.normalized_volume_fractions

    assert sum(fractions) == pytest.approx(1.0)
    assert fractions == pytest.approx(
        (
            1.0 / 1.72,
            0.57 / 1.72,
            0.15 / 1.72,
        )
    )


@pytest.mark.parametrize(
    ("component", "replacement", "error_type", "message"),
    [
        (
            "active_volume",
            ActiveVolume(width_x_mm=647.0, width_y_mm=648.0, depth_z_mm=166.5),
            ValueError,
            "width_x",
        ),
        (
            "readout",
            ReadoutGeometry(
                number_of_superlayers=9,
                number_of_layers=18,
                cells_per_layer=72,
                cell_pitch_mm=9.0,
                photomultipliers=323,
                anodes_per_photomultiplier=4,
                approximate_fibers_per_cell=35,
                superlayer_fiber_axes=SUPERLAYER_FIBER_AXES,
            ),
            ValueError,
            "anode",
        ),
        (
            "sampling_structure",
            SamplingStructure(
                superlayer_thickness_mm=18.0,
                absorber_foils_per_superlayer=11,
                absorber_foil_thickness_mm=1.0,
                fiber_planes_per_superlayer=10,
                fiber_diameter_mm=1.0,
                fiber_horizontal_pitch_mm=1.35,
                fiber_row_spacing_mm=1.73,
                adjacent_row_stagger_fraction=0.5,
                standard_absorber_material="lead",
                terminal_foil_material="aluminum",
            ),
            ValueError,
            "active ECAL depth",
        ),
    ],
)
def test_rejects_inconsistent_detector_components(
    component: str,
    replacement: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Fail when independently specified detector descriptions disagree."""

    geometry = make_geometry()

    with pytest.raises(error_type, match=message):
        replace(geometry, **{component: replacement})


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (
            lambda: ActiveVolume(
                width_x_mm=648.0,
                width_y_mm=648.0,
                depth_z_mm=0.0,
            ),
            "finite and positive",
        ),
        (
            lambda: ReadoutGeometry(
                number_of_superlayers=9,
                number_of_layers=17,
                cells_per_layer=72,
                cell_pitch_mm=9.0,
                photomultipliers=324,
                anodes_per_photomultiplier=4,
                approximate_fibers_per_cell=35,
                superlayer_fiber_axes=SUPERLAYER_FIBER_AXES,
            ),
            "twice",
        ),
        (
            lambda: SamplingStructure(
                superlayer_thickness_mm=18.5,
                absorber_foils_per_superlayer=11,
                absorber_foil_thickness_mm=1.0,
                fiber_planes_per_superlayer=10,
                fiber_diameter_mm=1.0,
                fiber_horizontal_pitch_mm=1.35,
                fiber_row_spacing_mm=1.73,
                adjacent_row_stagger_fraction=float("nan"),
                standard_absorber_material="lead",
                terminal_foil_material="aluminum",
            ),
            "finite",
        ),
        (
            lambda: MaterialProperties(
                average_density_g_cm3=6.8,
                relative_volume_lead=1.0,
                relative_volume_scintillating_fiber=0.57,
                relative_volume_optical_glue=0.15,
                effective_critical_energy_mev=-1.0,
            ),
            "finite and positive",
        ),
        (
            lambda: CoordinateSystem(
                origin="center",
                positive_z_direction="into_ecal",
                theta_reference_axis="positive_z",
            ),
            "front_face_center",
        ),
    ],
)
def test_rejects_invalid_component_values(
    constructor,
    message: str,
) -> None:
    """Validate local component invariants before composition."""

    with pytest.raises((TypeError, ValueError), match=message):
        constructor()


def test_rejects_non_sequence_superlayer_fiber_axes(
    tmp_path: Path,
) -> None:
    """Require YAML sequence syntax for the ordered superlayer axes."""

    def change(config: dict[str, object]) -> None:
        config["readout"]["superlayer_fiber_axes"] = "xyxyxyxyx"

    path = write_modified_config(tmp_path, change)

    with pytest.raises(GeometryConfigError, match="YAML sequence"):
        load_geometry(path)


def test_rejects_missing_config_file(tmp_path: Path) -> None:
    """Report a clear error for an absent geometry file."""

    with pytest.raises(FileNotFoundError, match="not found"):
        load_geometry(tmp_path / "missing.yaml")


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    """Convert parser failures into a geometry-specific exception."""

    path = tmp_path / "geometry.yaml"
    path.write_text("active_volume: [", encoding="utf-8")

    with pytest.raises(GeometryConfigError, match="Could not parse"):
        load_geometry(path)


def test_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    """Do not silently interpret a future geometry schema."""

    path = write_modified_config(
        tmp_path,
        lambda config: config.__setitem__("schema_version", 3),
    )

    with pytest.raises(GeometryConfigError, match="Unsupported"):
        load_geometry(path)


def test_rejects_wrong_units(tmp_path: Path) -> None:
    """Keep units explicit at the scientific configuration boundary."""

    def change(config: dict[str, object]) -> None:
        config["units"]["length"] = "cm"

    path = write_modified_config(tmp_path, change)

    with pytest.raises(GeometryConfigError, match="units must equal"):
        load_geometry(path)


def test_rejects_missing_required_key(tmp_path: Path) -> None:
    """Reject incomplete detector descriptions."""

    def change(config: dict[str, object]) -> None:
        del config["sampling_structure"]["fiber_diameter"]

    path = write_modified_config(tmp_path, change)

    with pytest.raises(GeometryConfigError, match="missing keys"):
        load_geometry(path)


def test_rejects_unexpected_key(tmp_path: Path) -> None:
    """Reject misspelled or undocumented scientific assumptions."""

    def change(config: dict[str, object]) -> None:
        config["active_volume"]["width_z"] = 123.0

    path = write_modified_config(tmp_path, change)

    with pytest.raises(GeometryConfigError, match="unexpected keys"):
        load_geometry(path)
