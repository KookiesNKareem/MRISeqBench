import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mriseqbench.catalog import Catalog, load_yaml, manifest_directory
from mriseqbench.models import MODELS, Timing

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def catalog_root(tmp_path):
    for folder in MODELS:
        shutil.copytree(
            manifest_directory(ROOT, folder), manifest_directory(tmp_path, folder)
        )
    return tmp_path


def update(root, folder, name, mutate):
    path = manifest_directory(root, folder) / f"{name}.yaml"
    value = load_yaml(path)
    mutate(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_explicit_cases_and_axes():
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "core"))
    assert len(cases) == 8
    assert [c["object_level"] for c in cases[:6]] == [
        "P1",
        "P2",
        "P2",
        "P3",
        "P4",
        "P4",
    ]
    assert {c["sequence_level"] for c in cases} == {"S1", "S2"}
    assert all(c["evaluation"]["calibrated"] is False for c in cases)


@pytest.mark.parametrize(
    "folder,name,mutate",
    [
        ("tasks", "gre_t1w", lambda d: d.update(typo=1)),
        ("tasks", "gre_t1w", lambda d: d.update(object="missing@1")),
        ("tasks", "gre_t1w", lambda d: d.update(object="tissue_discs@2")),
        ("tasks", "gre_t1w", lambda d: d["sequence"].update(level="S6")),
        ("physics", "b0_smooth", lambda d: d.update(level="P0")),
        ("physics", "b1_smooth", lambda d: d["effects"][0].update(center_scale=0.1)),
        ("suites", "core", lambda d: d["cases"].append(d["cases"][0])),
        (
            "objects",
            "tissue_discs",
            lambda d: d.update(phantom="missing@1"),
        ),
        (
            "evaluators",
            "t1_contrast",
            lambda d: d["metrics"][0].update(min=float("nan")),
        ),
    ],
)
def test_reject_invalid_catalog(catalog_root, folder, name, mutate):
    update(catalog_root, folder, name, mutate)
    with pytest.raises(ValueError):
        Catalog(catalog_root)


def test_reject_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: first\nid: second\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_yaml(path)


def test_timing_requires_explicit_tolerance():
    with pytest.raises(ValidationError):
        Timing(target=5)
    with pytest.raises(ValidationError):
        Timing(max=-1)


def test_smoke_cannot_claim_perturbed_physics():
    catalog = Catalog(ROOT)
    with pytest.raises(ValueError, match="ideal physics"):
        catalog.resolve(
            catalog.get("tasks", "pulseq_smoke"), catalog.get("physics", "b0_smooth")
        )


def test_invalid_yaml_is_readable_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unterminated")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_yaml(path)


@pytest.mark.parametrize("level", ["S1", "S2", "S3", "S4", "S5"])
def test_all_sequence_levels_supported(catalog_root, level):
    update(
        catalog_root, "tasks", "gre_t1w", lambda d: d["sequence"].update(level=level)
    )
    assert Catalog(catalog_root).get("tasks", "gre_t1w").sequence.level == level


def test_complete_vocabulary_and_draft_examples():
    from typing import get_args

    from mriseqbench.models import SEQUENCE_CATEGORIES, Capability

    assert set(get_args(Capability)) == {
        tag for tags in SEQUENCE_CATEGORIES.values() for tag in tags
    }
    assert {"zte", "grase", "wave_caipi", "kt_pca", "ptx", "asl", "navigator"} <= set(
        get_args(Capability)
    )
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "axes_examples"))
    assert {case["sequence_level"] for case in cases} == {"S1", "S2", "S3", "S4", "S5"}
    assert all(case["evaluation"]["calibrated"] is False for case in cases)
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_axes_keep_geometry_and_physics_separate():
    catalog = Catalog(ROOT)
    ideal = catalog.resolve(
        catalog.get("tasks", "rich_geometry_gre"), catalog.get("physics", "ideal")
    )
    moving = catalog.resolve(
        catalog.get("tasks", "rich_geometry_gre"), catalog.get("physics", "deformation")
    )
    assert (ideal["geometry_level"], ideal["physics_level"], ideal["object_level"]) == (
        "P1",
        "P0",
        "P1",
    )
    assert (
        moving["geometry_level"],
        moving["physics_level"],
        moving["object_level"],
    ) == ("P1", "P4", "P4")


@pytest.mark.parametrize(
    "profile",
    [
        "b0_offset",
        "b0_localized",
        "b1_low",
        "b1_localized",
        "rigid_steps",
        "deformation",
        "b0_drift",
        "b1_drift",
        "dynamic_tissue",
    ],
)
def test_extended_profiles_resolve(profile):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), catalog.get("physics", profile)
    )
    assert case["physics"]["id"] == profile
    assert case["physics"]["effects"]


@pytest.mark.parametrize(
    "folder,name,mutate",
    [
        (
            "physics",
            "b1_localized",
            lambda d: d["effects"][0].update(amplitude_scale=-1),
        ),
        (
            "physics",
            "b0_localized",
            lambda d: d["effects"][0].update(sigma_mm=[20, 30, 40]),
        ),
        (
            "physics",
            "rigid_steps",
            lambda d: d["effects"][0]["poses"][1].update(time_s=0),
        ),
        (
            "physics",
            "rigid_steps",
            lambda d: d["effects"][0]["poses"][1].update(translation_mm=[0, 0, 1]),
        ),
        (
            "physics",
            "deformation",
            lambda d: d["effects"][0].update(strain_amplitude=[1, 0]),
        ),
        ("physics", "b1_drift", lambda d: d["effects"][0].update(amplitude_scale=1)),
        (
            "physics",
            "dynamic_tissue",
            lambda d: d["effects"][0].update(material="missing"),
        ),
        (
            "objects",
            "nested_ellipses",
            lambda d: d.update(phantom="nested_ellipses@2"),
        ),
        (
            "objects",
            "nested_ellipses",
            lambda d: d.update(shapes=[]),
        ),
        ("tasks", "volume_gre", lambda d: d["sequence"].update(fov_mm=[200, 200])),
        (
            "tasks",
            "partial_fourier",
            lambda d: d["sequence"]["sampling"].update(partial_fourier_fraction=0.5),
        ),
        (
            "tasks",
            "mrf_schedule",
            lambda d: d["sequence"]["application"].update(tr_schedule_ms=[12]),
        ),
    ],
)
def test_reject_extended_invalid_contracts(catalog_root, folder, name, mutate):
    update(catalog_root, folder, name, mutate)
    with pytest.raises(ValueError):
        Catalog(catalog_root)


def test_3d_object_and_acquisition_resolve():
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "volume_gre"), catalog.get("physics", "ideal")
    )
    assert len(case["task"]["sequence"]["matrix"]) == 3
    assert len(case["object"]["definition"]["grid"]) == 3


def test_asset_hash_path_and_dimension_contract(catalog_root):
    import hashlib

    from mriseqbench.models import Asset

    path = catalog_root / "labels.npz"
    # Catalog verifies bytes and declared metadata; it does not decode array contents.
    path.write_bytes(b"declared-test-asset")
    asset = {
        "path": "labels.npz",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "npz",
        "array": "labels",
        "units": "label",
        "shape": [4, 4],
        "spacing_mm": [1, 1],
        "origin_mm": [-2, -2],
    }
    catalog = Catalog(catalog_root)
    catalog.validate_asset(Asset.model_validate(asset))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        catalog.validate_asset(Asset.model_validate(asset))
    with pytest.raises(ValueError, match="relative"):
        Asset.model_validate(dict(asset, path="../labels.npz"))
    with pytest.raises(ValueError, match="dimensions"):
        Asset.model_validate(dict(asset, spacing_mm=[1, 1, 1]))
    path.unlink()
    with pytest.raises(ValueError, match="missing asset"):
        catalog.validate_asset(Asset.model_validate(asset))


def test_object_selects_fixed_koma_constructor():
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), catalog.get("physics", "ideal")
    )
    assert case["object"]["phantom"] == "tissue_discs@1"
    assert case["object"]["definition"]["materials"] == ["wm", "gm", "csf", "fat"]
    assert "tissues" not in case["object"]


def test_fixed_phantom_must_fit_acquisition(catalog_root):
    update(
        catalog_root,
        "tasks",
        "gre_t1w",
        lambda d: d["sequence"].update(fov_mm=[100, 100]),
    )
    with pytest.raises(ValueError, match="beyond FOV"):
        Catalog(catalog_root)


def test_map_fields_have_explicit_units():
    from mriseqbench.models import Physics

    asset = {
        "path": "field.npz",
        "sha256": "0" * 64,
        "format": "npz",
        "array": "b0",
        "units": "Hz",
        "shape": [4, 4],
        "spacing_mm": [1, 1],
        "origin_mm": [-2, -2],
    }
    profile = {
        "schema_version": 1,
        "id": "map",
        "version": 1,
        "description": "test",
        "level": "P2",
        "effects": [{"kind": "b0", "model": "map", "asset": asset}],
    }
    assert Physics.model_validate(profile).effects[0].asset.units == "Hz"
    profile["effects"][0]["kind"] = "b1"
    with pytest.raises(ValueError, match="relative"):
        Physics.model_validate(profile)


def test_suite_can_reuse_task_with_explicit_object_override():
    from mriseqbench.models import Suite

    catalog = Catalog(ROOT)
    suite = Suite.model_validate(
        {
            "schema_version": 1,
            "id": "composition",
            "version": 1,
            "cases": [
                {
                    "task": "gre_t1w@1",
                    "objects": ["tissue_discs@1", "nested_ellipses@1"],
                    "physics": ["ideal@1"],
                }
            ],
        }
    )
    cases = catalog.cases(suite)
    assert len(cases) == 2
    assert cases[0]["case_id"] == "gre_t1w-v1--ideal-v1"
    assert cases[1]["case_id"] == "gre_t1w-v1--ideal-v1--object-nested_ellipses-v1"
    assert cases[1]["task"]["object"] == "nested_ellipses@1"
    assert cases[1]["object"]["id"] == "nested_ellipses"
    assert catalog.get("tasks", "gre_t1w").object == "tissue_discs@1"
    with pytest.raises(ValueError, match="dimensions"):
        catalog.resolve(
            catalog.get("tasks", "gre_t1w"),
            catalog.get("physics", "ideal"),
            "volume_ellipsoids@1",
        )


def test_full_suite_covers_every_benchmark_task_across_both_axes():
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "full"))
    assert len(cases) == 2760
    tasks = {
        task.id
        for task in catalog.documents["tasks"].values()
        if catalog.get("evaluators", task.evaluation).scope == "benchmark"
    }
    assert {case["task"]["id"] for case in cases} == tasks
    assert len({case["case_id"] for case in cases}) == len(cases)
    for task in tasks:
        selected = [case for case in cases if case["task"]["id"] == task]
        assert {case["object_level"] for case in selected} == {
            "P0",
            "P1",
            "P2",
            "P3",
            "P4",
        }
        assert all(case["evaluation"]["scope"] == "benchmark" for case in selected)
        assert all(
            case["object"]["definition"]["dimensions"]
            == len(case["task"]["sequence"]["matrix"])
            for case in selected
        )
        assert {case["hardware"]["id"] for case in selected} == {
            "standard",
            "low_field",
            "standard_1_5t",
            "high_performance",
        }


def test_hardware_override_preserves_default_and_resolves_contract():
    from mriseqbench.models import Suite

    catalog = Catalog(ROOT)
    task = catalog.get("tasks", "gre_t1w")
    suite = Suite.model_validate(
        {
            "schema_version": 1,
            "id": "hardware_axis",
            "version": 1,
            "cases": [
                {
                    "task": "gre_t1w@1",
                    "physics": ["ideal@1"],
                    "hardware": ["standard@1", "low_field@1", "high_performance@1"],
                }
            ],
        }
    )
    cases = catalog.cases(suite)
    assert cases[0]["case_id"] == "gre_t1w-v1--ideal-v1"
    assert cases[1]["case_id"].endswith("--hardware-low_field-v1")
    assert cases[1]["task"]["hardware"] == "low_field@1"
    assert cases[1]["hardware"]["field_strength_T"] == 0.55
    assert cases[2]["hardware"]["max_gradient_mT_per_m"] == 80
    assert "benchmark assumptions" in cases[1]["hardware"]["provenance"]["notes"]
    assert task.hardware == "standard@1"
    assert (
        catalog.resolve(task, catalog.get("physics", "ideal"), hardware_ref="standard")[
            "case_id"
        ]
        == cases[0]["case_id"]
    )
    assert catalog.find_case(cases[1]["case_id"]) == cases[1]
    with pytest.raises(ValueError, match="unknown hardware"):
        catalog.resolve(task, catalog.get("physics", "ideal"), hardware_ref="missing@1")
    suite.cases[0].hardware.append("low_field@1")
    with pytest.raises(ValueError, match="duplicate suite case"):
        catalog.cases(suite)
