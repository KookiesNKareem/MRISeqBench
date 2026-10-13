import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mriseqbench.catalog import Catalog, load_yaml, manifest_directory
from mriseqbench.models import MODELS, Timing
from tests.fixtures.legacy_catalog import legacy_task

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
    assert len(cases) == 2
    assert {c["sequence_level"] for c in cases} == {"S1", "S2"}
    assert all(c["image_policy"] == "static" for c in cases)
    assert all(c["object"]["id"].startswith("cardiac_2d") for c in cases)
    assert all(c["physics"]["id"] == "static_b0" for c in cases)
    assert all("object_level" not in c for c in cases)


@pytest.mark.parametrize(
    "change",
    [
        {"objects": ["tissue_discs@1"]},
        {"physics": ["ideal@1"]},
        {"physics": ["rigid_motion@1"]},
    ],
)
def test_registered_suites_cannot_reintroduce_phantom_axes(catalog_root, change):
    update(catalog_root, "suites", "core", lambda d: d["cases"][0].update(change))
    with pytest.raises(ValueError, match="benchmark suites require"):
        Catalog(catalog_root)


def test_physics_axis_is_b0_only_and_rejects_old_p4():
    from mriseqbench.models import Physics

    catalog = Catalog(ROOT)
    for profile in catalog.documents["physics"].values():
        assert profile.level in {None, "P0", "P2", "P3"}
        assert all(effect.kind != "b1" for effect in profile.effects)
        if profile.level == "P2":
            assert {effect.kind for effect in profile.effects} == {"b0"}
    profile = catalog.get("physics", "b0_smooth").model_dump()
    profile["level"] = "P4"
    with pytest.raises(ValidationError):
        Physics.model_validate(profile)
    profile["level"] = "P2"
    profile["effects"] = [{"kind": "b1", "model": "constant", "scale": 0.8}]
    with pytest.raises(ValidationError):
        Physics.model_validate(profile)
    profile["level"] = "P3"
    profile["effects"] = [
        {
            "kind": "time_variation",
            "model": "sinusoidal",
            "target": "b1_scale",
            "amplitude_scale": 0.1,
            "period_s": 1,
        }
    ]
    with pytest.raises(ValidationError):
        Physics.model_validate(profile)


@pytest.mark.parametrize(
    "folder,name,mutate",
    [
        ("tasks", "gre_t1w", lambda d: d.update(typo=1)),
        ("tasks", "gre_t1w", lambda d: d.update(object="missing@1")),
        ("tasks", "gre_t1w", lambda d: d.update(object="tissue_discs@2")),
        ("tasks", "gre_t1w", lambda d: d["sequence"].update(level="S6")),
        ("physics", "b0_smooth", lambda d: d.update(level="P0")),
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


def test_smoke_uses_static_cardiac_configuration():
    catalog = Catalog(ROOT)
    (case,) = catalog.cases(catalog.get("suites", "smoke"))
    assert case["object"]["id"].startswith("cardiac_2d")
    assert case["physics"]["id"] == "static_b0"
    assert case["evaluation"]["scope"] == "smoke"


def test_invalid_yaml_is_readable_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unterminated")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_yaml(path)


@pytest.mark.parametrize("level", ["S1", "S2", "S3", "S4"])
def test_all_sequence_levels_supported(catalog_root, level):
    def change_level(document):
        document["sequence"]["level"] = level
        if level == "S4":
            application = load_yaml(ROOT / "benchmark/tasks/adiabatic_inversion.yaml")
            document["sequence"]["capabilities"] = application["sequence"][
                "capabilities"
            ]
            document["sequence"]["temporal"] = application["sequence"]["temporal"]

    update(catalog_root, "tasks", "gre_t1w", change_level)
    assert Catalog(catalog_root).get("tasks", "gre_t1w").sequence.level == level


def test_removed_temporal_tier_and_application_renumbering():
    from typing import get_args

    from mriseqbench.models import SEQUENCE_CATEGORIES, Capability, SequenceSpec

    assert tuple(SEQUENCE_CATEGORIES) == ("S1", "S2", "S3", "S4")
    assert "inversion" in SEQUENCE_CATEGORIES["S4"]
    assert not {"cine", "kt_blast", "kt_sense", "twist", "keyhole"} & set(
        get_args(Capability)
    )
    original = load_yaml(ROOT / "benchmark/tasks/adiabatic_inversion.yaml")["sequence"]
    assert SequenceSpec.model_validate(original).level == "S4"
    with pytest.raises(ValidationError):
        SequenceSpec.model_validate({**original, "level": "S5"})
    for name in (
        "radial_cine",
        "interleaved_cine",
        "real_time_radial",
        "keyhole_dynamic",
        "kt_blast_sampling",
        "kt_sense_sampling",
        "twist_dynamic",
        "inversion_prepared",
        "multi_echo_gre",
        "mrf_schedule",
    ):
        assert not (ROOT / f"benchmark/tasks/{name}.yaml").exists()
        assert not (ROOT / f"benchmark/tasks/{name}_research.yaml").exists()


@pytest.mark.parametrize(
    "family", ["adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping"]
)
def test_cardiac_application_contracts(family):
    from mriseqbench.models import Task

    detailed = Task.model_validate(load_yaml(ROOT / f"benchmark/tasks/{family}.yaml"))
    research = Task.model_validate(
        load_yaml(ROOT / f"tests/fixtures/legacy_research_tasks/{family}_research.yaml")
    )
    assert detailed.sequence.level == research.sequence.level == "S4"
    assert {"triggering", "gating"} <= set(detailed.sequence.capabilities)
    assert "bssfp" in detailed.sequence.capabilities
    assert "spoiled_gre" not in detailed.sequence.capabilities
    assert "Balance net gradient area on all three axes" in detailed.objective
    assert "180 degrees" in detailed.objective
    assert "receiver phases coherent" in detailed.objective
    timing = detailed.sequence.temporal
    assert timing.trigger_source == "ecg"
    assert timing.rr_interval_ms == 1000
    assert timing.trigger_delay_ms == 600
    assert timing.acquisition_window_ms == 200
    assert research.sequence.temporal is None
    assert research.sequence.application is None
    assert any(
        "ECG R-wave" in requirement for requirement in research.identity_requirements
    )


@pytest.mark.parametrize(
    "family", ["adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping"]
)
@pytest.mark.parametrize("gre_mode", ["replace_bssfp", "add_spoiled_gre"])
def test_detailed_s4_magnetization_preparations_reject_spoiled_gre(family, gre_mode):
    from mriseqbench.models import Task

    task = load_yaml(ROOT / f"benchmark/tasks/{family}.yaml")
    if gre_mode == "replace_bssfp":
        task["sequence"]["capabilities"].remove("bssfp")
    task["sequence"]["capabilities"].append("spoiled_gre")
    with pytest.raises(ValidationError, match="require bSSFP readouts"):
        Task.model_validate(task)


@pytest.mark.parametrize(
    "changes",
    [
        {"rr_interval_ms": None},
        {"trigger_source": None},
        {"trigger_delay_ms": -1},
        {"acquisition_window_ms": 0},
        {"trigger_delay_ms": 900},
        {"rr_interval_ms": 500},
    ],
)
def test_invalid_cardiac_windows_rejected(changes):
    from mriseqbench.models import TemporalSpec

    timing = load_yaml(ROOT / "benchmark/tasks/adiabatic_inversion.yaml")["sequence"][
        "temporal"
    ]
    with pytest.raises(ValidationError):
        TemporalSpec.model_validate({**timing, **changes})


def test_s4_requires_ecg_contract_and_research_keeps_timing_public():
    from mriseqbench.models import Task

    path = ROOT / "benchmark/tasks/adiabatic_inversion.yaml"
    detailed = load_yaml(path)
    detailed["sequence"].pop("temporal")
    with pytest.raises(ValidationError, match="ECG timing contract"):
        Task.model_validate(detailed)
    detailed = load_yaml(path)
    detailed["sequence"]["capabilities"].remove("triggering")
    with pytest.raises(ValidationError, match="triggering and gating"):
        Task.model_validate(detailed)
    research = load_yaml(
        ROOT / "tests/fixtures/legacy_research_tasks/adiabatic_inversion_research.yaml"
    )
    research["sequence"]["temporal"] = load_yaml(path)["sequence"]["temporal"]
    with pytest.raises(ValidationError, match="hidden numerical"):
        Task.model_validate(research)


def test_complete_vocabulary_and_current_examples():
    from typing import get_args

    from mriseqbench.models import SEQUENCE_CATEGORIES, Capability

    assert set(get_args(Capability)) == {
        tag for tags in SEQUENCE_CATEGORIES.values() for tag in tags
    }
    assert {"zte", "grase", "wave_caipi", "ptx", "asl", "navigator"} <= set(
        get_args(Capability)
    )
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "axes_examples"))
    assert {case["sequence_level"] for case in cases} == {"S1", "S2", "S3", "S4"}
    assert all(case["evaluation"]["calibrated"] is True for case in cases)
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert all(
        case["evaluation"]["reconstruction"].startswith("bart_") for case in cases
    )
    assert all(case["evaluation"]["intrinsic_tissue_response"] for case in cases)


def test_axes_keep_geometry_and_physics_separate():
    catalog = Catalog(ROOT)
    ideal = catalog.resolve(
        legacy_task(catalog, "rich_geometry_gre"),
        catalog.get("physics", "ideal"),
        "nested_ellipses@1",
    )
    moving = catalog.resolve(
        legacy_task(catalog, "rich_geometry_gre"),
        catalog.get("physics", "deformation"),
        "nested_ellipses@1",
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
    ) == ("P1", "P3", "P3")


@pytest.mark.parametrize(
    "profile",
    [
        "b0_offset",
        "b0_localized",
        "rigid_steps",
        "deformation",
        "b0_drift",
        "dynamic_tissue",
    ],
)
def test_extended_profiles_resolve(profile):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"),
        catalog.get("physics", profile),
        "tissue_discs@1",
    )
    assert case["physics"]["id"] == profile
    assert case["physics"]["effects"]


@pytest.mark.parametrize(
    "folder,name,mutate",
    [
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
        ("tasks", "stack_of_stars", lambda d: d["sequence"].update(fov_mm=[200, 200])),
        (
            "tasks",
            "partial_fourier",
            lambda d: d["sequence"]["sampling"].update(partial_fourier_fraction=0.5),
        ),
        (
            "tasks",
            "t1_mapping",
            lambda d: d["sequence"]["application"].update(inversion_times_ms=[-1]),
        ),
    ],
)
def test_reject_extended_invalid_contracts(catalog_root, folder, name, mutate):
    update(catalog_root, folder, name, mutate)
    if folder == "physics":
        # Legacy profiles remain readable, but are absent from active suites.
        # Resolve one explicitly to exercise contextual dimensionality/material checks.
        update(
            catalog_root,
            "suites",
            "core",
            lambda d: d["cases"][0].update(
                physics=[f"{name}@1"], objects=["tissue_discs@1"]
            ),
        )
    with pytest.raises(ValueError):
        Catalog(catalog_root)


def test_3d_object_and_acquisition_resolve():
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "stack_of_stars"), catalog.get("physics", "ideal")
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
        catalog.get("tasks", "gre_t1w"),
        catalog.get("physics", "ideal"),
        "tissue_discs@1",
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
    profile["effects"][0]["asset"]["units"] = "relative"
    with pytest.raises(ValueError, match="B0 map units must be Hz"):
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
    assert cases[0]["case_id"] == "gre_t1w-v1--ideal-v1--object-tissue_discs-v1"
    assert cases[1]["case_id"] == "gre_t1w-v1--ideal-v1--object-nested_ellipses-v1"
    assert cases[1]["task"]["object"] == "nested_ellipses@1"
    assert cases[1]["object"]["id"] == "nested_ellipses"
    assert catalog.get("tasks", "gre_t1w").object == "cardiac_2d@1"
    with pytest.raises(ValueError, match="dimensions"):
        catalog.resolve(
            catalog.get("tasks", "gre_t1w"),
            catalog.get("physics", "ideal"),
            "volume_ellipsoids@1",
        )


def test_full_suite_covers_every_benchmark_task_on_static_cardiac_phantom():
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "full"))
    tasks = {
        task.id
        for task in catalog.documents["tasks"].values()
        if catalog.get("evaluators", task.evaluation).scope == "benchmark"
    }
    assert len(cases) == 3 * len(tasks)
    assert {case["task"]["id"] for case in cases} == tasks
    assert len({case["case_id"] for case in cases}) == len(cases)
    for task in tasks:
        selected = [case for case in cases if case["task"]["id"] == task]
        assert all(case["image_policy"] == "static" for case in selected)
        assert all("object_level" not in case for case in selected)
        assert all(
            case["object"]["id"].startswith(
                f"cardiac_{len(case['task']['sequence']['matrix'])}d"
            )
            for case in selected
        )
        assert all(case["physics"]["effects"] for case in selected)
        assert all(
            effect["kind"] == "b0"
            for case in selected
            for effect in case["physics"]["effects"]
        )
        assert all(
            case["object"]["definition"]["dimensions"]
            == len(case["task"]["sequence"]["matrix"])
            for case in selected
        )
        assert {case["hardware"]["id"] for case in selected} == {
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
                    "physics": ["static_b0@1"],
                    "hardware": [
                        "high_performance@1",
                        "low_field@1",
                        "standard_1_5t@1",
                    ],
                }
            ],
        }
    )
    cases = catalog.cases(suite)
    assert cases[0]["case_id"] == "gre_t1w-v1--static_b0-v1"
    assert cases[1]["case_id"].endswith("--hardware-low_field-v1")
    assert cases[1]["task"]["hardware"] == "low_field@1"
    assert cases[1]["hardware"]["field_strength_T"] == 0.55
    assert cases[2]["hardware"]["max_gradient_mT_per_m"] == 33
    assert "benchmark assumptions" in cases[1]["hardware"]["provenance"]["notes"]
    assert task.hardware == "high_performance@1"
    assert (
        catalog.resolve(
            task, catalog.get("physics", "static_b0"), hardware_ref="high_performance"
        )["case_id"]
        == cases[0]["case_id"]
    )
    assert catalog.find_case(cases[1]["case_id"]) == cases[1]
    with pytest.raises(ValueError, match="unknown hardware"):
        catalog.resolve(
            task, catalog.get("physics", "static_b0"), hardware_ref="missing@1"
        )
    suite.cases[0].hardware.append("low_field@1")
    with pytest.raises(ValueError, match="duplicate suite case"):
        catalog.cases(suite)


def test_active_tasks_have_no_research_variants():
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "full"))
    assert len(cases) == 75
    assert len(catalog.documents["tasks"]) == 26  # 25 benchmark tasks plus smoke.
    assert all(case["task"]["prompt_mode"] == "detailed" for case in cases)
    assert all(
        not task.id.endswith("_research")
        for task in catalog.documents["tasks"].values()
    )
    assert not list((ROOT / "benchmark/tasks").glob("*_research.yaml"))


def test_research_contract_rejects_hidden_targets():
    from mriseqbench.models import Task

    doc = Task.model_validate(
        load_yaml(ROOT / "tests/fixtures/legacy_research_tasks/gre_t1w_research.yaml")
    ).model_dump()
    doc["sequence"]["te_ms"] = {"target": 5, "tolerance": 1}
    with pytest.raises(ValidationError, match="hidden numerical"):
        Task.model_validate(doc)
