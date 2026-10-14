"""Active benchmark contracts and fail-closed manifest validation."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mriseqbench.catalog import Catalog, load_yaml, manifest_directory
from mriseqbench.models import MODELS, SEQUENCE_CATEGORIES, Metric, Task, Timing

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


def test_active_catalog_and_full_suite():
    catalog = Catalog(ROOT)
    assert len(catalog.documents["tasks"]) == 26
    assert len(catalog.documents["objects"]) == 8
    assert set(catalog.documents["physics"]) == {"static_b0@1", "static_b0_3d@1"}
    assert set(catalog.documents["evaluators"]) == {
        "pulseq_preflight@1",
        "s1@3",
        "s2@3",
        "s3@2",
        "s4@1",
    }
    cases = catalog.cases(catalog.get("suites", "full"))
    tasks = {
        task.id
        for task in catalog.documents["tasks"].values()
        if catalog.get("evaluators", task.evaluation).scope == "benchmark"
    }
    assert len(cases) == len({case["case_id"] for case in cases}) == 75
    assert {case["task"]["id"] for case in cases} == tasks
    assert {
        level: sum(case["sequence_level"] == level for case in cases)
        for level in SEQUENCE_CATEGORIES
    } == {"S1": 12, "S2": 27, "S3": 24, "S4": 12}
    for task in tasks:
        selected = [case for case in cases if case["task"]["id"] == task]
        assert {case["hardware"]["field_strength_T"] for case in selected} == {
            0.55,
            1.5,
            3,
        }
    for case in cases:
        dimensions = len(case["task"]["sequence"]["matrix"])
        assert case["image_policy"] == "static" and "object_level" not in case
        assert case["task"]["prompt_mode"] == "detailed"
        assert case["object"]["id"].startswith(f"cardiac_{dimensions}d_")
        assert case["object"]["definition"]["dimensions"] == dimensions
        assert case["physics"]["id"] == (
            "static_b0_3d" if dimensions == 3 else "static_b0"
        )
        assert all(effect["kind"] == "b0" for effect in case["physics"]["effects"])
        assert case["evaluation"]["calibrated"] is True
        assert case["evaluation"]["reconstruction"].startswith("bart_")
    vocabulary = {tag for tags in SEQUENCE_CATEGORIES.values() for tag in tags}
    assert vocabulary == {
        tag
        for task in catalog.documents["tasks"].values()
        for tag in task.sequence.capabilities
    }


def test_core_smoke_and_default_hardware():
    catalog = Catalog(ROOT)
    assert len(catalog.cases(catalog.get("suites", "core"))) == 2
    (smoke,) = catalog.cases(catalog.get("suites", "smoke"))
    assert smoke["evaluation"]["scope"] == "smoke"
    assert smoke["object"]["id"] == "cardiac_2d_3t"
    task = catalog.get("tasks", "gre_t1w")
    default = catalog.resolve(task)
    low = catalog.resolve(task, hardware_ref="low_field")
    assert task.hardware == "high_performance@1"
    assert default["case_id"] == "gre_t1w-v1--static_b0-v1"
    assert low["case_id"].endswith("--hardware-low_field-v1")
    assert low["task"]["hardware"] == "low_field@1"
    assert catalog.find_case(low["case_id"]) == low
    assert low["object"]["id"] == "cardiac_2d_055t"
    assert low["object"]["tissue_properties"]["fat"][
        "chemical_shift_hz"
    ] == pytest.approx(
        default["object"]["tissue_properties"]["fat"]["chemical_shift_hz"] * 0.55 / 3
    )


@pytest.mark.parametrize(
    "folder,name,mutate,message",
    [
        (
            "objects",
            "cardiac_2d",
            lambda d: d.update(phantom="uniform_disc@1"),
            "active tasks require cardiac anatomy",
        ),
        ("tasks", "gre_t1w", lambda d: d.update(typo=1), "Extra inputs"),
        ("tasks", "gre_t1w", lambda d: d.pop("evaluation"), "evaluation"),
        (
            "tasks",
            "gre_t1w",
            lambda d: d.update(evaluation="s2@3"),
            "evaluator must match",
        ),
        ("tasks", "gre_t1w", lambda d: d.update(object="missing@1"), "unknown objects"),
        (
            "tasks",
            "gre_t1w",
            lambda d: d.update(object="cardiac_3d@1"),
            "dimensions must match",
        ),
        (
            "tasks",
            "gre_t1w",
            lambda d: d["sequence"].update(fov_mm=[50, 50]),
            "extends beyond FOV",
        ),
        (
            "suites",
            "core",
            lambda d: d["cases"][0].update(objects=["cardiac_2d@1"]),
            "without an object axis",
        ),
        (
            "suites",
            "core",
            lambda d: d["cases"][0].update(physics=["static_b0_3d@1"]),
            "dimension-matched",
        ),
        (
            "suites",
            "core",
            lambda d: d["cases"].append(d["cases"][0]),
            "duplicate suite case",
        ),
    ],
)
def test_invalid_manifests_fail_closed(catalog_root, folder, name, mutate, message):
    update(catalog_root, folder, name, mutate)
    with pytest.raises(ValueError, match=message):
        Catalog(catalog_root)


def test_yaml_errors_are_actionable(tmp_path):
    path = tmp_path / "bad.yaml"
    for text, message in [
        ("id: first\nid: second\n", "duplicate YAML key"),
        ("key: [unterminated", "invalid YAML"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            load_yaml(path)


def test_explicit_task_and_numeric_contracts():
    task = load_yaml(ROOT / "benchmark/tasks/gre_t1w.yaml")
    for change in [{"prompt_mode": "research"}, {"identity_requirements": ["hidden"]}]:
        with pytest.raises(ValidationError):
            Task.model_validate({**task, **change})
    with pytest.raises(ValidationError):
        Timing(target=5)
    with pytest.raises(ValidationError):
        Metric(name="error", max=float("nan"))
    with pytest.raises(ValidationError):
        Metric(name="error", min=2, max=1)


@pytest.mark.parametrize("change", ["missing_ecg", "window", "gre_preparation"])
def test_cardiac_application_contract_rejects_invalid_design(change):
    task = load_yaml(ROOT / "benchmark/tasks/adiabatic_inversion.yaml")
    sequence = task["sequence"]
    if change == "missing_ecg":
        sequence.pop("temporal")
    elif change == "window":
        sequence["temporal"]["acquisition_window_ms"] = sequence["temporal"][
            "rr_interval_ms"
        ]
    else:
        sequence["capabilities"].remove("bssfp")
        sequence["capabilities"].append("spoiled_gre")
    with pytest.raises(ValidationError):
        Task.model_validate(task)


def test_field_specific_object_cannot_use_another_scanner():
    catalog = Catalog(ROOT)
    with pytest.raises(ValueError, match="scanner field strength must match"):
        catalog.resolve(
            catalog.get("tasks", "gre_t1w"),
            object_ref="cardiac_2d_3t@1",
            hardware_ref="low_field",
        )


def test_object_alias_cannot_bypass_scanner_specific_tissues():
    catalog = Catalog(ROOT)
    alias = catalog.get("objects", "cardiac_2d").model_copy(
        update={"id": "heart_alias"}
    )
    catalog.documents["objects"]["heart_alias@1"] = alias
    task = catalog.get("tasks", "gre_t1w").model_copy(
        update={"object": "heart_alias@1"}
    )
    resolved = catalog.resolve(task, hardware_ref="low_field")
    assert resolved["object"]["id"] == "cardiac_2d_055t"
    assert resolved["object"]["tissue_properties"]["myocardium"]["T1_s"] == 0.701


def test_editor_schemas_match_current_models():
    for kind, model in MODELS.items():
        schema = json.loads((ROOT / "schemas" / f"{kind}.schema.json").read_text())
        assert schema == model.model_json_schema()
