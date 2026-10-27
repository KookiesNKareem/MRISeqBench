"""Public labels must identify the existing benchmark contracts unambiguously."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.cli import main
from mriseqbench.harness.prepare import prompt, task_brief
from mriseqbench.labels import (
    HARDWARE_LABELS,
    PHANTOM_LABELS,
    SIMULATOR_LABELS,
    TASK_GROUPS,
    TASK_LABELS,
    arm_label,
    case_labels,
)
from mriseqbench.models import Arm, SuiteEntry, Task

ROOT = Path(__file__).resolve().parents[1]


def test_labels_cover_catalog_and_preserve_abstract_numbering():
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "full"))
    assert set(TASK_LABELS) == {case["task"]["id"] for case in cases}
    assert len(set(TASK_LABELS.values())) == 27
    assert {level: len(tasks) for level, tasks in TASK_GROUPS.items()} == {
        "T1": 4,
        "T2": 9,
        "T3": 9,
        "T4": 5,
    }
    assert TASK_LABELS["epi_multishot"] == "T2.2"
    assert TASK_LABELS["spiral_gre"] == "T2.6"
    assert TASK_LABELS["propeller_tse"] == "T2.7"
    assert TASK_LABELS["rf_excitation_2d"] == "T4.5"
    combinations = set()
    for case in cases:
        labels = case_labels(case)
        assert "labels" not in case
        assert labels["task"].startswith(labels["level"] + ".")
        assert (
            labels["hardware"]
            == {0.55: "H1", 1.5: "H2", 3: "H3"}[case["hardware"]["field_strength_T"]]
        )
        assert labels["phantom"] == (
            None if case["task"]["id"] == "rf_excitation_2d" else "P1"
        )
        combinations.add((labels["task"], labels["hardware"]))
    assert len(combinations) == 81
    assert set(HARDWARE_LABELS) == {case["hardware"]["id"] for case in cases}
    assert set(PHANTOM_LABELS) == {
        obj.id
        for obj in catalog.documents["objects"].values()
        if obj.phantom.startswith("cardiac_")
    }


def test_public_aliases_resolve_the_same_versioned_contracts():
    catalog = Catalog(ROOT)
    for identifier, label in TASK_LABELS.items():
        task = catalog.get("tasks", identifier)
        assert catalog.get("tasks", label) == task
        assert catalog.get("tasks", f"{label}@{task.version}") == task
    for identifier, label in HARDWARE_LABELS.items():
        assert catalog.get("hardware", label) == catalog.get("hardware", identifier)
    case = catalog.resolve(catalog.get("tasks", "T2.6"), hardware_ref="H1@1")
    assert case["case_id"] == "spiral_gre-v2--static_b0-v1--hardware-low_field-v1"
    assert catalog.find_case(case["case_id"]) == case
    assert "labels" not in task_brief(case)
    assert prompt(case).startswith(f"# {case['case_id']}\n")


def test_simulator_modifiers_are_independent_of_authoring_and_phantom():
    catalog = Catalog(ROOT)
    assert SIMULATOR_LABELS == {"mrzero": "S1", "koma": "S2", "komamripy": "S3"}
    assert arm_label(catalog.get("arms", "p1_pypulseq_mrzero")) == "P1 S1"
    assert arm_label(catalog.get("arms", "p2_pypulseq_koma")) == "P2 S3"
    assert arm_label(catalog.get("arms", "j7_pulseq_julia_mcp_koma")) == "J7 S2"
    assert arm_label(catalog.get("arms", "p0_pypulseq")) == "P0"
    assert arm_label(catalog.get("arms", "j6_koma_base_julia_mcp")) == "J6"


def test_configuration_labels_normalize_before_agent_contracts():
    entry = SuiteEntry.model_validate(
        {
            "task": "T2.6@2",
            "hardware": ["H1@1", "H3@1"],
            "physics": ["static_b0@1"],
        }
    )
    assert entry.task == "spiral_gre@2"
    assert entry.hardware == ["low_field@1", "high_performance@1"]
    for name, canonical_object in (
        ("spiral_gre", "cardiac_2d@1"),
        ("caipirinha", "cardiac_3d@1"),
    ):
        labeled = load_yaml(ROOT / f"benchmark/tasks/{name}.yaml")
        labeled["hardware"] = "H2@1"
        task = Task.model_validate(labeled)
        legacy = {
            **labeled,
            "object": canonical_object,
            "hardware": "standard_1_5t@1",
            "sequence": {**labeled["sequence"], "level": task.sequence.level},
        }
        assert task.model_dump() == Task.model_validate(legacy).model_dump()
        assert task.object == canonical_object
    for name, value in (
        ("p1_pypulseq_mrzero", "mrzero"),
        ("p2_pypulseq_koma", "komamripy"),
        ("j7_pulseq_julia_mcp_koma", "koma"),
    ):
        labeled = load_yaml(ROOT / f"benchmark/arms/{name}.yaml")
        assert (
            Arm.model_validate(labeled).model_dump()
            == Arm.model_validate({**labeled, "simulator": value}).model_dump()
        )
    invalid = load_yaml(ROOT / "benchmark/arms/p2_pypulseq_koma.yaml")
    with pytest.raises(ValidationError, match="incompatible"):
        Arm.model_validate({**invalid, "simulator": "S2"})
    with pytest.raises(ValidationError):
        SuiteEntry.model_validate({"task": "T2.99@2", "physics": ["static_b0@1"]})


def test_cli_uses_public_labels(capsys):
    assert main(["--root", str(ROOT), "list", "--suite", "full"]) == 0
    listing = capsys.readouterr().out
    assert "T2.6" in listing and "H1" in listing and "P1" in listing
    rf_line = next(line for line in listing.splitlines() if "rf_excitation_2d" in line)
    assert "T4.5" in rf_line and "assay" in rf_line
    assert main(["--root", str(ROOT), "describe", "T2.6", "--hardware", "H2"]) == 0
    brief = capsys.readouterr().out
    assert "# spiral_gre-v2--static_b0-v1--hardware-standard_1_5t-v1" in brief
    assert "field_strength_T: 1.5" in brief and "labels:" not in brief
    assert main(["--root", str(ROOT), "arms"]) == 0
    arms = capsys.readouterr().out
    assert "P1 S1 (p1_pypulseq_mrzero@1)" in arms
    assert "P2 S3 (p2_pypulseq_koma@1)" in arms
    assert "J7 S2 (j7_pulseq_julia_mcp_koma@1)" in arms
