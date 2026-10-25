"""Qualification reuse requires unchanged inputs and explicit source compatibility."""

import json
from copy import deepcopy

import pytest

from scripts.qualify_references import completed_result, sha

CURRENT = "a" * 64
PRIOR = "b" * 64


@pytest.fixture
def stored_reference(tmp_path):
    case = {
        "task": {"id": "example", "sequence": {"matrix": [128, 128]}},
        "hardware": {"id": "scanner"},
        "object": {"phantom": "cardiac"},
        "physics": {"effects": []},
        "evaluation": {"version": 4},
    }
    directory = tmp_path / "example" / "scanner"
    directory.mkdir(parents=True)
    (directory / "case.json").write_text(json.dumps(case))
    (directory / "sequence.seq").write_text("frozen pulse bytes\n")
    row = {
        "complete": True,
        "status": "passed",
        "source_sha256": CURRENT,
        "sequence_sha256": sha(directory / "sequence.seq"),
    }

    def save(**changes):
        value = {**row, **changes}
        (directory / "receipt.json").write_text(json.dumps(value))
        return value

    save()
    return tmp_path, directory, case, save


@pytest.mark.parametrize(
    "status,complete,expected",
    [
        ("passed", True, True),
        ("failed", True, True),
        ("unresolved", True, False),
        ("passed", False, False),
    ],
)
def test_same_source_reuses_only_completed_verdicts(
    stored_reference, status, complete, expected
):
    root, _, case, save = stored_reference
    save(status=status, complete=complete)
    assert completed_result(root, case, CURRENT) is expected


@pytest.mark.parametrize("status,expected", [("passed", True), ("failed", False)])
def test_foreign_source_needs_explicit_pass_compatibility(
    stored_reference, status, expected
):
    root, _, case, save = stored_reference
    save(status=status, source_sha256=PRIOR)
    assert not completed_result(root, case, CURRENT)
    assert completed_result(root, case, CURRENT, [PRIOR]) is expected


def test_changed_pulse_cannot_reuse_even_compatible_source(stored_reference):
    root, directory, case, save = stored_reference
    save(source_sha256=PRIOR)
    (directory / "sequence.seq").write_text("different pulse\n")
    assert not completed_result(root, case, CURRENT, [PRIOR])


@pytest.mark.parametrize(
    "key", ["object", "hardware", "physics", "evaluation", "sequence"]
)
def test_changed_case_cannot_reuse(stored_reference, key):
    root, _, case, _ = stored_reference
    case = deepcopy(case)
    (case["task"]["sequence"] if key == "sequence" else case[key])["changed"] = True
    assert not completed_result(root, case, CURRENT)


def test_recovered_receipt_has_the_same_reuse_guards(stored_reference):
    root, directory, case, save = stored_reference
    row = save(status="unresolved")
    row.update(status="passed", source_sha256=PRIOR)
    (directory / "recovered-receipt.json").write_text(json.dumps(row))
    assert not completed_result(root, case, CURRENT)
    assert completed_result(root, case, CURRENT, [PRIOR])
