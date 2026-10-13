"""Long native physics replay never consumes the participant authoring budget."""

import json
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness.grading import grade_frozen

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "task,physics",
    [
        ("gre_t1w", "static_b0"),
        ("spiral_gre", "static_b0"),
        ("caipirinha", "static_b0_3d"),
        ("t1_mapping", "static_b0"),
    ],
)
@pytest.mark.parametrize("external", [False, True])
def test_every_native_tier_can_finish_adaptive_replay(
    tmp_path, task, physics, external
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", task), catalog.get("physics", physics))
    submission = tmp_path / "sequence.seq"
    submission.write_bytes(b"immutable submitted sequence")
    calls = []

    def worker(command, directory, budget, *args, **kwargs):
        snapshot = Path(command[command.index("--submission") + 1])
        calls.append((snapshot.read_bytes(), budget, tuple(command)))
        if len(calls) == 1:
            return {"status": "error"}
        (directory / "result.json").write_text(json.dumps({"status": "passed"}))
        return {"status": "completed"}

    result = grade_frozen(
        submission,
        case,
        ROOT,
        tmp_path / "grade",
        0.01,
        worker,
        backend_path=tmp_path / "backend.json" if external else None,
    )
    assert result == {"status": "passed", "worker_recoveries": 1}
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][0] == submission.read_bytes()
    assert calls[0][1] == (0.01 if external else None)
    assert ("--backend" in calls[0][2]) is external


def test_smoke_grading_keeps_its_finite_budget(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "pulseq_smoke"), catalog.get("physics", "static_b0")
    )
    budgets = []

    def worker(command, directory, budget, *args, **kwargs):
        budgets.append(budget)
        (directory / "result.json").write_text(json.dumps({"status": "smoke_passed"}))
        return {"status": "completed"}

    result = grade_frozen(
        tmp_path / "sequence.seq", case, ROOT, tmp_path / "grade", 0.01, worker
    )
    assert result["status"] == "smoke_passed"
    assert budgets == [0.01]
