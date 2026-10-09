import copy
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness.runner import run
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]


def test_runner_separates_repeats_and_refuses_overwrite(tmp_path, submission):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.repeats = 2
    experiment.agent.command = [
        sys.executable,
        "-c",
        "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
        str(submission),
        "{submission}",
    ]
    output, results = run(catalog, experiment, tmp_path / "runs")
    assert [r["status"] for r in results] == ["smoke_passed", "smoke_passed"]
    assert len({r["workspace"] for r in results}) == 2
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
    assert all(
        (Path(r["workspace"]) / "evaluation/case.json").exists() for r in results
    )
    with pytest.raises(FileExistsError):
        run(catalog, experiment, output)


def test_failed_agent_does_not_get_evaluated(tmp_path):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.agent.command = [sys.executable, "-c", "raise SystemExit(7)"]
    _, results = run(catalog, experiment, tmp_path / "runs")
    assert results[0]["status"] == "agent_error"
    assert not results[0]["checks"]


def test_deadline_and_launch_error(tmp_path):
    result = execute(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        0.1,
        tmp_path / "slow",
    )
    assert result["status"] == "timeout"
    assert (
        execute(["/nonexistent/agent"], tmp_path, 1, tmp_path / "missing")["status"]
        == "launch_error"
    )
