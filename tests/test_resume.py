import json
from pathlib import Path

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.harness.runner import run
from mriseqbench.models import Experiment

ROOT = Path(__file__).resolve().parents[1]
PAID = {"route": "/chat/completions", "request_id": "x", "status": 200}


def test_resume_regrades_frozen_bytes_and_caps_paid_retries(tmp_path):
    catalog = Catalog(ROOT)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    out = tmp_path / "campaign"
    assert run(catalog, experiment, out)[2]
    trial = next(out.glob("*--r0"))
    agent_log = (trial / "agent.execution.json").read_text()
    (trial / "result.json").unlink()
    for path in trial.glob("**/evaluation/result.json"):
        path.unlink()
    for path in trial.glob("control/submissions/*/result.json"):
        path.unlink()
    _, reports, complete = run(catalog, experiment, out)
    assert complete and reports[0]["resumed"] == "regraded_frozen_submission"
    assert (trial / "agent.execution.json").read_text() == agent_log
    for name in ("--r1", "--r1.retry1"):  # Two paid interruptions exhaust one retry.
        (out / (trial.name[:-4] + name)).mkdir()
        (out / (trial.name[:-4] + name) / "api.jsonl").write_text(json.dumps(PAID))
    _, reports, complete = run(catalog, experiment, out, repeats=2)
    assert complete and reports[-1]["status"] == "infrastructure_interrupted"
