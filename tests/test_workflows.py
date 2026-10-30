"""Agentic contracts, physical edit controls and trusted multi-turn state transitions."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.evaluation.excitation_target import (
    excitation_metrics,
    target_arrays,
    target_at,
)
from mriseqbench.harness import feedback, workflow_rpc
from mriseqbench.harness.grading import EvaluationIncomplete
from mriseqbench.harness.runner import recover, run
from mriseqbench.harness.workflow import WorkflowServer
from mriseqbench.models import Experiment, FeedbackConfig
from mriseqbench.workflows import compile_workflow, summarize_workflows, workflow_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return Catalog(ROOT)


def plan(catalog, name="gre_t1w"):
    return workflow_for(catalog, catalog.resolve(catalog.get("tasks", name)))


def server(catalog, tmp_path, monkeypatch, verdicts=None, name="gre_t1w"):
    workflow = plan(catalog, name)
    verdicts = iter(verdicts or ["passed"] * 4)

    def grade(sequence, case, root, directory, *_args, **_kwargs):
        value = {
            "status": next(verdicts),
            "case_id": case["case_id"],
            "checks": [{"name": "sar_limit", "passed": True, "detail": "private"}],
            "metrics": {},
        }
        (directory / "result.json").write_text(json.dumps(value))
        return value

    monkeypatch.setattr(feedback, "grade_frozen", grade)
    return WorkflowServer(
        workflow["stages"][0]["case"],
        FeedbackConfig(),
        ROOT,
        tmp_path / "control",
        workflow=workflow,
        stage_timeout_s=10,
    )


def submit(s):
    return s.handle("submit", s.token, b"test frozen bytes")[1]


def advance(s, after):
    return s.handle("next", s.token, json.dumps({"after_stage": after}).encode())


def test_all_27_tasks_all_hardware_and_cumulative_undo(catalog):
    original = deepcopy(catalog.documents["tasks"])
    cases = catalog.cases(catalog.get("suites", "full"))
    assert len(cases) == 81
    assert len(catalog.documents["workflows"]) == 27
    for case in cases:
        w = workflow_for(catalog, case)
        assert w["qualification"] == "draft" and len(w["stages"]) == 4
        seq = [s["case"]["task"]["sequence"] for s in w["stages"]]
        assert len({json.dumps(v, sort_keys=True) for v in seq}) == 4
        if case["task"]["id"] == "rf_excitation_2d":
            assert seq[3]["excitation_target"] == {
                "amplitude": 0.5,
                "offset_mm": [10.0, 0.0],
            }
        else:
            assert seq[3]["fov_mm"] == seq[0]["fov_mm"]
            assert {k: v for k, v in seq[3].items() if k != "fov_mm"} == {
                k: v for k, v in seq[2].items() if k != "fov_mm"
            }
        assert all(s["case"]["evaluation"] == case["evaluation"] for s in w["stages"])
    assert original == catalog.documents["tasks"]


def test_judge_changes_and_unknown_fields_are_rejected(catalog):
    base = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    definition = catalog.get("workflows", "gre_t1w").model_copy(deep=True)
    for patch in (
        {"evaluation.metrics": []},
        {"matrix": [64, 64]},
        {"fov_mm": [50, 50]},
        {"tr_ms": {"typo": 20}},
    ):
        definition.stages[0].patch = patch
        with pytest.raises(ValueError):
            compile_workflow(catalog, base, definition)


def test_stage_gating_preserves_context_and_hides_future(
    catalog, tmp_path, monkeypatch
):
    with server(catalog, tmp_path, monkeypatch) as s:
        assert advance(s, 0)[0] == 409
        assert s.handle("next", "wrong", b'{"after_stage":0}')[0] == 403
        for i in range(4):
            result = submit(s)
            assert result["status"] == ("stage_passed" if i < 3 else "submitted")
            assert s.submitted.is_set() == (i == 3)
            frozen = tmp_path / "control/submissions" / f"attempt-{i + 1:03d}"
            assert json.loads((frozen / "receipt.json").read_text())["stage_index"] == i
            assert (
                json.loads((frozen / "case.json").read_text())["workflow_stage"][
                    "index"
                ]
                == i
            )
            if i < 3:
                assert submit(s)["status"] == "awaiting_next_turn"
                assert len(s.attempts) == i + 1
                code, response = advance(s, i)
                assert code == 200 and response["stage_index"] == i + 1
                public = json.loads(response["files"]["case.json"])
                assert "sar" not in public["hardware"] and "stages" not in public
                assert public["workflow_stage"]["index"] == i + 1
                assert advance(s, i) == (code, response)  # Idempotent delivery.
        report = s.report()
        assert report["status"] == "passed"
        summary = summarize_workflows([report])
        assert summary["passed_workflows"] == 1
        assert all(r["reached"] == 1 and r["passed"] == 1 for r in summary["stages"])


def test_per_stage_retries_and_not_reached(catalog, tmp_path, monkeypatch):
    with server(
        catalog,
        tmp_path,
        monkeypatch,
        ["failed", "failed", "passed", "failed", "failed", "failed"],
    ) as s:
        assert submit(s)["submissions_remaining"] == 2
        assert submit(s)["submissions_remaining"] == 1
        assert submit(s)["status"] == "stage_passed"
        advance(s, 0)
        assert s.runtime_status()["submissions_remaining"] == 3
        assert submit(s)["submissions_remaining"] == 2
        submit(s)
        assert submit(s)["status"] == "submitted"
        assert [r["status"] for r in s.report()["workflow"]["stages"]] == [
            "passed",
            "failed",
            "not_reached",
            "not_reached",
        ]
        assert recover(tmp_path, 3)[0] == "workflow"


def test_partial_pass_never_recovers_as_complete(catalog, tmp_path, monkeypatch):
    with server(catalog, tmp_path, monkeypatch) as s:
        submit(s)
        assert recover(tmp_path, 3) is None  # Lost conversation, rerun whole chain.
        (tmp_path / "session.json").write_text('{"ended":"completed"}')
        kind, report = recover(tmp_path, 3)
        assert kind == "workflow" and report["status"] == "failed"
        assert report["workflow"]["stages"][0]["status"] == "passed"


def test_infrastructure_never_becomes_workflow_failure(catalog, tmp_path, monkeypatch):
    with server(catalog, tmp_path, monkeypatch) as s:

        def broken(*args, **kwargs):
            raise EvaluationIncomplete("GPU lost")

        monkeypatch.setattr(feedback, "grade_frozen", broken)
        with pytest.raises(EvaluationIncomplete):
            submit(s)
        assert recover(tmp_path, 3) is None


def test_stage_deadline_and_grading_credit(catalog, tmp_path, monkeypatch):
    with server(catalog, tmp_path, monkeypatch) as s:
        s.start_agent_clock(40)
        s.agent_started -= 11
        assert s.stage_expired()
        s.excluded_seconds = 6
        assert not s.stage_expired()
        submit(s)
        advance(s, 0)
        assert s.runtime_status()["stage_remaining_s"] > 9
        assert s.runtime_status()["charged_wall_s"] >= 5


def test_rf_target_uses_amplitude_and_offset_without_fitted_gain():
    arrays = target_arrays()
    x, y = np.meshgrid(arrays["x_mm"], arrays["y_mm"], indexing="ij")
    points = np.stack((x, y), axis=-1)
    mask = arrays["object_mask"]
    small = target_arrays({"amplitude": 0.35, "offset_mm": [0, 0]})[
        "target_mxy_magnitude"
    ]
    np.testing.assert_allclose(small, arrays["target_mxy_magnitude"] * 0.7, atol=1e-12)
    translated = target_at(points, {"amplitude": 0.35, "offset_mm": [10, 0]})
    assert (
        excitation_metrics(translated[mask], translated[mask], 0.35)[
            "excitation_magnitude_nrmse"
        ]
        == 0
    )
    # Unchanged response and full rollback both fail a tight target-match control.
    assert (
        excitation_metrics(small[mask], translated[mask], 0.35)[
            "excitation_magnitude_nrmse"
        ]
        > 0.2
    )
    final = target_at(points, {"amplitude": 0.5, "offset_mm": [10, 0]})
    assert (
        excitation_metrics(arrays["target_mxy_magnitude"][mask], final[mask])[
            "excitation_magnitude_nrmse"
        ]
        > 0.2
    )


def test_draft_run_requires_explicit_development_override(catalog, tmp_path):
    experiment = Experiment.model_validate(
        load_yaml(ROOT / "experiments/pi_agentic.yaml")
    )
    with pytest.raises(ValueError, match="draft"):
        run(catalog, experiment, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_rpc_conversion_preserves_other_arguments():
    command = [
        "pi",
        "--model",
        "model",
        "--session-dir",
        "transcript",
        "--mode",
        "json",
        "-p",
        "{prompt_text}",
    ]
    assert workflow_rpc.rpc_command(command) == [
        "pi",
        "--model",
        "model",
        "--session-dir",
        "transcript",
        "--mode",
        "rpc",
    ]


def test_rpc_waits_for_settled_and_sends_real_user_turns(tmp_path, monkeypatch):
    # A subprocess protocol double deliberately emits agent_end before settled.
    script = tmp_path / "fake_pi.py"
    script.write_text("""import json, sys
for line in sys.stdin:
    command=json.loads(line)
    with open("commands.jsonl","a") as f: f.write(line)
    print(json.dumps({"type":"response","success":True}),flush=True)
    print(json.dumps({"type":"agent_end","willRetry":True,"messages":[]}),flush=True)
    print(json.dumps({"type":"agent_settled"}),flush=True)
""")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "PROMPT.md").write_text("initial user request")
    (tmp_path / ".harness.json").write_text("{}")
    count = 0

    def request(config, kind, payload=None):
        nonlocal count
        if kind == "status":
            return {"awaiting_next_turn": count < 3, "stage_index": count}
        count += 1
        return {
            "status": "next_stage",
            "stage_index": count,
            "message": f"edit {count}",
            "files": {"task.yaml": f"stage: {count}"},
        }

    monkeypatch.setattr(workflow_rpc, "request", request)
    assert workflow_rpc.drive([sys.executable, str(script)], tmp_path) == 0
    commands = [
        json.loads(line)
        for line in (tmp_path / "commands.jsonl").read_text().splitlines()
    ]
    assert [c["message"] for c in commands] == [
        "initial user request",
        "edit 1",
        "edit 2",
        "edit 3",
    ]
    assert all(c["type"] == "prompt" for c in commands)


def test_waveform_controls_reject_unchanged_metadata_only_and_full_rollback(
    catalog, tmp_path
):
    from mriseqbench.evaluation.s1_sequence import inspect_sequence
    from tests.fixtures.s1_sequences import make_sequence

    cases = [s["case"] for s in plan(catalog)["stages"]]
    paths = []
    for i, case in enumerate(cases):
        seq = make_sequence(case, preparations=0, readout_flat=0.00128)
        path = tmp_path / f"stage-{i}.seq"
        seq.write(str(path))
        paths.append(path)
        checks = {c["name"]: c["passed"] for c in inspect_sequence(path, case).checks}
        assert checks["fov_mm"] and checks["tr_ms"]
    # Same waveform under the new contract fails; a header change cannot fix it.
    unchanged = {
        c["name"]: c["passed"] for c in inspect_sequence(paths[0], cases[1]).checks
    }
    assert not unchanged["fov_mm"]
    metadata = (
        paths[0]
        .read_text()
        .replace("[DEFINITIONS]", "[DEFINITIONS]\nFOV 0.22 0.22 0.2")
    )
    path = tmp_path / "metadata-only.seq"
    path.write_text(metadata)
    assert not next(
        c["passed"]
        for c in inspect_sequence(path, cases[1]).checks
        if c["name"] == "fov_mm"
    )
    rollback = {
        c["name"]: c["passed"] for c in inspect_sequence(paths[0], cases[3]).checks
    }
    assert rollback["fov_mm"] and not rollback["tr_ms"]


@pytest.mark.parametrize(
    "mode,expected", [("generate_edit", 4), ("supplied_edit", 4), ("final_spec", 1)]
)
def test_runner_completes_chain_and_preserves_intermediate_artifacts(
    catalog, tmp_path, monkeypatch, mode, expected
):
    from mriseqbench.harness.runner import run_one

    def grade(sequence, case, root, directory, *_args, **_kwargs):
        value = {
            "status": "passed",
            "case_id": case["case_id"],
            "checks": [],
            "metrics": {},
        }
        (directory / "result.json").write_text(json.dumps(value))
        return value

    monkeypatch.setattr(feedback, "grade_frozen", grade)
    script = tmp_path / "fake_agent.py"
    script.write_text("""import json, urllib.request
from pathlib import Path
c=json.loads(Path(".harness.json").read_text())
def call(route,body):
 req=urllib.request.Request(f"http://127.0.0.1:{c['port']}/{route}",data=body,headers={"X-MRISeqBench-Token":c["token"]})
 return json.loads(urllib.request.urlopen(req).read())
for i in range(4):
 response=call("submit",f"sequence-stage-{i}".encode())
 if response['status']=='submitted': break
 assert response['status']=='stage_passed'
 previous=response["stage_index"]
 response=call("next",json.dumps({"after_stage":previous}).encode())
 assert response['stage_index']==previous+1
""")
    base = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    options = {"allow_unqualified": True, "stage_timeout_s": 30, "mode": mode}
    if mode == "supplied_edit":
        import hashlib

        seq = tmp_path / "seed.seq"
        seq.write_bytes(b"known seed")
        src = tmp_path / "design.py"
        src.write_text("# editable seed source")

        def entry(path):
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        manifest = tmp_path / "seeds.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "arm": None,
                    "cases": {
                        base["case_id"]: {
                            "sequence": entry(seq),
                            "sources": {"design.py": entry(src)},
                        }
                    },
                }
            )
        )
        options["seed_manifest"] = str(manifest)
    experiment = Experiment.model_validate(
        {
            "schema_version": 1,
            "id": "test_workflow",
            "version": 1,
            "suite": "full@1",
            "workflow": options,
            "agent": {
                "uses_model": False,
                "command": [sys.executable, str(script)],
                "timeout_s": 120,
                "sandbox": {"backend": "none"},
            },
        }
    )
    base = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    report = run_one(catalog, experiment, base, tmp_path / "run", 0)
    assert report["status"] == "passed"
    assert len(report["submissions"]) == expected
    assert report["feedback"]["submissions_used"] == expected
    assert report["workflow"]["mode"] == mode
    if mode == "supplied_edit":
        assert report["workflow"]["stages"][0]["origin"] == "supplied"
        assert (
            tmp_path / "run/agent/design.py"
        ).read_text() == "# editable seed source"
    assert json.loads((tmp_path / "run/evaluation/case.json").read_text())[
        "workflow_stage"
    ]["index"] == (0 if mode == "final_spec" else 3)


def test_rpc_real_http_updates_contract_before_next_user_turn(
    catalog, tmp_path, monkeypatch
):
    with server(catalog, tmp_path, monkeypatch) as s:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".harness.json").write_text(
            json.dumps({"port": s.port, "token": s.token, "timeout_s": 10})
        )
        (tmp_path / "PROMPT.md").write_text("generate initial sequence")
        (tmp_path / "case.json").write_text(json.dumps(s.case))
        script = tmp_path / "fake_pi_http.py"
        script.write_text("""import json, sys, urllib.request
from pathlib import Path
config=json.loads(Path(".harness.json").read_text())
for i,line in enumerate(sys.stdin):
 user=json.loads(line)
 assert user["type"]=="prompt"
 case=json.loads(Path("case.json").read_text())
 assert case["workflow_stage"]["index"]==i
 req=urllib.request.Request(f"http://127.0.0.1:{config['port']}/submit",data=f"stage-{i}".encode(),headers={"X-MRISeqBench-Token":config["token"]})
 result=json.loads(urllib.request.urlopen(req).read())
 assert result["status"]==("submitted" if i==3 else "stage_passed")
 print(json.dumps({"type":"agent_end","messages":[]}),flush=True)
 print(json.dumps({"type":"agent_settled"}),flush=True)
""")
        assert workflow_rpc.drive([sys.executable, str(script)], tmp_path) == 0
        assert s.report()["status"] == "passed"
        final = json.loads((tmp_path / "case.json").read_text())["task"]["sequence"]
        assert final["tr_ms"]["target"] == 20 and final["fov_mm"] == [200, 200]


def test_rf_public_target_refresh_and_redaction(catalog, tmp_path, monkeypatch):
    import base64
    import io

    with server(catalog, tmp_path, monkeypatch, name="rf_excitation_2d") as s:
        submit(s)
        _, response = advance(s, 0)
        with np.load(
            io.BytesIO(base64.b64decode(response["target_npz_base64"]))
        ) as data:
            assert data["target_mxy_magnitude"].max() <= 0.35
        assert "Target amplitude 0.35" in response["files"]["EVALUATION.md"]
        assert "divides by 0.35" in response["files"]["EVALUATION.md"]
        assert '"sar"' not in response["files"]["case.json"]


def test_seed_library_rejects_hash_changes_and_harness_destinations(tmp_path):
    import hashlib

    from mriseqbench.harness.workflow_seed import load_seed

    source = tmp_path / "source.py"
    source.write_bytes(b"source")
    entry = {"path": str(source), "sha256": hashlib.sha256(b"source").hexdigest()}
    document = {
        "schema_version": 1,
        "arm": None,
        "cases": {"case": {"sequence": entry, "sources": {"design.py": entry}}},
    }
    manifest = tmp_path / "seeds.json"
    manifest.write_text(json.dumps(document))
    assert load_seed(manifest, tmp_path, "case", None)[1] == {"design.py": b"source"}
    for name in [
        "../design.py",
        ".harness_client.py",
        "bin/design.py",
        "/tmp/design.py",
    ]:
        document["cases"]["case"]["sources"] = {name: entry}
        manifest.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="destination"):
            load_seed(manifest, tmp_path, "case", None)
    document["cases"]["case"]["sources"] = {"design.py": entry}
    manifest.write_text(json.dumps(document))
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_seed(manifest, tmp_path, "case", None)
