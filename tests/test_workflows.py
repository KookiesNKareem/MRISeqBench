"""Agentic contracts, physical edit controls and trusted multi-turn state transitions."""

import json
import sys
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.excitation_target import (
    excitation_metrics,
    target_arrays,
    target_at,
)
from mriseqbench.harness import feedback, workflow_rpc
from mriseqbench.harness.grading import EvaluationIncomplete
from mriseqbench.harness.runner import recover
from mriseqbench.harness.workflow import WorkflowServer
from mriseqbench.models import Experiment, FeedbackConfig
from mriseqbench.workflows import summarize_workflows, workflow_for

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
    )


def submit(s):
    return s.handle("submit", s.token, b"test frozen bytes")[1]


def advance(s, after):
    return s.handle("next", s.token, json.dumps({"after_stage": after}).encode())


def test_all_27_tasks_all_hardware_and_cumulative_undo(catalog):
    original = deepcopy(catalog.documents["tasks"])
    cases = catalog.cases(catalog.get("suites", "full"))
    assert len(cases) == 93
    assert (
        sum(bool(task.followups) for task in catalog.documents["tasks"].values()) == 31
    )
    for case in cases:
        w = workflow_for(catalog, case)
        assert len(w["stages"]) == 4
        assert w["stages"][0]["case"] == case
        assert all("followups" not in stage["case"]["task"] for stage in w["stages"])
        seq = [s["case"]["task"]["sequence"] for s in w["stages"]]
        assert len({json.dumps(v, sort_keys=True) for v in seq}) == 4
        if case["task"]["id"] == "ute_radial":
            assert all(s["te_ms"]["target"] + s["te_ms"]["tolerance"] <= 1 for s in seq)
        if case["task"]["id"] == "rf_excitation_2d":
            assert seq[3]["excitation_target"] == {
                "amplitude": 0.45,
                "offset_mm": [10.0, -10.0],
            }
        elif case["task"]["id"] in {"ute_radial", "spiral_gre"}:
            assert seq[1]["flip_angle_deg"] == seq[2]["flip_angle_deg"] == 10
            assert all(s["fov_mm"] == seq[0]["fov_mm"] for s in seq)
        else:
            # Stage 3 preserves the Stage 1/2 edited FOV while applying a real
            # slice-selection, readout, sampling, or timing follow-up edit.
            assert seq[3]["fov_mm"] == seq[2]["fov_mm"] != seq[0]["fov_mm"]
        assert all(s["case"]["evaluation"] == case["evaluation"] for s in w["stages"])
    assert original == catalog.documents["tasks"]


def test_judge_changes_and_unknown_fields_are_rejected(catalog, monkeypatch):
    base = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    task = catalog.get("tasks", "gre_t1w")
    for patch in (
        {"evaluation.metrics": []},
        {"matrix": [64, 64]},
        {"fov_mm": [50, 50]},
        {"tr_ms": {"typo": 20}},
    ):
        monkeypatch.setattr(task.followups[0], "patch", patch)
        with pytest.raises(ValueError):
            workflow_for(catalog, base)


def test_followups_require_a_change_even_at_previous_tolerance_edges(catalog):
    # The physical timing inspectors allow 0.001 ms of numerical slack.
    def bounds(value):
        lower, upper = 0, value.get("max") or float("inf")
        if value.get("target") is not None:
            lower = value["target"] - value["tolerance"]
            upper = min(upper, value["target"] + value["tolerance"])
        return lower - 0.001, upper + 0.001

    for case in catalog.cases(catalog.get("suites", "full")):
        stages = workflow_for(catalog, case)["stages"]
        for before, after in pairwise(stages):
            a, b = (s["case"]["task"]["sequence"] for s in (before, after))
            forced = a["fov_mm"] != b["fov_mm"]
            if a.get("flip_angle_deg") and b.get("flip_angle_deg"):
                forced |= abs(a["flip_angle_deg"] - b["flip_angle_deg"]) > 2
            for field in ("te_ms", "tr_ms"):
                if a.get(field) and b.get(field):
                    lo, hi = bounds(a[field])
                    other_lo, other_hi = bounds(b[field])
                    forced |= hi < other_lo or other_hi < lo
            forced |= a.get("application") != b.get("application")
            forced |= a.get("excitation_target") != b.get("excitation_target")
            forced |= a.get("slice_thickness_mm") != b.get("slice_thickness_mm")
            forced |= a.get("readout") != b.get("readout")
            forced |= a.get("sampling") != b.get("sampling")
            forced |= a.get("temporal") != b.get("temporal")
            assert forced, (case["case_id"], after["id"])


def test_followup_prompts_use_current_values_without_stale_objectives(catalog):
    from mriseqbench.harness.prepare import prompt

    for task in catalog.documents["tasks"].values():
        if not task.followups:
            continue
        for step, stage in zip(task.followups, plan(catalog, task.id)["stages"][1:]):
            message = prompt(stage["case"])
            assert step.request in message
            assert task.objective.strip() not in message
            assert "{'target'" not in message
            assert "Original family requirements" not in message
    changed = plan(catalog, "t1_mapping")["stages"][2]["case"]
    assert "150, 500, 1300" in changed["task"]["objective"]
    assert "TI 100, 400 and 1200" not in prompt(changed)


@pytest.mark.parametrize("hardware", ["low_field", "standard_1_5t", "high_performance"])
@pytest.mark.parametrize("name", ["ute_radial", "spiral_gre"])
def test_noncartesian_first_edit_has_feasible_physical_timing(
    catalog, tmp_path, hardware, name
):
    from mriseqbench.evaluation.preflight import preflight
    from mriseqbench.evaluation.s2_sequence import inspect_sequence
    from tests.fixtures.reference128_readouts import build

    base = catalog.resolve(catalog.get("tasks", name), hardware_ref=hardware)
    case = workflow_for(catalog, base)["stages"][1]["case"]
    path = tmp_path / "sequence.seq"
    build(case).write(str(path))
    checks = preflight(path, case)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    checks = inspect_sequence(path, case).checks
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


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
                json.loads((frozen / "case.json").read_text())
                == s.workflow["stages"][i]["case"]
            )
            if i < 3:
                assert submit(s)["status"] == "awaiting_next_turn"
                assert len(s.attempts) == i + 1
                code, response = advance(s, i)
                assert code == 200 and response["stage_index"] == i + 1
                public = json.loads(response["files"]["case.json"])
                assert "sar" not in public["hardware"] and "stages" not in public
                assert public["case_id"].endswith(f"--stage-{i + 1}")
                assert "followups" not in public["task"]
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
        final = submit(s)
        assert final["status"] == "submitted"
        assert final["evaluation"]["status"] == "failed"
        assert final["evaluation"]["checks"] == [
            {"name": "sar", "passed": True, "detail": "within the scanner SAR limit"}
        ]
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


def test_shared_deadline_and_grading_credit(catalog, tmp_path, monkeypatch):
    with server(catalog, tmp_path, monkeypatch) as s:
        s.start_agent_clock(40)
        s.agent_started -= 11
        assert s.runtime_status()["remaining_s"] == pytest.approx(29, abs=0.1)
        s.excluded_seconds = 6
        assert s.runtime_status()["remaining_s"] == pytest.approx(35, abs=0.1)
        submit(s)
        advance(s, 0)
        assert s.runtime_status()["remaining_s"] == pytest.approx(35, abs=0.1)
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
    # Correct translation with the old amplitude must still fail stage 2.
    assert (
        excitation_metrics(final[mask], translated[mask], 0.35)[
            "excitation_magnitude_nrmse"
        ]
        > 0.2
    )
    assert (
        excitation_metrics(arrays["target_mxy_magnitude"][mask], final[mask])[
            "excitation_magnitude_nrmse"
        ]
        > 0.2
    )


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
    assert (
        not rollback["fov_mm"]
        and not rollback["tr_ms"]
        and not rollback["slice_thickness_mm"]
    )

    # Implement the new TR, but accidentally revert the previous FOV edit.
    lost_fov = deepcopy(cases[2])
    lost_fov["task"]["sequence"]["fov_mm"] = cases[0]["task"]["sequence"]["fov_mm"]
    path = tmp_path / "lost-fov.seq"
    make_sequence(lost_fov, preparations=0, readout_flat=0.00128).write(str(path))
    checks = {c["name"]: c["passed"] for c in inspect_sequence(path, cases[2]).checks}
    assert checks["tr_ms"] and not checks["fov_mm"]

    # Requested edits cannot excuse a regression in the unchanged TE.
    path = tmp_path / "wrong-te.seq"
    make_sequence(
        cases[2], preparations=0, readout_flat=0.00128, echo_time=0.009
    ).write(str(path))
    checks = {c["name"]: c["passed"] for c in inspect_sequence(path, cases[2]).checks}
    assert checks["tr_ms"] and checks["fov_mm"] and not checks["te_ms"]


def test_new_flip_does_not_excuse_losing_the_sampling_mask(catalog, tmp_path):
    from mriseqbench.evaluation.s3_sequence import inspect_sequence
    from tests.fixtures.s3_sequences import make_sequence

    case = plan(catalog, "compressed_sensing")["stages"][2]["case"]
    for mutation in (None, "full_sampling"):
        path = tmp_path / f"{mutation}.seq"
        make_sequence(case, mutation, preparations=0).write(str(path))
        checks = {c["name"]: c["passed"] for c in inspect_sequence(path, case).checks}
        assert checks["flip_angle_deg"]
        assert checks["sampling_pattern"] == (mutation is None)


def test_runner_completes_chain_and_preserves_intermediate_artifacts(
    catalog, tmp_path, monkeypatch
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
    experiment = Experiment.model_validate(
        {
            "schema_version": 1,
            "id": "test_workflow",
            "version": 1,
            "suite": "full@1",
            "agentic": True,
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
    assert len(report["submissions"]) == 4
    assert report["feedback"]["submissions_used"] == 4
    assert json.loads((tmp_path / "run/evaluation/case.json").read_text())[
        "case_id"
    ].endswith("--stage-3")


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
 if i: assert case["case_id"].endswith(f"--stage-{i}")
 req=urllib.request.Request(f"http://127.0.0.1:{config['port']}/submit",data=f"stage-{i}".encode(),headers={"X-MRISeqBench-Token":config["token"]})
 result=json.loads(urllib.request.urlopen(req).read())
 assert result["status"]==("submitted" if i==3 else "stage_passed")
 print(json.dumps({"type":"agent_end","messages":[]}),flush=True)
 print(json.dumps({"type":"agent_settled"}),flush=True)
""")
        assert workflow_rpc.drive([sys.executable, str(script)], tmp_path) == 0
        assert s.report()["status"] == "passed"
        final = json.loads((tmp_path / "case.json").read_text())["task"]["sequence"]
        assert (
            final["tr_ms"]["target"] == 20
            and final["fov_mm"] == [220, 220]
            and final["slice_thickness_mm"] == 5
        )


def test_slice_selection_and_new_s4_application_tasks(catalog, tmp_path):
    from mriseqbench.evaluation.s1_sequence import inspect_sequence as inspect_s1
    from mriseqbench.evaluation.s4_sequence import inspect_sequence as inspect_s4
    from tests.fixtures import (
        reference128_advanced,
        reference128_readouts,
        s4_sequences,
    )

    # 1. Slice-profile & through-plane excitation fidelity on Stage 3 of gre_t1w
    gre_stages = plan(catalog, "gre_t1w")["stages"]
    stage2_case = gre_stages[2]["case"]
    stage3_case = gre_stages[3]["case"]
    stage2_path = tmp_path / "gre_stage2.seq"
    stage3_path = tmp_path / "gre_stage3.seq"
    reference128_readouts.build(stage2_case).write(str(stage2_path))
    reference128_readouts.build(stage3_case).write(str(stage3_path))
    checks_good = {
        c["name"]: c["passed"] for c in inspect_s1(stage3_path, stage3_case).checks
    }
    assert checks_good["slice_thickness_mm"]
    # Non-selective excitation from stage 2 must fail slice_thickness_mm on stage 3
    checks_nonselective = {
        c["name"]: c["passed"] for c in inspect_s1(stage2_path, stage3_case).checks
    }
    assert not checks_nonselective["slice_thickness_mm"]

    # 2. New S4 tasks: positive and negative controls for water_selective_gre, dwi_epi, venc_flow_2d, cardiac_respiratory_bssfp
    for task_id, check_name, bad_mutation in (
        ("water_selective_gre", "water_selective_suppression", "non_selective"),
        ("dwi_epi", "diffusion_b_value", "wrong_b_value"),
        ("venc_flow_2d", "flow_encoding_venc", "wrong_venc"),
        (
            "cardiac_respiratory_bssfp",
            "frame_0_respiratory_gating",
            "ungated_respiratory",
        ),
    ):
        case = catalog.resolve(catalog.get("tasks", task_id))
        good_path = tmp_path / f"{task_id}_good.seq"
        bad_path = tmp_path / f"{task_id}_bad.seq"
        reference128_advanced.build(case).write(str(good_path))
        s4_sequences.make_sequence(case, mutation=bad_mutation).write(str(bad_path))
        good_checks = {
            c["name"]: c["passed"] for c in inspect_s4(good_path, case).checks
        }
        bad_checks = {
            c["name"]: c["passed"] for c in inspect_s4(bad_path, case).checks
        }
        assert all(good_checks.values()), (task_id, good_checks)
        assert good_checks[check_name]
        assert not bad_checks[check_name]


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


def test_followup_count_is_not_fixed(catalog, monkeypatch):
    task = catalog.get("tasks", "gre_t1w")
    monkeypatch.setattr(task, "followups", task.followups[:1])
    assert len(plan(catalog)["stages"]) == 2


def test_stage_usage_reuses_api_log_and_handles_late_or_missing_usage(
    catalog, tmp_path, monkeypatch
):
    from mriseqbench.harness.accounting import summarize_api

    path = tmp_path / "api.jsonl"

    def record(request_id, **fields):
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {"request_id": request_id, "route": "/chat/completions", **fields}
                )
                + "\n"
            )

    def finish(request_id, inputs, cost):
        record(
            request_id,
            event="finished",
            response_complete=True,
            usage={
                "input_tokens": inputs,
                "output_tokens": 10,
                "total_tokens": inputs + 10,
                "cost_credits": cost,
            },
        )

    with server(catalog, tmp_path, monkeypatch) as s:
        record("a", event="started")
        submit(s)
        advance(s, 0)
        # Usage can arrive after a turn boundary. Attribute by request, not arrival.
        finish("a", 100, 0.02)
        finish("b", 200, 0.03)
        report = s.report()
        first, second = report["workflow"]["stages"][:2]
        assert first["usage"]["totals"]["cost_credits"] == 0.02
        assert second["usage"]["totals"]["cost_credits"] == 0.03
        for key in ["input_tokens", "output_tokens", "total_tokens", "cost_credits"]:
            assert sum(
                row["usage"]["totals"][key] for row in [first, second]
            ) == pytest.approx(summarize_api(path)["totals"][key])
        record("c", event="finished", response_complete=True)
        second = s.report()["workflow"]["stages"][1]
        assert second["usage"]["totals"]["cost_credits"] is None
        assert second["usage"]["observed_totals"]["cost_credits"] == 0.03
        assert second["usage"]["requests_missing_usage"] == 1
        (tmp_path / "session.json").write_text('{"ended":"completed"}')
        _, recovered = recover(tmp_path, 3)
        assert recovered["workflow"]["stages"][1]["usage"] == second["usage"]
