"""Independent submission evaluation and physical backend orchestration."""

import json
from pathlib import Path

from pydantic import Field, StrictBool

from ..io import write_json
from ..models import Finite, Strict
from ..process import execute, expand
from .preflight import preflight


class BackendResult(Strict):
    schema_version: int = Field(ge=1, le=1)
    case_id: str
    reconstruction: str
    requirements: dict[str, StrictBool]
    effects: list[str]
    metrics: dict[str, Finite]


def evaluate(submission, case, root, workspace, backend=None):
    submission = Path(submission).resolve()
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    checks = (
        preflight(submission, case)
        if submission.is_file()
        else [{"name": "submission_exists", "passed": False, "detail": str(submission)}]
    )
    report = {
        "case_id": case["case_id"],
        "status": "failed",
        "checks": checks,
        "metrics": {},
    }
    write_json(workspace / "case.json", case)
    if not all(c["passed"] for c in checks):
        return report
    contract = case["evaluation"]
    if contract["scope"] == "smoke":
        report["status"] = "smoke_passed"
        return report
    if backend is None:
        report.update(
            status="incomplete", reason="Physical evaluation backend is not configured"
        )
        return report
    output = workspace / "backend-result.json"
    if output.exists():
        raise ValueError(f"backend output already exists: {output}")
    command = expand(
        backend.command,
        root,
        workspace,
        case=workspace / "case.json",
        submission=submission,
        result=output,
    )
    execution = execute(command, workspace, backend.timeout_s, workspace / "backend")
    if execution["status"] != "completed":
        report.update(status="evaluator_error", backend=execution)
        return report
    try:
        raw = json.loads(
            output.read_text(),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
        result = BackendResult.model_validate(raw)
        if (
            result.case_id != case["case_id"]
            or result.reconstruction != contract["reconstruction"]
        ):
            raise ValueError("backend case/reconstruction does not match contract")
        required = set(case["task"]["sequence"]["capabilities"]) | {"matrix", "fov_mm"}
        for field in (
            "flip_angle_deg",
            "te_ms",
            "tr_ms",
            "readout",
            "sampling",
            "temporal",
            "application",
        ):
            if case["task"]["sequence"][field] is not None:
                required.add(field)
        missing = required - result.requirements.keys()
        if missing:
            raise ValueError(f"backend omitted requirements: {sorted(missing)}")
        effects = {e["kind"] for e in case["physics"]["effects"]}
        if set(result.effects) != effects:
            raise ValueError("backend did not evaluate the specified physics effects")
        for name in sorted(result.requirements):
            checks.append(
                {
                    "name": name,
                    "passed": result.requirements[name],
                    "detail": "backend structure check",
                }
            )
        for metric in contract["metrics"]:
            name = metric["name"]
            if name not in result.metrics:
                raise ValueError(f"backend omitted metric: {name}")
            value = result.metrics[name]
            ok = (metric["min"] is None or value >= metric["min"]) and (
                metric["max"] is None or value <= metric["max"]
            )
            checks.append(
                {"name": name, "passed": ok, "detail": {"value": value, **metric}}
            )
        report["metrics"] = result.metrics
        report["status"] = (
            "failed"
            if not all(c["passed"] for c in checks)
            else "passed"
            if contract["calibrated"]
            else "uncalibrated"
        )
    except (OSError, ValueError) as exc:
        report.update(status="evaluator_error", reason=str(exc))
    return report
