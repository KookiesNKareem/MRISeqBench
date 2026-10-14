"""Independent submission evaluation and physical backend orchestration."""

import json
import time
from pathlib import Path

from pydantic import Field, StrictBool

from ..io import write_json
from ..models import Finite, Strict
from ..process import execute, expand
from ..storage import finalize_evaluation
from .preflight import preflight
from .sar import evaluate_sar


class BackendResult(Strict):
    schema_version: int = Field(ge=1, le=1)
    case_id: str
    reconstruction: str
    requirements: dict[str, StrictBool]
    effects: list[str]
    metrics: dict[str, Finite]
    reconstruction_provenance: dict | None = None


def validate_bart_provenance(value):
    """External judges must disclose the BART executable and actual commands."""
    import re

    from .bart import QUALIFIED_SOURCE_COMMIT, QUALIFIED_VERSION

    if not isinstance(value, dict) or value.get("engine") != "BART":
        raise ValueError("backend must report BART reconstruction provenance")
    if value.get("version") != QUALIFIED_VERSION:
        raise ValueError(f"backend must use qualified BART {QUALIFIED_VERSION}")
    if value.get("source_commit") not in (None, QUALIFIED_SOURCE_COMMIT):
        raise ValueError("backend BART source must match the qualified release")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("binary_sha256", ""))):
        raise ValueError("backend must identify its BART executable by SHA-256")
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("backend must report actual BART reconstruction commands")
    for command in commands:
        if (
            not isinstance(command, dict)
            or not isinstance(command.get("argv"), list)
            or not command["argv"]
            or not all(isinstance(token, str) for token in command["argv"])
            or type(command.get("calls")) is not int
            or command["calls"] <= 0
        ):
            raise ValueError("backend BART command records are invalid")
    return value


def intrinsic_cardiac_case(case):
    return bool(
        case["evaluation"].get("intrinsic_tissue_response", False)
        and case["object"]["phantom"].startswith("cardiac_")
    )


def s4_backend_contract(case):
    """Require every cardiac frame and application outcome from external judges."""
    from .s1_case import case_bounds, case_policy
    from .s1_sequence import family

    spec, contract = case["task"]["sequence"], case["evaluation"]
    fam = family(case)
    requirements = {
        "ecg_trigger_events",
        "nominal_rr_cadence",
        "cardiac_adc_window",
        "cardiac_frame_count",
        "numerical_convergence",
    }
    if fam in {"t1_mapping", "t2_mapping"}:
        requirements.add("preparation_recovery")
    if fam == "adiabatic_inversion":
        requirements.add("adiabatic_hyperbolic_secant")
    bounds = []
    for frame in range(spec["temporal"]["frames"]):
        requirements |= {
            f"frame_{frame}_full_cartesian_grid",
            f"frame_{frame}_balanced_bssfp",
        }
        preparation = (
            "inversion"
            if fam in {"adiabatic_inversion", "t1_mapping"}
            else "t2_preparation"
        )
        requirements.add(f"frame_{frame}_{preparation}")
        bounds.extend(
            {**bound, "name": f"frame_{frame}_case_" + bound["name"]}
            for bound in case_bounds(case_policy(case))
        )
        if intrinsic_cardiac_case(case):
            # Match the independent native phase oracle's validity safeguard.
            bounds.append(
                {
                    "name": f"frame_{frame}_case_phase_reference_coverage",
                    "min": 0.5,
                    "max": None,
                }
            )
        bounds.append(
            {
                "name": f"frame_{frame}_diagnostic_signal_fraction",
                "min": 0.01,
                "max": None,
            }
        )
        prepared = (
            fam != "t2_mapping"
            or spec["application"]["preparation_durations_ms"][frame] > 0
        )
        if prepared:
            bounds.extend(
                [
                    {
                        "name": f"frame_{frame}_preparation_effect_fraction",
                        "min": contract["preparation_effect_fraction_min"],
                        "max": None,
                    },
                    {
                        "name": f"frame_{frame}_min_preparation_rf_inversion_efficiency",
                        "min": 0.90,
                        "max": None,
                    },
                ]
            )
    if fam in {"t1_mapping", "t2_mapping"}:
        parameter = "t1" if fam == "t1_mapping" else "t2"
        bounds.append(
            {
                "name": f"myocardial_{parameter}_relative_error",
                "min": None,
                "max": contract["mapping_relative_error_max"],
            }
        )
    return requirements, bounds


def evaluate(
    submission,
    case,
    root,
    workspace,
    backend=None,
    *,
    feedback=False,
    time_budget_s=None,
):
    report = _evaluate(
        submission,
        case,
        root,
        workspace,
        backend,
        feedback=feedback,
        time_budget_s=time_budget_s,
    )
    finalize_evaluation(report, workspace)
    return report


def _evaluate(
    submission,
    case,
    root,
    workspace,
    backend=None,
    *,
    feedback=False,
    time_budget_s=None,
):
    """Judge submissions by default; feedback intentionally omits SAR entirely."""
    submission = Path(submission).resolve()
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + time_budget_s if time_budget_s is not None else None
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
    if not feedback:
        checks.extend(evaluate_sar(submission, case["hardware"]))
        if not all(c["passed"] for c in checks):
            return report
    contract = case["evaluation"]
    if contract["scope"] == "smoke":
        report["status"] = "smoke_passed"
        return report
    if backend is None and case["sequence_level"] == "S4":
        from .s4 import evaluate_s4

        try:
            physical = evaluate_s4(
                submission, case, root, workspace / "s4", deadline=deadline
            )
            physical["checks"] = checks + physical["checks"]
            report.update(physical)
        except Exception as exc:
            raise RuntimeError(f"S4 evaluation interrupted: {exc}") from exc
        return report
    if backend is None and case["sequence_level"] in {"S1", "S2", "S3"}:
        from .s1_sequence import FAMILIES as S1_FAMILIES
        from .s1_sequence import family
        from .s2_sequence import FAMILIES as S2_FAMILIES
        from .s3_sequence import FAMILIES as S3_FAMILIES

        level = case["sequence_level"]
        families = {"S1": S1_FAMILIES, "S2": S2_FAMILIES, "S3": S3_FAMILIES}
        if family(case) in families[level]:
            from .s1 import evaluate_s1
            from .s2 import evaluate_s2
            from .s3 import evaluate_s3

            judge = {"S1": evaluate_s1, "S2": evaluate_s2, "S3": evaluate_s3}[level]
            try:
                physical = judge(
                    submission, case, root, workspace / level.lower(), deadline=deadline
                )
                physical["checks"] = checks + physical["checks"]
                report.update(physical)
            except Exception as exc:  # Interrupted jobs have no verdict
                raise RuntimeError(f"{level} evaluation interrupted: {exc}") from exc
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
        raise RuntimeError(f"Physical backend did not complete: {execution}")
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
        if contract["reconstruction"].startswith("bart_"):
            report["reconstruction_provenance"] = validate_bart_provenance(
                result.reconstruction_provenance
            )
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
        if case["task"].get("prompt_mode") == "research":
            # Backend must inspect physical sequence identity and parameter suitability,
            # rather than compare agent-selected timings to hidden reference numbers.
            required |= {"sequence_identity", "reasonable_parameters"}
        if contract["id"] == "s4":
            cardiac_requirements, _cardiac_bounds = s4_backend_contract(case)
            required |= cardiac_requirements
        if intrinsic_cardiac_case(case):
            from .s1_case import uses_case_image

            if uses_case_image(case):
                required.add("numerical_convergence")
        if case["task"]["id"] == "reduced_fov":
            required.add("spatial_selection")
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
        metric_bounds = contract["metrics"]
        case_acceptance = False
        if contract["id"] in {"s1", "s2", "s3", "s4"}:
            from .s1_case import case_bounds, case_policy, uses_case_image

            case_acceptance = uses_case_image(case)
            if case_acceptance:
                metric_bounds = [
                    {**bound, "name": "case_" + bound["name"]}
                    for bound in case_bounds(case_policy(case))
                ]
                if intrinsic_cardiac_case(case):
                    metric_bounds.append(
                        {
                            "name": "case_phase_reference_coverage",
                            "min": 0.5,
                            "max": None,
                        }
                    )
            if contract["id"] == "s4":
                _, metric_bounds = s4_backend_contract(case)
        if case["task"]["id"] == "reduced_fov":
            # Exact scalar semantics of s3.mechanism_probes: useful selected
            # water signal and worst outside/mean inside signal over 27 spins.
            metric_bounds = [
                *metric_bounds,
                {"name": "selection_leakage_ratio", "min": None, "max": 0.1},
                {"name": "diagnostic_signal_fraction", "min": 0.01, "max": None},
            ]
        for metric in metric_bounds:
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
        if case_acceptance:
            report.update(
                image_acceptance_basis="case_physics",
                imaging_success_under_case_physics=all(
                    c["passed"] for c in checks if c["name"].startswith("case_")
                ),
            )
            if intrinsic_cardiac_case(case):
                converged = result.requirements["numerical_convergence"]
                report["numerically_converged"] = converged
                if not converged:
                    report["imaging_success_under_case_physics"] = None
                    if all(
                        value
                        for name, value in result.requirements.items()
                        if name != "numerical_convergence"
                    ):
                        report.update(
                            status="numerically_unresolved",
                            image_verdict="numerically_unresolved",
                        )
        if contract["id"] == "s4":
            image_names = {
                bound["name"] for bound in metric_bounds if "_case_" in bound["name"]
            }
            image_checks = [item for item in checks if item["name"] in image_names]
            application_names = {bound["name"] for bound in metric_bounds} - image_names
            application_checks = [
                item for item in checks if item["name"] in application_names
            ]
            converged = result.requirements["numerical_convergence"]
            report.update(
                image_checks=image_checks,
                application_checks=application_checks,
                numerically_converged=converged,
                image_acceptance_basis="case_physics",
                imaging_success_under_case_physics=(
                    all(item["passed"] for item in image_checks)
                    if converged and contract["calibrated"]
                    else None
                ),
                image_verdict="passed"
                if report["status"] == "passed"
                else report["status"],
            )
            structure_valid = all(
                value
                for name, value in result.requirements.items()
                if name != "numerical_convergence"
            )
            if not converged and structure_valid:
                report.update(
                    status="numerically_unresolved",
                    image_verdict="numerically_unresolved",
                )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Physical backend returned invalid output: {exc}") from exc
    return report
