"""Versioned cumulative workflow contracts; no model-generated tasks or judge patches."""

from copy import deepcopy

from .models import Task

# Only parameters consumed by the existing physical evaluators may be edited.
COMMON = {"fov_mm", "te_ms", "tr_ms", "duration_s", "flip_angle_deg"}
CARDIAC = {
    "fov_mm",
    "application.inversion_time_ms",
    "application.inversion_times_ms",
    "application.preparation_duration_ms",
    "application.preparation_durations_ms",
    "temporal.trigger_delay_ms",
    "temporal.acquisition_window_ms",
}


def apply_patch(sequence, patch, allowed):
    result = deepcopy(sequence)
    for path, value in patch.items():
        if path not in allowed:
            raise ValueError(f"unsupported workflow edit: {path}")
        keys = path.split(".")
        parent = result
        for key in keys[:-1]:
            if not isinstance(parent.get(key), dict):
                raise ValueError(f"workflow patch has no parent: {path}")  # noqa: TRY004
            parent = parent[key]
        if keys[-1] not in parent:
            raise ValueError(f"unknown workflow field: {path}")
        parent[keys[-1]] = deepcopy(value)
    return result


def compile_workflow(catalog, base_case, definition):
    """Validate all stages privately. Future stages never enter the public case."""
    if definition.task != f"{base_case['task']['id']}@{base_case['task']['version']}":
        raise ValueError("workflow base task version does not match")
    allowed = (
        {"excitation_target"}
        if base_case["task"]["id"] == "rf_excitation_2d"
        else CARDIAC
        if base_case["evaluation"]["id"] == "s4"
        else COMMON
    )
    workflow_id = f"{definition.id}@{definition.version}"
    chain_id = (
        base_case["case_id"] + f"--workflow-{definition.id}-v{definition.version}"
    )
    stages = []
    sequence = apply_patch(
        base_case["task"]["sequence"], definition.initial_patch, allowed
    )
    for index, stage in enumerate([None, *definition.stages]):
        if stage:
            previous = sequence
            sequence = apply_patch(sequence, stage.patch, allowed)
            if sequence == previous:
                raise ValueError(f"{definition.id}/{stage.id}: no-op edit")
        task = Task.model_validate({**base_case["task"], "sequence": sequence})
        catalog.validate_geometry(task, task.object)
        case = deepcopy(base_case)
        case["task"] = task.model_dump(mode="json")
        case["case_id"] = chain_id + f"--stage-{index}"
        case["workflow_stage"] = {
            "workflow": workflow_id,
            "index": index,
            "id": stage.id if stage else "generate",
        }
        if stage:
            case["task"]["objective"] = (
                "Continue the existing sequence and source. "
                + stage.request
                + "\nPreserve every requirement not changed by this request. "
                "The effective numeric specification below supersedes earlier numeric requests. "
                "Original task description (unchanged structural requirements still apply):\n"
                + base_case["task"]["objective"]
            )
        stages.append(
            {
                "id": stage.id if stage else "generate",
                "case": case,
                "request": stage.request if stage else base_case["task"]["objective"],
                "patch": stage.patch if stage else {},
            }
        )
    return {
        "id": workflow_id,
        "case_id": chain_id,
        "qualification": definition.qualification,
        "stages": stages,
    }


def workflow_for(catalog, case):
    matches = [
        w
        for w in catalog.documents["workflows"].values()
        if w.task == f"{case['task']['id']}@{case['task']['version']}"
        and w.version == 1
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one v1 workflow for {case['task']['id']}")
    return compile_workflow(catalog, case, matches[0])


def summarize_workflows(reports):
    reports = [r for r in reports if "workflow" in r]
    rows = []
    count = max((len(r["workflow"]["stages"]) for r in reports), default=0)
    for index in range(count):
        stages = [
            r["workflow"]["stages"][index]
            for r in reports
            if index < len(r["workflow"]["stages"])
        ]
        reached = [s for s in stages if s["status"] != "not_reached"]
        rows.append(
            {
                "index": index,
                "reached": len(reached),
                "supplied": sum(s.get("origin") == "supplied" for s in reached),
                "passed": sum(s["status"] == "passed" for s in reached),
                "failed": sum(s["status"] == "failed" for s in reached),
                "not_reached": len(stages) - len(reached),
                "submissions": sum(s.get("submissions", 0) for s in stages),
            }
        )
    return {
        "completed_workflows": len(reports),
        "passed_workflows": sum(r["status"] == "passed" for r in reports),
        "stages": rows,
    }


def execution_workflow(catalog, case, mode):
    workflow = workflow_for(catalog, case)
    workflow["mode"] = mode
    if mode == "final_spec":
        stage = deepcopy(workflow["stages"][-1])
        stage["id"] = "final_spec"
        stage["case"]["workflow_stage"].update(index=0, id="final_spec")
        stage["case"]["task"]["objective"] = (
            "Generate a fresh sequence from the complete numeric specification below. "
            "It supersedes numeric values in this original family description: "
            + case["task"]["objective"]
        )
        stage["request"] = stage["case"]["task"]["objective"]
        workflow["stages"] = [stage]
    return workflow
