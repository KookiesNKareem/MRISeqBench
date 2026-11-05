"""Resolve a task's follow-ups into cumulative grading contracts."""

from copy import deepcopy

from .models import Task

# These fields are checked by the physical evaluators.
COMMON = {
    "fov_mm",
    "te_ms",
    "tr_ms",
    "duration_s",
    "flip_angle_deg",
    "slice_thickness_mm",
    "readout.shots",
    "readout.echo_train_length",
    "readout.interleaves",
    "readout.spokes",
    "readout.turns",
    "readout.blade_width_lines",
    "readout.echo_spacing_ms",
    "sampling.acceleration",
    "sampling.calibration_lines",
    "sampling.partial_fourier_fraction",
    "sampling.caipi_shift",
    "sampling.reduced_fov_fraction",
    "sampling.mask_seed",
}
CARDIAC = {
    "fov_mm",
    "te_ms",
    "tr_ms",
    "duration_s",
    "flip_angle_deg",
    "slice_thickness_mm",
    "readout.shots",
    "readout.echo_train_length",
    "readout.echo_spacing_ms",
    "application.inversion_time_ms",
    "application.inversion_times_ms",
    "application.preparation_duration_ms",
    "application.preparation_durations_ms",
    "application.b_value_s_per_mm2",
    "application.venc_cm_per_s",
    "application.fat_suppression_max_ratio",
    "temporal.trigger_delay_ms",
    "temporal.acquisition_window_ms",
    "temporal.segments_per_frame",
}


def workflow_for(catalog, base_case):
    task = catalog.get(
        "tasks", f"{base_case['task']['id']}@{base_case['task']['version']}"
    )
    if not task.followups:
        raise ValueError(f"{task.id}: no agentic follow-ups defined")
    if len({stage.id for stage in task.followups}) != len(task.followups):
        raise ValueError(f"{task.id}: duplicate follow-up IDs")
    allowed = (
        {"excitation_target"}
        if task.id == "rf_excitation_2d"
        else CARDIAC
        if base_case["evaluation"]["id"] == "s4"
        else COMMON
    )
    stages = [{"id": "generate", "case": deepcopy(base_case)}]
    sequence = deepcopy(base_case["task"]["sequence"])
    for index, stage in enumerate(task.followups, 1):
        previous = deepcopy(sequence)
        for path, value in stage.patch.items():
            if path not in allowed:
                raise ValueError(f"unsupported follow-up edit: {path}")
            keys = path.split(".")
            parent = sequence
            for key in keys[:-1]:
                if (
                    parent is sequence
                    and key in {"readout", "sampling", "application", "temporal"}
                    and parent.get(key) is None
                ):
                    parent[key] = {}
                if not isinstance(parent.get(key), dict):
                    raise ValueError(f"follow-up has no parent: {path}")  # noqa: TRY004
                parent = parent[key]
            parent[keys[-1]] = deepcopy(value)
        if sequence == previous:
            raise ValueError(f"{task.id}/{stage.id}: no-op edit")
        effective = Task.model_validate({**base_case["task"], "sequence": sequence})
        catalog.validate_geometry(effective, effective.object)
        case = deepcopy(base_case)
        case["task"] = effective.model_dump(mode="json", exclude={"followups"})
        case["task"]["objective"] = (
            "Update the existing sequence. "
            + stage.request
            + "\nPreserve requirements not changed by this request. "
            "Keep the sequence-family requirements from the initial task. "
            "Use the current numeric specification below."
        )
        case["case_id"] += f"--stage-{index}"
        stages.append({"id": stage.id, "case": case})
    return {
        "id": f"{task.id}@{task.version}",
        "case_id": base_case["case_id"],
        "stages": stages,
    }


def stage_case(catalog, case, index):
    stages = workflow_for(catalog, case)["stages"]
    if not 0 <= index < len(stages):
        raise ValueError(f"stage must be between 0 and {len(stages) - 1}")
    return stages[index]["case"]


def summarize_workflows(reports):
    reports = [r for r in reports if "workflow" in r]
    rows = []
    for index in range(max((len(r["workflow"]["stages"]) for r in reports), default=0)):
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
                "passed": sum(s["status"] == "passed" for s in reached),
                "failed": sum(s["status"] == "failed" for s in reached),
                "not_reached": len(stages) - len(reached),
                "submissions": sum(s["submissions"] for s in stages),
            }
        )
    return {
        "completed_workflows": len(reports),
        "passed_workflows": sum(r["status"] == "passed" for r in reports),
        "stages": rows,
    }
