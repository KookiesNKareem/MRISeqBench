"""Frozen analytic-phantom controls; never an active cardiac qualification."""

from pathlib import Path

from mriseqbench.catalog import load_yaml
from mriseqbench.models import Evaluation, Hardware, Task

FIXTURES = Path(__file__).resolve().parent


def legacy_task(catalog, reference):
    # Install the retired scanner only in this test-local catalog, never in product code.
    catalog.documents["hardware"]["standard@1"] = Hardware.model_validate(
        load_yaml(FIXTURES / "legacy_hardware/standard.yaml")
    )
    name = reference.split("@")[0]
    if name.endswith("_research"):
        task = Task.model_validate(
            load_yaml(FIXTURES / "legacy_research_tasks" / f"{name}.yaml")
        )
    else:
        frozen = FIXTURES / "legacy_tasks" / f"{name}.yaml"
        if (
            reference not in catalog.documents["tasks"]
            and f"{name}@1" not in catalog.documents["tasks"]
            and frozen.exists()
        ):
            task = Task.model_validate(load_yaml(frozen))
        else:
            task = catalog.get("tasks", reference)
    return task.model_copy(update={"hardware": "standard@1"})


def legacy_evaluator(name):
    return Evaluation.model_validate(
        load_yaml(FIXTURES / "legacy_evaluators" / f"{name}.yaml")
    ).model_dump(mode="json")


def resolve_legacy_case(
    catalog, task, physics="ideal", object_ref=None, *, hardware_ref=None
):
    """Resolve historical geometry and policy with explicit regression-only status."""
    if isinstance(task, str):
        task = legacy_task(catalog, task)
    else:
        legacy_task(catalog, "gre_t1w")  # Load the local retired hardware fixture.
        task = task.model_copy(update={"hardware": "standard@1"})
    family = task.id.removesuffix("_research")
    object_ref = object_ref or (
        "volume_ellipsoids@1"
        if len(task.sequence.matrix) == 3
        else "nested_ellipses@1"
        if family == "rich_geometry_gre"
        else "tissue_discs@1"
    )
    physics = catalog.get("physics", physics) if isinstance(physics, str) else physics
    case = catalog.resolve(task, physics, object_ref, hardware_ref=hardware_ref)
    evaluator = case["evaluation"]["id"]
    case["evaluation"] = legacy_evaluator(evaluator)
    if evaluator == "s3":
        # Synthetic decision-policy testing only; published cardiac S3 stays false.
        case["evaluation"]["calibrated"] = True
    case["image_policy"] = case["object_level"]
    case["task"]["evaluation"] = f"{evaluator}@{case['evaluation']['version']}"
    return case
