"""Frozen analytic-phantom controls; never an active cardiac qualification."""

from pathlib import Path

from mriseqbench.catalog import load_yaml
from mriseqbench.models import MODELS, Evaluation, Hardware

FIXTURES = Path(__file__).resolve().parent


def install_legacy_catalog(catalog):
    """Install tiny analytic inputs explicitly in one numerical test's catalog."""
    for kind in ("objects", "physics"):
        for path in (FIXTURES / "legacy_catalog" / kind).glob("*.yaml"):
            document = MODELS[kind].model_validate(load_yaml(path))
            reference = f"{document.id}@{document.version}"
            catalog.documents[kind].setdefault(reference, document)
    catalog.documents["hardware"].setdefault(
        "standard@1",
        Hardware.model_validate(load_yaml(FIXTURES / "legacy_hardware/standard.yaml")),
    )
    return catalog


def legacy_task(catalog, reference):
    install_legacy_catalog(catalog)
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
    object_ref = object_ref or (
        "volume_ellipsoids@1" if len(task.sequence.matrix) == 3 else "tissue_discs@1"
    )
    physics = catalog.get("physics", physics) if isinstance(physics, str) else physics
    case = catalog.resolve(task, physics, object_ref, hardware_ref=hardware_ref)
    evaluator = case["evaluation"]["id"]
    case["evaluation"] = legacy_evaluator(evaluator)
    if evaluator == "s3":
        # Synthetic decision-policy regression only; not a cardiac qualification.
        case["evaluation"]["calibrated"] = True
    case["image_policy"] = case["object_level"]
    case["task"]["evaluation"] = f"{evaluator}@{case['evaluation']['version']}"
    return case
