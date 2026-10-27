"""Resolve explicit references; never merge manifests through inheritance."""

import json
from pathlib import Path

import yaml

from .labels import catalog_reference
from .models import MODELS, PhantomDefinition, Physics, Suite, Task


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML mapping keys must be strings")  # noqa: TRY004 -- malformed document, not API argument
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def load_yaml(path):
    try:
        return yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc


def manifest_directory(root, kind):
    return (
        Path(root) / "experiments"
        if kind == "experiments"
        else Path(root) / "benchmark" / kind
    )


def task_evaluation(evaluator, task_id):
    """The evaluator contract with the task's calibrated image limits applied."""
    contract = evaluator.model_dump(mode="json")
    override = (contract.pop("family_image_criteria", None) or {}).get(task_id)
    if override is not None:
        contract["case_image_criteria"] = {
            **contract["case_image_criteria"],
            "static": override,
        }
    return contract


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        registry_path = self.root / "benchmark/objects/registry.json"
        self.phantoms = {
            ref: PhantomDefinition.model_validate(value)
            for ref, value in json.loads(registry_path.read_text()).items()
        }
        self.documents = {}
        for kind, model in MODELS.items():
            entries = {}
            for path in sorted(manifest_directory(self.root, kind).glob("*.yaml")):
                doc = model.model_validate(load_yaml(path))
                ref = f"{doc.id}@{doc.version}"
                # A later version of a document lives beside the first as id@N.yaml.
                if path.stem not in (doc.id, ref):
                    raise ValueError(f"{path}: filename must be {doc.id} or {ref}")
                if ref in entries:
                    raise ValueError(f"{path}: duplicate {ref}")
                entries[ref] = doc
            self.documents[kind] = entries
        if not self.documents["tasks"]:
            raise ValueError(f"no tasks found in {self.root}/benchmark/tasks")
        self.validate_references()
        anatomy = self.root / "benchmark/objects/anatomy.json"
        if anatomy.exists():
            from .evaluation.s1_anatomy import validate_assets

            validate_assets(self.root)

    def get(self, kind, ref):
        ref = catalog_reference(kind, ref)
        # CLI convenience: a bare id selects its latest version.
        entries = self.documents[kind]
        if "@" not in ref:
            matches = [doc for key, doc in entries.items() if key.split("@")[0] == ref]
            if matches:
                return max(matches, key=lambda doc: doc.version)
        if ref not in entries:
            raise ValueError(f"unknown {kind} reference {ref!r}")
        return entries[ref]

    def validate_references(self):
        for obj in self.documents["objects"].values():
            if obj.phantom not in self.phantoms:
                raise ValueError(f"unknown KomaMRI phantom {obj.phantom!r}")
        for task in self.documents["tasks"].values():
            for kind, ref in [
                ("objects", task.object),
                ("hardware", task.hardware),
                ("evaluators", task.evaluation),
            ]:
                self.get(kind, ref)
            rf_task = task.id == "rf_excitation_2d" and task.object == "uniform_disc@1"
            if not rf_task and not self.get("objects", task.object).phantom.startswith(
                "cardiac_"
            ):
                raise ValueError(f"{task.id}: active tasks require cardiac anatomy")
            evaluator = self.get("evaluators", task.evaluation)
            if (
                evaluator.scope == "benchmark"
                and evaluator.id != task.sequence.level.lower()
                and not (rf_task and evaluator.id == "rf_excitation")
            ):
                raise ValueError(
                    f"{task.id}: evaluator must match sequence tier {task.sequence.level}"
                )
            self.validate_geometry(task)
        for suite in self.documents["suites"].values():
            self.cases(suite)
        for experiment in self.documents["experiments"].values():
            self.get("suites", experiment.suite)
            if experiment.arm:
                arm = self.get("arms", experiment.arm)
                if (arm.simulator == "none") != (
                    experiment.arm_runtime.simulator_image is None
                ):
                    raise ValueError(
                        "simulator_image must be supplied exactly for simulation-enabled arms"
                    )

    def validate_geometry(self, task, object_ref=None):
        obj = self.get("objects", object_ref or task.object)
        if task.id == "rf_excitation_2d" and obj.phantom != "uniform_disc@1":
            raise ValueError(
                "AI excitation requires the fixed uniform water-disc phantom"
            )
        definition = self.phantoms[obj.phantom]
        fov = task.sequence.fov_mm
        if definition.dimensions != len(fov):
            raise ValueError(f"{task.id}: object and acquisition dimensions must match")
        if any(
            lo < -f / 2 or hi > f / 2 for (lo, hi), f in zip(definition.bounds_mm, fov)
        ):
            raise ValueError(f"{task.id}: phantom extends beyond FOV")

    def resolve(
        self,
        task: Task,
        physics: Physics | None = None,
        object_ref=None,
        hardware_ref=None,
    ):
        if physics is None:
            profile = "static_b0_3d" if len(task.sequence.matrix) == 3 else "static_b0"
            physics = self.get("physics", profile)
        obj = self.get("objects", object_ref or task.object)
        hardware = self.get("hardware", hardware_ref or task.hardware)
        if obj.phantom.startswith("cardiac_"):
            fields = {0.55: "055t", 1.5: "15t", 3.0: "3t"}
            if hardware.field_strength_T not in fields:
                raise ValueError(
                    "No validated cardiac tissue profile for scanner field strength"
                )
            dimensions = self.phantoms[obj.phantom].dimensions
            field_ref = f"cardiac_{dimensions}d_{fields[hardware.field_strength_T]}@1"
            if (
                obj.phantom not in {"cardiac_2d@1", "cardiac_3d@1"}
                and obj.phantom != field_ref
            ):
                raise ValueError(
                    "Cardiac tissue profile and scanner field strength must match"
                )
            obj = self.get("objects", field_ref)
        self.validate_geometry(task, f"{obj.id}@{obj.version}")
        evaluator = self.get("evaluators", task.evaluation)
        for effect in physics.effects:
            gradient = getattr(effect, "gradient_hz_per_m", None)
            if gradient is not None and len(gradient) != len(task.sequence.fov_mm):
                raise ValueError(
                    f"{task.id}/{physics.id}: effect dimensions do not match FOV"
                )
        case_id = f"{task.id}-v{task.version}--{physics.id}-v{physics.version}"
        if object_ref and object_ref != task.object:
            case_id += f"--object-{obj.id}-v{obj.version}"
        case_id += f"--hardware-{hardware.id}-v{hardware.version}"
        object_contract = obj.model_dump(mode="json", exclude_none=True)
        if obj.phantom.startswith("cardiac_"):
            from .evaluation.s1_anatomy import load_anatomy

            anatomy = load_anatomy(str(self.root), obj.phantom)
            object_contract["field_strength_T"] = hardware.field_strength_T
            object_contract["tissue_properties"] = {
                name: {
                    "proton_density": float(values[0]),
                    "T1_s": float(values[1]),
                    "T2_s": float(values[2]),
                    "chemical_shift_hz": float(
                        values[3] * hardware.field_strength_T / 3.0
                    ),
                }
                for name, values in zip(
                    anatomy.materials, anatomy.properties, strict=True
                )
            }
            object_contract["source_sha256"] = anatomy.source_sha256
        task_contract = task.model_dump(mode="json")
        task_contract["object"] = f"{obj.id}@{obj.version}"
        task_contract["hardware"] = f"{hardware.id}@{hardware.version}"
        physics_contract = physics.model_dump(mode="json", exclude_none=True)
        if physics.reference_field_strength_T is not None:
            scale = hardware.field_strength_T / physics.reference_field_strength_T
            for effect in physics_contract["effects"]:
                if effect["kind"] != "b0" or effect["model"] not in {
                    "linear",
                    "gaussian",
                    "constant",
                }:
                    raise ValueError(
                        "field-scaled profiles require static parametric B0"
                    )
                for name in ("offset_hz", "amplitude_hz"):
                    if name in effect:
                        effect[name] *= scale
                if "gradient_hz_per_m" in effect:
                    effect["gradient_hz_per_m"] = [
                        v * scale for v in effect["gradient_hz_per_m"]
                    ]
            physics_contract["field_strength_T"] = hardware.field_strength_T
            physics_contract["field_scale"] = scale
        image_policy = (
            "static"
            if obj.phantom.startswith("cardiac_")
            or "static" in (evaluator.case_image_criteria or {})
            else None
        )
        return {
            "case_id": case_id,
            "sequence_level": task.sequence.level,
            "image_policy": image_policy,
            "task": task_contract,
            "seed": 42,
            "object": {
                **object_contract,
                "definition": self.phantoms[obj.phantom].model_dump(mode="json"),
            },
            "physics": physics_contract,
            "hardware": hardware.model_dump(mode="json", exclude_none=True),
            "evaluation": task_evaluation(evaluator, task.id),
        }

    def cases(self, suite: Suite):
        cases, seen = [], set()
        for entry in suite.cases:
            task = self.get("tasks", entry.task)
            expected = (
                "static_b0_3d@1" if len(task.sequence.matrix) == 3 else "static_b0@1"
            )
            rf_task = task.id == "rf_excitation_2d" and task.object == "uniform_disc@1"
            if entry.objects is not None or (
                not rf_task and not task.object.startswith("cardiac_")
            ):
                raise ValueError(
                    "benchmark suites require fixed cardiac anatomy without an object axis"
                )
            if entry.physics != [expected]:
                raise ValueError(
                    "benchmark suites require one dimension-matched static B0 profile"
                )
            for hardware_ref in entry.hardware or [task.hardware]:
                case = self.resolve(
                    task, self.get("physics", expected), hardware_ref=hardware_ref
                )
                if case["case_id"] in seen:
                    raise ValueError(f"duplicate suite case {case['case_id']}")
                seen.add(case["case_id"])
                cases.append(case)
        return cases

    def find_case(self, case_id):
        for suite in self.documents["suites"].values():
            for case in self.cases(suite):
                if case["case_id"] == case_id:
                    return case
        raise ValueError(f"unknown suite case {case_id!r}; use list --suite core")
