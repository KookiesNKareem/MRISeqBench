"""Resolve explicit references; never merge manifests through inheritance."""

import hashlib
import json
from pathlib import Path

import yaml

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
                if path.stem != doc.id:
                    raise ValueError(f"{path}: filename must match id {doc.id}")
                entries[f"{doc.id}@{doc.version}"] = doc
            self.documents[kind] = entries
        if not self.documents["tasks"]:
            raise ValueError(f"no tasks found in {self.root}/benchmark/tasks")
        self.validate_references()
        anatomy = self.root / "benchmark/objects/anatomy.json"
        if anatomy.exists():
            from .evaluation.s1_anatomy import validate_assets

            validate_assets(self.root)

    def get(self, kind, ref):
        # CLI convenience: a bare id is accepted only when it is unambiguous.
        entries = self.documents[kind]
        if "@" not in ref:
            matches = [doc for key, doc in entries.items() if key.split("@")[0] == ref]
            if len(matches) == 1:
                return matches[0]
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

    def validate_asset(self, asset):
        path = (self.root / asset.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("asset path escapes catalog root")
        if not path.is_file():
            raise ValueError(f"missing asset: {asset.path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset.sha256:
            raise ValueError(f"asset hash mismatch: {asset.path}")

    def validate_geometry(self, task, object_ref=None):
        obj = self.get("objects", object_ref or task.object)
        if obj.phantom not in self.phantoms:
            raise ValueError(f"unknown KomaMRI phantom {obj.phantom!r}")
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
        seed=42,
        hardware_ref=None,
    ):
        if physics is None:
            profile = "static_b0_3d" if len(task.sequence.matrix) == 3 else "static_b0"
            physics = self.get("physics", profile)
        obj = self.get("objects", object_ref or task.object)
        hardware = self.get("hardware", hardware_ref or task.hardware)
        if obj.id.startswith("cardiac_"):
            fields = {0.55: "055t", 1.5: "15t", 3.0: "3t"}
            if hardware.field_strength_T not in fields:
                raise ValueError(
                    "No validated cardiac tissue profile for scanner field strength"
                )
            dimensions = self.phantoms[obj.phantom].dimensions
            field_ref = f"cardiac_{dimensions}d_{fields[hardware.field_strength_T]}@1"
            if obj.id not in {"cardiac_2d", "cardiac_3d"} and obj.phantom != field_ref:
                raise ValueError(
                    "Cardiac tissue profile and scanner field strength must match"
                )
            obj = self.get("objects", field_ref)
        self.validate_geometry(task, f"{obj.id}@{obj.version}")
        evaluator = self.get("evaluators", task.evaluation)
        # Validate object, field and motion dimensions.
        for effect in physics.effects:
            dimensions = None
            for name in (
                "gradient_hz_per_m",
                "gradient_scale_per_m",
                "center_mm",
                "translation_amplitude_mm",
                "strain_amplitude",
            ):
                if hasattr(effect, name):
                    dimensions = len(getattr(effect, name))
            if dimensions is not None and dimensions != len(task.sequence.fov_mm):
                raise ValueError(
                    f"{task.id}/{physics.id}: effect dimensions do not match FOV"
                )
            if (
                effect.kind == "motion"
                and effect.model == "rigid_keyframes"
                and len(task.sequence.fov_mm) != 3
            ) and any(
                p.translation_mm[2] or p.rotation_xyz_deg[0] or p.rotation_xyz_deg[1]
                for p in effect.poses
            ):
                raise ValueError(
                    "through-plane motion requires a 3D object/acquisition"
                )
            if hasattr(effect, "asset"):
                if len(effect.asset.shape) != len(task.sequence.fov_mm):
                    raise ValueError("field asset dimensions do not match FOV")
                self.validate_asset(effect.asset)
            if effect.kind == "time_variation" and effect.material:
                materials = self.phantoms[obj.phantom].materials
                if effect.material not in materials:
                    raise ValueError("time variation references an unknown material")
        case_id = f"{task.id}-v{task.version}--{physics.id}-v{physics.version}"
        if object_ref and object_ref != task.object:
            case_id += f"--object-{obj.id}-v{obj.version}"
        if f"{hardware.id}@{hardware.version}" != task.hardware:
            case_id += f"--hardware-{hardware.id}-v{hardware.version}"
        object_contract = obj.model_dump(mode="json", exclude_none=True)
        if obj.id.startswith("cardiac_"):
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
        legacy_levels = {}
        if obj.level is not None and physics.level is not None:
            legacy_levels = {
                "object_level": f"P{max(int(obj.level[1:]), int(physics.level[1:]))}",
                "geometry_level": obj.level,
                "physics_level": physics.level,
            }
        image_policy = (
            "static"
            if obj.id.startswith("cardiac_")
            or "static" in (evaluator.case_image_criteria or {})
            else legacy_levels.get("object_level")
        )
        return {
            "case_id": case_id,
            "sequence_level": task.sequence.level,
            **legacy_levels,
            "image_policy": image_policy,
            "task": task_contract,
            "seed": seed,
            "object": {
                **object_contract,
                "definition": self.phantoms[obj.phantom].model_dump(mode="json"),
            },
            "physics": physics_contract,
            "hardware": hardware.model_dump(mode="json", exclude_none=True),
            "evaluation": self.get("evaluators", task.evaluation).model_dump(
                mode="json"
            ),
        }

    def cases(self, suite: Suite):
        cases, seen = [], set()
        for entry in suite.cases:
            task = self.get("tasks", entry.task)
            if f"{suite.id}@{suite.version}" in self.documents["suites"]:
                expected = (
                    "static_b0_3d@1"
                    if len(task.sequence.matrix) == 3
                    else "static_b0@1"
                )
                if entry.objects is not None or not task.object.startswith("cardiac_"):
                    raise ValueError(
                        "benchmark suites require fixed cardiac anatomy without an object axis"
                    )
                if entry.physics != [expected]:
                    raise ValueError(
                        "benchmark suites require one dimension-matched static B0 profile"
                    )
            for object_ref in entry.objects or [task.object]:
                for ref in entry.physics:
                    for hardware_ref in entry.hardware or [task.hardware]:
                        case = self.resolve(
                            task,
                            self.get("physics", ref),
                            object_ref,
                            hardware_ref=hardware_ref,
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
