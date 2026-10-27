"""Stable repository labels for configuration, CLI listings and analysis.

Manifest references and evaluator routing retain their historical identifiers.
Do not renumber existing tasks when extending a level: append new tasks instead.
"""

TASK_GROUPS = {
    "T1": ("gre_t1w", "gre_t2star", "spin_echo_t2w", "bssfp"),
    "T2": (
        "epi_single_shot",
        "epi_multishot",
        "tse_t2w",
        "grase",
        "radial_readout",
        "spiral_gre",
        "propeller_tse",
        "stack_of_stars",
        "ute_radial",
    ),
    "T3": (
        "sense_sampling",
        "grappa_sampling",
        "partial_fourier",
        "compressed_sensing",
        "variable_density",
        "elliptical_kspace",
        "caipirinha",
        "reduced_fov",
        "rfov_epi",
    ),
    "T4": (
        "t1_mapping",
        "t2_mapping",
        "t2_prepared",
        "adiabatic_inversion",
        "rf_excitation_2d",
    ),
}
TASK_LABELS = {
    task: f"{level}.{number}"
    for level, tasks in TASK_GROUPS.items()
    for number, task in enumerate(tasks, 1)
}
LEVEL_LABELS = {f"S{n}": f"T{n}" for n in range(1, 5)}
HARDWARE_LABELS = {
    "low_field": "H1",
    "standard_1_5t": "H2",
    "high_performance": "H3",
}
SIMULATOR_LABELS = {"mrzero": "S1", "koma": "S2", "komamripy": "S3"}
# 2D/3D and field-dependent tissue profiles are variants of the same anatomy.
PHANTOM_LABELS = {
    f"cardiac_{dimension}d{suffix}": "P1"
    for dimension in (2, 3)
    for suffix in ("", "_055t", "_15t", "_3t")
}
PHANTOM_OBJECTS = {"P1": {2: "cardiac_2d", 3: "cardiac_3d"}}


def catalog_reference(kind, reference, *, dimensions=2):
    """Resolve configuration labels before creating runtime contracts."""
    if not isinstance(reference, str):
        return reference  # Let the manifest model report an invalid input type.
    labels = {"tasks": TASK_LABELS, "hardware": HARDWARE_LABELS}.get(kind, {})
    identifier, separator, version = reference.partition("@")
    aliases = {label: identifier for identifier, label in labels.items()}
    if kind == "objects":
        aliases = {
            label: variants[dimensions] for label, variants in PHANTOM_OBJECTS.items()
        }
    return aliases.get(identifier, identifier) + separator + version


def case_labels(case):
    """Compute repository labels without modifying a resolved runtime contract.

    The fixed uniform-disc RF assay and infrastructure smoke task are unnumbered;
    neither is an additional benchmark anatomy or one of the 27 numbered tasks.
    """
    task = case["task"]
    return {
        "task": TASK_LABELS.get(task["id"]),
        "level": LEVEL_LABELS[task["sequence"]["level"]],
        "hardware": HARDWARE_LABELS[case["hardware"]["id"]],
        "phantom": PHANTOM_LABELS.get(case["object"]["phantom"].split("@")[0]),
    }


def arm_label(arm):
    """Simulator modifiers describe access, independently of authoring arm IDs."""
    label = arm.id.split("_", 1)[0].upper()
    simulator = SIMULATOR_LABELS.get(arm.simulator)
    return f"{label} {simulator}" if simulator else label
