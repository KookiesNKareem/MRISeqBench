"""Rebuild pinned cardiac label assets from the repository-local supplied phantom.

Run with the benchmark Python environment. This records every regenerated file
hash in anatomy.json; inspect that manifest and the assets together before use.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

root = Path(__file__).resolve().parent
previous = (
    json.loads((root / "anatomy.json").read_text())
    if (root / "anatomy.json").exists()
    else {}
)
source = root / "assets/cardiac_source.phantom"
with h5py.File(source) as f:
    xyz = [f["position/" + a][:] for a in ("x", "y", "z")]
    axes = [np.unique(a).astype(float) for a in xyz]
    props = np.unique(
        np.stack([f["contrast/" + k][:] for k in ["ρ", "T1", "T2", "Δw"]], axis=1),
        axis=0,
    ).astype(float)
    grid = np.zeros(tuple(map(len, axes)), np.uint8)
    grid[tuple(np.searchsorted(ax, a) for ax, a in zip(axes, xyz))] = (
        np.searchsorted(props[:, 1], f["contrast/T1"][:]) + 1
    )
# Uniform spacing is explicit, avoiding rounded Float32 coordinate perturbations.
spacing = np.array([0.0015, 0.0015, 0.00015])
mid = np.array([(a[0] + a[-1]) / 2 for a in axes])
print("source grid", grid.shape, "source center", mid)
meta = {
    "schema_version": 1,
    "source": {
        "path": "assets/cardiac_source.phantom",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_name": "Cardiac 3 T case 01, static 10x-z, fat shift, no B0 map",
        "source_spins": len(xyz[0]),
        "source_center_m": mid.tolist(),
        "coordinate_normalization": "Subtract the midpoint of each source coordinate axis; replace float32 spacing roundoff with the declared 1.5 x 1.5 x 0.15 mm uniform spacing. No crop, scale, deformation, or motion.",
    },
    "materials": ["fat", "myocardium", "blood"],
    "properties_columns": ["proton_density", "T1_s", "T2_s", "chemical_shift_hz_at_3T"],
    "properties": np.column_stack([props[:, :3], props[:, 3] / (2 * np.pi)]).tolist(),
    "objects": {},
}
meta["field_profiles"] = previous.get("field_profiles", {})
if "literature" in previous:
    meta["literature"] = previous["literature"]
meta["field_profiles"]["3t"] = {
    **previous.get("field_profiles", {}).get("3t", {}),
    "field_strength_T": 3.0,
    "properties": meta["properties"],
    "provenance": "Exact supplied source contrast values.",
}
for profile, field in (("15t", 1.5), ("055t", 0.55)):
    meta["field_profiles"].setdefault(
        profile,
        {
            "field_strength_T": field,
            "properties": None,
            "provenance": "Pending field-specific literature values.",
        },
    )
for dims in (2, 3):
    ref = f"cardiac_{dims}d@1"
    arr = grid[:, :, 545] if dims == 2 else grid
    p = root / "assets" / f"cardiac_{dims}d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "labels", data=arr, compression="gzip", compression_opts=9, shuffle=True
        )
        f["voxel_size_m"] = spacing[:dims]
        f["center_m"] = np.zeros(dims)
    meta["objects"][ref] = {
        "path": str(p.relative_to(root)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "dimensions": dims,
        "grid": list(arr.shape),
        "voxel_size_m": spacing[:dims].tolist(),
        "center_m": [0] * dims,
        "slice_index_zero_based": 545 if dims == 2 else None,
        "field_profile": "3t",
    }
    (root / f"cardiac_{dims}d.yaml").write_text(
        f"# yaml-language-server: $schema=../../schemas/objects.schema.json\nschema_version: 1\nid: cardiac_{dims}d\nversion: 1\ndescription: Static cardiac {'short-axis slice' if dims == 2 else 'volume'} with myocardium, blood and fat; intrinsic fat chemical shift at the scanner field.\nphantom: {ref}\n"
    )
    registry = json.loads((root / "registry.json").read_text())
    extent = np.array(arr.shape) * spacing[:dims] * 1000 / 2
    registry[ref] = {
        "dimensions": dims,
        "grid": list(arr.shape),
        "fov_mm": [200] * dims,
        "bounds_mm": [[-float(v), float(v)] for v in extent],
        "materials": meta["materials"],
    }
    for suffix, field in (("055t", 0.55), ("15t", 1.5), ("3t", 3.0)):
        name = f"cardiac_{dims}d_{suffix}"
        alias = name + "@1"
        meta["objects"][alias] = deepcopy(meta["objects"][ref])
        meta["objects"][alias]["field_profile"] = suffix
        registry[alias] = deepcopy(registry[ref])
        (root / (name + ".yaml")).write_text(
            f"# yaml-language-server: $schema=../../schemas/objects.schema.json\nschema_version: 1\nid: {name}\nversion: 1\ndescription: Static cardiac {'short-axis slice' if dims == 2 else 'volume'} at {field:g} T with field-specific tissue relaxation and fat chemical shift.\nphantom: {alias}\n"
        )
    (root / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
(root / "anatomy.json").write_text(json.dumps(meta, indent=2) + "\n")
