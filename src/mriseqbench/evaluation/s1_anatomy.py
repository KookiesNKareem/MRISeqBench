"""Pinned cardiac voxel continuum, shared by native replay and independent targets.

Labels are piecewise constant over physical cells. Randomized integration samples
cells and positions within them; the Fourier oracle integrates every cell exactly.
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import finufft
import h5py
import numpy as np


@dataclass(frozen=True)
class Anatomy:
    labels: np.ndarray
    voxel_size_m: np.ndarray
    center_m: np.ndarray
    materials: list
    properties: np.ndarray
    source_sha256: str


def is_anatomical(case_or_ref):
    ref = (
        case_or_ref
        if isinstance(case_or_ref, str)
        else case_or_ref["object"]["phantom"]
    )
    return ref in {
        f"cardiac_{dims}d{suffix}@1"
        for dims in (2, 3)
        for suffix in ("", "_055t", "_15t", "_3t")
    }


def validate_assets(root):
    """Validate all pinned source/derived bytes; return the metadata digest."""
    directory = Path(root) / "benchmark/objects"
    data = (directory / "anatomy.json").read_bytes()
    meta = json.loads(data)
    items = [meta["source"], *meta["objects"].values()]
    if meta.get("literature"):
        items.append(meta["literature"])
    for item in items:
        _checked_asset(directory, item)
    return hashlib.sha256(data).hexdigest()


def _checked_asset(directory, entry):
    relative = Path(entry["path"])
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Anatomical asset path must remain inside benchmark/objects")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Anatomical asset SHA-256 mismatch: {path}")
    return data


def load_anatomy(root, ref):
    """Verify current bytes before consulting a cache keyed by their content."""
    directory = Path(root) / "benchmark/objects"
    meta_bytes = (directory / "anatomy.json").read_bytes()
    meta = json.loads(meta_bytes)
    data = _checked_asset(directory, meta["objects"][ref])
    return _decode_anatomy(ref, meta_bytes, data)


@lru_cache(maxsize=8)
def _decode_anatomy(ref, meta_bytes, data):
    meta = json.loads(meta_bytes)
    entry = meta["objects"][ref]
    with h5py.File(BytesIO(data)) as handle:
        labels = handle["labels"][...]
        spacing = handle["voxel_size_m"][...]
        center = handle["center_m"][...]
    if (
        tuple(labels.shape) != tuple(entry["grid"])
        or labels.ndim != entry["dimensions"]
        or not np.issubdtype(labels.dtype, np.integer)
        or labels.min() < 0
        or labels.max() > len(meta["materials"])
    ):
        raise ValueError(
            "Anatomical asset geometry or tissue labels disagree with metadata"
        )
    if (
        spacing.shape != (labels.ndim,)
        or center.shape != (labels.ndim,)
        or not np.isfinite(spacing).all()
        or not np.isfinite(center).all()
        or np.any(spacing <= 0)
        or not np.array_equal(spacing, entry["voxel_size_m"])
        or not np.array_equal(center, entry["center_m"])
    ):
        raise ValueError("Anatomical asset coordinates disagree with metadata")
    profile = meta.get("field_profiles", {}).get(entry.get("field_profile"))
    properties_data = (
        profile["properties"] if profile is not None else meta["properties"]
    )
    if properties_data is None:
        raise ValueError(
            f"Anatomical field-specific tissue properties are not configured: {ref}"
        )
    properties = np.asarray(properties_data, dtype=float)
    if (
        properties.shape != (len(meta["materials"]), 4)
        or not np.isfinite(properties).all()
        or np.any(properties[:, :3] <= 0)
    ):
        raise ValueError(
            "Anatomical properties must be finite positive rho/T1/T2 plus chemical shift"
        )
    for array in (labels, spacing, center, properties):
        array.flags.writeable = False
    return Anatomy(
        labels,
        spacing,
        center,
        meta["materials"],
        properties,
        hashlib.sha256(meta_bytes + data).hexdigest(),
    )


def _geometry(anatomy):
    shape = np.asarray(anatomy.labels.shape)
    return shape, anatomy.center_m - shape * anatomy.voxel_size_m / 2


def strata_per_cell(samples, cells, dims):
    """Per-axis strata so that cells * prod(strata) is close to the sample budget."""
    per_cell = max(1.0, samples / cells)
    side = max(1, int(np.floor(per_cell ** (1 / dims))))
    strata = [side] * dims
    # Spend the remainder on the leading axes so each budget doubling refines.
    for axis in range(dims):
        if np.prod(strata) * (strata[axis] + 1) / strata[axis] <= per_cell:
            strata[axis] += 1
    return strata


def sampled_anatomy(case, root, samples, replicate, *, directory=None):
    """Jittered-grid quadrature: each labeled cell splits into equal strata.

    Every stratum receives one uniform point, independently per replicate, with
    its exact volume as weight. The estimate is unbiased, and within-pixel
    structure (e.g. spoiler-induced isochromat dependence) is integrated far
    more evenly than with unstratified points.
    """
    from .s1_phantom import Spins

    anatomy = load_anatomy(str(root), case["object"]["phantom"])
    dims = anatomy.labels.ndim
    shape, lower = _geometry(anatomy)
    seed = case["evaluation"].get("spatial_seed")
    seed = 1729 if seed is None else seed
    spec = case["task"]["sequence"]
    image_voxel = np.prod(np.asarray(spec["fov_mm"]) / 1000 / spec["matrix"])
    cell_volume = np.prod(anatomy.voxel_size_m)
    flat = anatomy.labels.ravel()
    active = np.flatnonzero(flat > 0)
    if not len(active):
        raise ValueError("Registered anatomy has no labeled integration cells")
    strata = np.asarray(strata_per_cell(samples, len(active), dims))
    per_cell = int(np.prod(strata))
    total = len(active) * per_cell

    def allocate(shape, dtype):
        if directory is None:
            return np.empty(shape, dtype=dtype)
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Anonymous disk-backed mappings are reclaimable under the worker's RAM
        # cap and disappear with the array. No source/data file is overwritten.
        with tempfile.TemporaryFile(dir=directory) as handle:
            return np.memmap(handle, dtype=dtype, mode="w+", shape=shape)

    positions = allocate((total, 3), np.float64)
    labels = allocate((total,), np.int64)
    weights = allocate((total,), np.float64)
    offsets = np.indices(strata).reshape(dims, -1).T  # (per_cell, dims)
    rng = np.random.default_rng([seed, replicate])
    chunk = max(1, 1048576 // per_cell)
    for start in range(0, len(active), chunk):
        cells_ids = active[start : start + chunk]
        cells = np.column_stack(np.unravel_index(cells_ids, shape))
        jitter = rng.random((len(cells_ids), per_cell, dims))
        unit = (offsets[None] + jitter) / strata  # position within the cell
        target = slice(start * per_cell, (start + len(cells_ids)) * per_cell)
        positions[target] = 0
        positions[target, :dims] = (
            lower + (cells[:, None, :] + unit) * anatomy.voxel_size_m
        ).reshape(-1, dims)
        labels[target] = np.repeat(flat[cells_ids] - 1, per_cell)
        weights[target] = cell_volume / per_cell / image_voxel
    return Spins(
        positions,
        labels,
        anatomy.materials,
        anatomy.properties,
        weights,
        tuple(spec["matrix"]),
        anatomy.source_sha256,
    )


def _overlap(native_count, native_size, native_center, image_count, image_fov):
    native_edges = (
        native_center + (np.arange(native_count + 1) - native_count / 2) * native_size
    )
    image_edges = (np.arange(image_count + 1) / image_count - 0.5) * image_fov
    return np.maximum(
        0,
        np.minimum(image_edges[1:, None], native_edges[None, 1:])
        - np.maximum(image_edges[:-1, None], native_edges[None, :-1]),
    )


def fraction_phantom(case, root):
    """Exact anatomical cell/image-voxel overlap, positioned at image centres.

    This deterministic geometry is for masks and the voxel-centre phase oracle;
    it is never used to integrate encoded Bloch signals.
    """
    from .s1_phantom import Spins

    anatomy = load_anatomy(str(root), case["object"]["phantom"])
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    dims = len(matrix)
    overlaps = [
        _overlap(n, size, center, int(m), extent)
        for n, size, center, m, extent in zip(
            anatomy.labels.shape,
            anatomy.voxel_size_m,
            anatomy.center_m,
            matrix,
            fov,
            strict=True,
        )
    ]
    positions, labels, weights = [], [], []
    image_voxel = np.prod(fov / matrix)
    for label in range(len(anatomy.materials)):
        volume = (anatomy.labels == label + 1).astype(float)
        # Reduce the finely oversampled z axis first to bound memory and work.
        for axis in reversed(range(dims)):
            volume = np.moveaxis(
                np.tensordot(overlaps[axis], volume, axes=(1, axis)), 0, axis
            )
        indices = np.column_stack(np.nonzero(volume > image_voxel * 1e-12))
        xyz = np.zeros((len(indices), 3))
        xyz[:, :dims] = ((indices + 0.5) / matrix - 0.5) * fov
        positions.append(xyz)
        labels.append(np.full(len(indices), label, dtype=np.int64))
        weights.append(volume[tuple(indices.T)] / image_voxel)
    return Spins(
        np.concatenate(positions),
        np.concatenate(labels),
        anatomy.materials,
        anatomy.properties,
        np.concatenate(weights),
        tuple(matrix),
        anatomy.source_sha256,
    )


def anatomical_reference(
    acq, case, root, response, *, clip_axis=None, clip_bounds_m=None
):
    """Exact Fourier integral of the voxel continuum, optionally slab clipped.

    NUFFT evaluates the finite sum over voxel centres; cell sinc factors integrate
    their interiors. Slab boundary cells use their actual intersection widths.
    """
    anatomy = load_anatomy(str(root), case["object"]["phantom"])
    dims = anatomy.labels.ndim
    shape, lower = _geometry(anatomy)
    spec = case["task"]["sequence"]
    image_voxel = np.prod(np.asarray(spec["fov_mm"]) / 1000 / spec["matrix"])
    k = np.asarray(acq.k[:, :dims], dtype=float)
    raw = np.zeros(len(k), complex)
    axes = [
        lower[d] + (np.arange(shape[d]) + 0.5) * anatomy.voxel_size_m[d]
        for d in range(dims)
    ]
    groups = [(None, None)]
    if clip_axis is not None:
        if clip_bounds_m is None or not 0 <= clip_axis < dims:
            raise ValueError("A valid anatomical clipping axis and bounds are required")
        lo, hi = clip_bounds_m
        size = anatomy.voxel_size_m[clip_axis]
        left = np.maximum(axes[clip_axis] - size / 2, lo)
        right = np.minimum(axes[clip_axis] + size / 2, hi)
        widths = np.maximum(0, right - left)
        groups = [
            (indices, float(widths[indices[0]]))
            for width in np.unique(np.round(widths, 14))
            if width > 0
            for indices in [
                np.flatnonzero(np.isclose(widths, width, rtol=0, atol=1e-14))
            ]
        ]
        axes[clip_axis] = (left + right) / 2
    for label in range(len(anatomy.materials)):
        for included, width in groups:
            active = anatomy.labels == label + 1
            if included is not None:
                keep = np.zeros(shape[clip_axis], bool)
                keep[included] = True
                view = [1] * dims
                view[clip_axis] = len(keep)
                active &= keep.reshape(view)
            index = np.nonzero(active)
            if not len(index[0]):
                continue
            sizes = anatomy.voxel_size_m.copy()
            if width is not None:
                sizes[clip_axis] = width
            factors = np.prod(sizes * np.sinc(k * sizes), axis=1) / image_voxel
            # DC is exact and avoids asking a type-3 planner for zero bandwidth.
            nonzero = np.any(k != 0, axis=1)
            transform = np.full(len(k), len(index[0]), dtype=complex)
            if nonzero.any():
                points = [np.ascontiguousarray(axes[d][index[d]]) for d in range(dims)]
                targets = [
                    np.ascontiguousarray(2 * np.pi * k[nonzero, d]) for d in range(dims)
                ]
                operation = finufft.nufft2d3 if dims == 2 else finufft.nufft3d3
                transform[nonzero] = operation(
                    *points,
                    np.ones(len(index[0]), complex),
                    *targets,
                    isign=-1,
                    eps=1e-10,
                    nthreads=1,
                )
            raw += transform * factors * response[label]
    return raw
