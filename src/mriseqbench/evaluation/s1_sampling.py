"""Analytic phantom encoding and randomized, variance-controlled Bloch integration.

The residual control variate preserves every simulated coherence pathway. Independent
scrambles permit unbiased quadratic image scores without squaring sampling noise.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import j1
from scipy.stats import qmc, t

from .s1_phantom import Spins


def source(case, root):
    path = Path(root) / "benchmark/objects/shapes.json"
    data = path.read_bytes()
    shapes = json.loads(data)
    return (
        shapes,
        shapes["objects"][case["object"]["phantom"]],
        hashlib.sha256(data).hexdigest(),
    )


def sampled_phantom(case, root, samples, replicate):
    from .s1_anatomy import is_anatomical, sampled_anatomy

    if is_anatomical(case):
        return sampled_anatomy(case, root, samples, replicate)
    shapes, primitives, digest = source(case, root)
    spec = case["task"]["sequence"]
    dims = len(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    seed = case["evaluation"].get("spatial_seed")
    seed = 1729 if seed is None else seed
    materials = list(shapes["materials"])
    # Integrate each registered primitive directly. Rejecting FOV points at
    # material boundaries introduces discontinuities and noisy tissue volumes,
    # particularly for small inclusions. Exact primitive volumes remove that
    # source of variance without replacing any Bloch dynamics.
    count = 2 ** int(np.floor(np.log2(samples / len(primitives))))
    positions, labels, weights = [], [], []
    voxel = np.prod(fov / spec["matrix"])
    for index, shape in enumerate(primitives):
        unit = qmc.Sobol(
            d=dims, scramble=True, seed=seed + replicate + 65537 * index
        ).random_base2(int(np.log2(count)))
        radii = np.asarray(shape["radii_mm"]) / 1000
        if shape["kind"] == "box":
            coords = (2 * unit - 1) * radii
            volume = np.prod(2 * radii)
        elif dims == 2:
            angle, radius = 2 * np.pi * unit[:, 0], np.sqrt(unit[:, 1])
            coords = radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
            coords *= radii
            volume = np.pi * np.prod(radii)
        else:
            angle, z, radius = (
                2 * np.pi * unit[:, 0],
                2 * unit[:, 1] - 1,
                np.cbrt(unit[:, 2]),
            )
            radial = np.sqrt(np.maximum(0, 1 - z**2))
            coords = radius[:, None] * np.column_stack(
                (radial * np.cos(angle), radial * np.sin(angle), z)
            )
            coords *= radii
            volume = 4 * np.pi / 3 * np.prod(radii)
        coords += np.asarray(shape["center_mm"]) / 1000
        xyz = np.zeros((count, 3))
        xyz[:, :dims] = coords
        positions.append(xyz)
        labels.append(
            np.full(count, materials.index(shape["material"]), dtype=np.int64)
        )
        weights.append(np.full(count, volume / voxel / count))
        # The registry uses disjoint shapes or fully contained inclusions, just
        # as analytic_reference does. Bloch dynamics are linear in equilibrium
        # density, so subtract the overwritten parent response at identical nodes.
        parent = None
        for earlier in primitives[:index]:
            q = (np.asarray(shape["center_mm"]) - earlier["center_mm"]) / earlier[
                "radii_mm"
            ]
            if (np.abs(q) <= 1).all() if earlier["kind"] == "box" else q @ q <= 1:
                parent = earlier["material"]
        if parent is not None:
            positions.append(xyz.copy())
            labels.append(np.full(count, materials.index(parent), dtype=np.int64))
            weights.append(np.full(count, -volume / voxel / count))
    return Spins(
        np.concatenate(positions),
        np.concatenate(labels),
        materials,
        np.array([shapes["materials"][m] for m in materials]),
        np.concatenate(weights),
        tuple(spec["matrix"]),
        digest,
    )


def primitive_fourier(shape, k):
    radii = np.asarray(shape["radii_mm"]) / 1000
    center = np.asarray(shape["center_mm"]) / 1000
    dims = len(radii)
    phase = np.exp(-2j * np.pi * (k[:, :dims] @ center))
    if shape["kind"] == "box":
        return phase * np.prod(2 * radii * np.sinc(2 * k[:, :dims] * radii), axis=1)
    q = 2 * np.pi * np.linalg.norm(k[:, :dims] * radii, axis=1)
    ratio = np.ones(len(q))
    nonzero = q > 1e-4
    if dims == 2:
        ratio[nonzero] = 2 * j1(q[nonzero]) / q[nonzero]
        ratio[~nonzero] = 1 - q[~nonzero] ** 2 / 8
        volume = np.pi * np.prod(radii)
    else:
        x = q[nonzero]
        ratio[nonzero] = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        ratio[~nonzero] = 1 - q[~nonzero] ** 2 / 10
        volume = 4 * np.pi / 3 * np.prod(radii)
    return phase * volume * ratio


def analytic_reference(acq, case, root, response):
    from .s1_anatomy import anatomical_reference, is_anatomical

    if is_anatomical(case):
        return anatomical_reference(acq, case, root, response)
    shapes, primitives, _digest = source(case, root)
    materials = list(shapes["materials"])
    spec = case["task"]["sequence"]
    voxel = np.prod(np.asarray(spec["fov_mm"]) / 1000 / spec["matrix"])
    result = np.zeros(len(acq.adc), dtype=complex)
    for index, shape in enumerate(primitives):
        # Registered composites consist of disjoint shapes or fully contained
        # inclusions. Remove the overwritten parent material in each inclusion.
        parent = None
        c = np.asarray(shape["center_mm"])
        for earlier in primitives[:index]:
            q = (c - earlier["center_mm"]) / earlier["radii_mm"]
            if (np.abs(q) <= 1).all() if earlier["kind"] == "box" else q @ q <= 1:
                parent = earlier["material"]
        coefficient = response[materials.index(shape["material"])].copy()
        if parent is not None:
            coefficient -= response[materials.index(parent)]
        result += primitive_fourier(shape, acq.k) / voxel * coefficient
    return result


def quadratic(values):
    """Unbiased squared norm of the integral, and delete-one estimates.

    Each row is an independent randomized quadrature, not an ADC sample. Cross
    products remove the positive image-energy bias caused by quadrature noise.
    """
    flat = np.asarray(values).reshape(len(values), -1)
    n = len(flat)
    summed = flat.sum(axis=0)
    norms = np.sum(np.abs(flat) ** 2, axis=1)
    total = float(np.vdot(summed, summed).real)
    value = (total - norms.sum()) / (n * (n - 1))
    loo = (
        total - 2 * np.real(flat.conj() @ summed) + norms - (norms.sum() - norms)
    ) / ((n - 1) * (n - 2))
    return float(value), loo


def interval(value, loo):
    n = len(loo)
    se = np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    radius = float(t.ppf(0.995, n - 1) * se)
    return float(value - radius), float(value + radius)


def image_scores(images, reference, support):
    denom = max(float(np.linalg.norm(reference) ** 2), 1e-30)
    error, error_loo = quadratic(images - reference)
    energy, energy_loo = quadratic(images)
    outside, outside_loo = quadratic(images[:, ~support])
    error_ci = interval(error / denom, error_loo / denom)
    signal_ci = interval(energy / denom, energy_loo / denom)
    fraction = outside / max(energy, 1e-30)
    fraction_ci = interval(fraction, outside_loo / np.maximum(energy_loo, 1e-30))
    return {
        "complex_image_nrmse": np.sqrt(max(0.0, error / denom)),
        "normalized_squared_error": error / denom,
        "signal_fraction": np.sqrt(max(0.0, energy / denom)),
        "outside_object_energy_fraction": max(0.0, fraction),
    }, {
        "complex_image_nrmse": [np.sqrt(max(0.0, x)) for x in error_ci],
        "normalized_squared_error": list(error_ci),
        "signal_fraction": [np.sqrt(max(0.0, x)) for x in signal_ci],
        "outside_object_energy_fraction": [max(0.0, x) for x in fraction_ci],
    }
