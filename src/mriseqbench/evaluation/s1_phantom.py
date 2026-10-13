"""Judge quadrature using the same analytic shapes/materials as the Julia loader."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Spins:
    positions: np.ndarray
    labels: np.ndarray
    materials: list
    properties: np.ndarray
    weight: float | np.ndarray
    grid: tuple
    source_sha256: str


def phantom(case, root, oversampling=2, *, gauss_legendre=True):
    from .s1_anatomy import fraction_phantom, is_anatomical

    if is_anatomical(case):
        return fraction_phantom(case, root)
    path = Path(root) / "benchmark/objects/shapes.json"
    source = path.read_bytes()
    shapes = json.loads(source)
    ref = case["object"]["phantom"]
    spec = case["task"]["sequence"]
    dims = len(spec["matrix"])
    cells = np.asarray(spec["matrix"]) * (2 if dims == 2 and gauss_legendre else 1)
    grid = tuple(int(n * oversampling) for n in cells)
    if np.prod(grid) > 4_000_000:
        raise ValueError(
            "S1 quadrature exceeds the 4 million spatial-node CPU budget; use a smaller refinement or an external backend"
        )
    fov = np.asarray(spec["fov_mm"]) / 1000
    if gauss_legendre:
        # Gauss-Legendre subcell quadrature: nonperiodic isochromat spacing,
        # exact low polynomial moments, and no random image noise.
        nodes, weights = np.polynomial.legendre.leggauss(oversampling)
        axes = [
            ((np.arange(n)[:, None] + 0.5 + nodes[None, :] / 2) / n - 0.5).ravel() * f
            for n, f in zip(cells, fov, strict=True)
        ]
        axis_weights = [
            np.tile(weights / 2 * spec["matrix"][j] / n, n) for j, n in enumerate(cells)
        ]
        quadrature = np.prod(
            np.stack(np.meshgrid(*axis_weights, indexing="ij"), axis=-1), axis=-1
        ).ravel()
    else:
        axes = [
            ((np.arange(n) + 0.5) / n - 0.5) * f for n, f in zip(grid, fov, strict=True)
        ]
        quadrature = np.full(
            int(np.prod(grid)), float(np.prod(np.asarray(spec["matrix"]) / grid))
        )
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dims)
    labels = np.full(len(coords), -1, dtype=np.int64)
    materials = list(shapes["materials"])
    for shape in shapes["objects"][ref]:
        q = (coords * 1000 - shape["center_mm"]) / shape["radii_mm"]
        inside = (
            np.all(np.abs(q) <= 1, axis=1)
            if shape["kind"] == "box"
            else np.sum(q * q, axis=1) <= 1
        )
        labels[inside] = materials.index(shape["material"])
    keep = labels >= 0
    xyz = np.zeros((keep.sum(), 3))
    xyz[:, :dims] = coords[keep]
    return Spins(
        xyz,
        labels[keep],
        materials,
        np.array([shapes["materials"][m] for m in materials]),
        quadrature[keep],
        grid,
        hashlib.sha256(source).hexdigest(),
    )


def rotation(angles):
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def vec3(values):
    return np.pad(np.asarray(values, dtype=float), (0, 3 - len(values)))


def environment(times, effects, materials):
    """Scanner-frame maps evaluated on moved coordinates, continuously in time."""
    count = len(times)
    transform = np.tile(np.eye(3), (count, 1, 1))
    shift = np.zeros((count, 3))
    b0 = np.zeros(count)
    gradient = np.zeros(3)
    gaussian = np.zeros(8)  # amplitude, centre xyz, sigma xyz, enabled
    pd = np.ones((count, len(materials)))
    for e in effects:
        kind, model = e["kind"], e["model"]
        if kind == "motion":
            if model in ("rigid_sinusoidal", "affine_sinusoidal"):
                s = np.sin(
                    2 * np.pi * times / e["period_s"]
                    + np.deg2rad(e.get("phase_deg", 0))
                )
                if model == "affine_sinusoidal":
                    transform[:, range(3), range(3)] = 1 + s[:, None] * vec3(
                        e["strain_amplitude"]
                    )
                else:
                    shift = s[:, None] * vec3(e["translation_amplitude_mm"]) / 1000
                    for j, a in enumerate(s * e["rotation_amplitude_deg"]):
                        transform[j] = rotation([0, 0, a])
            elif model == "rigid_keyframes":
                poses = e["poses"]
                for j, t in enumerate(times):
                    k = max(
                        0,
                        np.searchsorted([p["time_s"] for p in poses], t, side="right")
                        - 1,
                    )
                    a = np.asarray(poses[k]["rotation_xyz_deg"], dtype=float)
                    tr = np.asarray(poses[k]["translation_mm"], dtype=float)
                    if e["interpolation"] == "linear" and k < len(poses) - 1:
                        f = np.clip(
                            (t - poses[k]["time_s"])
                            / (poses[k + 1]["time_s"] - poses[k]["time_s"]),
                            0,
                            1,
                        )
                        a = (1 - f) * a + f * np.asarray(
                            poses[k + 1]["rotation_xyz_deg"]
                        )
                        tr = (1 - f) * tr + f * np.asarray(
                            poses[k + 1]["translation_mm"]
                        )
                    transform[j] = rotation(a)
                    shift[j] = tr / 1000
            else:
                raise ValueError(f"Unsupported S1 motion: {model}")
        elif kind == "b0":
            b0 += e.get("offset_hz", 0)
            if model == "linear":
                gradient += vec3(e["gradient_hz_per_m"])
            elif model == "gaussian":
                gaussian[:7] = np.r_[
                    e["amplitude_hz"],
                    vec3(e["center_mm"]) / 1000,
                    np.r_[
                        np.asarray(e["sigma_mm"]) / 1000,
                        np.ones(3 - len(e["sigma_mm"])),
                    ],
                ]
                gaussian[7] = 1
            elif model != "constant":
                raise ValueError(f"Unsupported S1 field: {model}")
        elif kind == "time_variation" and model == "sinusoidal":
            s = np.sin(
                2 * np.pi * times / e["period_s"] + np.deg2rad(e.get("phase_deg", 0))
            )
            if e["target"] == "b0_offset":
                b0 += e["amplitude_hz"] * s
            elif e["target"] == "proton_density":
                pd[:, materials.index(e["material"])] *= 1 + e["amplitude_scale"] * s
            else:
                raise ValueError(f"Unsupported S1 time variation: {e['target']}")
        else:
            raise ValueError(f"Unsupported S1 effect: {kind}/{model}")
    return transform, shift, b0, gradient, gaussian, pd
