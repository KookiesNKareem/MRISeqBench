"""S3 sampling evaluation with fixed linear receive encoding and full-grid targets.

The image policy is a benchmark CG-SENSE baseline, including GRAPPA/CS mask
families. It never fits a reconstruction to a phantom or an image target.
"""

import json
from copy import copy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import finufft
import numpy as np

from ..io import write_json
from . import bart
from .s1 import evaluate_s1
from .s1_case import phase_reference as shared_phase_reference
from .s1_image import canonical_response, expected_receiver_phase, nrmse
from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_probes import radical_inverse, reference_index
from .s1_sampling import analytic_reference
from .s1_sequence import check, family
from .s3_sequence import PARALLEL, REDUCED_FOV, inspect_sequence, receive_model

RECEIVE_AMPLITUDE = 0.8
RECEIVE_NORMALIZATION = np.sqrt(4 * (1 + RECEIVE_AMPLITUDE**2 / 2))
# Fixed Tikhonov policy keeps BART's complex64 solve stable for irregular masks.
REGULARIZATION = 1e-3
# BART CG uses a zero stopping tolerance; excessive iterations lose conjugacy
# after reaching the complex64 floor on irregular masks. Fixed 50 is validated
# against dense normal equations, including phase-rotated inputs.
MAX_ITERATIONS = 50
REFERENCE_CHUNK_SPINS = 262144


def _coil_angles(dims, parallel=True):
    return (
        list(product(np.arange(4) * np.pi / 2, repeat=dims - 1)) if parallel else [()]
    )


def parallel_imaging(acq):
    return acq.measurements["family"] in PARALLEL or acq.measurements.get(
        "parallel_imaging", False
    )


def receive_terms(dims, parallel=True):
    """Finite Fourier representation of positive, smooth synthetic receive maps.

    Four translated cosine lobes per phase axis; sum of squared maps is one.
    These are fixed in material coordinates (co-moving under motion), explicitly
    a synthetic benchmark model rather than a scanner coil or noise model.
    """
    if not parallel:
        return [[(np.zeros(3), 1.0)]]
    channels = []
    for angles in _coil_angles(dims):
        terms = [(np.zeros(3), 1.0)]
        for axis, angle in enumerate(angles, start=1):
            expanded = []
            for shift, coefficient in terms:
                for harmonic in (-1, 0, 1):
                    delta = shift.copy()
                    delta[axis] += harmonic
                    value = (
                        1.0
                        if harmonic == 0
                        else RECEIVE_AMPLITUDE / 2 * np.exp(-1j * harmonic * angle)
                    )
                    expanded.append(
                        (delta, coefficient * value / RECEIVE_NORMALIZATION)
                    )
            terms = expanded
        channels.append(terms)
    return channels


def _receive_map(positions, fov, angles):
    """One exact separable cosine coil, without complex harmonic temporaries."""
    value = np.ones(len(positions))
    for axis, angle in enumerate(angles, start=1):
        phase = 2 * np.pi * positions[:, axis] / fov[axis] - angle
        value *= (1 + RECEIVE_AMPLITUDE * np.cos(phase)) / RECEIVE_NORMALIZATION
    return value


def receive_maps(positions, fov, parallel=True):
    xyz = np.asarray(positions)
    angles = _coil_angles(len(fov), parallel)
    maps = np.empty((len(angles), len(xyz)))
    for channel, phases in enumerate(angles):
        maps[channel] = _receive_map(xyz, fov, phases)
    return maps


def gaussian_receive_maps(positions, fov, parallel=True):
    """VD's fixed eight localized y coils, normalized to unit summed power."""
    if len(fov) != 2:
        raise ValueError("The Gaussian receive profile is defined for 2D VD")
    if not parallel:
        return np.ones((1, len(positions)))
    centres = (np.arange(8) + 0.5) / 8 * fov[1] - fov[1] / 2
    maps = np.exp(
        -0.5
        * ((np.asarray(positions)[:, 1][None, :] - centres[:, None]) / (fov[1] / 16))
        ** 2
    )
    return maps / np.sqrt(np.sum(maps * maps, axis=0))


def acquisition_maps(acq):
    return (
        gaussian_receive_maps
        if acq.measurements["family"] == "variable_density"
        else receive_maps
    )


def signal_shape(acq):
    return (
        8
        if acq.measurements["family"] == "variable_density"
        else len(receive_terms(len(acq.offset), parallel_imaging(acq))),
        len(acq.adc),
    )


def simulate_channels(acq, spins, effects=(), **steps):
    fov = np.asarray(acq.measurements["fov_mm"]) / 1000
    maps = acquisition_maps(acq)(spins.positions, fov, parallel_imaging(acq))
    if len(maps) == 1:
        return simulate(acq, spins, effects, **steps)[None, :]
    # Native per-spin acquisition blocks share one Bloch state across receive
    # channels. Receive maps are applied only at acquisition, with no dynamics
    # synthesized from ideal or reconstructed images.
    return simulate(acq, spins, effects, receive_weights=maps, **steps)


def reference_channels(acq, spins, spec, response):
    """Share bounded NUFFT plans across receive-weight channels."""
    fov = np.asarray(spec["fov_mm"]) / 1000
    gaussian = acq.measurements["family"] == "variable_density"
    angles = (
        list(range(8)) if gaussian else _coil_angles(len(fov), parallel_imaging(acq))
    )
    weights = np.broadcast_to(spins.weight, (len(spins.positions),))
    raw = np.zeros((len(angles), len(acq.adc)), complex)
    dims = len(fov)
    transform = finufft.nufft2d3 if dims == 2 else finufft.nufft3d3
    frequencies = [np.ascontiguousarray(2 * np.pi * acq.k[:, j]) for j in range(dims)]
    for start in range(0, len(weights), REFERENCE_CHUNK_SPINS):
        stop = start + REFERENCE_CHUNK_SPINS
        xyz, labels = spins.positions[start:stop], spins.labels[start:stop]
        for channel in range(0, len(angles), 4):
            maps = np.asarray(
                gaussian_receive_maps(xyz, fov)[channel : channel + 4]
                if gaussian
                else [
                    _receive_map(xyz, fov, phases)
                    for phases in angles[channel : channel + 4]
                ]
            )
            part = np.zeros((len(maps), len(acq.adc)), complex)
            for label in np.unique(labels):
                keep = labels == label
                values = transform(
                    *[np.ascontiguousarray(xyz[keep, j]) for j in range(dims)],
                    np.ascontiguousarray(
                        maps[:, keep] * weights[start:stop][keep], dtype=complex
                    ),
                    *frequencies,
                    eps=1e-10,
                    isign=-1,
                    nthreads=1,
                )
                part += values * response[label]
            # Preserve material summation inside each original spatial chunk.
            raw[channel : channel + len(maps)] += part
    return raw


def analytic_channels(acq, case, root, response):
    if family(case) == "variable_density":
        return gaussian_analytic_channels(acq, case, root, response)
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    output = []
    harmonics = {}
    for terms in receive_terms(len(fov), parallel_imaging(acq)):
        raw = np.zeros(len(acq.adc), complex)
        for shift, coefficient in terms:
            key = tuple(shift)
            if key not in harmonics:
                shifted = copy(acq)
                shifted.k = acq.k.copy()
                shifted.k[:, : len(fov)] -= shift[: len(fov)] / fov
                harmonics[key] = analytic_reference(shifted, case, root, response)
            # Preserve the original coefficient/addition order for every coil.
            raw += coefficient * harmonics[key]
        output.append(raw)
    return np.asarray(output)


def gaussian_analytic_channels(acq, case, root, response, order=8):
    """Continuous Gaussian receive: exact x-cell integral and converged y quadrature."""
    from .s1_anatomy import _geometry, load_anatomy
    from .s1_control import fourier

    anatomy = load_anatomy(root, case["object"]["phantom"])
    _, lower = _geometry(anatomy)
    spec = case["task"]["sequence"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    size = anatomy.voxel_size_m
    voxel = np.prod(fov / np.asarray(spec["matrix"]))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    raw = np.zeros((8, len(acq.adc)), complex)
    for label in range(len(anatomy.materials)):
        index = np.column_stack(np.nonzero(anatomy.labels == label + 1))
        if not len(index):
            continue
        centres = lower + (index + 0.5) * size
        points = np.repeat(centres, order, axis=0)
        points[:, 1] += np.tile(nodes, len(centres)) * size[1] / 2
        maps = gaussian_receive_maps(points, fov)
        weight = np.tile(weights / 2, len(centres)) * np.prod(size) / voxel
        value = fourier(points, maps * weight, acq.k[:, :2])
        raw += value * np.sinc(acq.k[:, 0] * size[0]) * response[label]
    return raw


def fourier_factors(matrix, offset):
    n = np.asarray(matrix)
    indices = np.indices(tuple(n)).reshape(len(n), -1).T
    positions = (indices + 0.5) / n - 0.5
    spatial = np.exp(-2j * np.pi * (positions @ offset)).reshape(tuple(n))
    half_voxel = np.exp(-1j * np.pi * np.sum((indices - n // 2) / n, axis=1)).reshape(
        tuple(n)
    )
    return spatial, half_voxel


def forward(image, offset):
    spatial, phase = fourier_factors(image.shape, offset)
    return (
        bart.fft(image * spatial, inverse=False, axes=tuple(range(image.ndim))) * phase
    )


def inverse(grid, offset):
    spatial, phase = fourier_factors(grid.shape, offset)
    return (
        bart.fft(grid * phase.conj(), inverse=True, axes=tuple(range(grid.ndim)))
        * spatial.conj()
    )


def selection_region(acq, matrix):
    region = np.ones(matrix, bool)
    if acq.measurements["family"] in REDUCED_FOV:
        n = matrix[1]
        width = round(n * acq.measurements["reduced_fov_fraction"])
        ys = np.arange(n)
        region[:] = ((ys >= n // 2 - width // 2) & (ys < n // 2 - width // 2 + width))[
            None, :
        ]
    return region


def scoring_regions(acq, case, support, masks):
    region = selection_region(acq, support.shape)
    return support & region, {name: mask & region for name, mask in masks.items()}


def autocalibrated(acq):
    return acq.measurements["family"] == "grappa_sampling"


def reconstruct(raw, acq, matrix, *, calibrate=True):
    raw = np.asarray(raw)
    dims = len(matrix)
    grid = np.zeros((raw.shape[0], *matrix), complex)
    grid[(slice(None), *acq.indices.T)] = raw
    fam = acq.measurements["family"]
    if not parallel_imaging(acq):
        image = inverse(grid[0], acq.offset)
        if fam in REDUCED_FOV:
            image *= (
                selection_region(acq, matrix) / acq.measurements["reduced_fov_fraction"]
            )
        return image
    # BART's fixed L2 SENSE policy uses prescribed maps/mask, never fitted image
    # gains, field correction or a phantom-dependent regularizer.
    if not hasattr(acq, "_bart_sense_inputs"):
        maps = pixel_maps(acq, matrix)
        mask = np.asarray(acq.measurements["sampling_mask"], bool)
        spatial, phase = fourier_factors(matrix, acq.offset)
        shape = (*matrix, *((1,) * (3 - dims)), maps.shape[0])
        sensitivity = np.moveaxis(maps * spatial, 0, -1).reshape(shape)
        pattern = mask.reshape((*matrix, *((1,) * (4 - dims))))
        acq._bart_sense_inputs = sensitivity, pattern, phase, shape
    sensitivity, pattern, phase, shape = acq._bart_sense_inputs
    # PICS uses a unitary Fourier operator; acquisition amplitudes use an
    # unnormalized DFT. These fixed coordinate/unit conversions use no data fit.
    data = np.moveaxis(grid * phase.conj() / np.sqrt(np.prod(matrix)), 0, -1).reshape(
        shape
    )
    spatial = None
    if autocalibrated(acq) and calibrate:
        # Auto-calibrating policy: ESPIRiT maps from the acquired central
        # calibration lines replace the known maps. The common lattice phase
        # stays in the image and is removed after the solve.
        spatial = fourier_factors(matrix, acq.offset)[0]
        try:
            sensitivity = bart.espirit_maps(data)
        except RuntimeError as error:
            if "Calibration region not found" not in str(error):
                raise
            # No fully sampled centre: maps cannot be calibrated, so the
            # auto-calibrating policy yields no image.
            return np.zeros(matrix, complex)
    image = bart.run(
        [
            "pics",
            "-l2",
            "-r",
            str(REGULARIZATION),
            "-i",
            str(MAX_ITERATIONS),
            "-w",
            "1",
            "-p",
            "{mask}",
            "{data}",
            "{maps}",
            "{out}",
        ],
        {"data": data, "maps": sensitivity, "mask": pattern},
    )
    image = image.reshape(matrix)
    return image if spatial is None else image * spatial.conj()


def pixel_maps(acq, matrix):
    """Known receive maps sampled at the image pixel centres."""
    dims = len(matrix)
    indices = np.indices(matrix).reshape(dims, -1).T
    fov = np.asarray(acq.measurements["fov_mm"]) / 1000
    xyz = np.zeros((len(indices), 3))
    xyz[:, :dims] = ((indices + 0.5) / matrix - 0.5) * fov
    return acquisition_maps(acq)(xyz, fov).reshape((-1, *matrix))


def known_map_raw(image, acq, matrix):
    """Noise-free multi-coil samples of ``image`` under the known receive maps."""
    return np.asarray(
        [
            forward(m * image, acq.offset)[tuple(acq.indices.T)]
            for m in pixel_maps(acq, matrix)
        ]
    )


def temporal_image_error(delta, acq, matrix, reference, coarse=None):
    """Project timestep error through the fixed reconstruction policy.

    Normalize by the independent image target, as the image quality scores do.
    This also remains meaningful when disturbed acquisition signal is weak.
    For the auto-calibrated GRAPPA policy the maps depend on the data, so the
    ESPIRiT reconstructions of the coarse and half-step acquisitions are
    compared directly. ``coarse`` is the coarse-step acquisition when the
    caller supplies it; otherwise the operating point is the target's own
    noise-free multi-coil k-space. The linear known-map projection is kept as
    a lower-bound diagnostic and the larger of the two is returned.
    """
    norm = max(np.linalg.norm(reference), 1e-30)
    known = float(np.linalg.norm(reconstruct(delta, acq, matrix, calibrate=False)))
    if not (parallel_imaging(acq) and autocalibrated(acq)):
        return known / norm
    base = (
        np.asarray(coarse)
        if coarse is not None
        else known_map_raw(np.asarray(reference), acq, tuple(matrix))
    )
    calibrated = float(
        np.linalg.norm(
            reconstruct(base + delta, acq, matrix) - reconstruct(base, acq, matrix)
        )
    )
    acq.measurements["temporal_image_error_terms"] = {
        "calibrated": calibrated / norm,
        "known_map_projection": known / norm,
        "operating_point": "coarse_acquisition" if coarse is not None else "target",
    }
    return max(calibrated, known) / norm


def clipped_reference(acq, case, root, response, fraction):
    """Independent anatomical Fourier reference clipped to the ideal inner slab."""
    from .s1_anatomy import anatomical_reference

    half = case["task"]["sequence"]["fov_mm"][1] / 2000 * fraction
    return anatomical_reference(
        acq, case, root, response, clip_axis=1, clip_bounds_m=(-half, half)
    )


def reference_context(acq, spec):
    """The single central echo defining both the S3 target and its phase oracle.

    Selected once (earliest k-space-origin sample) and passed explicitly to the
    magnitude/contrast target and to the phase builder, so a half-sample
    lattice with several near-centre samples cannot feed them different
    TE/RF histories.
    """
    return reference_index(acq, spec)


def phase_reference(acq, case, root, reference, *, temporal_scale=1):
    return shared_phase_reference(
        acq,
        case,
        root,
        reference,
        temporal_scale=temporal_scale,
        sample_indices=np.array([reference_context(acq, case["task"]["sequence"])]),
    )


def _target_grid(matrix, offset, fov, response, chosen):
    """Full Cartesian k-space grid with every sample at the reference echo."""
    indices = np.indices(tuple(matrix)).reshape(len(matrix), -1).T
    k = np.zeros((len(indices), 3))
    k[:, : len(matrix)] = (indices - matrix // 2 + offset) / fov
    constant = np.broadcast_to(response[:, chosen, None], (len(response), len(indices)))
    return SimpleNamespace(adc=np.arange(len(indices)), k=k), constant


def reference_image(acq, case, root, response, reference_raw):
    """Complete Fourier grid of the object, independent of the submitted mask."""
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    chosen = reference_context(acq, spec)
    if family(case) in REDUCED_FOV:
        fraction = acq.measurements["reduced_fov_fraction"]
        inner = matrix.copy()
        inner[1] = round(matrix[1] * fraction)
        fov[1] *= fraction
        offset = acq.offset.copy()
        offset[1] *= fraction
        grid, constant = _target_grid(inner, offset, fov, response, chosen)
        raw = clipped_reference(grid, case, root, constant, fraction)
        image = np.zeros(tuple(matrix), complex)
        lo = matrix[1] // 2 - inner[1] // 2
        image[:, lo : lo + inner[1]] = inverse(raw.reshape(tuple(inner)), offset)
        return image
    grid, constant = _target_grid(matrix, acq.offset, fov, response, chosen)
    raw = analytic_reference(grid, case, root, constant)
    return inverse(raw.reshape(tuple(matrix)), acq.offset) * selection_region(
        acq, tuple(matrix)
    )


# Paired midpoint grids per in-plane voxel axis. A uniform n-point grid
# aliases dephasing of exactly n, 2n, ... cycles per voxel; coprime pairs
# alias together only at multiples of n*m cycles, so agreement between the
# two grids certifies the voxel integral. Refined only on disagreement.
SELECTION_GRIDS = ((24, 31), (48, 61), (96, 127))
SELECTION_AGREEMENT = 0.01
SPOILING_COUNTS = (1021, 2053)


def voxel_offsets(voxel, points):
    """Midpoint grid spanning one in-plane voxel (x, y); 2D objects have no z extent."""
    u = (np.arange(points) + 0.5) / points - 0.5
    grid = np.stack(np.meshgrid(u, u, indexing="ij"), axis=-1).reshape(-1, 2)
    offsets = np.zeros((len(grid), 3))
    offsets[:, :2] = grid * np.asarray(voxel)[:2]
    return offsets


def voxel_widths(case):
    """Per-axis voxel size fov/matrix in metres; zero z extent for 2D tasks."""
    spec = case["task"]["sequence"]
    width = np.zeros(3)
    dims = len(spec["matrix"])
    width[:dims] = np.asarray(spec["fov_mm"]) / 1000 / np.asarray(spec["matrix"])
    return width


def spoiling_probes(acq, case, materials, properties):
    """Spoiled-GRE probe over the anisotropic image voxel (x, y and, in 3D, z).

    Same criterion as the S2 probe (mean central-echo signal >= 0.01 PD and
    voxel-integrated Bloch replay within 0.20 complex error of perfect
    spoiling at 1021- and 2053-point Halton quadratures), but the quadrature
    spans fov/matrix on each axis, e.g. 1.5625 x 1.5625 x 6.25 mm in 3D.
    """
    fam = family(case)
    response = canonical_response(acq, properties, fam, finite_rf=True)
    props = np.array([[1.0, 0.8, 0.08]])
    ideal = canonical_response(acq, props, fam, finite_rf=True)[0]
    centres = np.asarray(acq.measurements["centre_sample_indices"])
    signal = float(np.mean(np.abs(ideal[centres])))
    width = voxel_widths(case)
    errors = []
    for count in SPOILING_COUNTS:
        points = (
            np.array(
                [
                    [radical_inverse(i, b) - 0.5 for b in (2, 3, 5)]
                    for i in range(1, count + 1)
                ]
            )
            * width
        )
        spins = Spins(
            points,
            np.zeros(count, dtype=int),
            ["diagnostic"],
            props,
            1 / count,
            (count,),
            "diagnostic",
        )
        a = copy(acq)
        a.adc, a.adc_phase, a.k = (
            acq.adc[centres],
            expected_receiver_phase(acq)[centres],
            acq.k[centres],
        )
        errors.append(
            nrmse(simulate(a, spins, demodulate_encoding=True), ideal[centres])
        )
    checks = [
        check("usable_echo_signal", signal >= 0.01, {"signal": signal, "min": 0.01}),
        check(
            "spoiled_gre",
            max(errors) <= 0.2,
            {"relative_errors": errors, "max": 0.2, "voxel_m": width.tolist()},
        ),
    ]
    metrics = {
        "diagnostic_signal_fraction": signal,
        "spoiling_relative_error": max(errors),
    }
    return checks, metrics, response


def selection_signal(acq, case, xyz, ids, points, effects):
    """Voxel-averaged probe magnitudes at the chosen samples, one grid size."""
    offsets = voxel_offsets(voxel_widths(case), points)
    ensemble = (xyz[:, None, :] + offsets[None]).reshape(-1, 3)
    spins = Spins(
        ensemble,
        np.zeros(len(ensemble), int),
        ["diagnostic"],
        np.array([[1.0, 0.8, 0.08]]),
        1.0,
        (len(ensemble),),
        "s3-selection",
    )
    a = copy(acq)
    a.adc, a.adc_phase, a.k = acq.adc[ids], acq.adc_phase[ids], acq.k[ids]
    raw = simulate(a, spins, effects, individual=True).reshape(
        len(ids), len(xyz), len(offsets)
    )
    demodulation = np.exp(2j * np.pi * (a.k[:, :3] @ offsets.T))
    return np.abs(np.mean(raw * demodulation[:, None, :], axis=2))


def selection_scores(signal, inside_count):
    inside_values = signal[:, :inside_count]
    exterior = signal[:, inside_count:]
    return {
        "inside": float(np.mean(inside_values)),
        "outside": float(np.max(exterior)),
        "leakage": float(
            np.max(
                np.max(exterior, axis=1)
                / np.maximum(np.mean(inside_values, axis=1), 1e-15)
            )
        ),
        "inside_min": float(np.min(inside_values)),
    }


def mechanism_probes(acq, case, materials, properties):
    if family(case) not in REDUCED_FOV:
        # Mechanism probes isolate spoiling from receiver corruption. Only this
        # diagnostic uses the RF-derived phase; image replay keeps submitted ADC
        # phases, so receiver errors remain in the independently scored image.
        return spoiling_probes(acq, case, materials, properties)
    # Selective RF must suppress spins outside the requested inner volume. The
    # encoded replay retains the submitted gradients and receiver phases.
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    fraction = acq.measurements["reduced_fov_fraction"]
    width = fraction * fov[1]
    inside_y = np.linspace(-0.35, 0.35, 9) * width
    outside_y = np.r_[
        np.linspace(-0.48 * fov[1], -0.65 * width, 9),
        np.linspace(0.65 * width, 0.48 * fov[1], 9),
    ]
    xyz = np.zeros((len(inside_y) + len(outside_y), 3))
    xyz[:, 1] = np.r_[inside_y, outside_y]
    inside_count = len(inside_y)
    if family(case) == "rfov_epi":
        # Both axes must select. Include x-only stopband probes so a 1D y-slab
        # excitation cannot masquerade as a 2D RF pulse on compact anatomy.
        inside_points = np.array(
            [(x, y, 0.0) for x in (-0.056, 0.0, 0.056) for y in inside_y]
        )
        outside_points = np.array(
            [(x, y, 0.0) for x in (-0.056, 0.0, 0.056) for y in outside_y]
            + [
                (x, y, 0.0)
                for x in (-0.096, -0.092, 0.092, 0.096)
                for y in (-0.025, 0.0, 0.025)
            ]
        )
        xyz = np.vstack((inside_points, outside_points))
        inside_count = len(inside_points)
    # Each probe is one image voxel: in-plane isochromat grids whose complex
    # signal is averaged after removing the nominal encoding phase. Crusher-
    # dephased (unrefocused) pathways cancel as they do in the image, while
    # coherent exterior magnetization is still measured.
    centres = np.asarray(acq.measurements["centre_sample_indices"])
    lines = np.unique([0, len(centres) // 2, len(centres) - 1])
    ids = centres[lines]
    effects = case["physics"]["effects"] if family(case) == "rfov_epi" else ()
    decay = np.ones(len(ids))
    reference_times = None
    if family(case) == "rfov_epi":
        # Isolate finite-RF selection from the longer 128-grid readout: refer
        # each echo to the end of the excitation that produced its line. With
        # uniform diagnostic T2 and no further RF, this decay factor is exact.
        excitations = [p for p in acq.pulses if not p.refocusing]
        reference_times = np.array(
            [excitations[max(int(acq.excitation[j]), 0)].end for j in lines]
        )
        decay = np.exp((acq.adc[ids] - reference_times) / 0.08)
    history, chosen, converged = [], None, False
    for pair in SELECTION_GRIDS:
        scores = [
            selection_scores(
                selection_signal(acq, case, xyz, ids, n, effects) * decay[:, None],
                inside_count,
            )
            for n in pair
        ]
        history.append({"grid_points": list(pair), "scores": scores})
        inside = max(s["inside"] for s in scores)
        converged = abs(scores[0]["leakage"] - scores[1]["leakage"]) <= (
            SELECTION_AGREEMENT
        ) and abs(scores[0]["inside"] - scores[1]["inside"]) <= (
            SELECTION_AGREEMENT * max(inside, 1e-15)
        )
        # Conservative combination of the two quadratures.
        chosen = {
            "inside": min(s["inside"] for s in scores),
            "outside": max(s["outside"] for s in scores),
            "leakage": max(s["leakage"] for s in scores),
            "inside_min": min(s["inside_min"] for s in scores),
        }
        if converged:
            break
    response = canonical_response(acq, properties, family(case), finite_rf=True)
    checks = [
        check(
            "spatial_selection",
            converged and chosen["inside"] >= 0.01 and chosen["leakage"] <= 0.1,
            {
                "inside_signal": chosen["inside"],
                "outside_signal": chosen["outside"],
                "outside_to_inside_max": 0.1,
                "ratio": chosen["leakage"],
                "quadrature_converged": converged,
                "quadrature": history,
            },
        )
    ]
    if family(case) == "rfov_epi":
        checks.append(
            check(
                "rfov_inside_signal",
                chosen["inside_min"] >= 0.2,
                {
                    "min_signal": chosen["inside_min"],
                    "minimum": 0.2,
                    "reference_times_s": reference_times.tolist(),
                },
            )
        )
    return (
        checks,
        {
            "selection_leakage_ratio": chosen["leakage"],
            "diagnostic_signal_fraction": chosen["inside"],
        },
        response,
    )


def evaluate_s3(submission, case, root, workspace, *, deadline=None):
    result = evaluate_s1(
        submission,
        case,
        root,
        workspace,
        deadline=deadline,
        inspector=inspect_sequence,
        probes=mechanism_probes,
        reconstructor=reconstruct,
        sampler=reference_channels,
        analytic_sampler=analytic_channels,
        reference_builder=reference_image,
        phase_builder=phase_reference,
        region_builder=scoring_regions,
        simulator=simulate_channels,
        replay_shape=signal_shape,
        temporal_norm=temporal_image_error,
        encoded_integrator="bloch",
    )
    fam = family(case)
    result["limitations"] = (
        result.get("limitations", [])
        + (
            ["s3_image_thresholds_are_provisional_until_calibration"]
            if not case["evaluation"]["calibrated"]
            else []
        )
        + [
            "BART_fixed_L2_SENSE_for_parallel_and_CS_masks; BART_zero_fill_FFT_for_partial_and_elliptical_support",
            *(
                ["grappa_sampling_maps_ESPIRiT_autocalibrated_from_submitted_data"]
                if fam == "grappa_sampling"
                else []
            ),
            "synthetic_receive_maps_are_comoving; no_scanner_coil_or_noise_claim",
        ]
    )
    provenance_path = Path(workspace) / "provenance.json"
    receive = receive_model(fam)
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance["receive_encoding"] = {
            "model": receive,
            "dynamics": "KomaMRICore native GPU Bloch: static fused receive or dynamic BlochDict"
            if fam in PARALLEL
            else "KomaMRICore.Bloch",
            "accumulation": "native fused exact static receive; bounded native per-spin ADC for dynamic receive"
            if fam in PARALLEL
            else "native summed Bloch replay",
        }
        write_json(provenance_path, provenance)
    write_json(
        Path(workspace) / "policy.json",
        {
            "evaluation_contract": result["evaluation_contract"],
            "calibrated": case["evaluation"]["calibrated"],
            "reconstruction": "bart_sampling_sense@1",
            "regularization": REGULARIZATION,
            "receive_model": receive,
            "solver": "BART pics fixed L2 SENSE",
            "sensitivity_maps": (
                f"BART ecalib ESPIRiT from acquired calibration (m=1, kernel "
                f"{bart.ESPIRIT_KERNEL}, region <= {bart.ESPIRIT_CALIBRATION})"
                if fam == "grappa_sampling"
                else "known synthetic receive maps"
            ),
            "solver_max_iterations": MAX_ITERATIONS,
            "temporal_raw_nrmse_max": 0.01,
            "temporal_image_nrmse_max": 0.01,
            "temporal_image_normalization": "independent ideal reference norm",
            **(
                {"receive_coils": 8, "receive_sigma_y_fov": 1 / 16}
                if fam == "variable_density"
                else {"receive_amplitude": RECEIVE_AMPLITUDE}
            ),
            "limitations": result["limitations"],
        },
    )
    result["artifacts"]["policy"] = str((Path(workspace) / "policy.json").resolve())
    return result
