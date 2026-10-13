"""S3 sampling evaluation with fixed linear receive encoding and full-grid targets.

The image policy is a benchmark CG-SENSE baseline, including GRAPPA/CS mask
families. It never fits a reconstruction to a phantom or an image target.
"""

import json
from copy import copy
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import roots_legendre

from ..io import write_json
from . import bart
from .s1 import evaluate_s1
from .s1_image import canonical_response, expected_receiver_phase
from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_sampling import analytic_reference, source
from .s1_sequence import check, family
from .s2 import mechanism_probes as gre_probes
from .s2 import phase_reference, reference_index
from .s2 import reference_signal as spatial_reference
from .s3_sequence import PARALLEL, inspect_sequence

RECEIVE_AMPLITUDE = 0.8
RECEIVE_NORMALIZATION = np.sqrt(4 * (1 + RECEIVE_AMPLITUDE**2 / 2))
# Fixed Tikhonov policy keeps BART's complex64 solve stable for irregular masks.
REGULARIZATION = 1e-3
# BART CG uses a zero stopping tolerance; excessive iterations lose conjugacy
# after reaching the complex64 floor on irregular masks. Fixed 50 is validated
# against dense normal equations, including phase-rotated inputs.
MAX_ITERATIONS = 50


def receive_terms(dims, parallel=True):
    """Finite Fourier representation of positive, smooth synthetic receive maps.

    Four translated cosine lobes per phase axis; sum of squared maps is one.
    These are fixed in material coordinates (co-moving under motion), explicitly
    a synthetic benchmark model rather than a scanner coil or noise model.
    """
    if not parallel:
        return [[(np.zeros(3), 1.0)]]
    channels = []
    for angles in product(np.arange(4) * np.pi / 2, repeat=dims - 1):
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


def receive_maps(positions, fov, parallel=True):
    dims = len(fov)
    xyz = np.asarray(positions)
    maps = []
    for terms in receive_terms(dims, parallel):
        value = sum(
            c * np.exp(2j * np.pi * (xyz[:, :dims] @ (shift[:dims] / fov)))
            for shift, c in terms
        )
        maps.append(value.real)
    return np.asarray(maps)


def signal_shape(acq):
    return (
        len(receive_terms(len(acq.offset), acq.measurements["family"] in PARALLEL)),
        len(acq.adc),
    )


def simulate_channels(acq, spins, effects=(), **steps):
    fov = np.asarray(acq.measurements["fov_mm"]) / 1000
    maps = receive_maps(spins.positions, fov, acq.measurements["family"] in PARALLEL)
    if len(maps) == 1:
        return simulate(acq, spins, effects, **steps)[None, :]
    # Native per-spin acquisition blocks share one Bloch state across receive
    # channels. Receive maps are applied only at acquisition, with no dynamics
    # synthesized from ideal or reconstructed images.
    return simulate(acq, spins, effects, receive_weights=maps, **steps)


def reference_channels(acq, spins, spec, response):
    maps = receive_maps(
        spins.positions,
        np.asarray(spec["fov_mm"]) / 1000,
        acq.measurements["family"] in PARALLEL,
    )
    weights = np.broadcast_to(spins.weight, (len(spins.positions),))
    return np.asarray(
        [
            spatial_reference(
                acq, replace(spins, weight=weights * sensitivity), spec, response
            )
            for sensitivity in maps
        ]
    )


def analytic_channels(acq, case, root, response):
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    output = []
    harmonics = {}
    for terms in receive_terms(len(fov), family(case) in PARALLEL):
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
    if acq.measurements["family"] == "reduced_fov":
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


def reconstruct(raw, acq, matrix):
    raw = np.asarray(raw)
    dims = len(matrix)
    grid = np.zeros((raw.shape[0], *matrix), complex)
    grid[(slice(None), *acq.indices.T)] = raw
    fam = acq.measurements["family"]
    if fam not in PARALLEL:
        image = inverse(grid[0], acq.offset)
        if fam == "reduced_fov":
            image *= (
                selection_region(acq, matrix) / acq.measurements["reduced_fov_fraction"]
            )
        return image
    # BART's fixed L2 SENSE policy uses prescribed maps/mask, never fitted image
    # gains, field correction or a phantom-dependent regularizer.
    if not hasattr(acq, "_bart_sense_inputs"):
        indices = np.indices(matrix).reshape(dims, -1).T
        fov = np.asarray(acq.measurements["fov_mm"]) / 1000
        xyz = np.zeros((len(indices), 3))
        xyz[:, :dims] = ((indices + 0.5) / matrix - 0.5) * fov
        maps = receive_maps(xyz, fov).reshape((-1, *matrix))
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
    return image.reshape(matrix)


def temporal_image_error(delta, acq, matrix, reference):
    """Project timestep error through the fixed linear reconstruction policy.

    Normalize by the independent image target, as the image quality scores do.
    This also remains meaningful when disturbed acquisition signal is weak.
    """
    return float(
        np.linalg.norm(reconstruct(delta, acq, matrix))
        / max(np.linalg.norm(reference), 1e-30)
    )


def clipped_reference(acq, case, root, response, fraction, *, order=512):
    """Independent Fourier integral of primitives clipped to the ideal inner slab.

    For ellipses the readout-axis integral is analytic; only the phase-axis
    cross-section uses deterministic Gauss-Legendre quadrature. It does not use
    submitted RF, measured signals, reconstructed images or candidate geometry.
    """
    from .s1_anatomy import anatomical_reference, is_anatomical

    if is_anatomical(case):
        half = case["task"]["sequence"]["fov_mm"][1] / 2000 * fraction
        return anatomical_reference(
            acq, case, root, response, clip_axis=1, clip_bounds_m=(-half, half)
        )
    shapes, primitives, _ = source(case, root)
    materials = list(shapes["materials"])
    spec = case["task"]["sequence"]
    voxel = np.prod(np.asarray(spec["fov_mm"]) / 1000 / spec["matrix"])
    half = spec["fov_mm"][1] / 2000 * fraction
    unit, weights = roots_legendre(order)
    raw = np.zeros(len(acq.adc), complex)
    kx, ky = acq.k[:, 0], acq.k[:, 1]
    for index, shape in enumerate(primitives):
        cx, cy = np.asarray(shape["center_mm"]) / 1000
        rx, ry = np.asarray(shape["radii_mm"]) / 1000
        lo, hi = max(cy - ry, -half), min(cy + ry, half)
        if lo >= hi:
            continue
        parent = None
        for earlier in primitives[:index]:
            q = (np.asarray(shape["center_mm"]) - earlier["center_mm"]) / earlier[
                "radii_mm"
            ]
            if (np.abs(q) <= 1).all() if earlier["kind"] == "box" else q @ q <= 1:
                parent = earlier["material"]
        coefficient = response[materials.index(shape["material"])].copy()
        if parent is not None:
            coefficient -= response[materials.index(parent)]
        if shape["kind"] == "box":
            fourier = (
                2
                * rx
                * np.sinc(2 * kx * rx)
                * (hi - lo)
                * np.sinc(ky * (hi - lo))
                * np.exp(-2j * np.pi * (kx * cx + ky * (hi + lo) / 2))
            )
        else:
            ys = (lo + hi) / 2 + (hi - lo) / 2 * unit
            width = rx * np.sqrt(np.maximum(0, 1 - ((ys - cy) / ry) ** 2))
            cross_section = (
                2 * width[None, :] * np.sinc(2 * kx[:, None] * width[None, :])
            )
            fourier = np.exp(-2j * np.pi * kx * cx) * (
                (cross_section * np.exp(-2j * np.pi * ky[:, None] * ys[None, :]))
                @ ((hi - lo) / 2 * weights)
            )
        raw += fourier / voxel * coefficient
    return raw


def reference_image(acq, case, root, response, reference_raw):
    """Complete Fourier grid of the object, independent of the submitted mask."""
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    if family(case) == "reduced_fov":
        fraction = acq.measurements["reduced_fov_fraction"]
        inner = matrix.copy()
        inner[1] = round(matrix[1] * fraction)
        fov = np.asarray(spec["fov_mm"]) / 1000
        fov[1] *= fraction
        offset = acq.offset.copy()
        offset[1] *= fraction
        indices = np.indices(tuple(inner)).reshape(2, -1).T
        k = np.zeros((len(indices), 3))
        k[:, :2] = (indices - inner // 2 + offset) / fov
        grid = SimpleNamespace(adc=np.arange(len(indices)), k=k)
        chosen = reference_index(acq, spec)
        constant = np.broadcast_to(
            response[:, chosen, None], (len(response), len(indices))
        )
        raw = clipped_reference(grid, case, root, constant, fraction)
        image = np.zeros(tuple(matrix), complex)
        lo = matrix[1] // 2 - inner[1] // 2
        image[:, lo : lo + inner[1]] = inverse(raw.reshape(tuple(inner)), offset)
        return image
    indices = np.indices(tuple(matrix)).reshape(len(matrix), -1).T
    k = np.zeros((len(indices), 3))
    k[:, : len(matrix)] = (indices - matrix // 2 + acq.offset) / (
        np.asarray(spec["fov_mm"]) / 1000
    )
    grid = SimpleNamespace(adc=np.arange(len(indices)), k=k)
    chosen = reference_index(acq, spec)
    constant = np.broadcast_to(response[:, chosen, None], (len(response), len(indices)))
    raw = analytic_reference(grid, case, root, constant)
    return inverse(raw.reshape(tuple(matrix)), acq.offset) * selection_region(
        acq, tuple(matrix)
    )


def mechanism_probes(acq, case, materials, properties):
    if family(case) != "reduced_fov":
        # Mechanism probes isolate spoiling from receiver corruption. Only this
        # diagnostic uses the RF-derived phase; image replay keeps submitted ADC
        # phases, so receiver errors remain in the independently scored image.
        diagnostic = copy(acq)
        diagnostic.adc_phase = expected_receiver_phase(acq)
        return gre_probes(diagnostic, case, materials, properties)
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
    props = np.array([[1.0, 0.8, 0.08]])
    spins = Spins(
        xyz,
        np.zeros(len(xyz), int),
        ["diagnostic"],
        props,
        1.0,
        (len(xyz),),
        "s3-selection",
    )
    a = copy(acq)
    centres = np.asarray(acq.measurements["centre_sample_indices"])
    ids = centres[np.unique([0, len(centres) // 2, len(centres) - 1])]
    a.adc, a.adc_phase, a.k = acq.adc[ids], acq.adc_phase[ids], acq.k[ids]
    signal = np.abs(simulate(a, spins, individual=True))
    inside = float(np.mean(signal[:, : len(inside_y)]))
    outside = float(np.max(signal[:, len(inside_y) :]))
    leakage = outside / max(inside, 1e-15)
    # Refocusing adds pi to the independent excitation -i convention.
    response = canonical_response(
        acq,
        properties,
        family(case),
        finite_rf=case["evaluation"].get("intrinsic_tissue_response", False),
    )
    parity = np.zeros(len(acq.adc), int)
    for pulse in acq.pulses:
        after = acq.adc >= pulse.center
        if pulse.refocusing:
            parity[after] += 1
        else:
            parity[after] = 0
    response *= (-1.0) ** parity
    checks = [
        check(
            "spatial_selection",
            inside >= 0.01 and leakage <= 0.1,
            {
                "inside_signal": inside,
                "outside_signal": outside,
                "outside_to_inside_max": 0.1,
                "ratio": leakage,
            },
        )
    ]
    return (
        checks,
        {"selection_leakage_ratio": leakage, "diagnostic_signal_fraction": inside},
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
    result["limitations"] = (
        result.get("limitations", [])
        + (
            ["s3_image_thresholds_are_provisional_until_calibration"]
            if not case["evaluation"]["calibrated"]
            else []
        )
        + [
            "BART_fixed_L2_SENSE_for_parallel_and_CS_masks; BART_zero_fill_FFT_for_partial_and_elliptical_support",
            "synthetic_receive_maps_are_comoving; no_scanner_coil_or_noise_claim",
        ]
    )
    provenance_path = Path(workspace) / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance["receive_encoding"] = {
            "model": "s3_trigonometric_comoving@1"
            if family(case) in PARALLEL
            else "uniform@1",
            "dynamics": "KomaMRICore.BlochDict"
            if family(case) in PARALLEL
            else "KomaMRICore.Bloch",
            "accumulation": "bounded native per-spin ADC blocks multiplied by fixed receive maps"
            if family(case) in PARALLEL
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
            "receive_model": "s3_trigonometric_comoving@1"
            if family(case) in PARALLEL
            else "uniform@1",
            "solver": "BART pics fixed L2 SENSE",
            "solver_max_iterations": MAX_ITERATIONS,
            "temporal_raw_nrmse_max": 0.01,
            "temporal_image_nrmse_max": 0.01,
            "temporal_image_normalization": "independent ideal reference norm",
            "receive_amplitude": RECEIVE_AMPLITUDE,
            "limitations": result["limitations"],
        },
    )
    result["artifacts"]["policy"] = str((Path(workspace) / "policy.json").resolve())
    return result
