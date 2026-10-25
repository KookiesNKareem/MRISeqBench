"""S2 readout judge on the shared registered-phantom KomaMRI replay pipeline."""

import os
from copy import copy
from types import SimpleNamespace

import finufft
import numpy as np

from .s1 import evaluate_s1
from .s1_case import phase_reference as shared_phase_reference
from .s1_image import canonical_response, expected_receiver_phase, nrmse
from .s1_koma import reconstruct as cartesian_reconstruct
from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_probes import radical_inverse, reference_index
from .s1_sampling import analytic_reference
from .s1_sequence import check, family
from .s2_sequence import CARTESIAN, FAMILIES, REFOCUSED, inspect_sequence

NONCARTESIAN = FAMILIES - CARTESIAN


def reference_signal(acq, spins, spec, response):
    """Sample the independent spatial control variate at arbitrary ADC nodes."""
    dims = len(spec["matrix"])
    weights = np.broadcast_to(spins.weight, spins.labels.shape)
    raw = np.zeros(len(acq.adc), complex)
    # Type 3 preserves physical coordinates and arbitrary trajectory samples.
    transform = finufft.nufft2d3 if dims == 2 else finufft.nufft3d3
    for label in np.unique(spins.labels):
        keep = spins.labels == label
        values = transform(
            *[np.ascontiguousarray(spins.positions[keep, j]) for j in range(dims)],
            weights[keep].astype(complex),
            *[np.ascontiguousarray(2 * np.pi * acq.k[:, j]) for j in range(dims)],
            eps=1e-10,
            isign=-1,
            nthreads=max(
                1, min(4, int(os.environ.get("MRISEQBENCH_REFERENCE_THREADS", "1")))
            ),
        )
        raw += values * response[label]
    return raw


def reconstruct(raw, acq, matrix):
    if acq.measurements["family"] in CARTESIAN:
        return cartesian_reconstruct(raw, acq, matrix)
    from .bart import reconstruct_noncartesian

    return reconstruct_noncartesian(raw, acq, matrix, acq.measurements["fov_mm"])


def reference_image(acq, case, root, response, reference_raw):
    # The image target uses a full independent Cartesian grid. Reconstructing the
    # submitted trajectory as its own target would excuse a missing sector/PSF.
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    indices = np.indices(tuple(matrix)).reshape(len(matrix), -1).T
    k = np.zeros((len(indices), 3))
    k[:, : len(matrix)] = (indices - matrix // 2) / (np.asarray(spec["fov_mm"]) / 1000)
    grid = SimpleNamespace(
        adc=np.arange(len(indices)), k=k, indices=indices, offset=np.zeros(len(matrix))
    )
    target = reference_response(acq, spec, response)
    constant_response = np.broadcast_to(target[:, None], (len(response), len(indices)))
    raw = analytic_reference(grid, case, root, constant_response)
    return cartesian_reconstruct(raw, grid, tuple(map(int, matrix)))


CENTRE_RADIUS = 0.75


def _temporal_centre(acq, candidates):
    """Tied candidate nearest the readout's temporal centroid (timestamps only)."""
    times = acq.adc[candidates]
    distance = abs(times - times.mean())
    return int(candidates[np.flatnonzero(distance <= distance.min() + 1e-12)[0]])


def contrast_indices(acq, spec):
    """Echo contexts that set both the image target and the B0 phase oracle.

    One sample per contributing readout; ties within a readout use the sample
    nearest their temporal centroid. Receiver data never enter the selection.

    Cartesian (S1 and S2 EPI/TSE/GRASE): the single earliest sample within
    1e-3 samples of the global minimum k radius (the documented central echo;
    for a centred grid the ky=0 echo), so a half-sample lattice cannot mix
    echoes acquired at different times.
    Non-Cartesian S2 families: every readout whose nearest in-plane (kx, ky)
    sample lies within 0.75 samples of the origin, at that sample (within 1e-3
    of the readout's own minimum): every radial, UTE, spiral and
    stack-of-stars readout in every partition, and the central line of each
    PROPELLER blade. Their image is the mix of these readouts, so the target
    uses their mean response rather than the first readout's transient.
    """
    fam = acq.measurements.get("family")
    if fam in NONCARTESIAN:
        q = acq.k[:, :2] * (np.asarray(spec["fov_mm"])[:2] / 1000)
        radius = np.linalg.norm(q, axis=1)
        chosen = []
        for line in acq.lines:
            line = np.asarray(line)
            nearest = radius[line].min()
            if nearest <= CENTRE_RADIUS:
                chosen.append(
                    _temporal_centre(acq, line[radius[line] <= nearest + 1e-3])
                )
        if chosen:
            return np.asarray(chosen, dtype=int)
    # Cartesian: the single earliest sample nearest the k-space origin.
    return np.array([reference_index(acq, spec)])


# The phase oracle and the target share one context selection.
reference_indices = contrast_indices


def reference_response(acq, spec, response):
    """Per-material complex target response: mean over contrast_indices."""
    return np.mean(response[:, contrast_indices(acq, spec)], axis=1)


def phase_reference(acq, case, root, reference, *, temporal_scale=1):
    return shared_phase_reference(
        acq,
        case,
        root,
        reference,
        temporal_scale=temporal_scale,
        sample_indices=reference_indices(acq, case["task"]["sequence"]),
    )


def mechanism_probes(acq, case, materials, properties):
    fam = family(case)
    response = canonical_response(acq, properties, fam, finite_rf=True)
    props = np.array([[1.0, 0.8, 0.08]])
    ideal = canonical_response(acq, props, fam, finite_rf=True)[0]
    centres = np.asarray(acq.measurements["centre_sample_indices"])
    signal = float(np.mean(np.abs(ideal[centres])))
    metrics = {"diagnostic_signal_fraction": signal}
    checks = [
        check("usable_echo_signal", signal >= 0.01, {"signal": signal, "min": 0.01})
    ]
    if fam not in REFOCUSED:
        # Per-axis voxel widths (FOV/matrix); a 2D plane has zero z extent.
        spec = case["task"]["sequence"]
        dims = len(spec["matrix"])
        width = np.zeros(3)
        width[:dims] = np.asarray(spec["fov_mm"]) / 1000 / np.asarray(spec["matrix"])
        errors = []
        for count in (1021, 2053):
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
        metrics["spoiling_relative_error"] = max(errors)
        checks.append(
            check(
                "spoiled_gre",
                max(errors) <= 0.2,
                {"relative_errors": errors, "max": 0.2, "voxel_m": width.tolist()},
            )
        )
    else:
        # Refocusing must demonstrate B0 rephasing on every distinct spin echo,
        # including later train echoes; GRASE GEs straddle each spin echo.
        refs = [p for p in acq.pulses if p.refocusing]
        targets = []
        for g in np.unique(acq.excitation):
            exc = [p for p in acq.pulses if not p.refocusing][g]
            last_adc = acq.adc[acq.lines[np.flatnonzero(acq.excitation == g)[-1]][-1]]
            delta = -exc.center
            echoes = []
            for p in refs:
                if exc.center < p.center < last_adc:
                    delta = -2 * p.center - delta
                    echoes.append(-delta)
            if echoes:
                targets.append((exc, np.asarray(echoes)))
        retentions = []
        for exc, echoes in targets:
            a = copy(acq)
            # Later RF cannot affect an earlier echo: sample every causal prefix
            # in one native timeline, preserving each shot's serialized RF phase.
            a.pulses = [p for p in acq.pulses if exc.start <= p.start < echoes[-1]]
            a.adc, a.adc_phase, a.k = (
                echoes,
                np.zeros(len(echoes)),
                np.zeros((len(echoes), 3)),
            )
            spins = Spins(
                np.zeros((65, 3)),
                np.zeros(65, dtype=int),
                ["diagnostic"],
                props,
                1 / 65,
                (65,),
                "diagnostic",
            )
            baseline = simulate(a, spins, zero_gradients=True)
            actual = simulate(
                a,
                spins,
                zero_gradients=True,
                frequency_offsets=np.linspace(-50, 50, 65),
            )
            retentions.extend(
                (np.abs(actual) / np.maximum(np.abs(baseline), 1e-15)).tolist()
            )
        metrics["min_refocusing_retention"] = min(retentions, default=0.0)
        checks.append(
            check(
                "refocusing_response",
                bool(retentions) and min(retentions) >= 0.9,
                {"retentions": retentions, "min": 0.9},
            )
        )
        if fam == "tse_t2w":
            weighting_samples = np.array(
                [reference_index(acq, case["task"]["sequence"])]
            )
            sensitivity = []
            for axis in (1, 2):
                varied = props.copy()
                varied[:, axis] *= 1.1
                altered = canonical_response(acq, varied, fam, finite_rf=True)[0]
                sensitivity.append(
                    float(
                        np.mean(
                            np.abs(
                                np.abs(altered[weighting_samples])
                                - np.abs(ideal[weighting_samples])
                            )
                        )
                    )
                )
            checks.append(
                check(
                    "t2_weighting",
                    sensitivity[1] > sensitivity[0],
                    {"t1": sensitivity[0], "t2": sensitivity[1]},
                )
            )
    return checks, metrics, response


def evaluate_s2(submission, case, root, workspace, *, deadline=None):
    # The default temporal gate projects timestep error through `reconstruct`.
    encoding = {"reconstructor": reconstruct, "sampler": reference_signal}
    if (
        family(case) == "epi_single_shot"
        and ((case["task"]["sequence"].get("sampling") or {}).get("acceleration") or 1)
        > 1
    ):
        from . import s3

        encoding.update(
            reconstructor=s3.reconstruct,
            sampler=s3.reference_channels,
            analytic_sampler=s3.analytic_channels,
            simulator=s3.simulate_channels,
            replay_shape=s3.signal_shape,
            temporal_norm=s3.temporal_image_error,
        )
    return evaluate_s1(
        submission,
        case,
        root,
        workspace,
        deadline=deadline,
        inspector=inspect_sequence,
        probes=mechanism_probes,
        **encoding,
        reference_builder=reference_image,
        phase_builder=phase_reference,
        # S2 thresholds and encoded replay retain the native Bloch integrator.
        encoded_integrator="bloch",
    )
