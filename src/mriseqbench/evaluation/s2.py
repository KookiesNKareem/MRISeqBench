"""S2 readout judge on the shared registered-phantom KomaMRI replay pipeline."""

import os
from copy import copy
from types import SimpleNamespace

import finufft
import numpy as np

from .s1 import evaluate_s1
from .s1_case import phase_reference as shared_phase_reference
from .s1_case import uses_case_image
from .s1_image import canonical_response, expected_receiver_phase, nrmse
from .s1_koma import reconstruct as cartesian_reconstruct
from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_probes import radical_inverse
from .s1_sampling import analytic_reference
from .s1_sequence import MissingExcitation, check, family
from .s2_sequence import CARTESIAN, REFOCUSED, inspect_sequence


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
    chosen = reference_index(acq, spec)
    constant_response = np.broadcast_to(
        response[:, chosen, None], (len(response), len(indices))
    )
    raw = analytic_reference(grid, case, root, constant_response)
    return cartesian_reconstruct(raw, grid, tuple(map(int, matrix)))


def reference_index(acq, spec):
    """Earliest centre sample, with a fixed reciprocal-FOV tie tolerance.

    Ring samples and repeated radial centres must not select an arbitrary RF
    transient or readout time because of floating-point trajectory noise.
    """
    q = acq.k[:, : len(spec["matrix"])] * (np.asarray(spec["fov_mm"]) / 1000)
    radius = np.linalg.norm(q, axis=1)
    return int(np.flatnonzero(radius <= radius.min() + 1e-3)[0])


def reference_indices(acq, spec):
    """Representative low-k echo in each acquired centre-bearing readout.

    Repeated centres use the sample nearest their temporal centroid; the choice
    uses only trajectory/timestamps. Each readout contributes one RF-only echo
    context, so later B0 drift is represented and a long ring/rosette readout is
    not assigned its first ADC's phase. Receiver data never enter the selection.
    """
    q = acq.k[:, : len(spec["matrix"])] * (np.asarray(spec["fov_mm"]) / 1000)
    radius = np.linalg.norm(q, axis=1)
    centre = radius <= radius.min() + 1e-3
    chosen = []
    for line in acq.lines:
        candidates = np.asarray(line)[centre[line]]
        if len(candidates):
            times = acq.adc[candidates]
            distance = abs(times - times.mean())
            tied = np.flatnonzero(distance <= distance.min() + 1e-12)
            chosen.append(int(candidates[tied[0]]))
    return np.asarray(chosen, dtype=int)


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
    ideal = canonical_response(acq, props, fam)[0]
    centres = np.asarray(acq.measurements["centre_sample_indices"])
    signal = float(np.mean(np.abs(ideal[centres])))
    metrics = {"diagnostic_signal_fraction": signal}
    checks = [
        check("usable_echo_signal", signal >= 0.01, {"signal": signal, "min": 0.01})
    ]
    if fam not in REFOCUSED:
        width = (
            np.ones(3)
            * np.asarray(case["task"]["sequence"]["fov_mm"])[0]
            / case["task"]["sequence"]["matrix"][0]
            / 1000
        )
        if len(case["task"]["sequence"]["matrix"]) == 2:
            width[2] = 0.0
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
                {"relative_errors": errors, "max": 0.2},
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
            for p in refs:
                if exc.center < p.center < last_adc:
                    delta = -2 * p.center - delta
                    targets.append((exc, -delta))
        retentions = []
        for exc, echo in targets:
            a = copy(acq)
            a.pulses = [p for p in acq.pulses if exc.start <= p.start < echo]
            a.adc, a.adc_phase, a.k = np.array([echo]), np.zeros(1), np.zeros((1, 3))
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
            retentions.append(float(abs(actual[0]) / max(abs(baseline[0]), 1e-15)))
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
                altered = canonical_response(acq, varied, fam)[0]
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
    try:
        result = evaluate_s1(
            submission,
            case,
            root,
            workspace,
            deadline=deadline,
            inspector=inspect_sequence,
            probes=mechanism_probes,
            reconstructor=reconstruct,
            sampler=reference_signal,
            reference_builder=reference_image,
            phase_builder=phase_reference,
            # S2 thresholds and encoded replay retain the native Bloch integrator.
            encoded_integrator="bloch",
        )
    except MissingExcitation as exc:
        return {
            "status": "failed",
            "checks": [check("excitation_before_adc", False, str(exc))],
            "metrics": {},
            "sequence_valid": False,
            "ideal_imaging_success": None,
            "imaging_success_under_case_physics": None,
            "image_verdict": "not_evaluated",
            "image_acceptance_basis": "case_physics"
            if uses_case_image(case)
            else "ideal_control",
            "evaluation_contract": f"s2@{case['evaluation']['version']}",
        }
    return result
