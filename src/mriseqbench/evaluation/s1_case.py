"""Case-aware RF phase reference and independent image-quality requirements.

The phase oracle contains RF and declared fields, never submitted ADC phases or
encoding gradients. GRE/SE isolate the selected echo (ideal spoiling); bSSFP
retains its complete RF history. It is a finite-RF KomaMRI response, not a fitted
phase map. Geometry, magnitude, contrast and dropout retain nominal-object targets.
"""

from copy import copy
from itertools import combinations

import numpy as np

from .s1_image import expected_receiver_phase
from .s1_koma import simulate
from .s1_phantom import Spins, phantom
from .s1_sampling import image_scores, interval
from .s1_sequence import family


def image_policy(case):
    return case.get("image_policy", case.get("object_level"))


def case_policy(case):
    policies = case["evaluation"].get("case_image_criteria") or {}
    key = image_policy(case)
    return policies.get(key, policies.get("static"))


def uses_case_image(case):
    return (
        case["evaluation"]["id"] in {"s2", "s3", "s4"}
        or case["evaluation"].get("version", 1) >= 2
    ) and case_policy(case) is not None


def phase_reference(
    acq,
    case,
    root,
    reference,
    *,
    temporal_scale=1,
    sample_index=None,
    sample_indices=None,
):
    """Predict field-induced phase at the echo nearest the centre of k-space.

    This is a voxel-centre phase target, with fixed material volume fractions.
    It is deliberately distinct from a continuum encoded-image simulation.
    A refined temporal replay checks numerical error in this reference model.
    Field-dependent phase is retained; motion encoding phase is not excused.
    """
    if sample_indices is not None and sample_index is not None:
        raise ValueError("Choose a single echo or acquisition echo contexts")
    idx = np.asarray(
        sample_indices
        if sample_indices is not None
        else [
            int(np.argmin(np.linalg.norm(acq.k, axis=1)))
            if sample_index is None
            else sample_index
        ],
        dtype=int,
    )
    lines = [next(j for j, ids in enumerate(acq.lines) if i in ids) for i in idx]
    line = lines[0]
    a = copy(acq)
    phase_windows = None
    intrinsic_response = case["evaluation"].get("intrinsic_tissue_response", False)
    if family(case) != "bssfp" and not intrinsic_response:
        excitations = [p for p in acq.pulses if not p.refocusing]
        contexts = {}
        for i, current_line in zip(idx, lines, strict=True):
            excitation = excitations[acq.excitation[current_line]]
            contexts[excitation.start] = max(
                contexts.get(excitation.start, 0), acq.adc[i]
            )
        a.pulses = [
            p
            for p in acq.pulses
            if any(start <= p.start < end for start, end in contexts.items())
        ]
        if sample_indices is not None:
            phase_windows = [
                [float(start), float(end)] for start, end in sorted(contexts.items())
            ]
    else:
        a.pulses = [p for p in acq.pulses if p.start < acq.adc[idx[-1]]]
    a.adc = acq.adc[idx]
    a.adc_phase = expected_receiver_phase(acq)[idx]
    a.k = np.zeros((len(idx), 3))
    matrix = tuple(case["task"]["sequence"]["matrix"])
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    geometry = phantom(case, root, 4)
    loc = np.floor((geometry.positions[:, : len(matrix)] / fov + 0.5) * matrix).astype(
        int
    )
    voxel = np.ravel_multi_index(tuple(loc.T), matrix)
    groups = voxel * len(geometry.materials) + geometry.labels
    unique, inverse = np.unique(groups, return_inverse=True)
    weights = np.bincount(
        inverse, weights=np.broadcast_to(geometry.weight, groups.shape)
    )
    loc = np.array(np.unravel_index(unique // len(geometry.materials), matrix)).T
    positions = np.zeros((len(unique), 3))
    positions[:, : len(matrix)] = ((loc + 0.5) / matrix - 0.5) * fov
    spins = Spins(
        positions,
        unique % len(geometry.materials),
        geometry.materials,
        geometry.properties,
        weights,
        matrix,
        geometry.source_sha256,
    )

    def replay(refinement=1):
        images = []
        for physical, effects in enumerate(((), case["physics"]["effects"])):
            # Intrinsic chemical shift is nominal material contrast in current
            # cardiac contracts. Legacy contracts retain their original baseline.
            replay_spins = copy(spins)
            if (
                not physical
                and spins.properties.shape[1] > 3
                and not case["evaluation"].get("intrinsic_tissue_response", False)
            ):
                replay_spins.properties = spins.properties.copy()
                replay_spins.properties[:, 3] = 0
            values = simulate(
                a,
                replay_spins,
                effects,
                zero_gradients=True,
                individual=True,
                step_s=0.001 / temporal_scale / refinement,
                rf_step_s=20e-6 / temporal_scale / refinement,
                phase_windows=phase_windows,
                ideal_spoiling=intrinsic_response and family(case) != "bssfp",
            ).mean(axis=0)
            image = np.zeros(matrix, complex)
            np.add.at(image, tuple(loc.T), values)
            images.append(image)
        base, physical = images
        floor = max(float(np.max(np.abs(base))) * 1e-6, 1e-15)
        valid = (np.abs(base) > floor) & (
            np.abs(physical) > np.maximum(0.01 * np.abs(base), floor)
        )
        ratio = physical * base.conj()
        phasor = np.ones(matrix, complex)
        phasor[valid] = ratio[valid] / np.abs(ratio[valid])
        return phasor, valid

    fine, valid = replay()
    refined, refined_valid = replay(2)
    weights = np.abs(reference) ** 2

    def discrepancy(other, other_valid):
        # A validity-mask disagreement is counted, rather than silently dropped.
        delta = np.abs(fine - other) ** 2
        delta[valid != other_valid] = 4.0
        return float(np.sqrt(np.sum(weights * delta) / max(np.sum(weights), 1e-30)))

    temporal = discrepancy(refined, refined_valid)
    temporal_history = [{"scale": temporal_scale, "phase_chord_rms": temporal}]
    refinement = 2
    while temporal > 0.01:
        fine, valid = refined, refined_valid
        refinement *= 2
        refined, refined_valid = replay(refinement)
        temporal = discrepancy(refined, refined_valid)
        temporal_history.append(
            {
                "scale": temporal_scale * refinement / 2,
                "phase_chord_rms": temporal,
            }
        )
    coverage = float(np.sum(weights[valid]) / max(np.sum(weights), 1e-30))
    return (
        fine,
        valid,
        {
            "model": (
                "koma_rf_acquisition_phase"
                if sample_indices is not None
                else "koma_rf_voxel_phase"
            )
            + ("@2" if intrinsic_response else "@1"),
            "engine": "KomaMRICore",
            "echo_time_s": float(a.adc[0]),
            "line": line,
            "sample_indices": idx.tolist(),
            "echo_contexts": len(phase_windows) if phase_windows is not None else 1,
            "intrinsic_tissue_response": intrinsic_response,
            "rf_history": "complete; ideal transverse spoiling before excitation"
            if intrinsic_response and family(case) != "bssfp"
            else "complete coherent history"
            if family(case) == "bssfp"
            else "independent equilibrium echo contexts",
            "temporal_phase_chord_rms": temporal,
            "reference_energy_coverage": coverage,
            "spatial_model": (
                "voxel centres with exact anatomical-cell/image-voxel fractions; no continuum phase-reference claim"
                if case["object"]["phantom"].startswith("cardiac_")
                else "voxel centres with fixed order-4 material fractions; no continuum phase-reference claim"
            ),
            "converged": temporal <= 0.01,
            "coverage_adequate": coverage >= 0.5,
            "temporal_refinement_history": temporal_history,
            "limits": {
                "temporal_phase_chord_rms": 0.01,
                "reference_energy_coverage": 0.5,
            },
            "receiver_phase": "independent RF coherence convention; no fitted phase or gain",
            "geometry_frame": "original object coordinates; no registration",
        },
    )


def jackknife_stat(images, function):
    """Leading-bias-corrected nonlinear score with numerical jackknife interval."""
    n = len(images)
    total = images.sum(axis=0)
    full = float(function(total / n))
    loo = np.array([function((total - image) / (n - 1)) for image in images])
    estimate = float(n * full - (n - 1) * loo.mean())
    return estimate, list(interval(estimate, loo))


def case_image_scores(images, reference, support, phasor, valid, masks):
    """Complex and phase fidelity use only the precomputed physics phase target.

    Magnitude/dropout/contrast keep the nominal reference. Phase validity depends
    only on the oracle, never the candidate image. Nonlinear statistics are
    jackknife estimates, not described as exactly unbiased quadratic scores.
    """
    target = reference * phasor
    phase_support = (
        valid & support & (np.abs(reference) >= 0.05 * np.max(np.abs(reference)))
    )
    # Phase-free background still contributes to complex error and ghost energy.
    scored = valid | ~support
    values, intervals = image_scores(images[:, scored], target[scored], support[scored])
    magnitude = np.abs(reference)
    denom = max(float(np.sum(magnitude**2)), 1e-30)
    mag, mag_ci = jackknife_stat(
        images, lambda im: np.sum((np.abs(im) - magnitude) ** 2) / denom
    )
    values["normalized_squared_magnitude_error"] = mag
    values["magnitude_image_nrmse"] = float(np.sqrt(max(0.0, mag)))
    intervals["normalized_squared_magnitude_error"] = mag_ci
    intervals["magnitude_image_nrmse"] = [float(np.sqrt(max(0.0, x))) for x in mag_ci]
    weights = magnitude[phase_support] ** 2
    weight = max(float(weights.sum()), 1e-30)
    phase, phase_ci = jackknife_stat(
        images,
        lambda im: (
            np.sum(
                weights
                * np.angle(im[phase_support] * target[phase_support].conj()) ** 2
            )
            / weight
        ),
    )
    values["normalized_squared_phase_error"] = phase
    values["phase_error_rad"] = float(np.sqrt(max(0.0, phase)))
    intervals["normalized_squared_phase_error"] = phase_ci
    intervals["phase_error_rad"] = [float(np.sqrt(max(0.0, x))) for x in phase_ci]
    rois = {
        name: mask
        for name, mask in masks.items()
        if mask.any() and magnitude[mask].mean() > 0.05 * magnitude.max()
    }
    means = {name: float(magnitude[mask].mean()) for name, mask in rois.items()}
    pairs = [
        (a, b)
        for a, b in combinations(rois, 2)
        if abs((means[a] - means[b]) / (means[a] + means[b])) >= 0.1
    ]

    def tissue_signal(im):
        return min(
            (
                float(np.abs(im[mask]).mean()) / means[name]
                for name, mask in rois.items()
            ),
            default=1.0,
        )

    def contrast(im):
        actual = {name: float(np.abs(im[mask]).mean()) for name, mask in rois.items()}
        return min(
            (
                (actual[a] - actual[b])
                / max(actual[a] + actual[b], 1e-30)
                / ((means[a] - means[b]) / (means[a] + means[b]))
                for a, b in pairs
            ),
            default=1.0,
        )

    for name, function in (
        ("min_tissue_signal_fraction", tissue_signal),
        ("contrast_retention", contrast),
    ):
        values[name], intervals[name] = jackknife_stat(images, function)
    # Full-domain leakage/signal must not shrink when reference phase is undefined.
    full_values, full_intervals = image_scores(images, reference, support)
    for name in ("outside_object_energy_fraction", "signal_fraction"):
        values[name], intervals[name] = full_values[name], full_intervals[name]
    return (
        values,
        intervals,
        {
            "tissues": list(rois),
            "contrast_pairs": pairs,
            "phase_voxels": int(phase_support.sum()),
        },
    )


def case_bounds(policy):
    return [
        {"name": name, "min": minimum, "max": maximum}
        for name, minimum, maximum in (
            ("complex_image_nrmse", None, policy["complex_nrmse_max"]),
            ("magnitude_image_nrmse", None, policy["magnitude_nrmse_max"]),
            ("phase_error_rad", None, policy["phase_rms_rad_max"]),
            ("min_tissue_signal_fraction", policy["tissue_signal_fraction_min"], None),
            ("contrast_retention", policy["contrast_retention_min"], None),
            (
                "outside_object_energy_fraction",
                None,
                policy["outside_energy_fraction_max"],
            ),
        )
    ]
