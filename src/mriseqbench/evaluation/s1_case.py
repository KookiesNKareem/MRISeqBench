"""Case-aware RF phase reference and independent image-quality requirements.

The phase oracle contains RF and declared fields, never submitted ADC phases or
encoding gradients. GRE/SE isolate the selected echo (ideal spoiling); bSSFP
retains its complete RF history. It is a finite-RF KomaMRI response, not a fitted
phase map. Geometry, magnitude, contrast and dropout retain nominal-object targets.
"""

from copy import copy, deepcopy
from itertools import combinations

import numpy as np
from pypulseq import Sequence, make_delay

from .s1_image import expected_receiver_phase
from .s1_koma import simulate
from .s1_phantom import Spins, phantom
from .s1_sampling import image_scores, interval, quadratic
from .s1_sequence import family


def case_policy(case):
    policies = case["evaluation"].get("case_image_criteria") or {}
    return policies.get(case.get("image_policy"), policies.get("static"))


def uses_case_image(case):
    return (
        case["evaluation"]["id"] in {"s2", "s3", "s4"}
        or case["evaluation"].get("version", 1) >= 2
    ) and case_policy(case) is not None


def phase_replay(acq, spins, effects, **options):
    """Share identical RF-only states, retaining every original volume fraction."""
    if (
        not options.get("zero_gradients")
        or not options.get("individual")
        or any(
            e["kind"] != "b0" or e["model"] not in {"constant", "linear"}
            for e in effects
        )
    ):
        return simulate(acq, spins, effects, **options)
    active = np.zeros(3, dtype=bool)
    for effect in effects:
        if effect["model"] == "linear":
            gradient = np.asarray(effect["gradient_hz_per_m"])
            active[: len(gradient)] |= gradient != 0
    _, indices, inverse = np.unique(
        np.rec.fromarrays([spins.labels, *spins.positions[:, active].T]),
        return_index=True,
        return_inverse=True,
    )
    reduced = copy(spins)
    reduced.positions, reduced.labels, reduced.weight = (
        spins.positions[indices],
        spins.labels[indices],
        1.0,
    )
    values = simulate(acq, reduced, effects, **options)
    return values[:, inverse] * np.broadcast_to(spins.weight, spins.labels.shape)


def phase_reference(
    acq, case, root, reference, *, temporal_scale=1, sample_indices=None
):
    """Predict field-induced phase at the echo nearest the centre of k-space.

    This is a voxel-centre phase target, with fixed material volume fractions.
    It is deliberately distinct from a continuum encoded-image simulation.
    A refined temporal replay checks numerical error in this reference model.
    Field-dependent phase is retained; motion encoding phase is not excused.
    """
    idx = np.asarray(
        sample_indices
        if sample_indices is not None
        else [int(np.argmin(np.linalg.norm(acq.k, axis=1)))],
        dtype=int,
    )
    line = next(j for j, ids in enumerate(acq.lines) if idx[0] in ids)
    a = copy(acq)
    a.pulses = [p for p in acq.pulses if p.start < acq.adc[idx[-1]]]
    a.adc = acq.adc[idx]
    a.adc_phase = expected_receiver_phase(acq)[idx]
    a.k = np.zeros((len(idx), 3))
    matrix, spins, loc = voxel_spins(case, root)

    def replay(refinement=1):
        images = []
        for physical, effects in enumerate(((), case["physics"]["effects"])):
            replay_acq = a
            spatial_excitation = physical and family(case) == "rfov_epi"
            if spatial_excitation:
                # The fixed rFOV target requires actual 2D excitation. Its
                # independent phase oracle retains selection gradients while
                # discarding every subsequent image-encoding gradient.
                from .s3_sequence import rf_block_endpoints

                endpoint = max(
                    end
                    for start, end in rf_block_endpoints(a.seq).items()
                    if start < a.adc[-1]
                )
                selection = Sequence(a.seq.system)
                clock = 0.0
                for number in a.seq.block_events:
                    block = a.seq.get_block(number)
                    duration = a.seq.block_durations[number]
                    events = [make_delay(duration)]
                    for name in ("rf", "adc"):
                        event = getattr(block, name)
                        if event is not None:
                            events.append(deepcopy(event))
                    if clock < endpoint - 1e-12:
                        events.extend(
                            deepcopy(g)
                            for g in (block.gx, block.gy, block.gz)
                            if g is not None
                        )
                    selection.add_block(*events)
                    clock += duration
                replay_acq = copy(a)
                replay_acq.seq, replay_acq.source_path = selection, None
            values = phase_replay(
                replay_acq,
                spins,
                effects,
                zero_gradients=not spatial_excitation,
                individual=True,
                step_s=0.001 / temporal_scale / refinement,
                rf_step_s=20e-6 / temporal_scale / refinement,
                ideal_spoiling=family(case) not in {"bssfp", "rfov_epi"},
            ).mean(axis=0)
            image = np.zeros(matrix, complex)
            np.add.at(image, tuple(loc.T), values)
            images.append(image)
        return rf_phasor(*images, matrix)

    fine, valid, summary = converge_phase(replay, reference, temporal_scale, case)
    return (
        fine,
        valid,
        {
            "model": (
                "koma_rf_acquisition_phase"
                if sample_indices is not None
                else "koma_rf_voxel_phase"
            )
            + "@2",
            "engine": "KomaMRICore",
            "echo_time_s": float(a.adc[0]),
            "line": line,
            "sample_indices": idx.tolist(),
            "rf_history": "complete coherent history"
            if family(case) == "bssfp"
            else "complete; ideal transverse spoiling before excitation",
            **summary,
        },
    )


def voxel_spins(case, root):
    """Voxel-centre spins carrying exact order-4 material/voxel volume fractions."""
    matrix = tuple(case["task"]["sequence"]["matrix"])
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    geometry = phantom(case, root)
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
    return matrix, spins, loc


def rf_phasor(base, physical, matrix):
    floor = max(float(np.max(np.abs(base))) * 1e-6, 1e-15)
    valid = (np.abs(base) > floor) & (
        np.abs(physical) > np.maximum(0.01 * np.abs(base), floor)
    )
    ratio = physical * base.conj()
    phasor = np.ones(matrix, complex)
    phasor[valid] = ratio[valid] / np.abs(ratio[valid])
    return phasor, valid


# References need at most 2 halvings (image replay and phase oracle); the cap
# bounds grading of a replay that never converges (no time limit by design).
MAX_TEMPORAL_HALVINGS = 6


def converge_phase(replay, reference, temporal_scale, case):
    """Halve RF-only phase-oracle steps until the weighted phase chord settles."""
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
    while temporal > 0.01 and refinement < 2**MAX_TEMPORAL_HALVINGS:
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
            "temporal_phase_chord_rms": temporal,
            "reference_energy_coverage": coverage,
            "spatial_model": (
                "voxel centres with exact anatomical-cell/image-voxel fractions; no continuum phase-reference claim"
                if case["object"]["phantom"].startswith("cardiac_")
                else "voxel centres with fixed order-4 material fractions; no continuum phase-reference claim"
            ),
            "converged": temporal <= 0.01 and coverage >= 0.5,
            "coverage_adequate": coverage >= 0.5,
            "temporal_refinement_history": temporal_history,
            "timestep_cap_reached": temporal > 0.01,
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


COMPLEX_NRMSE_RULE = (
    "Complex NRMSE covers the full image and divides by the full target energy: "
    "oracle-valid voxels and voxels outside object support contribute their "
    "complex error, and foreground voxels that are not oracle-valid contribute "
    "their magnitude error (|image| - |target|). "
)


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
    # See COMPLEX_NRMSE_RULE: an invalid phase region can neither be excused
    # nor shrink the normalisation.
    scored = valid | ~support
    unphased = ~scored
    values, intervals = image_scores(images[:, scored], target[scored], support[scored])
    magnitude = np.abs(reference)
    denom = max(float(np.sum(magnitude**2)), 1e-30)
    error, error_loo = quadratic(images[:, scored] - target[scored])
    if unphased.any():
        n = len(images)
        total = images.sum(axis=0)

        def unphased_error(im):
            return float(np.sum((np.abs(im[unphased]) - magnitude[unphased]) ** 2))

        loo = np.array([unphased_error((total - im) / (n - 1)) for im in images])
        error += n * unphased_error(total / n) - (n - 1) * loo.mean()
        error_loo = error_loo + loo

    def squared_and_root(squared, root, value, ci):
        values[squared], intervals[squared] = value, list(ci)
        values[root] = float(np.sqrt(max(0.0, value)))
        intervals[root] = [float(np.sqrt(max(0.0, x))) for x in ci]

    squared_and_root(
        "normalized_squared_error",
        "complex_image_nrmse",
        error / denom,
        interval(error / denom, error_loo / denom),
    )
    squared_and_root(
        "normalized_squared_magnitude_error",
        "magnitude_image_nrmse",
        *jackknife_stat(
            images, lambda im: np.sum((np.abs(im) - magnitude) ** 2) / denom
        ),
    )
    weights = magnitude[phase_support] ** 2
    weight = max(float(weights.sum()), 1e-30)
    squared_and_root(
        "normalized_squared_phase_error",
        "phase_error_rad",
        *jackknife_stat(
            images,
            lambda im: (
                np.sum(
                    weights
                    * np.angle(im[phase_support] * target[phase_support].conj()) ** 2
                )
                / weight
            ),
        ),
    )
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


def contrast_required(case):
    return family(case) not in {
        "t1_mapping",
        "t2_mapping",
        "reduced_fov",
        "rfov_epi",
    }


def case_bounds(policy, *, contrast=True):
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
        if contrast or name != "contrast_retention"
    ]
