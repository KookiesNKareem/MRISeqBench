"""Built-in S1 judge: waveform identity and uncertainty-controlled image verdicts."""

import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..io import write_json
from ..models import ImagePrecision
from ..storage import atomic_savez_compressed
from . import bart
from .s1_case import (
    case_bounds,
    case_image_scores,
    case_policy,
    phase_reference,
    uses_case_image,
)
from .s1_image import (
    image_metrics,
    material_masks,
    nrmse,
    reconstruct,
    reference_signal,
)
from .s1_koma import simulate, versions
from .s1_numerics import ReplayCache
from .s1_phantom import phantom
from .s1_probes import mechanism_probes
from .s1_sampling import analytic_reference, image_scores, interval, sampled_phantom
from .s1_sequence import check, family, inspect_sequence

IMPLEMENTATION_SHA256 = hashlib.sha256(
    b"".join(
        p.name.encode() + p.read_bytes()
        for p in sorted(
            [
                *Path(__file__).parent.glob("s[1234]*.*"),
                Path(__file__).with_name("bart.py"),
            ]
        )
    )
).hexdigest()


def decision(bounds, metrics, prefix=""):
    """Apply the published cutoffs to sufficiently precise point estimates."""
    outcomes = []
    for bound in bounds:
        name = bound["name"].removeprefix(prefix)
        if name not in metrics:
            raise ValueError(f"S1 contract requests unsupported metric: {name}")
        value = metrics[name]
        outcomes.append(
            (bound["min"] is None or value >= bound["min"])
            and (bound["max"] is None or value <= bound["max"])
        )
    return all(outcomes)


def score_precision(metrics, intervals, tolerances):
    """Numerical precision depends on interval size, never acceptance cutoffs.

    Error precision uses the untruncated quadratic statistic, since square-root
    precision is ill-conditioned near zero. No clipping can hide its uncertainty.
    """
    if "normalized_squared_error" not in intervals:
        raise ValueError("Squared error uncertainty is required for image precision")
    errors = {
        name: max(abs(metrics[name] - lo), abs(hi - metrics[name]))
        for name, (lo, hi) in intervals.items()
        if name in tolerances
    }
    return all(
        np.isfinite(error) and error <= tolerances[name]
        for name, error in errors.items()
    ), errors


def certified_decision(bounds, intervals, prefix=""):
    """An already decisive numerical interval needs no exact score estimate."""
    outcomes = []
    for bound in bounds:
        lo, hi = intervals[bound["name"].removeprefix(prefix)]
        if (bound["min"] is not None and hi < bound["min"]) or (
            bound["max"] is not None and lo > bound["max"]
        ):
            outcomes.append(False)
        elif (bound["min"] is None or lo >= bound["min"]) and (
            bound["max"] is None or hi <= bound["max"]
        ):
            outcomes.append(True)
        else:
            outcomes.append(None)
    return False if False in outcomes else None if None in outcomes else True


def resolve_decision(bounds, metrics, intervals, precise, prefix=""):
    if precise:
        return decision(bounds, metrics, prefix), "precision_target"
    certified = certified_decision(bounds, intervals, prefix)
    return certified, "interval_certified" if certified is not None else "unresolved"


def contrast_interval(images, masks):
    def ratio(image):
        magnitude = np.abs(image)
        return float(
            magnitude[masks["wm"]].mean() / max(magnitude[masks["gm"]].mean(), 1e-15)
        )

    total = images.sum(axis=0)
    value = ratio(total / len(images))
    loo = np.array([ratio((total - image) / (len(images) - 1)) for image in images])
    return value, interval(value, loo)


def evaluate_s1(
    submission,
    case,
    root,
    workspace,
    *,
    deadline=None,
    inspector=None,
    probes=None,
    reconstructor=None,
    sampler=None,
    reference_builder=None,
    phase_builder=None,
    encoded_integrator="magnus4",
    simulator=None,
    analytic_sampler=None,
    region_builder=None,
    replay_shape=None,
    temporal_norm=None,
):
    if encoded_integrator not in {"bloch", "magnus4"}:
        raise ValueError("Unsupported encoded integrator")
    simulator = simulator or simulate
    analytic_sampler = analytic_sampler or analytic_reference
    inspector = inspector or inspect_sequence
    probes = probes or mechanism_probes
    reconstructor = reconstructor or reconstruct
    sampler = sampler or reference_signal
    phase_builder = phase_builder or phase_reference
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    acq = inspector(submission, case)
    checks = acq.checks
    case_acceptance = uses_case_image(case)
    report = {
        "checks": checks,
        "metrics": {},
        "sequence_valid": False,
        "ideal_imaging_success": None,
        "imaging_success_under_case_physics": None,
        "status": "failed",
        "image_verdict": "not_evaluated",
        "image_acceptance_basis": "case_physics"
        if case_acceptance
        else "ideal_control",
        "evaluation_contract": f"{case['evaluation']['id']}@{case['evaluation']['version']}",
        "nominal_contrast_model": "finite_rf_intrinsic_materials_central_echo@1"
        if case["evaluation"].get("intrinsic_tissue_response", False)
        else "legacy_nominal_materials@1",
        "reconstruction": acq.measurements.get(
            "reconstruction", "bart_cartesian_fft@1"
        ),
        "artifacts": {"sequence": str((workspace / "sequence.json").resolve())},
    }
    write_json(workspace / "sequence.json", acq.measurements)
    if not all(c["passed"] for c in checks):
        return report
    # Masks are used only for display/ROI diagnostics, never as a spin integration
    # grid. The signal is integrated independently below.
    mask_spins = phantom(case, root, 4)
    probe_checks, diagnostic, response = probes(
        acq, case, mask_spins.materials, mask_spins.properties
    )
    checks.extend(probe_checks)
    if case["task"].get("prompt_mode") == "research":
        checks.append(
            check(
                "sequence_identity",
                all(c["passed"] for c in checks),
                case["task"]["identity_requirements"],
            )
        )
    report["sequence_valid"] = all(c["passed"] for c in checks)
    if not report["sequence_valid"]:
        report["metrics"] = diagnostic
        return report
    spec, contract = case["task"]["sequence"], case["evaluation"]
    matrix = tuple(spec["matrix"])
    reference_raw = analytic_sampler(acq, case, root, response)
    if reference_builder is None and contract.get("intrinsic_tissue_response", False):
        # Independent full Cartesian grid at central-echo material contrast;
        # encoding blur and displacement cannot enter their own target.
        from .s2 import reference_image

        reference_builder = reference_image
    reference = (
        reference_builder(acq, case, root, response, reference_raw)
        if reference_builder is not None
        else reconstructor(reference_raw, acq, matrix)
    )
    masks = material_masks(mask_spins, spec)
    support = np.zeros(matrix, dtype=bool)
    dims = len(matrix)
    loc = np.floor(
        (mask_spins.positions[:, :dims] / (np.asarray(spec["fov_mm"]) / 1000) + 0.5)
        * matrix
    ).astype(int)
    support[tuple(loc.T)] = True
    if region_builder is not None:
        support, masks = region_builder(acq, case, support, masks)
    effects = case["physics"]["effects"]
    samples = contract.get("spatial_samples") or 32768
    replicates = contract.get("spatial_replicates") or 8
    max_samples = contract.get("spatial_max_samples") or min(4 * samples, 1048576)
    tolerances = ImagePrecision.model_validate(
        contract.get("image_precision") or {}
    ).model_dump()
    seed = contract.get("spatial_seed")
    seed = 1729 if seed is None else seed
    temporal_scale = 2 ** (contract.get("temporal_refinement") or 0)
    base_steps = {"step_s": 0.001 / temporal_scale, "rf_step_s": 20e-6 / temporal_scale}
    replay = ReplayCache(
        workspace / "replays",
        IMPLEMENTATION_SHA256
        + hashlib.sha256(Path(submission).read_bytes()).hexdigest()
        + json.dumps(
            {"case": case, "encoded_integrator": encoded_integrator}, sort_keys=True
        ),
        lambda *args, **steps: simulator(
            *args, integration_method=encoded_integrator, **steps
        ),
        chunk_size=min(max_samples, 32768),
        signal_shape=replay_shape(acq) if replay_shape is not None else None,
    )
    robustness_bounds = [
        {"name": "complex_image_nrmse", "min": None, "max": 0.2},
        {"name": "signal_fraction", "min": 0.01, "max": None},
        {"name": "outside_object_energy_fraction", "min": None, "max": 0.4},
    ]
    if contract["id"] in {"s2", "s3"}:
        # P0/P1 have no perturbations. Their diagnostic case image must use the
        # same S2/S3 policy as the identical ideal image, including numerical stop.
        robustness_bounds = [
            {**bound, "name": bound["name"].removeprefix("ideal_")}
            for bound in contract["metrics"]
        ]
    phasor, phase_valid, phase_info = (
        np.ones(matrix, complex),
        np.ones(matrix, bool),
        None,
    )
    case_details = None
    if case_acceptance:
        robustness_bounds = case_bounds(case_policy(case))
        phasor, phase_valid, phase_info = phase_builder(
            acq,
            case,
            root,
            reference,
            temporal_scale=temporal_scale,
        )
    history = []
    temporal_history = []
    stop_reason = "precision_target"
    while True:
        ideal_runs, physical_runs = [], []
        ideal_raw_runs, physical_raw_runs = [], []
        koma_ideal_runs, koma_physical_runs, sampled_reference_runs = [], [], []
        counts = []
        for replicate in range(replicates):
            spins = sampled_phantom(case, root, samples, replicate)
            sampled_reference = sampler(acq, spins, spec, response)
            ideal_spins = spins
            chemical_shift = spins.properties.shape[1] > 3 and np.any(
                spins.properties[:, 3]
            )
            if chemical_shift and not contract.get("intrinsic_tissue_response", False):
                properties = spins.properties.copy()
                properties[:, 3] = 0
                ideal_spins = replace(spins, properties=properties)
            ideal_sampled = replay(acq, ideal_spins, **base_steps)
            physical_sampled = (
                replay(acq, spins, effects, **base_steps)
                if effects or chemical_shift
                else ideal_sampled
            )
            if replicate == 0:
                # Refine integration before accepting any spatial-stage scores.
                # A single unsuccessful halving is a request for more work.
                while True:
                    refined_sampled = replay(
                        acq,
                        spins,
                        effects,
                        **{name: step / 2 for name, step in base_steps.items()},
                    )
                    temporal_raw_error = nrmse(refined_sampled, physical_sampled)
                    temporal_image_error = (
                        temporal_norm(
                            refined_sampled - physical_sampled, acq, matrix, reference
                        )
                        if temporal_norm is not None
                        else 0.0
                    )
                    temporal_error = max(temporal_raw_error, temporal_image_error)
                    temporal_history.append(
                        {
                            **base_steps,
                            "nrmse": temporal_error,
                            **(
                                {
                                    "raw_nrmse": temporal_raw_error,
                                    "image_nrmse": temporal_image_error,
                                }
                                if temporal_norm is not None
                                else {}
                            ),
                        }
                    )
                    write_json(
                        workspace / "progress.json",
                        {
                            "spatial_refinement": history,
                            "temporal_refinement": temporal_history,
                            "current_samples": samples,
                            "cache_hits": replay.hits,
                        },
                    )
                    if temporal_error <= 0.01:
                        break
                    base_steps = {name: step / 2 for name, step in base_steps.items()}
                    physical_sampled = refined_sampled
                    ideal_sampled = (
                        replay(acq, ideal_spins, **base_steps)
                        if effects or chemical_shift
                        else physical_sampled
                    )
            # A control variate, not substituted dynamics: only the known Fourier
            # sampling error is removed. All finite-RF/coherence residuals survive.
            ideal_raw = reference_raw + ideal_sampled - sampled_reference
            physical_raw = reference_raw + physical_sampled - sampled_reference
            koma_ideal_runs.append(ideal_sampled)
            koma_physical_runs.append(physical_sampled)
            sampled_reference_runs.append(sampled_reference)
            ideal_raw_runs.append(ideal_raw)
            physical_raw_runs.append(physical_raw)
            ideal_runs.append(reconstructor(ideal_raw, acq, matrix))
            physical_runs.append(reconstructor(physical_raw, acq, matrix))
            counts.append(len(spins.positions))
        ideal_runs, physical_runs = np.asarray(ideal_runs), np.asarray(physical_runs)
        ideal_metrics, ideal_intervals = image_scores(ideal_runs, reference, support)
        if case_acceptance:
            physical_metrics, physical_intervals, case_details = case_image_scores(
                physical_runs, reference, support, phasor, phase_valid, masks
            )
        else:
            physical_metrics, physical_intervals = image_scores(
                physical_runs, reference, support
            )
        bounds = list(contract["metrics"])
        if family(case) == "gre_t1w" and all(
            m in masks and masks[m].any() for m in ("wm", "gm")
        ):
            value, ci = contrast_interval(ideal_runs, masks)
            ideal_metrics["wm_gm_ratio"] = value
            ideal_intervals["wm_gm_ratio"] = ci
            bounds.append({"name": "ideal_wm_gm_ratio", "min": 1.15, "max": None})
        ideal_precise, ideal_errors = score_precision(
            ideal_metrics, ideal_intervals, tolerances
        )
        case_precise, case_errors = score_precision(
            physical_metrics, physical_intervals, tolerances
        )
        ideal_ok, ideal_basis = resolve_decision(
            bounds, ideal_metrics, ideal_intervals, ideal_precise, "ideal_"
        )
        case_ok, case_basis = resolve_decision(
            robustness_bounds, physical_metrics, physical_intervals, case_precise
        )
        if (
            phase_info is not None
            and phase_info.get("reference_energy_coverage", 1) < 0.5
        ):
            case_ok, case_basis = False, "phase_reference_coverage"
        history.append(
            {
                "samples": samples,
                "replicates": len(ideal_runs),
                "spins": counts,
                "ideal_intervals": ideal_intervals,
                "case_intervals": physical_intervals,
                "ideal_precision_errors": ideal_errors,
                "case_precision_errors": case_errors,
                "decision_basis": {"ideal": ideal_basis, "case": case_basis},
            }
        )
        write_json(
            workspace / "progress.json",
            {
                "spatial_refinement": history,
                "temporal_refinement": temporal_history,
                "cache_hits": replay.hits,
            },
        )
        if case_ok is not None and (case_acceptance or ideal_ok is not None):
            stop_reason = (
                "precision_target"
                if case_precise and (case_acceptance or ideal_precise)
                else "interval_certified"
            )
            break
        samples *= 2
    converged = (
        temporal_error <= 0.01
        and case_ok is not None
        and (case_acceptance or ideal_ok is not None)
        and (phase_info is None or phase_info["converged"])
    )
    ideal, physical = ideal_runs.mean(axis=0), physical_runs.mean(axis=0)
    metrics = {
        **diagnostic,
        **{
            f"ideal_{k}": v
            for k, v in image_metrics(ideal, reference, mask_spins, spec).items()
        },
        **{
            f"case_{k}": v
            for k, v in image_metrics(physical, reference, mask_spins, spec).items()
        },
        # Complex error uses unbiased cross-products; nonlinear case scores use
        # bias-corrected jackknife estimates, not the displayed mean-image score.
        **{f"ideal_{k}": float(v) for k, v in ideal_metrics.items()},
        **{f"case_{k}": float(v) for k, v in physical_metrics.items()},
        "temporal_nrmse": temporal_error,
        **(
            {
                "temporal_raw_nrmse": temporal_raw_error,
                "temporal_image_nrmse": temporal_image_error,
            }
            if temporal_norm is not None
            else {}
        ),
        "case_vs_ideal_image_nrmse": nrmse(np.abs(physical), np.abs(ideal)),
        "case_signal_retention": float(
            np.linalg.norm(physical) / max(np.linalg.norm(ideal), 1e-15)
        ),
    }
    report.update(
        metrics=metrics,
        numerical_intervals={"ideal": ideal_intervals, "case": physical_intervals},
        numerically_converged=converged,
        numerical_precision={
            "targets": tolerances,
            "ideal_errors": ideal_errors,
            "case_errors": case_errors,
        },
        score_precision_met={"ideal": ideal_precise, "case": case_precise},
        decision_basis={"ideal": ideal_basis, "case": case_basis},
        spatial_sampling={
            "samples_per_replicate": samples,
            "replicates": len(ideal_runs),
            "requested_replicates": replicates,
            "seed": seed,
            "initial_budget_per_replicate": max_samples,
            "chunk_size": replay.chunk_size,
            "exceeded_initial_budget": samples > max_samples,
            "stop_reason": stop_reason,
        },
        effects=sorted({e["kind"] for e in effects}),
        reconstruction_provenance=bart.provenance(),
        case_phase_reference=phase_info,
        case_scoring_regions=case_details,
        limitations=[]
        if converged
        else [
            "case_phase_reference_not_resolved"
            if phase_info is not None and not phase_info["converged"]
            else "image_score_precision_or_temporal_error_not_resolved"
        ],
    )
    report["ideal_image_checks"] = [
        {
            "name": bound["name"],
            "passed": resolve_decision(
                [bound], ideal_metrics, ideal_intervals, ideal_precise, "ideal_"
            )[0]
            if temporal_error <= 0.01 and ideal_ok is not None
            else None,
            "detail": {
                "value": metrics[bound["name"]],
                "numerical_interval": ideal_intervals[
                    bound["name"].removeprefix("ideal_")
                ],
                "numerically_resolved": temporal_error <= 0.01 and ideal_ok is not None,
                **bound,
            },
        }
        for bound in bounds
    ]
    report["case_image_checks"] = [
        {
            "name": "case_" + bound["name"],
            "passed": resolve_decision(
                [bound], physical_metrics, physical_intervals, case_precise
            )[0]
            if converged
            else None,
            "detail": {
                "value": physical_metrics[bound["name"]],
                "numerical_interval": physical_intervals[bound["name"]],
                "numerically_resolved": converged,
                **bound,
            },
        }
        for bound in robustness_bounds
    ]
    if phase_info is not None:
        report["case_image_checks"].append(
            {
                "name": "case_phase_reference_coverage",
                "passed": phase_info.get("reference_energy_coverage", 1) >= 0.5,
                "detail": {
                    "value": phase_info.get("reference_energy_coverage", 1),
                    "min": 0.5,
                },
            }
        )
    report["image_checks"] = (
        report["case_image_checks"] if case_acceptance else report["ideal_image_checks"]
    )
    if converged and contract["calibrated"]:
        report.update(
            ideal_imaging_success=bool(ideal_ok) if ideal_ok is not None else None,
            imaging_success_under_case_physics=bool(case_ok),
            image_verdict="passed" if case_ok else "failed",
            status="passed"
            if report["sequence_valid"] and (case_ok if case_acceptance else ideal_ok)
            else "failed",
        )
    elif not converged:
        raise RuntimeError("Adaptive replay exited before numerical convergence")
    else:
        report.update(status="uncalibrated", image_verdict="uncalibrated")
    report["artifacts"] = {
        name: str((workspace / filename).resolve())
        for name, filename in (
            ("sequence", "sequence.json"),
            ("images", "images.npz"),
            ("provenance", "provenance.json"),
        )
    }
    atomic_savez_compressed(
        workspace / "images.npz",
        reference=reference,
        case_reference=reference * phasor,
        case_reference_phasor=phasor,
        case_reference_phase_valid=phase_valid,
        object_support=support,
        ideal=ideal,
        physical=physical,
        reference_raw=reference_raw,
        ideal_raw=np.mean(ideal_raw_runs, axis=0),
        physical_raw=np.mean(physical_raw_runs, axis=0),
        koma_ideal_raw_replicates=np.asarray(koma_ideal_runs),
        koma_physical_raw_replicates=np.asarray(koma_physical_runs),
        sampled_reference_raw_replicates=np.asarray(sampled_reference_runs),
        ideal_replicates=ideal_runs,
        physical_replicates=physical_runs,
        temporal_raw_difference=refined_sampled - koma_physical_runs[0],
        adc_times_s=acq.adc,
        k_cycles_per_m=acq.k,
        matrix=np.asarray(matrix),
        fov_mm=np.asarray(spec["fov_mm"]),
    )
    write_json(
        workspace / "provenance.json",
        {
            "implementation": f"{contract['id']}_koma@1",
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "reconstruction": report["reconstruction"],
            "reconstruction_provenance": report["reconstruction_provenance"],
            "case_sha256": hashlib.sha256(
                json.dumps(case, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "finufft", "pypulseq", "scipy")
            },
            "julia_manifest_sha256": hashlib.sha256(
                (Path(root) / "Manifest.toml").read_bytes()
            ).hexdigest(),
            "submission_sha256": hashlib.sha256(
                Path(submission).read_bytes()
            ).hexdigest(),
            "phantom": case["object"]["phantom"],
            "shapes_sha256": mask_spins.source_sha256,
            "koma_versions": versions(),
            "spatial_integration": (
                "tissue-stratified independently scrambled Sobol over anatomical cells and their interiors; exact tissue volumes and voxel-cell Fourier control variate; full Bloch residual retained"
                if case["object"]["phantom"].startswith("cardiac_")
                else "shape-stratified independently scrambled Sobol with exact primitive volumes and signed parent subtraction; analytic family-encoding control variate; full Bloch residual retained"
            ),
            "encoded_integrator": "KomaMRICore.Bloch"
            if encoded_integrator == "bloch"
            else "KomaMRICore.BlochMagnus4",
            "spatial_samples": samples,
            "spatial_replicates": len(ideal_runs),
            "requested_spatial_replicates": replicates,
            "seeds": list(range(seed, seed + len(ideal_runs))),
            "refinement_history": history,
            "temporal_refinement_history": temporal_history,
            "replay_cache_hits": replay.hits,
            "score_intervals": "Approximate 99% Student-t delete-one jackknife intervals; complex/energy scores use quadratic cross-products, nonlinear case scores use leading-bias correction; numerical uncertainty, not task uncertainty",
            "rf_step_s": [base_steps["rf_step_s"], base_steps["rf_step_s"] / 2],
            "dynamic_step_s": [base_steps["step_s"], base_steps["step_s"] / 2],
            "convergence_limits": {
                "temporal_nrmse": 0.01,
                **(
                    {"temporal_raw_nrmse": 0.01, "temporal_image_nrmse": 0.01}
                    if temporal_norm is not None
                    else {}
                ),
                "initial_spatial_budget": max_samples,
            },
            "numerically_converged": converged,
            "image_precision_targets": tolerances,
            "decision_rule": "interval-certified verdict where possible; otherwise point estimate at published cutoff after numerical precision target is met",
            "effects": effects,
            "reference": "analytic Fourier transforms of registered shapes with independent ideal family response; no fitted gain or registration",
            "case_phase_reference": phase_info,
            "case_scoring_regions": case_details,
            "image_acceptance_basis": report["image_acceptance_basis"],
            "evaluation_contract": report["evaluation_contract"],
            "case_image_criteria": contract.get("case_image_criteria"),
            "t2star_scope": "65-isochromat +/-50 Hz diagnostic only; images retain declared T2 and B0",
            "calibrated": contract["calibrated"],
            "family": family(case),
        },
    )
    return report
