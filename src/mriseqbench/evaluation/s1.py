"""Built-in S1 judge: waveform identity and uncertainty-controlled image verdicts."""

import hashlib
import importlib.metadata
import json
import os
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np

from ..io import finite_json, write_json
from ..models import ImagePrecision
from ..storage import atomic_savez_compressed
from . import bart
from .s1_case import (
    MAX_TEMPORAL_HALVINGS,
    case_bounds,
    case_image_scores,
    case_policy,
    contrast_required,
    uses_case_image,
)
from .s1_control import build_controls, corrected_raw
from .s1_image import image_metrics, material_masks, nrmse, reference_signal
from .s1_koma import NativeRejection, reconstruct, simulate, versions
from .s1_numerics import ReplayCache
from .s1_phantom import phantom
from .s1_probes import mechanism_probes
from .s1_sampling import analytic_reference, image_scores, sampled_phantom
from .s1_sequence import MissingExcitation, check, family, inspect_sequence

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


def decision(bound, metrics, prefix=""):
    """Apply a published cutoff to a sufficiently precise point estimate."""
    name = bound["name"].removeprefix(prefix)
    if name not in metrics:
        raise ValueError(f"S1 contract requests unsupported metric: {name}")
    value = metrics[name]
    return bool(
        (bound["min"] is None or value >= bound["min"])
        and (bound["max"] is None or value <= bound["max"])
    )


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


def certified_decision(bound, intervals, prefix=""):
    """An already decisive numerical interval needs no exact score estimate."""
    lo, hi = intervals[bound["name"].removeprefix(prefix)]
    if (bound["min"] is not None and hi < bound["min"]) or (
        bound["max"] is not None and lo > bound["max"]
    ):
        return False
    if (bound["min"] is None or lo >= bound["min"]) and (
        bound["max"] is None or hi <= bound["max"]
    ):
        return True
    return None


# Jackknife-corrected nonlinear scores and the squared statistic each derives from.
NONLINEAR = {
    "magnitude_image_nrmse": "normalized_squared_magnitude_error",
    "phase_error_rad": "normalized_squared_phase_error",
}


def resolve_decision(bounds, metrics, intervals, precise, prefix=""):
    """Use precise point estimates or a decisive numerical interval.

    Nonlinear squared errors have nonnegative support. Projecting their interval
    onto that support preserves its coverage; a negative corrected estimate is
    never accepted merely because its displayed square root clips to zero.
    """
    outcomes = []
    point_valid = True
    for bound in bounds:
        name = bound["name"].removeprefix(prefix)
        local_intervals = intervals
        squared = NONLINEAR.get(name)
        admissible = True
        if squared is not None and squared in metrics:
            lo, hi = intervals[squared]
            value = metrics[squared]
            if not np.isfinite([value, lo, hi]).all() or lo > hi:
                outcomes.append(None)
                point_valid = False
                continue
            local_intervals = {name: (np.sqrt(max(0.0, lo)), np.sqrt(max(0.0, hi)))}
            admissible = value >= 0
            point_valid &= admissible
        outcomes.append(
            decision(bound, metrics, prefix)
            if precise and admissible
            else certified_decision(bound, local_intervals, prefix)
        )
    if precise and point_valid and None not in outcomes:
        return all(outcomes), "precision_target"
    outcome = False if False in outcomes else None if None in outcomes else True
    return outcome, "interval_certified" if outcome is not None else "unresolved"


def failed_report(
    case, checks, *, reconstruction="bart_cartesian_fft@1", artifacts=None
):
    """Complete shared S1-S4 report schema, initialised as a failed verdict."""
    return {
        "checks": checks,
        "image_checks": [],
        "ideal_image_checks": [],
        "case_image_checks": [],
        "metrics": {},
        "sequence_valid": False,
        "ideal_imaging_success": None,
        "imaging_success_under_case_physics": None,
        "numerically_converged": None,
        "status": "failed",
        "image_verdict": "not_evaluated",
        "image_acceptance_basis": "case_physics"
        if uses_case_image(case)
        else "ideal_control",
        "evaluation_contract": f"{case['evaluation']['id']}@{case['evaluation']['version']}",
        "nominal_contrast_model": "finite_rf_intrinsic_materials_central_echo@1",
        "reconstruction": reconstruction,
        "artifacts": artifacts or {},
        "limitations": [],
    }


def reconstructed_temporal_error(reconstructor):
    def error(delta, acq, matrix, reference, coarse=None):
        image = reconstructor(delta, acq, matrix)
        return float(np.linalg.norm(image) / max(np.linalg.norm(reference), 1e-30))

    return error


def structure_failure(case, name, detail):
    return failed_report(case, [check(name, False, detail)])


def evaluate_s1(submission, case, root, workspace, **kwargs):
    """S1-S3 judge; a native parse rejection of the submitted bytes fails it.

    Host failures (OOM, CUDA, interrupt, I/O, worker exit, timeout) propagate.
    """
    Path(workspace).mkdir(parents=True, exist_ok=True)  # Callers add artifacts.
    try:
        # Every report path is strict-JSON safe (NaN/inf metrics become null).
        return finite_json(_evaluate_s1(submission, case, root, workspace, **kwargs))
    except NativeRejection as exc:
        return structure_failure(case, exc.check, str(exc))


def _evaluate_s1(
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
    control_builder=build_controls,
):
    if encoded_integrator not in {"bloch", "magnus4"}:
        raise ValueError("Unsupported encoded integrator")
    simulator = simulator or simulate
    analytic_sampler = analytic_sampler or analytic_reference
    inspector = inspector or inspect_sequence
    probes = probes or mechanism_probes
    reconstructor = reconstructor or reconstruct
    sampler = sampler or reference_signal
    if phase_builder is None:
        # S1 shares the S2 context selector: target and phase oracle replay
        # the same central echo.
        from .s2 import phase_reference as phase_builder
    workspace = Path(workspace)

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Image replay exceeded its grading budget")

    check_deadline()
    try:
        acq = inspector(submission, case)
    except MissingExcitation as exc:
        return structure_failure(
            case, getattr(exc, "check", "excitation_before_adc"), str(exc)
        )
    # Candidate geometry can leave measurements undefined (e.g. no k travel);
    # report those as null so strict JSON never turns them into infra errors.
    acq.checks = finite_json(acq.checks)
    acq.measurements = finite_json(acq.measurements)
    checks = acq.checks
    # Gate on the timestep difference through the task's reconstruction,
    # normalised by the independent target (as image scores are), in addition
    # to raw-signal NRMSE; the raw denominator is the coarse signal.
    temporal_norm = temporal_norm or reconstructed_temporal_error(reconstructor)
    case_acceptance = uses_case_image(case)
    report = failed_report(
        case,
        checks,
        reconstruction=acq.measurements.get("reconstruction", "bart_cartesian_fft@1"),
        artifacts={"sequence": str((workspace / "sequence.json").resolve())},
    )
    write_json(workspace / "sequence.json", acq.measurements)
    if not all(c["passed"] for c in checks):
        return report
    # Masks are used only for display/ROI diagnostics, never as a spin integration
    # grid. The signal is integrated independently below.
    check_deadline()
    mask_spins = phantom(case, root)
    probe_checks, diagnostic, response = probes(
        acq, case, mask_spins.materials, mask_spins.properties
    )
    checks.extend(probe_checks)
    report["sequence_valid"] = all(c["passed"] for c in checks)
    if not report["sequence_valid"]:
        report["metrics"] = diagnostic
        return report
    spec, contract = case["task"]["sequence"], case["evaluation"]
    matrix = tuple(spec["matrix"])
    reference_raw = analytic_sampler(acq, case, root, response)
    if reference_builder is None:
        # Independent full Cartesian grid at central-echo material contrast;
        # encoding blur and displacement cannot enter their own target.
        from .s2 import reference_image as reference_builder
    reference = reference_builder(acq, case, root, response, reference_raw)
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
    contrast = contrast_required(case)
    if not contrast:
        tolerances.pop("contrast_retention", None)
    seed = contract.get("spatial_seed")
    seed = 1729 if seed is None else seed
    temporal_scale = 2 ** (contract.get("temporal_refinement") or 0)
    base_steps = {"step_s": 0.001 / temporal_scale, "rf_step_s": 20e-6 / temporal_scale}
    submission_sha256 = hashlib.sha256(Path(submission).read_bytes()).hexdigest()
    replay = ReplayCache(
        workspace / "replays",
        IMPLEMENTATION_SHA256
        + submission_sha256
        + json.dumps(
            {"case": case, "encoded_integrator": encoded_integrator}, sort_keys=True
        ),
        lambda *args, **steps: simulator(
            *args, integration_method=encoded_integrator, **steps
        ),
        chunk_size=min(max_samples, 32768),
        signal_shape=replay_shape(acq) if replay_shape is not None else None,
        deadline=deadline,
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
        robustness_bounds = case_bounds(case_policy(case), contrast=contrast)
        phasor, phase_valid, phase_info = phase_builder(
            acq,
            case,
            root,
            reference,
            temporal_scale=temporal_scale,
        )
        if not phase_info["converged"]:
            report.update(
                status="numerically_unresolved",
                numerically_converged=False,
                failure_kind="unresolved_at_timestep_cap"
                if phase_info.get("timestep_cap_reached")
                else "phase_reference_not_resolved",
                case_phase_reference=phase_info,
            )
            return report
    # The control's integral is separate from the independent target encoding.
    check_deadline()
    controls = control_builder(acq, case, root) if control_builder else None
    check_deadline()
    control_description = (
        {
            "method": "known_integral_fid_history@1",
            "history": controls[0].history,
            "independent_target_preserved": True,
            "native_residual_preserved": True,
        }
        if controls
        else {"method": "intended_fourier@1"}
    )
    report["spatial_control_variate"] = control_description
    history = []
    temporal_history = []
    if checkpoint_path := os.environ.get("MRISEQBENCH_NUMERICAL_CHECKPOINT"):
        checkpoint = json.loads(Path(checkpoint_path).read_text())
        assert (
            checkpoint["case_sha256"]
            == hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        )
        assert checkpoint["sequence_sha256"] == submission_sha256
        history = checkpoint["progress"]["spatial_refinement"]
        temporal_history = checkpoint["progress"]["temporal_refinement"]
        assert history[-1]["decision_basis"]["case"] == "unresolved"
        if history[-1]["samples"] >= max_samples:
            raise ValueError(
                "Checkpoint already reached the declared spatial sample cap"
            )
        samples = min(history[-1]["samples"] * 2, max_samples)
        base_steps = {name: temporal_history[-1][name] for name in base_steps}
    stage_limit = int(os.environ.get("MRISEQBENCH_MAX_SPATIAL_STAGE", "0"))
    while True:
        check_deadline()
        ideal_runs, physical_runs = [], []
        ideal_raw_runs, physical_raw_runs = [], []
        koma_ideal_runs, koma_physical_runs, sampled_reference_runs = [], [], []
        ideal_control_runs, physical_control_runs = [], []
        counts = []
        pool, pending, spin_batch = None, [], None
        cancelled = Event()

        def control_check(cancelled=cancelled):
            check_deadline()
            if cancelled.is_set():
                raise CancelledError()

        def control_samples(spins, check=control_check):
            check()
            ideal = controls[0].sampled(spins, check=check)
            physical = (
                controls[1].sampled(spins, check=check)
                if controls[1] is not controls[0]
                else ideal
            )
            return ideal, physical

        try:
            if controls and dims == 3:
                pool = ThreadPoolExecutor(max_workers=4)
                spin_batch = [
                    sampled_phantom(
                        case, root, samples, rep, directory=workspace / ".sampling"
                    )
                    for rep in range(replicates)
                ]
                pending = [pool.submit(control_samples, spins) for spins in spin_batch]
            for replicate in range(replicates):
                check_deadline()
                spins = (
                    spin_batch[replicate]
                    if pending
                    else sampled_phantom(
                        case,
                        root,
                        samples,
                        replicate,
                        directory=workspace / ".sampling",
                    )
                )
                sampled_reference = (
                    None if controls else sampler(acq, spins, spec, response)
                )
                chemical_shift = spins.properties.shape[1] > 3 and np.any(
                    spins.properties[:, 3]
                )
                ideal_sampled = replay(acq, spins, **base_steps)
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
                        temporal_image_error = temporal_norm(
                            refined_sampled - physical_sampled,
                            acq,
                            matrix,
                            reference,
                            coarse=physical_sampled,
                        )
                        temporal_error = max(temporal_raw_error, temporal_image_error)
                        temporal_history.append(
                            {
                                **base_steps,
                                "nrmse": temporal_error,
                                "raw_nrmse": temporal_raw_error,
                                "image_nrmse": temporal_image_error,
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
                        halvings = round(
                            float(
                                np.log2(0.001 / temporal_scale / base_steps["step_s"])
                            )
                        )
                        # The comparison just made used halving `halvings + 1`;
                        # stop there, as the phase oracle does.
                        if halvings + 1 >= MAX_TEMPORAL_HALVINGS:
                            # Grading has no time limit, so a replay that never
                            # converges must stop at a declared timestep cap.
                            report.update(
                                status="numerically_unresolved",
                                numerically_converged=False,
                                failure_kind="unresolved_at_timestep_cap",
                                reason=(
                                    "Temporal NRMSE stayed above 0.01 after "
                                    f"{MAX_TEMPORAL_HALVINGS} timestep halvings"
                                ),
                                temporal_refinement=temporal_history,
                                progress_checkpoint=str(workspace / "progress.json"),
                            )
                            return report
                        base_steps = {
                            name: step / 2 for name, step in base_steps.items()
                        }
                        physical_sampled = refined_sampled
                        ideal_sampled = (
                            replay(acq, spins, **base_steps)
                            if effects or chemical_shift
                            else physical_sampled
                        )
                # A control variate, not substituted dynamics: only the known Fourier
                # sampling error is removed. All finite-RF/coherence residuals survive.
                if controls:
                    ideal_control, physical_control = controls
                    if pending:
                        ideal_sampled_control, physical_sampled_control = pending[
                            replicate
                        ].result(
                            timeout=None
                            if deadline is None
                            else max(0, deadline - time.monotonic())
                        )
                    else:
                        ideal_sampled_control = ideal_control.sampled(spins)
                        physical_sampled_control = (
                            physical_control.sampled(spins)
                            if physical_control is not ideal_control
                            else ideal_sampled_control
                        )
                    ideal_raw = corrected_raw(
                        ideal_sampled, ideal_sampled_control, ideal_control.known
                    )
                    physical_raw = corrected_raw(
                        physical_sampled,
                        physical_sampled_control,
                        physical_control.known,
                    )
                    ideal_control_runs.append(ideal_sampled_control)
                    physical_control_runs.append(physical_sampled_control)
                else:
                    ideal_raw = reference_raw + ideal_sampled - sampled_reference
                    physical_raw = reference_raw + physical_sampled - sampled_reference
                    sampled_reference_runs.append(sampled_reference)
                koma_ideal_runs.append(ideal_sampled)
                koma_physical_runs.append(physical_sampled)
                ideal_raw_runs.append(ideal_raw)
                physical_raw_runs.append(physical_raw)
                ideal_runs.append(reconstructor(ideal_raw, acq, matrix))
                physical_runs.append(reconstructor(physical_raw, acq, matrix))
                counts.append(len(spins.positions))
                # Do not retain the previous multi-million-node sample while the
                # next independent scramble is generated.
                del spins
        finally:
            cancelled.set()
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            spin_batch = None
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
        if samples >= max_samples or (stage_limit and samples >= stage_limit):
            images_path = workspace / "images.npz"
            atomic_savez_compressed(
                images_path,
                reference=reference,
                case_reference=reference * phasor,
                ideal=ideal_runs.mean(axis=0),
                physical=physical_runs.mean(axis=0),
                ideal_replicates=ideal_runs,
                physical_replicates=physical_runs,
            )
            report["artifacts"]["images"] = str(images_path.resolve())
            report.update(
                status="numerically_unresolved",
                numerically_converged=False,
                failure_kind="numerical_precision_not_reached",
                progress_checkpoint=str(workspace / "progress.json"),
            )
            return report
        samples = min(samples * 2, max_samples)
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
        "temporal_raw_nrmse": temporal_raw_error,
        "temporal_image_nrmse": temporal_image_error,
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
        ideal_replicates=ideal_runs,
        physical_replicates=physical_runs,
        temporal_raw_difference=refined_sampled - koma_physical_runs[0],
        adc_times_s=acq.adc,
        k_cycles_per_m=acq.k,
        matrix=np.asarray(matrix),
        fov_mm=np.asarray(spec["fov_mm"]),
        **(
            {
                "known_control_raw": controls[1].known,
                "ideal_known_control_raw": controls[0].known,
                "sampled_control_raw_replicates": np.asarray(physical_control_runs),
                "ideal_sampled_control_raw_replicates": np.asarray(ideal_control_runs),
            }
            if controls
            else {
                "sampled_reference_raw_replicates": np.asarray(sampled_reference_runs)
            }
        ),
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
            "submission_sha256": submission_sha256,
            "phantom": case["object"]["phantom"],
            "shapes_sha256": mask_spins.source_sha256,
            "koma_versions": versions(),
            "spatial_integration": "material-stratified native-cell CDF below the cell count, otherwise jittered strata within every cell; independent replicates, exact tissue volumes and Fourier control variate; full Bloch residual retained",
            "spatial_control_variate": control_description,
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
                "temporal_raw_nrmse": 0.01,
                "temporal_image_nrmse": 0.01,
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
