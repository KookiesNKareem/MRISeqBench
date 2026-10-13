"""Native prepared cardiac bSSFP judge, with fixed FFT and full RF history.

Every cardiac frame has an independent full-grid target at its central echo.
Encoding gradients, submitted receiver errors and readout modulation never enter
that target. Intrinsic chemical shift remains part of the nominal tissue model.
"""

import hashlib
import json
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar

from ..io import write_json
from .bart import provenance as bart_provenance
from .s1 import evaluate_s1, resolve_decision
from .s1_image import expected_receiver_phase, material_masks
from .s1_koma import reconstruct, simulate
from .s1_phantom import Spins, phantom
from .s1_sampling import analytic_reference, interval
from .s1_sequence import MissingExcitation, check, family
from .s4_sequence import frame_acquisition, inspect_sequence


class FullFrameReplay:
    """Exact native encoded sums shared by ADC subsets of one acquisition.

    Only summed, fully encoded replay uses this cache. RF-only targets/oracles
    retain their separate calls. Cached arrays contain the original ADC samples;
    no interpolation, RF decimation or field/reconstruction correction occurs.
    """

    def __init__(self, acquisition, directory):
        self.acquisition = acquisition
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits, self.calls = 0, 0
        self.identity = hashlib.sha256(
            Path(acquisition.source_path).read_bytes()
            + repr(acquisition.seq.system.B0).encode()
            + Path(__file__).read_bytes()
            + Path(__file__).with_name("s1_koma.py").read_bytes()
            + Path(__file__).with_name("s1_koma.jl").read_bytes()
        ).digest()

    def __call__(self, acquisition, spins, effects=(), **steps):
        if set(steps) - {"step_s", "rf_step_s", "integration_method"} or any(
            effect.get("kind") != "b0" for effect in effects
        ):
            return simulate(acquisition, spins, effects, **steps)
        indices = np.searchsorted(self.acquisition.adc, acquisition.adc)
        prefix = [
            pulse
            for pulse in self.acquisition.pulses
            if pulse.start < acquisition.adc[-1]
        ]
        if (
            np.any(indices >= len(self.acquisition.adc))
            or not np.array_equal(self.acquisition.adc[indices], acquisition.adc)
            or not np.array_equal(self.acquisition.k[indices], acquisition.k)
            or not np.array_equal(
                self.acquisition.adc_phase[indices], acquisition.adc_phase
            )
            or len(prefix) != len(acquisition.pulses)
            or any(a is not b for a, b in zip(prefix, acquisition.pulses, strict=True))
        ):
            return simulate(acquisition, spins, effects, **steps)
        digest = hashlib.sha256(self.identity)
        digest.update(
            json.dumps(
                [effects, steps, spins.materials, spins.source_sha256], sort_keys=True
            ).encode()
        )
        for value in (spins.positions, spins.labels, spins.weight, spins.properties):
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        path = self.directory / f"{digest.hexdigest()}.npy"
        values = None
        if path.exists():
            try:
                values = np.load(path, allow_pickle=False)
                if (
                    values.shape != self.acquisition.adc.shape
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid full-frame encoded replay")
                self.hits += 1
            except (OSError, ValueError, EOFError):
                path.unlink(missing_ok=True)
                values = None
        if values is None:
            values = simulate(self.acquisition, spins, effects, **steps)
            self.calls += 1
            if (
                values.shape != self.acquisition.adc.shape
                or not np.isfinite(values).all()
            ):
                raise RuntimeError(
                    "Native full-frame replay returned invalid encoded samples"
                )
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(path)
        return values[indices].copy()


def _central(acq):
    return int(np.argmin(np.linalg.norm(acq.k, axis=1)))


def canonical_response(acq, properties):
    """Finite-RF Bloch; all preparations/transients and intrinsic fat retained."""
    a = copy(acq)
    a.adc_phase = expected_receiver_phase(acq)
    n = len(properties)
    spins = Spins(
        np.zeros((n, 3)),
        np.arange(n),
        [str(i) for i in range(n)],
        properties,
        np.ones(n),
        (n,),
        "s4-rf-only",
    )
    return simulate(a, spins, zero_gradients=True, individual=True).T


def reference_image(acq, case, root, response, _reference_raw):
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    indices = np.indices(tuple(matrix)).reshape(len(matrix), -1).T
    k = np.zeros((len(indices), 3))
    k[:, : len(matrix)] = (indices - matrix // 2) / (np.asarray(spec["fov_mm"]) / 1000)
    grid = SimpleNamespace(
        adc=np.arange(len(indices)), k=k, indices=indices, offset=np.zeros(len(matrix))
    )
    constant = np.broadcast_to(
        response[:, _central(acq), None], (len(response), len(indices))
    )
    return reconstruct(
        analytic_reference(grid, case, root, constant), grid, tuple(map(int, matrix))
    )


def _inversion_response(pulse, field):
    """Independent finite-waveform rotation assay, including the RF frequency sweep.

    The unitary assay isolates inversion efficiency; image replay additionally
    includes relaxation, complete RF history, encoding, B0 and chemical shift.
    """
    offsets = np.linspace(-520, 70, 17) * field / 3
    magnetization = np.tile([0.0, 0.0, 1.0], (len(offsets), 1))
    mids = (pulse.edges[:-1] + pulse.edges[1:]) / 2
    for signal, dt, mid in zip(pulse.signal, np.diff(pulse.edges), mids, strict=True):
        rf = signal * np.exp(
            1j * (pulse.phase + 2 * np.pi * pulse.frequency * (mid - pulse.start))
        )
        axes = (
            2
            * np.pi
            * dt
            * np.column_stack(
                (
                    np.full(len(offsets), rf.real),
                    np.full(len(offsets), rf.imag),
                    offsets,
                )
            )
        )
        angles = np.linalg.norm(axes, axis=1)
        unit = axes / np.maximum(angles[:, None], 1e-30)
        cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
        magnetization = (
            magnetization * cosine
            - np.cross(unit, magnetization) * sine
            + unit * np.sum(unit * magnetization, axis=1)[:, None] * (1 - cosine)
        )
    return float(np.min(-magnetization[:, 2])), offsets.tolist()


def mechanism_probes(acq, case, _materials, properties):
    response = canonical_response(acq, properties)
    frame = acq.measurements["frame"]
    prep_ids = frame["preparation_pulse_indices"]
    checks, metrics = [], {}
    myocardial = _materials.index("myocardium")
    usable = float(abs(response[myocardial, _central(acq)]) / properties[myocardial, 0])
    checks.append(
        check(
            "usable_echo_signal",
            usable >= 0.01,
            {"fraction_of_equilibrium_pd": usable, "min": 0.01},
        )
    )
    metrics["diagnostic_signal_fraction"] = usable
    if prep_ids:
        a = copy(acq)
        a.pulses = [p for i, p in enumerate(acq.pulses) if i not in prep_ids]
        unprepared = canonical_response(a, properties)
        effect = float(
            abs(
                response[myocardial, _central(acq)]
                - unprepared[myocardial, _central(acq)]
            )
            / properties[myocardial, 0]
        )
        cutoff = case["evaluation"]["preparation_effect_fraction_min"]
        checks.append(
            check(
                "preparation_bloch_effect",
                effect >= cutoff,
                {"fraction_of_equilibrium_pd": effect, "min": cutoff},
            )
        )
        metrics["preparation_effect_fraction"] = effect
        for pulse_index in prep_ids:
            if not acq.pulses[pulse_index].refocusing:
                continue
            efficiency, offsets = _inversion_response(
                acq.pulses[pulse_index], case["hardware"]["field_strength_T"]
            )
            checks.append(
                check(
                    f"preparation_rf_{pulse_index}_inversion_efficiency",
                    efficiency >= 0.90,
                    {
                        "min_inverted_mz_fraction": efficiency,
                        "min": 0.90,
                        "assay_offsets_hz": offsets,
                    },
                )
            )
            metrics[f"preparation_rf_{pulse_index}_inversion_efficiency"] = efficiency
    return checks, metrics, response


def _mapping_model(acquisitions, frames, case, root):
    """Fixed anatomical/RF dictionary, independent of spatial replay samples."""
    geometry = phantom(case, root, 4)
    mask = material_masks(geometry, case["task"]["sequence"])["myocardium"]
    if not mask.any():
        return None
    label = geometry.materials.index("myocardium")
    axis = 1 if family(case) == "t1_mapping" else 2
    truth = float(geometry.properties[label, axis])
    grid = np.geomspace(0.1 if axis == 1 else 0.005, 4.0 if axis == 1 else 0.3, 241)
    props = np.tile(geometry.properties[label], (len(grid), 1))
    props[:, axis] = grid
    spectra = []
    for acq, report in zip(acquisitions, frames, strict=True):
        a = copy(acq)
        i = _central(acq)
        a.adc = acq.adc[[i]]
        a.adc_phase = expected_receiver_phase(acq)[[i]]
        a.k = np.zeros((1, 3))
        a.pulses = [pulse for pulse in acq.pulses if pulse.start < a.adc[-1]]
        coefficients = canonical_response(a, props)[:, 0]
        fixed = canonical_response(a, geometry.properties)[label, 0]
        unit_response = np.zeros((len(geometry.materials), len(acq.adc)), complex)
        unit_response[label] = 1
        basis = reference_image(acq, case, root, unit_response, None)
        with np.load(report["artifacts"]["images"]) as images:
            # Exact anatomical-cell Fourier basis preserves partial volumes and
            # the standard FFT's fixed PSF. The other tissues keep their clean
            # central-echo reference; encoded readout dynamics never enter here.
            other = images["reference"] - fixed * basis
            model = (
                other[None]
                + coefficients.reshape((-1,) + (1,) * basis.ndim) * basis[None]
            )
            spectra.append(np.mean(np.abs(model[:, mask]), axis=1))
    return {
        "dictionary": np.asarray(spectra),
        "grid": grid,
        "mask": mask,
        "truth": truth,
        "axis": axis,
    }


def _mapping_decision(relative, bounds, limit, precision):
    radius = max(abs(relative - bounds[0]), abs(bounds[1] - relative))
    precise = bool(np.isfinite(radius) and radius <= precision)
    outcome, basis = resolve_decision(
        [{"name": "relative_error", "min": None, "max": limit}],
        {"relative_error": relative},
        {"relative_error": bounds},
        precise,
    )
    return outcome, basis, radius


def _mapping(acquisitions, frames, case, root, *, cache=None):
    """Fit one relaxation parameter, with declared numerical precision.

    No gain, phase, registration or field correction is fitted. Acceptance is
    unchanged; precision controls when a point score can resolve an interval
    straddling the limit, using the shared S1 numerical decision convention.
    """
    cache = {} if cache is None else cache
    if "model" not in cache:
        cache["model"] = _mapping_model(acquisitions, frames, case, root)
    model = cache["model"]
    if model is None:
        return [
            check("myocardial_mapping_region", False, "No interior myocardial ROI")
        ], {}
    dictionary, grid, mask, truth, axis = (
        model[name] for name in ("dictionary", "grid", "mask", "truth", "axis")
    )
    signals = []
    for report in frames:
        with np.load(report["artifacts"]["images"]) as images:
            signals.append(
                np.mean(np.abs(images["physical_replicates"][:, mask]), axis=1)
            )
    observations = np.asarray(signals)

    def estimate(values):
        residuals = np.sum((dictionary - values[:, None]) ** 2, axis=0)
        best = int(np.argmin(residuals))
        lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        fit = minimize_scalar(
            lambda value: sum(
                (np.interp(value, grid, row) - observation) ** 2
                for row, observation in zip(dictionary, values, strict=True)
            ),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-8},
        )
        return float(fit.x)

    n = observations.shape[1]
    total = observations.sum(axis=1)
    point = estimate(total / n)
    loo = np.array([estimate((total - observations[:, i]) / (n - 1)) for i in range(n)])
    relative = abs(point - truth) / truth
    loo_error = abs(loo - truth) / truth
    bounds = interval(relative, loo_error)
    limit = case["evaluation"]["mapping_relative_error_max"]
    precision = case["evaluation"]["mapping_relative_error_precision"]
    passed, basis, radius = _mapping_decision(relative, bounds, limit, precision)
    name = (
        "myocardial_t1_relative_error" if axis == 1 else "myocardial_t2_relative_error"
    )
    return [
        {
            "name": name,
            "passed": passed,
            "detail": {
                "estimate_ms": point * 1000,
                "truth_ms": truth * 1000,
                "relative_error": relative,
                "numerical_interval": list(bounds),
                "numerical_precision": radius,
                "precision_target": precision,
                "decision_basis": basis,
                "max": limit,
                "fitting": "one relaxation parameter; known PD and other relaxation time; no fitted gain/phase/B0",
            },
        }
    ], {name: relative, name.replace("relative_error", "estimate_ms"): point * 1000}


def _evaluate_frames(
    acq,
    submission,
    case,
    root,
    workspace,
    *,
    deadline=None,
    samples=None,
    simulator=None,
):
    acquisitions, results = [], []
    for number, frame in enumerate(acq.measurements["frames"]):
        acquisition = frame_acquisition(acq, frame)
        acquisitions.append(acquisition)
        local_case = deepcopy(case)
        if samples is not None:
            local_case["evaluation"]["spatial_samples"] = samples
        result = evaluate_s1(
            submission,
            local_case,
            root,
            Path(workspace) / f"frame-{number}",
            deadline=deadline,
            inspector=lambda _path, _case, a=acquisition: a,
            probes=mechanism_probes,
            reference_builder=reference_image,
            phase_builder=phase_reference,
            encoded_integrator="bloch",
            simulator=simulator,
        )
        results.append(result)
        if not result["sequence_valid"]:
            break
    return acquisitions, results


def _mapping_refinement(
    acq,
    submission,
    acquisitions,
    results,
    case,
    root,
    workspace,
    *,
    deadline=None,
    simulator=None,
):
    """Refine independent quadrature until mapping precision or its verdict resolves.

    The maximum sample setting is the shared evaluator's initial budget, not an
    acceptance shortcut. Each stage has separate frame artifacts; checkpointed
    progress records the unresolved interval before starting further simulation.
    """
    cache, history = {}, []
    initial = case["evaluation"].get("spatial_samples") or 32768
    budget = case["evaluation"].get("spatial_max_samples") or min(4 * initial, 1048576)
    while True:
        checks, metrics = _mapping(acquisitions, results, case, root, cache=cache)
        samples = [
            result["spatial_sampling"]["samples_per_replicate"] for result in results
        ]
        history.append(
            {
                "samples_per_replicate": samples,
                "checks": checks,
                "metrics": metrics,
                "frame_artifacts": [result.get("artifacts", {}) for result in results],
            }
        )
        numerical = {
            "precision_target": case["evaluation"]["mapping_relative_error_precision"],
            "initial_budget_per_replicate": budget,
            "exceeded_initial_budget": any(
                max(stage["samples_per_replicate"]) > budget for stage in history
            ),
            "refinement_history": history,
        }
        write_json(Path(workspace) / "mapping-progress.json", numerical)
        if all(item["passed"] is not None for item in checks):
            return acquisitions, results, checks, metrics, numerical
        count = 2 * max(samples)
        numerical["next_samples_per_replicate"] = count
        write_json(Path(workspace) / "mapping-progress.json", numerical)
        acquisitions, results = _evaluate_frames(
            acq,
            submission,
            case,
            root,
            Path(workspace) / f"mapping-stage-{len(history)}",
            deadline=deadline,
            samples=count,
            simulator=simulator,
        )
        if not all(result["sequence_valid"] for result in results):
            return acquisitions, results, checks, metrics, numerical


def evaluate_s4(submission, case, root, workspace, *, deadline=None):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        acq = inspect_sequence(submission, case)
    except MissingExcitation as exc:
        return {
            "status": "failed",
            "image_verdict": "not_evaluated",
            "checks": [check("excitation_before_adc", False, str(exc))],
            "metrics": {},
            "sequence_valid": False,
            "application_checks": [],
            "ideal_imaging_success": None,
            "imaging_success_under_case_physics": None,
            "image_acceptance_basis": "case_physics",
            "evaluation_contract": "s4@1",
            "reconstruction": "bart_cartesian_fft@1",
        }
    report = {
        "status": "failed",
        "image_verdict": "not_evaluated",
        "checks": acq.checks,
        "metrics": {},
        "sequence_valid": all(c["passed"] for c in acq.checks),
        "ideal_imaging_success": None,
        "imaging_success_under_case_physics": None,
        "image_acceptance_basis": "case_physics",
        "evaluation_contract": "s4@1",
        "reconstruction": "bart_cartesian_fft@1",
        "application_checks": [],
    }
    write_json(workspace / "sequence.json", acq.measurements)
    if not report["sequence_valid"]:
        return report
    encoded_replay = (
        FullFrameReplay(acq, workspace / "full-encoded-replays")
        if len(acq.measurements["frames"]) > 1
        else None
    )
    acquisitions, results = _evaluate_frames(
        acq,
        submission,
        case,
        root,
        workspace,
        deadline=deadline,
        simulator=encoded_replay,
    )
    mapping_checks, mapping_metrics = [], {}
    if all(result["sequence_valid"] for result in results) and family(case) in {
        "t1_mapping",
        "t2_mapping",
    }:
        acquisitions, results, mapping_checks, mapping_metrics, numerical = (
            _mapping_refinement(
                acq,
                submission,
                acquisitions,
                results,
                case,
                root,
                workspace,
                deadline=deadline,
                simulator=encoded_replay,
            )
        )
        report["mapping_numerical_precision"] = numerical
    report["frames"] = results
    report["sequence_valid"] = all(result["sequence_valid"] for result in results)
    for number, result in enumerate(results):
        report["metrics"].update(
            {
                f"frame_{number}_{name}": value
                for name, value in result["metrics"].items()
            }
        )
        inversion_efficiencies = [
            value
            for name, value in result["metrics"].items()
            if name.startswith("preparation_rf_")
            and name.endswith("inversion_efficiency")
        ]
        if inversion_efficiencies:
            report["metrics"][
                f"frame_{number}_min_preparation_rf_inversion_efficiency"
            ] = min(inversion_efficiencies)
        report["application_checks"].extend(
            {**item, "name": f"frame_{number}_{item['name']}"}
            for item in result["checks"]
        )
    report["application_checks"].extend(mapping_checks)
    report["metrics"].update(mapping_metrics)
    if not report["sequence_valid"]:
        if any("images" in frame.get("artifacts", {}) for frame in results):
            report["reconstruction_provenance"] = bart_provenance()
        return report
    report["image_checks"] = [
        {**item, "name": f"frame_{number}_{item['name']}"}
        for number, result in enumerate(results)
        for item in result["image_checks"]
    ]
    report["numerically_converged"] = all(
        result["numerically_converged"] for result in results
    )
    application = [item["passed"] for item in report["application_checks"]]
    image = [item["passed"] for item in report["image_checks"]]
    all_outcomes = application + image
    if None in all_outcomes:
        report.update(
            status="numerically_unresolved",
            image_verdict="numerically_unresolved",
            numerically_converged=False,
        )
    elif not all(all_outcomes):
        report.update(
            status="failed",
            image_verdict="failed",
            imaging_success_under_case_physics=False,
        )
    elif case["evaluation"]["calibrated"]:
        report.update(
            status="passed",
            image_verdict="passed",
            imaging_success_under_case_physics=True,
            ideal_imaging_success=all(
                all(c["passed"] for c in result["ideal_image_checks"])
                for result in results
            ),
        )
    else:
        report.update(status="uncalibrated", image_verdict="uncalibrated")
    report["s4_implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("s4_sequence.py").read_bytes()
        + Path(__file__).with_name("bart.py").read_bytes()
    ).hexdigest()
    # Include dictionary FFT calls as well as every reconstructed cardiac frame.
    report["reconstruction_provenance"] = bart_provenance()
    if encoded_replay is not None:
        report["encoded_replay_cache"] = {
            "model": "exact_full_acquisition_adc_subsets@1",
            "native_full_replays": encoded_replay.calls,
            "cache_hits": encoded_replay.hits,
            "directory": str(encoded_replay.directory.resolve()),
        }
    write_json(workspace / "s4.json", report)
    return report


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
        for effects in ((), case["physics"]["effects"]):
            # Both baselines retain intrinsic fat shift. Only the physical
            # oracle adds declared scanner B0, without changing tissue magnitude.
            replay_spins = copy(spins)
            values = simulate(
                a,
                replay_spins,
                effects,
                zero_gradients=True,
                individual=True,
                step_s=0.001 / temporal_scale / refinement,
                rf_step_s=20e-6 / temporal_scale / refinement,
                phase_windows=phase_windows,
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
            "model": "koma_rf_acquisition_phase@1"
            if sample_indices is not None
            else "koma_rf_voxel_phase@1",
            "engine": "KomaMRICore",
            "echo_time_s": float(a.adc[0]),
            "line": line,
            "sample_indices": idx.tolist(),
            "echo_contexts": len(phase_windows) if phase_windows is not None else 1,
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
