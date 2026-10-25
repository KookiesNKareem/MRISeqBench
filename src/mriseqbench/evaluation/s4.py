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

from ..io import finite_json, write_json
from .bart import provenance as bart_provenance
from .s1 import evaluate_s1, resolve_decision
from .s1_case import converge_phase, rf_phasor, voxel_spins
from .s1_image import expected_receiver_phase, material_masks, rf_only_response
from .s1_koma import inversion_assay, reconstruct, simulate
from .s1_phantom import phantom
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


def preparation_spoils(acq):
    return [
        t
        for t in getattr(acq, "measurements", {}).get("preparation_spoil_times_s", [])
        if t < acq.adc[-1]
    ]


def canonical_response(acq, properties):
    return rf_only_response(acq, properties, spoil_times_s=preparation_spoils(acq))


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


# Adiabatic-inversion B1 robustness grid (relative transmit scale, 0.8-1.2).
ADIABATIC_B1_SCALES = tuple(np.round(np.linspace(0.8, 1.2, 9), 2).tolist())
INVERSION_MIN = 0.90


def assay_offsets(field):
    """17 B0 offsets over [-520, 70] Hz at 3 T, scaled linearly with field."""
    return np.linspace(-520, 70, 17) * field / 3


def inversion_grid(pulse, offsets, b1_scales=(1.0,), assay=None):
    """Inverted Mz fraction for every (B1 scale, B0 offset) pair.

    A B1 scale multiplies the complex RF waveform only; timing, phase and the
    RF frequency sweep are unchanged. Returns an array of shape
    (len(b1_scales), len(offsets)).
    """
    assay = assay or inversion_assay
    rows = []
    for scale in b1_scales:
        scaled = copy(pulse)
        scaled.signal = np.asarray(pulse.signal) * float(scale)
        rows.append(np.asarray(assay(scaled, offsets), dtype=float))
    return np.vstack(rows)


def _inversion_response(pulse, field, b1_scales=(1.0,)):
    """Independent finite-waveform rotation assay, including the RF frequency sweep.

    The unitary assay isolates inversion efficiency; image replay additionally
    includes relaxation, complete RF history, encoding, B0 and chemical shift.
    """
    offsets = assay_offsets(field)
    grid = inversion_grid(pulse, offsets, b1_scales)
    return float(np.min(grid)), offsets.tolist(), grid.min(axis=1).tolist()


def mechanism_probes(acq, case, materials, properties):
    response = canonical_response(acq, properties)
    frame = acq.measurements["frame"]
    prep_ids = frame["preparation_pulse_indices"]
    checks, metrics = [], {}
    myocardial = materials.index("myocardium")
    central = _central(acq)
    usable = float(abs(response[myocardial, central]) / properties[myocardial, 0])
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
            abs(response[myocardial, central] - unprepared[myocardial, central])
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
            # Adiabatic pulses are defined by B1-insensitive inversion; T1-map
            # inversions and T2-prep refocusers need not be adiabatic.
            scales = (
                ADIABATIC_B1_SCALES if family(case) == "adiabatic_inversion" else (1.0,)
            )
            efficiency, offsets, per_scale = _inversion_response(
                acq.pulses[pulse_index],
                case["hardware"]["field_strength_T"],
                scales,
            )
            checks.append(
                check(
                    f"preparation_rf_{pulse_index}_inversion_efficiency",
                    efficiency >= INVERSION_MIN,
                    {
                        "min_inverted_mz_fraction": efficiency,
                        "min": INVERSION_MIN,
                        "assay_offsets_hz": offsets,
                        "assay_b1_scales": list(scales),
                        "min_inverted_mz_fraction_by_b1_scale": per_scale,
                    },
                )
            )
            metrics[f"preparation_rf_{pulse_index}_inversion_efficiency"] = efficiency
    return checks, metrics, response


def _mapping_model(acquisitions, frames, case, root):
    """Fixed anatomical/RF dictionary, independent of spatial replay samples."""
    geometry = phantom(case, root)
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


def _roi_magnitudes(ensembles, exclude=None):
    """ROI-mean magnitude of each frame's complex replicate-ensemble mean.

    Replicates are averaged as complex images before the magnitude is taken,
    so independent spatial-sampling noise does not bias the magnitude upward.
    ``exclude`` drops one replicate (leave-one-out ensemble).
    """
    values = []
    for replicates in ensembles:
        keep = np.ones(len(replicates), bool)
        if exclude is not None:
            keep[exclude] = False
        values.append(np.mean(np.abs(np.mean(replicates[keep], axis=0))))
    return np.asarray(values)


def _fit_relaxation(dictionary, grid, values):
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


def _mapping_estimates(ensembles, dictionary, grid):
    """Point fit from the full complex ensemble; jackknife refits every
    leave-one-out ensemble through the complete magnitude/ROI/fit pipeline."""
    n = len(ensembles[0])
    if n < 2 or any(len(item) != n for item in ensembles):
        raise ValueError("Mapping needs >=2 equal-size replicate ensembles")
    point = _fit_relaxation(dictionary, grid, _roi_magnitudes(ensembles))
    loo = np.array(
        [
            _fit_relaxation(dictionary, grid, _roi_magnitudes(ensembles, exclude=i))
            for i in range(n)
        ]
    )
    return point, loo


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
    ensembles = []
    for report in frames:
        with np.load(report["artifacts"]["images"]) as images:
            ensembles.append(np.asarray(images["physical_replicates"][:, mask]))
    point, loo = _mapping_estimates(ensembles, dictionary, grid)
    relative = abs(point - truth) / truth
    loo_error = abs(loo - truth) / truth
    bounds = interval(relative, loo_error)
    limit = case["evaluation"]["mapping_relative_error_max"]
    precision = case["evaluation"]["mapping_relative_error_precision"]
    radius = max(abs(relative - bounds[0]), abs(bounds[1] - relative))
    passed, basis = resolve_decision(
        [{"name": "relative_error", "min": None, "max": limit}],
        {"relative_error": relative},
        {"relative_error": bounds},
        bool(np.isfinite(radius) and radius <= precision),
    )
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


def _sample_budget(case):
    initial = case["evaluation"].get("spatial_samples") or 32768
    return case["evaluation"].get("spatial_max_samples") or min(4 * initial, 1048576)


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
    probe_case = id(acq), json.dumps(case, sort_keys=True)
    if getattr(acq, "_frame_probe_case", None) != probe_case:
        acq._frame_probe_case = probe_case
        acq._frame_probe_results = {}
        if hasattr(acq, "_phase_reference_batch"):
            del acq._phase_reference_batch
    if not hasattr(acq, "_phase_reference_batch"):
        acq._phase_reference_batch = {
            "acquisition": copy(acq),
            "indices": np.asarray(
                [frame["central_adc_index"] for frame in acq.measurements["frames"]]
            ),
            "images": {},
        }
    phase_batch = acq._phase_reference_batch
    probe_results = acq._frame_probe_results
    for number, frame in enumerate(acq.measurements["frames"]):
        acquisition = frame_acquisition(acq, frame)
        acquisitions.append(acquisition)
        local_case = deepcopy(case)
        if samples is not None:
            local_case["evaluation"]["spatial_samples"] = min(
                samples, _sample_budget(case)
            )

        def frame_probes(*args, frame_number=number):
            if frame_number not in probe_results:
                probe_results[frame_number] = mechanism_probes(*args)
            return deepcopy(probe_results[frame_number])

        result = evaluate_s1(
            submission,
            local_case,
            root,
            Path(workspace) / f"frame-{number}",
            deadline=deadline,
            inspector=lambda _path, _case, a=acquisition: a,
            probes=frame_probes,
            reference_builder=reference_image,
            phase_builder=lambda *args, **kwargs: phase_reference(
                *args, batch=phase_batch, **kwargs
            ),
            encoded_integrator="bloch",
            simulator=simulator,
        )
        results.append(result)
        # A certified failed frame already decides the verdict: later frames
        # cannot restore the conjunction of all frame requirements.
        if (
            not result["sequence_valid"]
            or result.get("status") == "numerically_unresolved"
            or certified_failure(result)
        ):
            break
    return acquisitions, results


def certified_failure(result):
    """A numerically converged frame with a certified failed image check."""
    return bool(result.get("numerically_converged")) and any(
        c.get("passed") is False for c in result["image_checks"]
    )


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
    """Refine mapping within the declared sample cap; uncertainty never passes."""
    cache, history, budget = {}, [], _sample_budget(case)
    while True:
        # A numerically certified failed frame already falsifies the
        # conjunction of image and mapping requirements, at the initial
        # sampling or after a refinement stage. Further mapping refinement
        # cannot change that verdict; retain the measured checks and metrics.
        failed_frames = [
            {
                "frame": i,
                "failed_checks": [
                    c["name"] for c in result["image_checks"] if c["passed"] is False
                ],
            }
            for i, result in enumerate(results)
            if certified_failure(result)
        ]
        if failed_frames:
            return (
                acquisitions,
                results,
                [],
                {},
                {
                    "stop_reason": "certified_image_failure",
                    "mapping_not_evaluated": True,
                    "certified_failed_frames": failed_frames,
                    "acceptance_unchanged": "All image and mapping requirements must pass; a certified image failure suffices to fail",
                },
            )
        if any(result.get("status") == "numerically_unresolved" for result in results):
            return (
                acquisitions,
                results,
                [],
                {},
                {
                    "stop_reason": "frame_numerically_unresolved",
                    "mapping_not_evaluated": True,
                },
            )
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
            "max_samples_per_replicate": budget,
            "refinement_history": history,
        }
        write_json(Path(workspace) / "mapping-progress.json", numerical)
        if all(item["passed"] is not None for item in checks):
            return acquisitions, results, checks, metrics, numerical
        if min(samples) >= budget:
            numerical["stop_reason"] = "spatial_max_samples"
            write_json(Path(workspace) / "mapping-progress.json", numerical)
            return acquisitions, results, checks, metrics, numerical
        count = min(2 * max(samples), budget)
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


def evaluate_s4(submission, case, root, workspace, *, physio_resolution, deadline=None):
    """``physio_resolution`` comes from evaluate(), which resolves the ECG first."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "failed",
        "image_verdict": "not_evaluated",
        "checks": [],
        "metrics": {},
        "sequence_valid": False,
        "ideal_imaging_success": None,
        "imaging_success_under_case_physics": None,
        "image_acceptance_basis": "case_physics",
        "evaluation_contract": f"s4@{case['evaluation']['version']}",
        "reconstruction": "bart_cartesian_fft@1",
        "application_checks": [],
    }
    try:
        acq = inspect_sequence(submission, case)
    except MissingExcitation as exc:
        report["checks"] = [check("excitation_before_adc", False, str(exc))]
        return report
    # Undefined candidate measurements are null, never NaN, in every artifact.
    acq.checks = finite_json(acq.checks)
    acq.measurements = finite_json(acq.measurements)
    report.update(
        checks=acq.checks,
        sequence_valid=all(c["passed"] for c in acq.checks),
        physio=physio_resolution,
    )
    report["checks"].insert(
        0, check("physio_trigger_resolution", True, physio_resolution)
    )
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
    if (
        all(result["sequence_valid"] for result in results)
        and not any(
            result.get("status") == "numerically_unresolved" for result in results
        )
        and family(case) in {"t1_mapping", "t2_mapping"}
    ):
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
    unresolved = [
        number
        for number, result in enumerate(results)
        if result.get("status") == "numerically_unresolved"
    ]
    if unresolved:
        # Keep a frame's timestep-cap cause; otherwise the normalized verdict
        # would wrongly report the spatial sample cap.
        kinds = {number: results[number].get("failure_kind") for number in unresolved}
        temporal = [
            n for n, kind in kinds.items() if kind == "unresolved_at_timestep_cap"
        ]
        report.update(
            status="numerically_unresolved",
            image_verdict="numerically_unresolved",
            numerically_converged=False,
            image_checks=[],
            failure_kind="unresolved_at_timestep_cap"
            if temporal
            else "frame_numerically_unresolved",
            unresolved_frames=unresolved,
            unresolved_frame_failure_kinds={str(n): k for n, k in kinds.items()},
        )
        return report
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
    # As in resolve_decision: one certified failure decides the conjunction,
    # however many other outcomes stayed unresolved at the sample cap.
    if False in all_outcomes:
        report.update(
            status="failed",
            image_verdict="failed",
            imaging_success_under_case_physics=False,
        )
    elif None in all_outcomes:
        report.update(
            status="numerically_unresolved",
            image_verdict="numerically_unresolved",
            numerically_converged=False,
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


def phase_reference(acq, case, root, reference, *, temporal_scale=1, batch=None):
    """Predict field-induced phase at the echo nearest the centre of k-space.

    This is a voxel-centre phase target, with fixed material volume fractions.
    It is deliberately distinct from a continuum encoded-image simulation.
    A refined temporal replay checks numerical error in this reference model.
    Field-dependent phase is retained; motion encoding phase is not excused.
    """
    idx = np.asarray([_central(acq)])
    line = next(j for j, ids in enumerate(acq.lines) if idx[0] in ids)
    source = acq if batch is None else batch["acquisition"]
    nodes = idx if batch is None else batch["indices"]
    if batch is not None:
        context = int(np.searchsorted(source.adc[nodes], acq.adc[idx[0]]))
        if context >= len(nodes) or source.adc[nodes[context]] != acq.adc[idx[0]]:
            raise ValueError("Batched phase reference must select an exact frame echo")
    a = copy(source)
    a.pulses = [p for p in source.pulses if p.start < source.adc[nodes[-1]]]
    a.adc = source.adc[nodes]
    a.adc_phase = expected_receiver_phase(source)[nodes]
    a.k = np.zeros((len(nodes), 3))
    matrix, spins, loc = voxel_spins(case, root)
    spoils = preparation_spoils(a)

    def replay(refinement=1):
        key = temporal_scale, refinement
        images = batch["images"].get(key) if batch is not None else None
        if images is None:
            images = []
            for effects in ((), case["physics"]["effects"]):
                # All frame echo samples share the exact preceding RF history.
                # Later RF cannot affect earlier samples; each frame's scoring
                # and temporal-convergence check remain independent below.
                values = simulate(
                    a,
                    spins,
                    effects,
                    zero_gradients=True,
                    individual=True,
                    step_s=0.001 / temporal_scale / refinement,
                    rf_step_s=20e-6 / temporal_scale / refinement,
                    **({"spoil_times_s": spoils} if spoils else {}),
                )
                if batch is None:
                    image = np.zeros(matrix, complex)
                    np.add.at(image, tuple(loc.T), values.mean(axis=0))
                else:
                    image = np.zeros((len(nodes), *matrix), complex)
                    np.add.at(image, (slice(None), *loc.T), values)
                images.append(image)
            if batch is not None:
                batch["images"][key] = images
        base, physical = (
            images if batch is None else [image[context] for image in images]
        )
        return rf_phasor(base, physical, matrix)

    fine, valid, summary = converge_phase(replay, reference, temporal_scale, case)
    return (
        fine,
        valid,
        {
            "model": "koma_rf_voxel_phase@1",
            "engine": "KomaMRICore",
            "echo_time_s": float(acq.adc[idx[0]]),
            "line": line,
            "sample_indices": idx.tolist(),
            "echo_contexts": 1,
            **summary,
        },
    )
