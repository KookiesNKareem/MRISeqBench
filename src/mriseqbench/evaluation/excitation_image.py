"""Image-based spatial excitation: repeated excitation/GRE line/recovery."""

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pypulseq import Opts, Sequence

from ..io import write_json
from ..storage import atomic_savez_compressed
from .bart import reconstruct_cartesian
from .excitation import target_spins
from .excitation_target import excitation_metrics, target_arrays, target_sha256
from .s1_koma import NativeRejection, simulate, versions, worker


def image_acquisition(submission, case):
    """Inspect public readout geometry without treating spatial RF as hard RF."""
    seq = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    seq.read(str(submission))
    matrix = tuple(case["task"]["sequence"]["matrix"])
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    pulses, adc, phases, lines, shots = [], [], [], [], []
    clock, previous_end = 0.0, None
    pending = []
    for number in seq.block_events:
        block = seq.get_block(number)
        if getattr(block, "trigger", None):
            raise ValueError("Spatial-excitation imaging has no trigger timeline")
        rf = block.rf
        if rf is not None and np.any(np.abs(rf.signal) > 0):
            pulse = SimpleNamespace(
                block_number=number,
                start=clock + rf.delay,
                end=clock + rf.delay + rf.shape_dur,
                frequency=float(
                    rf.freq_offset
                    + rf.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
                ),
                phase_correction=float(
                    rf.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
                ),
            )
            if (
                not pending
                and previous_end is not None
                and pulse.start < previous_end + 2.0 - 1e-9
            ):
                raise ValueError(
                    f"Shot {len(lines) + 1}: need 2 s recovery after the preceding ADC block; got {pulse.start - previous_end:.9g} s"
                )
            pending.append(pulse)
            pulses.append(pulse)
        if block.adc is not None:
            a = block.adc
            if not pending:
                raise ValueError(
                    f"Readout {len(lines) + 1} needs a spatial-excitation RF module"
                )
            if a.num_samples != matrix[0]:
                raise ValueError(
                    f"Each Cartesian readout must acquire {matrix[0]} samples; got {a.num_samples}"
                )
            local = (np.arange(a.num_samples) + 0.5) * a.dwell
            times = clock + a.delay + local
            if pending[-1].end > times[0] + 1e-9:
                raise ValueError("RF must end before the readout starts")
            if pending[-1].end - pending[0].start > 0.04 + 1e-9:
                raise ValueError("Each spatial RF module must fit within 40 ms")
            if times[-1] - pending[-1].end > 0.005 + 1e-9:
                raise ValueError(
                    "Acquire the complete readout within 5 ms after the RF module"
                )
            ids = np.arange(len(adc), len(adc) + a.num_samples)
            lines.append(ids)
            adc.extend(times)
            frequency = (
                a.freq_offset + a.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            phase = (
                a.phase_offset + a.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            phases.extend(
                np.broadcast_to(
                    phase
                    + 2 * np.pi * frequency * local
                    + (a.phase_modulation if np.size(a.phase_modulation) else 0),
                    local.shape,
                )
            )
            shots.append(
                {
                    "rf_start_s": pending[0].start,
                    "rf_end_s": pending[-1].end,
                    "adc_end_s": float(times[-1]),
                }
            )
            pending = []
            previous_end = clock + seq.block_durations[number]
        clock += seq.block_durations[number]
    if pending or len(lines) != matrix[1]:
        raise ValueError(
            f"Need exactly {matrix[1]} excitation/readout shots; got {len(lines)} complete shots"
        )
    if previous_end is None or clock - previous_end < 2.0 - 1e-9:
        raise ValueError("The last readout also needs a 2 s recovery tail")
    # Use the public Pulseq RF-center convention for imaging coordinates;
    # arbitrary excitation phase remains free in the magnitude comparison.
    k = np.asarray(seq.calculate_kspace()[0], dtype=float).T
    grid = k[:, :2] * fov + np.asarray(matrix) // 2
    indices = np.rint(grid).astype(int)
    if not np.isfinite(grid).all() or not np.allclose(grid, indices, rtol=0, atol=1e-3):
        raise ValueError(
            "ADC samples must lie on the 128x128 Cartesian grid for the declared 200 mm FOV"
        )
    if np.any(indices < 0) or np.any(indices >= matrix):
        raise ValueError("Readout samples extend outside the declared Cartesian matrix")
    if len(np.unique(indices, axis=0)) != np.prod(matrix):
        raise ValueError("The readout must cover every Cartesian sample exactly once")
    for ids in lines:
        line = indices[ids]
        if len(np.unique(line[:, 1])) != 1 or set(line[:, 0]) != set(range(matrix[0])):
            raise ValueError(
                "Each shot must acquire one complete Cartesian x-readout line"
            )
    acq = SimpleNamespace(
        seq=seq,
        source_path=str(submission),
        pulses=pulses,
        adc=np.asarray(adc),
        adc_phase=np.asarray(phases),
        k=k,
        indices=indices,
        offset=np.zeros(2),
    )
    return acq, shots


def evaluate_excitation_image(submission, case, root, workspace, *, deadline=None):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "case_id": case["case_id"],
        "status": "failed",
        "checks": [],
        "metrics": {},
    }
    try:
        acq, shots = image_acquisition(submission, case)
    except ValueError as exc:
        report["checks"].append(
            {"name": "excitation_image_protocol", "passed": False, "detail": str(exc)}
        )
        return report
    report["checks"].append(
        {
            "name": "excitation_image_protocol",
            "passed": True,
            "detail": "128 spatial-excitation/GRE-line shots, full Cartesian grid and 2 s recovery per shot",
        }
    )
    arrays = target_arrays()
    target = arrays["target_mxy_magnitude"]
    spins, _ = target_spins(
        case["evaluation"].get("phantom_slab_mm"),
        case["evaluation"].get("phantom_z_samples") or 32,
    )
    bounds = case["evaluation"]["metrics"]
    previous, stages, converged = None, [], False
    for step in (10e-6, 5e-6, 2.5e-6, 1.25e-6, 0.625e-6):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                "Spatial-excitation image replay exceeded its grading budget"
            )
        try:
            signal = simulate(
                acq, spins, case["physics"]["effects"], step_s=step, rf_step_s=step
            )
        except NativeRejection as exc:
            report["checks"].append(
                {"name": exc.check, "passed": False, "detail": str(exc)}
            )
            return report
        image = reconstruct_cartesian(signal, acq, target.shape)
        metrics = excitation_metrics(image, target)
        # Save every reconstructed stage for the human monitor; private arrays
        # stay behind the judge boundary. Simulator output remains independent.
        atomic_savez_compressed(
            workspace / "images.npz",
            physical=image,
            reference=target,
            case_reference=target,
            signal=signal,
            k_cycles_per_m=acq.k,
        )
        errors = (
            None
            if previous is None
            else {
                name: max(1e-4, 2 * abs(value - previous[name]))
                for name, value in metrics.items()
            }
        )
        stages.append({"step_s": step, "metrics": metrics, "estimated_error": errors})
        write_json(workspace / "numerics.json", stages)
        report["artifacts"] = {"images": str((workspace / "images.npz").resolve())}
        report["metrics"] = metrics
        write_json(workspace / "result.json", report)
        if errors is not None:
            converged = max(errors.values()) <= 0.002 and all(
                abs(metrics[b["name"]] - b[key]) > errors[b["name"]]
                for b in bounds
                for key in ("min", "max")
                if b[key] is not None
            )
        if converged:
            break
        previous = metrics
    report["checks"].append(
        {"name": "numerical_convergence", "passed": converged, "detail": stages[-1]}
    )
    for bound in bounds:
        value = metrics[bound["name"]]
        report["checks"].append(
            {
                "name": bound["name"],
                "passed": (bound["min"] is None or value >= bound["min"])
                and (bound["max"] is None or value <= bound["max"]),
                "detail": {"value": value, **bound},
            }
        )
    report.update(
        status="passed" if all(c["passed"] for c in report["checks"]) else "failed",
        reconstruction="bart_cartesian_fft@1",
        numerically_converged=converged,
        target_sha256=target_sha256(),
        submission_sha256=hashlib.sha256(Path(submission).read_bytes()).hexdigest(),
        shots=shots,
        versions=versions(),
        execution=worker().execution,
    )
    if not case["evaluation"]["calibrated"] and report["status"] == "passed":
        report["status"] = "uncalibrated"
    if not converged:
        report.update(
            status="numerically_unresolved",
            reason="Image metrics did not resolve under temporal refinement",
        )
    write_json(workspace / "result.json", report)
    return report
