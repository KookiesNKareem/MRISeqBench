"""Native finite-RF replay of the AI spatial-excitation task; no reconstruction."""

import hashlib
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pypulseq import Opts, Sequence

from ..io import write_json
from .excitation_target import (
    excitation_metrics,
    save_target,
    target_arrays,
    target_sha256,
)
from .s1_koma import simulate, versions, worker
from .s1_phantom import Spins


def diagnostic_acquisition(submission, case, workspace):
    """Append a trusted one-sample ADC; participant ADC/phase never defines scoring."""
    seq = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    seq.read(str(submission))
    pulses = []
    clock = 0.0
    for number in seq.block_events:
        block = seq.get_block(number)
        if block.adc is not None:
            raise ValueError("RF excitation submissions must have no ADC events")
        if getattr(block, "trigger", None):
            raise ValueError("RF excitation has no physiological trigger timeline")
        rf = block.rf
        if rf is not None and np.any(np.abs(rf.signal) > 0):
            pulses.append(
                SimpleNamespace(
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
            )
        clock += seq.block_durations[number]
    if not pulses:
        raise ValueError("RF excitation needs a nonzero RF waveform")
    duration = float(seq.duration()[0])
    # Append only BLOCKS/ADC rows; keep original RF, gradients, shapes and
    # extensions byte-identical. No waveform read/write round trip is allowed.
    text = Path(submission).read_text()
    major = re.search(r"(?m)^major\s+(\d+)", text)
    minor = re.search(r"(?m)^minor\s+(\d+)", text)
    if (
        major is None
        or minor is None
        or int(major[1]) != 1
        or int(minor[1]) not in (4, 5)
    ):
        raise ValueError("RF excitation diagnostic replay requires Pulseq 1.4 or 1.5")
    text = re.sub(r"(?ms)^\[SIGNATURE\]\s*\n.*?(?=^\[|\Z)", "", text)
    adc_id = max(seq.adc_library.data, default=0) + 1

    def append_row(source, section, row):
        match = re.search(rf"(?m)^\[{section}\][ \t]*\n", source)
        if match is None:
            return source.rstrip() + f"\n\n[{section}]\n{row}\n\n"
        end = re.search(r"(?m)^\[", source[match.end() :])
        stop = match.end() + end.start() if end else len(source)
        rows = list(re.finditer(r"(?m)^[ \t]*\d+[^\n]*\n", source[match.end() : stop]))
        insert = match.end() + rows[-1].end() if rows else match.end()
        return source[:insert] + row + "\n" + source[insert:]

    text = append_row(
        text, "BLOCKS", f"{max(seq.block_events) + 1} 1 0 0 0 0 {adc_id} 0"
    )
    adc_row = f"{adc_id} 1 2000 0 " + ("0 0 0 0 0" if int(minor[1]) == 5 else "0 0")
    text = append_row(text, "ADC", adc_row)
    text = re.sub(
        r"(?m)^TotalDuration[^\n]*",
        f"TotalDuration {duration + seq.block_duration_raster:.12g}",
        text,
    )
    path = Path(workspace) / "diagnostic.seq"
    path.write_text(text)
    diagnostic = Sequence(seq.system)
    diagnostic.read(str(path))
    np.testing.assert_allclose(
        diagnostic.adc_times()[0], [duration + 1e-6], atol=1e-12, rtol=0
    )
    return SimpleNamespace(
        seq=diagnostic,
        source_path=str(path),
        pulses=pulses,
        adc=np.array([duration + 1e-6]),
        adc_phase=np.zeros(1),
    )


def target_spins():
    arrays = target_arrays()
    x, y = np.meshgrid(arrays["x_mm"], arrays["y_mm"], indexing="ij")
    mask = arrays["object_mask"]
    positions = np.column_stack((x[mask] / 1000, y[mask] / 1000, np.zeros(mask.sum())))
    return Spins(
        positions,
        np.zeros(mask.sum(), dtype=np.int64),
        ["water"],
        np.array([[1.0, 1.0, 0.1]]),
        1.0,
        (64, 64),
        target_sha256(),
    ), arrays["target_mxy_magnitude"][mask]


def evaluate_excitation(submission, case, root, workspace, *, deadline=None):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "case_id": case["case_id"],
        "status": "failed",
        "checks": [],
        "metrics": {},
    }
    try:
        acq = diagnostic_acquisition(submission, case, workspace)
    except ValueError as exc:
        report["checks"].append(
            {"name": "excitation_only", "passed": False, "detail": str(exc)}
        )
        return report
    report["checks"].append(
        {
            "name": "excitation_only",
            "passed": True,
            "detail": "Nonzero RF, no ADC or triggers",
        }
    )
    save_target(workspace / "target.npz")
    spins, target = target_spins()
    previous = None
    stages = []
    converged = False
    bounds = case["evaluation"]["metrics"]
    for step in (10e-6, 5e-6, 2.5e-6, 1.25e-6, 0.625e-6):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("RF excitation replay exceeded its grading budget")
        response = simulate(
            acq,
            spins,
            case["physics"]["effects"],
            step_s=step,
            rf_step_s=step,
            individual=True,
        )[0]
        metrics = excitation_metrics(response, target)
        np.savez_compressed(
            workspace / f"replay-{len(stages)}.npz",
            positions_m=spins.positions,
            mxy=response,
            target=target,
        )
        errors = (
            {
                name: max(1e-4, 2 * abs(value - previous[name]))
                for name, value in metrics.items()
            }
            if previous is not None
            else None
        )
        stages.append({"step_s": step, "metrics": metrics, "estimated_error": errors})
        write_json(workspace / "numerics.json", stages)
        if errors is not None:
            # Require both temporal stability and a verdict clear of every boundary.
            converged = max(errors.values()) <= 0.002 and all(
                (
                    m["min"] is None
                    or abs(metrics[m["name"]] - m["min"]) > errors[m["name"]]
                )
                and (
                    m["max"] is None
                    or abs(metrics[m["name"]] - m["max"]) > errors[m["name"]]
                )
                for m in bounds
            )
        if converged:
            break
        previous = metrics
    report["metrics"] = metrics
    report["numerically_converged"] = converged
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
        status=("passed" if case["evaluation"]["calibrated"] else "uncalibrated")
        if all(c["passed"] for c in report["checks"])
        else "failed",
        target_sha256=target_sha256(),
        submission_sha256=hashlib.sha256(Path(submission).read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(
            Path(acq.source_path).read_bytes()
        ).hexdigest(),
        sample_time_s=float(acq.adc[0]),
        reconstruction="direct_magnetization@1",
        versions=versions(),
        execution=worker().execution,
    )
    if not converged:
        report.update(
            status="incomplete",
            reason="Excitation metrics did not resolve under temporal refinement",
        )
    write_json(workspace / "result.json", report)
    return report
