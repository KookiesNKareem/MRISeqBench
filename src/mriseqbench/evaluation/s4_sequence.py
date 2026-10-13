"""ECG windows, preparation identity and bSSFP derived from Pulseq events."""

from copy import copy

import numpy as np

from .s1_sequence import check, family, gradient_integrals
from .s1_sequence import inspect_sequence as inspect_cartesian

FAMILIES = {"adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping"}


def _phase(pulse):
    return pulse.phase + 2 * np.pi * pulse.frequency * (pulse.center - pulse.start)


def _phase_error(actual, expected):
    return float(abs(np.angle(np.exp(1j * (actual - expected)))))


def _hs_shape(pulse):
    """Sech amplitude and tanh frequency sweep; RF use/name labels are ignored."""
    mids = (pulse.edges[:-1] + pulse.edges[1:]) / 2
    amplitude = np.abs(pulse.signal)
    if len(amplitude) < 16 or amplitude.max() <= 0:
        return {"valid": False, "reason": "insufficient shaped RF samples"}
    x = 2 * (mids - pulse.start) / (pulse.end - pulse.start) - 1
    normalized = amplitude / amplitude.max()
    betas = np.linspace(1, 15, 281)
    errors = np.array(
        [np.sqrt(np.mean((normalized - 1 / np.cosh(b * x)) ** 2)) for b in betas]
    )
    beta = betas[errors.argmin()]
    frequency = np.gradient(np.unwrap(np.angle(pulse.signal)), mids) / (2 * np.pi)
    basis = np.column_stack((np.ones(len(x)), np.tanh(beta * x)))
    fit = basis @ np.linalg.lstsq(basis, frequency, rcond=None)[0]
    sweep = float(np.ptp(frequency))
    sweep_error = float(np.sqrt(np.mean((frequency - fit) ** 2)) / max(sweep, 1e-15))
    return {
        "valid": bool(errors.min() <= 0.05 and sweep_error <= 0.05 and sweep >= 100),
        "sech_amplitude_rms": float(errors.min()),
        "tanh_sweep_relative_rms": sweep_error,
        "sweep_hz": sweep,
        "beta": float(beta),
    }


def inspect_sequence(path, case):
    acq = inspect_cartesian(path, case)
    spec = case["task"]["sequence"]
    timing = spec["temporal"]
    rr = timing["rr_interval_ms"] / 1000
    delay = timing["trigger_delay_ms"] / 1000
    window = timing["acquisition_window_ms"] / 1000
    tolerance = max(acq.seq.block_duration_raster, acq.seq.rf_raster_time) * 2
    clocks, triggers, adc_offsets, adc_frequencies = 0.0, [], [], []
    for number in acq.seq.block_events:
        block = acq.seq.get_block(number)
        for event in (getattr(block, "trigger", None) or {}).values():
            if event.type == "trigger" and event.channel == "physio1":
                triggers.append(clocks + float(event.delay))
        if block.adc is not None:
            adc_offsets.append(
                float(
                    block.adc.phase_offset
                    + block.adc.phase_ppm
                    * 1e-6
                    * acq.seq.system.gamma
                    * acq.seq.system.B0
                )
            )
            adc_frequencies.append(
                float(
                    block.adc.freq_offset
                    + block.adc.freq_ppm
                    * 1e-6
                    * acq.seq.system.gamma
                    * acq.seq.system.B0
                )
            )
        clocks += acq.seq.block_durations[number]
    triggers = np.asarray(triggers)
    checks = [
        c
        for c in acq.checks
        if c["name"]
        in {
            "rf_before_adc",
            "echo_in_adc",
            "cartesian",
            "matrix",
            "fov_mm",
            "single_line_per_excitation",
        }
    ]
    checks.append(
        check(
            "ecg_trigger_events",
            len(triggers) > 0,
            {"source": "physio1", "trigger_times_s": triggers.tolist()},
        )
    )
    if not len(triggers):
        acq.checks = checks
        acq.measurements["frames"] = []
        return acq
    cadence = np.diff(triggers) / rr
    checks.append(
        check(
            "nominal_rr_cadence",
            np.all(np.abs(cadence - np.rint(cadence)) <= tolerance / rr)
            and np.all(cadence >= 1 - tolerance / rr),
            {"rr_ms": rr * 1000, "trigger_intervals_rr": cadence.tolist()},
        )
    )
    assignments = np.searchsorted(triggers, acq.adc, side="right") - 1
    valid = assignments >= 0
    elapsed = acq.adc - triggers[np.maximum(assignments, 0)]
    checks.append(
        check(
            "cardiac_adc_window",
            valid.all()
            and np.all(elapsed >= delay - tolerance)
            and np.all(elapsed <= delay + window + tolerance),
            {
                "adc_trigger_delay_ms": [
                    float(elapsed.min() * 1000),
                    float(elapsed.max() * 1000),
                ],
                "required_ms": [delay * 1000, (delay + window) * 1000],
            },
        )
    )
    pulse_centers = np.array([p.center for p in acq.pulses])
    acquired_triggers = np.unique(assignments[valid])
    if len(acquired_triggers) != timing["frames"]:
        checks.append(
            check(
                "cardiac_frame_count",
                False,
                {"frames": len(acquired_triggers), "required": timing["frames"]},
            )
        )
        acq.checks = checks
        acq.measurements["frames"] = []
        return acq
    frames = []
    prior_frame_end = 0.0
    prep_centers = []
    for frame_no, trigger_id in enumerate(acquired_triggers):
        line_ids = [
            j
            for j, line in enumerate(acq.lines)
            if np.all(assignments[line] == trigger_id)
        ]
        ids = np.concatenate([acq.lines[j] for j in line_ids])
        readout_ids = [
            int(
                np.searchsorted(pulse_centers, acq.adc[acq.lines[j][0]], side="right")
                - 1
            )
            for j in line_ids
        ]
        readout_times = pulse_centers[readout_ids]
        tr = float(np.median(np.diff(readout_times))) if len(readout_times) > 1 else 0.0
        start = readout_ids[0]
        while (
            start > 0
            and abs(pulse_centers[start] - pulse_centers[start - 1] - tr) <= tolerance
        ):
            start -= 1
        train_start = acq.pulses[start].start
        prep_ids = [
            j
            for j, pulse in enumerate(acq.pulses)
            if prior_frame_end <= pulse.start and pulse.end <= train_start - tolerance
        ]
        preparations = [acq.pulses[j] for j in prep_ids]
        unique = len(np.unique(acq.indices[ids], axis=0))
        checks.append(
            check(
                f"frame_{frame_no}_full_cartesian_grid",
                unique == np.prod(spec["matrix"])
                and len(ids) == np.prod(spec["matrix"]),
                {"unique_samples": unique, "samples": len(ids)},
            )
        )
        starts = np.array([acq.pulses[j].start for j in readout_ids])
        ends = np.r_[starts[1:], acq.pulses[readout_ids[-1]].block_start + tr]
        areas = gradient_integrals(acq.seq, ends) - gradient_integrals(acq.seq, starts)
        fov = np.r_[np.asarray(spec["fov_mm"]) / 1000, 0.2][:3]
        balance = float(np.max(np.abs(areas) * fov))
        phases = np.array([_phase(acq.pulses[j]) for j in readout_ids])
        alternating = (
            np.max(np.abs(np.abs(np.angle(np.exp(1j * np.diff(phases)))) - np.pi))
            <= 0.01
        )
        # Compare event phases in their common carrier frame. Comparing the first
        # ADC sample's accumulated demodulation phase with RF-centre phase would
        # incorrectly reject ordinary matched RF/ADC frequency tuning.
        receiver_error = max(
            _phase_error(adc_offsets[j], acq.pulses[pulse].phase)
            for j, pulse in zip(line_ids, readout_ids, strict=True)
        )
        matched_carrier = all(
            abs(adc_frequencies[j] - acq.pulses[pulse].frequency) <= 1e-6
            for j, pulse in zip(line_ids, readout_ids, strict=True)
        )
        echo_te = np.asarray(acq.measurements["echo_times_s"])[line_ids] - readout_times
        checks.append(
            check(
                f"frame_{frame_no}_balanced_bssfp",
                tr > 0
                and balance <= 0.01
                and alternating
                and receiver_error <= 0.01
                and matched_carrier
                and np.all(np.abs(echo_te - tr / 2) <= max(1e-5, 0.05 * tr)),
                {
                    "tr_ms": tr * 1000,
                    "max_gradient_area_fov": balance,
                    "alternating_rf": bool(alternating),
                    "receiver_phase_error_rad": receiver_error,
                    "matched_rf_adc_frequency": matched_carrier,
                },
            )
        )
        central = int(ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))])
        application = spec["application"]
        if family(case) in {"adiabatic_inversion", "t1_mapping"}:
            target_ti = (
                application["inversion_time_ms"]
                if family(case) == "adiabatic_inversion"
                else application["inversion_times_ms"][frame_no]
            )
            inversion = preparations[-1] if preparations else None
            inversion_ok = (
                inversion is not None
                and len(preparations) == 1
                and inversion.flip >= 170
            )
            measured_ti = (
                (acq.adc[central] - inversion.center) * 1000
                if inversion is not None
                else None
            )
            checks.append(
                check(
                    f"frame_{frame_no}_inversion",
                    inversion_ok and abs(measured_ti - target_ti) <= tolerance * 1000,
                    {
                        "ti_ms": measured_ti,
                        "required_ti_ms": target_ti,
                        "preparation_pulses": len(preparations),
                    },
                )
            )
            if inversion is not None:
                prep_centers.append(inversion.center)
                if family(case) == "adiabatic_inversion":
                    hs = _hs_shape(inversion)
                    checks.append(
                        check(
                            "adiabatic_hyperbolic_secant",
                            hs["valid"]
                            and abs(
                                (inversion.end - inversion.start) * 1000
                                - application["preparation_duration_ms"]
                            )
                            <= tolerance * 1000,
                            hs,
                        )
                    )
        else:
            target = (
                application["preparation_duration_ms"]
                if family(case) == "t2_prepared"
                else application["preparation_durations_ms"][frame_no]
            )
            valid_prep = not preparations if target == 0 else len(preparations) == 4
            measured = None
            if target > 0 and len(preparations) == 4:
                down, ref1, ref2, up = preparations
                measured = (up.center - down.center) * 1000
                interval = up.center - down.center
                valid_prep = (
                    abs(down.flip - 90) <= 2
                    and abs(up.flip - 90) <= 2
                    and ref1.flip >= 178
                    and ref2.flip >= 178
                    and abs(measured - target) <= tolerance * 1000
                    and abs(ref1.center - down.center - interval / 4) <= tolerance
                    and abs(ref2.center - down.center - 3 * interval / 4) <= tolerance
                    and _phase_error(_phase(up), _phase(down) + np.pi) <= 0.01
                    and _phase_error(_phase(ref1), _phase(down) + np.pi / 2) <= 0.01
                    and _phase_error(_phase(ref2), _phase(ref1)) <= 0.01
                )
                prep_centers.append(down.center)
            checks.append(
                check(
                    f"frame_{frame_no}_t2_preparation",
                    valid_prep,
                    {
                        "preparation_ms": measured if target else 0,
                        "required_ms": target,
                        "preparation_pulses": len(preparations),
                    },
                )
            )
        frames.append(
            {
                "adc_indices": ids.tolist(),
                "line_indices": line_ids,
                "central_adc_index": central,
                "preparation_pulse_indices": prep_ids,
                "readout_pulse_indices": readout_ids,
                "train_start_s": train_start,
                "trigger_time_s": float(triggers[trigger_id]),
            }
        )
        prior_frame_end = float(acq.adc[ids[-1]])
    checks.append(
        check(
            "cardiac_frame_count",
            len(frames) == timing["frames"],
            {"frames": len(frames), "required": timing["frames"]},
        )
    )
    if family(case) in {"t1_mapping", "t2_mapping"}:
        separations = np.diff(prep_centers)
        checks.append(
            check(
                "preparation_recovery",
                np.all(separations >= 3 * rr - tolerance),
                {"preparation_intervals_s": separations.tolist(), "min_s": 3 * rr},
            )
        )
    acq.checks = checks
    acq.measurements.update(
        frames=frames, family=family(case), reconstruction="bart_cartesian_fft@1"
    )
    return acq


def frame_acquisition(acq, frame):
    """Subset imaging ADCs while retaining every prior RF preparation and transient."""
    result = copy(acq)
    ids = np.asarray(frame["adc_indices"])
    result.adc, result.adc_phase = acq.adc[ids], acq.adc_phase[ids]
    result.k, result.indices = acq.k[ids], acq.indices[ids]
    result.pulses = [p for p in acq.pulses if p.start < result.adc[-1]]
    inverse = {int(old): new for new, old in enumerate(ids)}
    result.lines = [
        np.array([inverse[int(i)] for i in acq.lines[j]]) for j in frame["line_indices"]
    ]
    result.excitation = acq.excitation[frame["line_indices"]]
    result.checks = []
    result.measurements = {**acq.measurements, "frame": frame, "frames": [frame]}
    return result
