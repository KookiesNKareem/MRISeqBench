"""ECG windows, preparation identity and bSSFP derived from Pulseq events."""

from copy import copy
from types import SimpleNamespace

import numpy as np
from pypulseq import Sequence
from scipy.optimize import least_squares

from .s1_sequence import check, family, gradient_integrals
from .s1_sequence import inspect_sequence as inspect_cartesian


def _inversion_time_matches(measured_ms, target_ms, tolerance_ms):
    """Inclusive TI bound with a 1 ps guard for accumulated float arithmetic.

    This is numerical slack, far below the file rasters; it does not replace
    the physical tolerance or round a genuinely out-of-tolerance TI onto it.
    """
    return measured_ms is not None and abs(measured_ms - target_ms) <= (
        tolerance_ms + 1e-9
    )


def _phase(pulse):
    return pulse.phase + 2 * np.pi * pulse.frequency * (pulse.center - pulse.start)


def _phase_error(actual, expected):
    return float(abs(np.angle(np.exp(1j * (actual - expected)))))


def classify_preparation(pulses, window_ids, readout_flip, *, t2):
    """Assign pre-train RF a role from its waveform, never from labels.

    Preparation is an inversion-class pulse (nominal flip >=170 degrees) or,
    for T2 families, the first 90/>=178/>=178/90 consecutive block. Pulses no
    larger than the train's own flips (+2 degrees) are conditioning, e.g.
    alpha/2 catalyzation, ramp-up or flip-back. Other RF requires a measured
    RF-free crusher interval to establish an additional spoiled preparation.
    """
    flips = [pulses[j].flip for j in window_ids]
    prep = []
    if t2:
        for i in range(len(window_ids) - 3):
            down, ref1, ref2, up = flips[i : i + 4]
            if (
                abs(down - 90) <= 2
                and ref1 >= 178
                and ref2 >= 178
                and abs(up - 90) <= 2
            ):
                prep = window_ids[i : i + 4]
                break
    else:
        prep = [j for j, flip in zip(window_ids, flips) if flip >= 170]
    rest = [j for j in window_ids if j not in prep]
    conditioning = [
        j for j in rest if pulses[j].flip <= readout_flip + 2 and pulses[j].flip < 170
    ]
    unclassified = [j for j in rest if j not in conditioning]
    return prep, conditioning, unclassified


# A crusher must wrap transverse magnetization by >=2 full cycles across one
# in-plane image voxel (|m| * FOV / N >= 2 on x or y). The crushed coherence's
# echo is then displaced by twice the full sampled k-space width, i.e. >=1.5
# widths beyond its edge, so the ideal-dephasing oracle is a sound model.
# Through-plane (z) moments do not dephase a single-slice 2D object.
CRUSHER_MIN_CYCLES_PER_VOXEL = 2.0
CRUSHER_RELATIVE_SLACK = 1e-6


def crusher_cycles_per_voxel(area, spec):
    """Cycles of phase across one image voxel along each in-plane axis."""
    voxel = np.asarray(spec["fov_mm"], float) / 1000 / np.asarray(spec["matrix"])
    return np.abs(np.asarray(area)[: len(voxel)]) * voxel


def spoiled_preparations(acq, candidates, spec):
    """Additional prep RF may dephase at its following RF-free crusher only.

    The crusher must reach CRUSHER_MIN_CYCLES_PER_VOXEL along an in-plane axis;
    weaker RF-free moments do not establish a spoiled preparation.
    """
    ids, times = [], []
    for j in candidates:
        if j + 1 >= len(acq.pulses):
            continue
        pulse, following = acq.pulses[j : j + 2]
        area = np.diff(
            gradient_integrals(acq.seq, [pulse.end, following.start]), axis=0
        )[0]
        cycles = crusher_cycles_per_voxel(area, spec)
        if np.max(cycles) >= CRUSHER_MIN_CYCLES_PER_VOXEL * (
            1 - CRUSHER_RELATIVE_SLACK
        ):
            ids.append(j)
            times.append(float(following.start))
    return ids, times


def _hs_shape(pulse):
    """Sech amplitude and tanh frequency sweep; RF use/name labels are ignored."""
    mids = (pulse.edges[:-1] + pulse.edges[1:]) / 2
    amplitude = np.abs(pulse.signal)
    if len(amplitude) < 3 or amplitude.max() <= 0:
        return {"valid": False, "reason": "insufficient shaped RF samples"}
    x = 2 * (mids - pulse.start) / (pulse.end - pulse.start) - 1
    normalized = amplitude / amplitude.max()
    # Infer a positive shape scale from the observed envelope, rather than
    # limiting valid HS pulses to a hand-picked beta range. Normalize the model
    # on the same sample lattice, whose centre need not contain x=0.
    observed = (normalized > 0) & (normalized < 1) & (np.abs(x) > 0)
    scales = np.arccosh(1 / normalized[observed]) / np.abs(x[observed])
    scales = scales[np.isfinite(scales) & (scales > 0)]
    initial = float(np.median(scales)) if len(scales) else np.sqrt(np.finfo(float).eps)

    def envelope(beta):
        decay = np.exp(-np.abs(beta * x))
        values = 2 * decay / (1 + decay**2)
        return values / values.max()

    fit = least_squares(
        lambda value: envelope(value[0]) - normalized,
        [initial],
        bounds=(np.finfo(float).tiny, np.inf),
    )
    beta = float(fit.x[0])
    amplitude_error = float(np.sqrt(np.mean(fit.fun**2)))
    frequency = np.gradient(np.unwrap(np.angle(pulse.signal)), mids) / (2 * np.pi)
    basis = np.column_stack((np.ones(len(x)), np.tanh(beta * x)))
    fit = basis @ np.linalg.lstsq(basis, frequency, rcond=None)[0]
    sweep = float(np.ptp(frequency))
    sweep_error = float(np.sqrt(np.mean((frequency - fit) ** 2)) / max(sweep, 1e-15))
    return {
        "valid": bool(amplitude_error <= 0.05 and sweep_error <= 0.05 and sweep > 0),
        "sech_amplitude_rms": amplitude_error,
        "tanh_sweep_relative_rms": sweep_error,
        "sweep_hz": sweep,
        "beta": beta,
    }


def _failed(acq, checks):
    acq.checks = checks
    acq.measurements["frames"] = []
    return acq


def inspect_sequence(path, case):
    seq = Sequence()
    seq.read(str(path))
    empty = [
        int(number)
        for number, events in seq.block_events.items()
        if events[5] and float(seq.adc_library.data[int(events[5])][0]) <= 0
    ]
    if empty:
        # Zero-sample ADC events make every Cartesian reduction undefined.
        detail = {"blocks_with_nonpositive_samples": empty[:20], "min": 1}
        return SimpleNamespace(
            checks=[check("adc_sample_count", False, detail)],
            measurements={"frames": []},
        )
    acq = inspect_cartesian(path, case)
    spec = case["task"]["sequence"]
    timing = spec["temporal"]
    segments_per_frame = timing.get("segments_per_frame", 1)
    rr = timing["rr_interval_ms"] / 1000
    delay = timing["trigger_delay_ms"] / 1000
    window = timing["acquisition_window_ms"] / 1000
    tolerance = max(acq.seq.block_duration_raster, acq.seq.rf_raster_time) * 2
    system = acq.seq.system
    clocks, triggers, adc_offsets, adc_frequencies = 0.0, [], [], []
    for number in acq.seq.block_events:
        block = acq.seq.get_block(number)
        for event in (getattr(block, "trigger", None) or {}).values():
            if event.type == "trigger" and event.channel == "physio1":
                triggers.append(clocks + float(event.delay))
        if (adc := block.adc) is not None:
            gamma, b0 = system.gamma, system.B0
            adc_offsets.append(
                float(adc.phase_offset + adc.phase_ppm * 1e-6 * gamma * b0)
            )
            adc_frequencies.append(
                float(adc.freq_offset + adc.freq_ppm * 1e-6 * gamma * b0)
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
        check("imaging_adc_samples", len(acq.adc) > 0, {"samples": len(acq.adc)})
    )
    checks.append(
        check(
            "ecg_trigger_events",
            len(triggers) > 0,
            {"source": "physio1", "trigger_times_s": triggers.tolist()},
        )
    )
    if not all(item["passed"] for item in checks):
        # Established structural failures make dependent timing/role analysis
        # undefined; return them as a failed verdict rather than reducing over
        # possibly empty or wrapped indices.
        return _failed(acq, checks)
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
    if len(acquired_triggers) != timing["frames"] * segments_per_frame:
        checks.append(
            check(
                "cardiac_frame_count",
                False,
                {
                    "acquired_windows": len(acquired_triggers),
                    "required_windows": timing["frames"] * segments_per_frame,
                },
            )
        )
        return _failed(acq, checks)
    frames = []
    # Lines per frame (all non-readout grid points) split evenly over segments.
    segment_lines = max(2, int(np.prod(spec["matrix"][1:])) // segments_per_frame)
    preparation_spoil_times = []
    prior_frame_end = 0.0
    prep_centers = []
    for segment_no, trigger_id in enumerate(acquired_triggers):
        frame_no = segment_no // segments_per_frame
        check_start = len(checks)
        line_ids = [
            j
            for j, line in enumerate(acq.lines)
            if np.all(assignments[line] == trigger_id)
        ]
        if len(line_ids) != segment_lines:
            # Each segment acquires exactly its declared share of lines (also
            # guarantees >=2 lines for TR, phase cycling and balance).
            checks.append(
                check(
                    f"frame_{frame_no}_segment_{segment_no % segments_per_frame}"
                    "_acquisition",
                    False,
                    {
                        "complete_lines_in_window": len(line_ids),
                        "required_lines": segment_lines,
                        "trigger_time_s": float(triggers[trigger_id]),
                    },
                )
            )
            return _failed(acq, checks)
        ids = np.concatenate([acq.lines[j] for j in line_ids])
        readout_ids = [
            int(
                np.searchsorted(pulse_centers, acq.adc[acq.lines[j][0]], side="right")
                - 1
            )
            for j in line_ids
        ]
        readout_times = pulse_centers[readout_ids]
        tr = float(np.median(np.diff(readout_times)))
        start = readout_ids[0]
        while (
            start > 0
            and abs(pulse_centers[start] - pulse_centers[start - 1] - tr) <= tolerance
        ):
            start -= 1
        train_start = acq.pulses[start].start
        window_ids = [
            j
            for j, pulse in enumerate(acq.pulses)
            if prior_frame_end <= pulse.start and pulse.end <= train_start - tolerance
        ]
        prep_ids, conditioning_ids, unclassified_ids = classify_preparation(
            acq.pulses,
            window_ids,
            max(acq.pulses[j].flip for j in readout_ids),
            t2=family(case) in {"t2_prepared", "t2_mapping"},
        )
        # Additional spoiled preparation is compatible with coherent bSSFP.
        # Infer its crusher from the RF-free gradient moment; RF labels, carrier
        # frequency and pulse shape do not establish this role.
        spoiled_ids, spoil_times = spoiled_preparations(acq, unclassified_ids, spec)
        preparation_spoil_times.extend(spoil_times)
        unclassified_ids = [j for j in unclassified_ids if j not in spoiled_ids]
        preparations = [acq.pulses[j] for j in prep_ids]
        roles = {
            "preparation_pulse_indices": prep_ids,
            "conditioning_pulse_indices": conditioning_ids,
            "unclassified_pulse_indices": unclassified_ids,
            "unclassified_flips_deg": [acq.pulses[j].flip for j in unclassified_ids],
            "spoiled_preparation_pulse_indices": spoiled_ids,
        }
        starts = np.array([acq.pulses[j].start for j in readout_ids])
        # Every TR, including the last, spans readout RF start to RF start + TR.
        ends = np.r_[starts[1:], starts[-1] + tr]
        areas = gradient_integrals(acq.seq, ends) - gradient_integrals(acq.seq, starts)
        fov = np.r_[np.asarray(spec["fov_mm"]) / 1000, 0.2][:3]
        balance = float(np.max(np.abs(areas) * fov))
        phases = np.array([_phase(acq.pulses[j]) for j in readout_ids])
        alternating = (
            np.max(np.abs(np.abs(np.angle(np.exp(1j * np.diff(phases)))) - np.pi))
            <= 0.01
        )
        # The ADC phase offset must equal the effective RF phase at the pulse
        # centre (event phase, envelope phase and the frequency offset's accrual
        # from pulse start to centre), the convention COMMON_EVALUATION.md states
        # and expected_receiver_phase demodulates with. The ADC's own frequency
        # accrual is not included: comparing the first sample's accumulated
        # demodulation phase would reject ordinary matched RF/ADC tuning.
        receiver_error = max(
            _phase_error(adc_offsets[j], _phase(acq.pulses[pulse]))
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
                and matched_carrier,
                {
                    "tr_ms": tr * 1000,
                    "max_gradient_area_fov": balance,
                    "alternating_rf": bool(alternating),
                    "receiver_phase_error_rad": receiver_error,
                    "matched_rf_adc_frequency": matched_carrier,
                    "echo_te_ms": (echo_te * 1000).tolist(),
                    "limits": {
                        "max_gradient_area_fov": 0.01,
                        "phase_error_rad": 0.01,
                        "carrier_error_hz": 1e-6,
                    },
                    "conditions": {
                        "positive_tr": bool(tr > 0),
                        "balanced_gradients": balance <= 0.01,
                        "alternating_rf": bool(alternating),
                        "matched_receiver_phase": receiver_error <= 0.01,
                        "matched_rf_adc_frequency": matched_carrier,
                    },
                },
            )
        )
        central = int(ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))])
        # TI uses the same interpolated k-space-centre echo time as TE. The
        # nearest ADC sample is up to half a dwell away on a half-sample lattice.
        central_echo = float(
            acq.measurements["echo_times_s"][
                next(j for j in line_ids if central in acq.lines[j])
            ]
        )
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
                and not unclassified_ids
            )
            measured_ti = (
                (central_echo - inversion.center) * 1000
                if inversion is not None
                else None
            )
            ti_ok = _inversion_time_matches(measured_ti, target_ti, tolerance * 1000)
            checks.append(
                check(
                    f"frame_{frame_no}_inversion",
                    inversion_ok and ti_ok,
                    {
                        "ti_ms": measured_ti,
                        "required_ti_ms": target_ti,
                        "preparation_pulses": len(preparations),
                        **roles,
                        "nominal_flip_deg": inversion.flip
                        if inversion is not None
                        else None,
                        "min_nominal_flip_deg": 170,
                        "ti_error_ms": abs(measured_ti - target_ti)
                        if measured_ti is not None
                        else None,
                        "ti_tolerance_ms": tolerance * 1000,
                        "ti_numerical_slack_ms": 1e-9,
                        "conditions": {
                            "exactly_one_inversion": len(preparations) == 1,
                            "no_unclassified_rf": not unclassified_ids,
                            "inversion_time": ti_ok,
                        },
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
            valid_prep = not unclassified_ids and (
                not preparations if target == 0 else len(preparations) == 4
            )
            measured = None
            if target > 0 and len(preparations) == 4:
                down, ref1, ref2, up = preparations
                measured = (up.center - down.center) * 1000
                interval = up.center - down.center
                phase_errors = [
                    _phase_error(_phase(up), _phase(down) + np.pi),
                    _phase_error(_phase(ref1), _phase(down) + np.pi / 2),
                    _phase_error(_phase(ref2), _phase(ref1)),
                ]
                valid_prep = valid_prep and (
                    abs(down.flip - 90) <= 2
                    and abs(up.flip - 90) <= 2
                    and ref1.flip >= 178
                    and ref2.flip >= 178
                    and abs(measured - target) <= tolerance * 1000
                    and abs(ref1.center - down.center - interval / 4) <= tolerance
                    and abs(ref2.center - down.center - 3 * interval / 4) <= tolerance
                    and all(error <= 0.01 for error in phase_errors)
                )
                prep_centers.append(down.center)
            detail = {
                "preparation_ms": measured if target else 0,
                "required_ms": target,
                "preparation_pulses": len(preparations),
                **roles,
                "nominal_flips_deg": [p.flip for p in preparations],
                "timing_tolerance_ms": tolerance * 1000,
                "tip_flip_tolerance_deg": 2,
                "min_refocusing_flip_deg": 178,
                "phase_tolerance_rad": 0.01,
                "conditions": {
                    "preparation_event_count": len(preparations)
                    == (4 if target > 0 else 0),
                    "no_unclassified_rf": not unclassified_ids,
                },
            }
            if target > 0 and len(preparations) == 4:
                quarter_errors = [
                    (ref1.center - down.center - interval / 4) * 1000,
                    (ref2.center - down.center - 3 * interval / 4) * 1000,
                ]
                detail.update(
                    refocuser_center_errors_ms=quarter_errors,
                    phase_errors_rad=phase_errors,
                    preparation_duration_error_ms=abs(measured - target),
                )
                detail["conditions"].update(
                    {
                        "tip_down_flip": abs(down.flip - 90) <= 2,
                        "tip_up_flip": abs(up.flip - 90) <= 2,
                        "first_refocusing_flip": ref1.flip >= 178,
                        "second_refocusing_flip": ref2.flip >= 178,
                        "duration": abs(measured - target) <= tolerance * 1000,
                        "first_refocuser_center": abs(quarter_errors[0])
                        <= tolerance * 1000,
                        "second_refocuser_center": abs(quarter_errors[1])
                        <= tolerance * 1000,
                        "tip_up_phase": phase_errors[0] <= 0.01,
                        "first_refocusing_phase": phase_errors[1] <= 0.01,
                        "second_refocusing_phase": phase_errors[2] <= 0.01,
                    }
                )
            checks.append(
                check(
                    f"frame_{frame_no}_t2_preparation",
                    valid_prep,
                    detail,
                )
            )
        frames.append(
            {
                "adc_indices": ids.tolist(),
                "line_indices": line_ids,
                "central_adc_index": central,
                "preparation_pulse_indices": prep_ids,
                "conditioning_pulse_indices": conditioning_ids,
                "readout_pulse_indices": readout_ids,
                "train_start_s": train_start,
                "trigger_time_s": float(triggers[trigger_id]),
                "segment_index": segment_no % segments_per_frame,
            }
        )
        prior_frame_end = float(acq.adc[ids[-1]])
        if segments_per_frame > 1:
            for item in checks[check_start:]:
                name = item["name"]
                item["name"] = name.replace(
                    f"frame_{frame_no}_",
                    f"frame_{frame_no}_segment_{segment_no % segments_per_frame}_",
                    1,
                )
    for item in checks:
        detail = item.get("detail")
        if isinstance(detail, dict) and "conditions" in detail:
            detail["conditions"] = {k: bool(v) for k, v in detail["conditions"].items()}
            detail["failed_conditions"] = [
                k for k, v in detail["conditions"].items() if not v
            ]
    frames = _assemble_frames(acq, frames, spec, checks, segments_per_frame)
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
        frames=frames,
        family=family(case),
        reconstruction="bart_cartesian_fft@1",
        preparation_spoil_times_s=preparation_spoil_times,
    )
    return acq


def _assemble_frames(acq, segments, spec, checks, segments_per_frame):
    """Assemble measured trigger windows; require each imaging sample exactly once."""
    frames = []
    for number in range(spec["temporal"]["frames"]):
        parts = segments[
            number * segments_per_frame : (number + 1) * segments_per_frame
        ]
        ids = np.concatenate([part["adc_indices"] for part in parts]).astype(int)
        unique = len(np.unique(acq.indices[ids], axis=0))
        checks.append(
            check(
                f"frame_{number}_full_cartesian_grid",
                unique == np.prod(spec["matrix"])
                and len(ids) == np.prod(spec["matrix"]),
                {"unique_samples": unique, "samples": len(ids), "segments": len(parts)},
            )
        )
        if segments_per_frame > 1:
            for suffix in ("balanced_bssfp", "inversion", "t2_preparation"):
                measured = [
                    item
                    for item in checks
                    if item["name"].startswith(f"frame_{number}_segment_")
                    and item["name"].endswith("_" + suffix)
                ]
                if measured:
                    checks[:] = [item for item in checks if item not in measured]
                    checks.append(
                        check(
                            f"frame_{number}_{suffix}",
                            len(measured) == segments_per_frame
                            and all(item["passed"] for item in measured),
                            {"segments": [item["detail"] for item in measured]},
                        )
                    )
        frame = dict(parts[0])
        for key in (
            "adc_indices",
            "line_indices",
            "preparation_pulse_indices",
            "conditioning_pulse_indices",
            "readout_pulse_indices",
        ):
            frame[key] = [value for part in parts for value in part[key]]
        frame["central_adc_index"] = int(
            ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))]
        )
        if segments_per_frame > 1:
            frame["segments"] = parts
        frames.append(frame)
    hs = [item for item in checks if item["name"] == "adiabatic_hyperbolic_secant"]
    if len(hs) > 1:
        checks[:] = [item for item in checks if item not in hs]
        checks.append(
            check(
                "adiabatic_hyperbolic_secant",
                all(item["passed"] for item in hs),
                {"segments": [item["detail"] for item in hs]},
            )
        )
    return frames


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
