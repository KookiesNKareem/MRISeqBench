"""Independently authored ECG-triggered cardiac preparation/bSSFP controls.

Scanner-dependent TR places fat and water in usable ordinary bSSFP passbands.
All timing, phase cycles, preparation RF and input triggers are actual events.
"""

import copy

import numpy as np
from pypulseq import (
    Opts,
    Sequence,
    calc_duration,
    make_adc,
    make_adiabatic_pulse,
    make_block_pulse,
    make_delay,
    make_trapezoid,
    make_trigger,
)

MUTATIONS = (
    "missing_trigger",
    "wrong_trigger_channel",
    "late_adc",
    "tiny_flip",
    "missing_preparation",
    "wrong_preparation_time",
    "bad_refocusing",
    "unbalanced",
    "no_phase_cycle",
    "receiver_phase",
    "constant_contrast",
    "narrow_preparation",
)


def make_sequence(
    case,
    mutation=None,
    *,
    matrix=None,
    flip_deg=35,
    dummy_count=None,
    prep_gap_tr=3,
    ramp_count=None,
    tr_s=None,
):
    hardware = case["hardware"]
    spec = case["task"]["sequence"]
    family = case["task"]["id"]
    nx, ny = matrix or spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    system = Opts(
        B0=hardware["field_strength_T"],
        max_grad=hardware["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hardware["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hardware["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hardware["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hardware["adc_dead_time_us"] * 1e-6,
    )
    seq = Sequence(system)
    field = hardware["field_strength_T"]
    if dummy_count is None:
        # At 3 T finite-bandwidth T2 preparation leaves fat transients. A longer
        # smooth ramp settles the ordinary bSSFP readout before acquired lines.
        dummy_count = (
            128 if field == 3 and family in {"t2_prepared", "t2_mapping"} else 16
        )
    tr = {0.55: 0.0025, 1.5: 0.0046, 3.0: 0.0023}[field]
    if field == 1.5 and family == "t1_mapping":
        tr = 0.0035
    if tr_s is not None:
        tr = tr_s
    te = tr / 2
    flat = 0.0004 if field == 3 else 0.00064
    gradient_duration = 0.0004 if field == 3 else 0.0005
    rf = make_block_pulse(
        np.deg2rad(0.0001 if mutation == "tiny_flip" else flip_deg),
        duration=0.0002,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    read = make_trapezoid("x", flat_area=nx / fov[0], flat_time=flat, system=system)
    pre_x = make_trapezoid(
        "x", area=-read.area / 2, duration=gradient_duration, system=system
    )
    post_x = copy.copy(pre_x)
    adc = make_adc(nx, duration=flat, delay=read.rise_time, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    read_center = read.rise_time + flat / 2
    inversion = make_adiabatic_pulse(
        "hypsec",
        duration=0.01,
        beta=800.0 if mutation == "narrow_preparation" else 1000.0,
        mu=4.9,
        adiabaticity=4,
        delay=system.rf_dead_time,
        system=system,
        use="inversion",
    )
    refocusing = make_adiabatic_pulse(
        "hypsec",
        duration=0.008,
        beta=1100.0,
        mu=4.9,
        adiabaticity=4,
        delay=system.rf_dead_time,
        phase_offset=np.pi / 2,
        system=system,
        use="refocusing",
    )
    if mutation == "bad_refocusing":
        refocusing.signal = refocusing.signal * 0.25
    if mutation == "narrow_preparation":
        refocusing = make_adiabatic_pulse(
            "hypsec",
            duration=0.0045,
            beta=1000.0,
            mu=3.0,
            adiabaticity=6,
            delay=system.rf_dead_time,
            phase_offset=np.pi / 2,
            system=system,
            use="refocusing",
        )
    tipdown = make_block_pulse(
        np.pi / 2,
        duration=0.0003,
        delay=system.rf_dead_time,
        system=system,
        use="preparation",
    )
    tipup = copy.copy(tipdown)
    tipup.phase_offset = np.pi
    frames = 3 if family in {"t1_mapping", "t2_mapping"} else 1
    trigger_times = [1.0, 6.0, 11.0] if frames == 3 else [1.0]
    values = {
        "adiabatic_inversion": [600.0],
        "t1_mapping": [100.0, 400.0, 1200.0],
        "t2_prepared": [40.0],
        "t2_mapping": [0.0, 25.0, 50.0],
    }[family]
    clock = 0.0
    next_trigger = 0.0
    shot = 0

    def add(*events):
        nonlocal clock
        seq.add_block(*events)
        clock += calc_duration(*events)

    def wait(target):
        nonlocal next_trigger
        if target < clock - 1e-8:
            raise ValueError(
                f"Independent S4 control has overlapping events: {target - clock:g}s"
            )
        while next_trigger <= target + 1e-9:
            if next_trigger < clock - 1e-8:
                raise ValueError(
                    "An ECG input trigger would intersect an RF/readout block"
                )
            if next_trigger - clock > 1e-9:
                add(make_delay(next_trigger - clock))
            event = (
                make_delay(system.grad_raster_time)
                if mutation == "missing_trigger"
                else make_trigger(
                    "physio2" if mutation == "wrong_trigger_channel" else "physio1",
                    duration=system.grad_raster_time,
                    system=system,
                )
            )
            add(event)
            next_trigger += 1.0
        if target - clock > 1e-9:
            add(make_delay(target - clock))

    def preparation_pulse(center, pulse):
        pulse_center = pulse.delay + pulse.shape_dur / 2
        wait(center - pulse_center)
        if mutation == "missing_preparation":
            add(make_delay(calc_duration(pulse)))
        else:
            add(pulse)

    # Centric ordering keeps the image's effective preparation at its first echo.
    ky = sorted(range(-ny // 2, ny - ny // 2), key=lambda value: (abs(value), value))
    for frame, (trigger, value) in enumerate(zip(trigger_times, values, strict=True)):
        frame_dummy_count = (
            dummy_count[frame]
            if isinstance(dummy_count, (list, tuple))
            else dummy_count
        )
        frame_ramp_count = (
            ramp_count[frame] if isinstance(ramp_count, (list, tuple)) else ramp_count
        )
        if frame_ramp_count is None:
            frame_ramp_count = frame_dummy_count
        central_echo = trigger + 0.606 + (0.20 if mutation == "late_adc" else 0.0)
        first_center = central_echo - te
        catalyst_center = first_center - frame_dummy_count * tr
        actual_value = value
        if mutation == "constant_contrast" and frames > 1:
            actual_value = 400.0 if family == "t1_mapping" else 25.0
        if mutation == "wrong_preparation_time":
            actual_value += (
                25.0 if family in {"adiabatic_inversion", "t1_mapping"} else 30.0
            )
        if family in {"adiabatic_inversion", "t1_mapping"}:
            preparation_pulse(central_echo - actual_value / 1000, inversion)
        elif actual_value > 0:
            # A symmetric, two-refocusing T2 preparation, then a separate catalyst.
            end = catalyst_center - prep_gap_tr * tr
            start = end - actual_value / 1000
            preparation_pulse(start, tipdown)
            preparation_pulse(start + actual_value / 4000, refocusing)
            preparation_pulse(start + 3 * actual_value / 4000, refocusing)
            preparation_pulse(end, tipup)
        for line in range(-frame_dummy_count, ny):
            center = first_center + line * tr
            wait(center - rf_center)
            r = copy.copy(rf)
            phase = 0.0 if mutation == "no_phase_cycle" else np.pi * (shot % 2)
            r.phase_offset = phase
            if line < 0:
                # Smooth alpha ramp reduces preparation-to-bSSFP transients.
                step = line + frame_dummy_count + 1
                factor = (
                    (0.5 if step == 1 else 1.0)
                    if frame_ramp_count == 0
                    else min(1.0, step / frame_ramp_count)
                )
                r.signal = r.signal * factor
            a = copy.copy(adc)
            a.phase_offset = phase + (
                np.pi / 2 if mutation == "receiver_phase" else 0.0
            )
            yy = 0 if line < 0 else ky[line]
            pre_y = make_trapezoid(
                "y", area=yy / fov[1], duration=gradient_duration, system=system
            )
            post_y = make_trapezoid(
                "y", area=-yy / fov[1], duration=gradient_duration, system=system
            )
            add(r)
            add(pre_x, pre_y)
            wait(center + te - read_center)
            add(read, *([a] if line >= 0 else []))
            add(*([post_x] if mutation != "unbalanced" else []), post_y)
            end = center - rf_center + tr
            wait(end)
            shot += 1
    # Finish the final nominal cardiac cycle with actual RF-free recovery.
    # This is part of the physical protocol and its SAR averaging duration.
    most_dummies = (
        max(dummy_count) if isinstance(dummy_count, (list, tuple)) else dummy_count
    )
    recovery = 1 if family == "t2_prepared" and field == 3 and most_dummies > 16 else 0
    wait(float(np.ceil(clock)) + recovery)
    seq.set_definition("Name", f"Independent static cardiac {family} control")
    seq.set_definition("FramePreparation_ms", values)
    seq.set_definition("NominalRR_s", 1.0)
    seq.set_definition("ReadoutTR_s", tr)
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    peak = max(
        np.max(np.abs(seq.get_block(number).rf.signal))
        for number in seq.block_events
        if seq.get_block(number).rf is not None
    )
    if peak / system.gamma * 1e6 > hardware["max_b1_uT"] + 1e-8:
        raise ValueError("Independent positive control exceeds the supplied B1 limit")
    if clock > spec["duration_s"] + 1e-8:
        raise ValueError("Independent control exceeds the task duration")
    return seq
