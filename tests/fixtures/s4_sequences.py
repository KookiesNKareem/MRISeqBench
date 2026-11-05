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
    make_gauss_pulse,
    make_trapezoid,
    make_trigger,
)

from tests.fixtures.binomial_rf import sparse_binomial_rf
from tests.fixtures.gradients import trapezoid


def _make_water_selective_gre(case, mutation=None):
    hardware = case["hardware"]
    spec = case["task"]["sequence"]
    nx, ny = spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    field = hardware["field_strength_T"]
    system = Opts(
        B0=field,
        max_grad=hardware["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hardware["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hardware["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hardware["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hardware["adc_dead_time_us"] * 1e-6,
    )
    seq = Sequence(system)
    flip = np.deg2rad(spec.get("flip_angle_deg") or 20)
    te = spec["te_ms"]["target"] / 1000
    tr = spec["tr_ms"]["target"] / 1000
    fat_hz = abs(case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"])
    if mutation == "non_selective":
        rf = make_block_pulse(
            flip,
            duration=0.0004,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    else:
        rf = sparse_binomial_rf(
            flip,
            1 if field == 0.55 else 2,
            fat_hz,
            system,
            "excitation",
            centre_raster=2 * system.block_duration_raster,
        )
    readout_flat = 0.00032 if field == 3 else 0.00064
    read = make_trapezoid(
        "x", flat_area=nx / fov[0], flat_time=readout_flat, system=system
    )
    spoiler_factors = [1, 1.37, 1.91, 1.11, 1.73, 1.23, 1.59]
    spoil_area = nx / fov[0] * 1.3
    gdur = max(
        0.0006,
        np.ceil(
            calc_duration(
                make_trapezoid(
                    "x", area=spoil_area * max(spoiler_factors), system=system
                )
            )
            / 1e-5
        )
        * 1e-5
        + system.grad_raster_time,
    )
    adc = make_adc(nx, duration=read.flat_time, delay=read.rise_time, system=system)
    pre_x = make_trapezoid("x", area=-read.area / 2, duration=gdur, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    read_center = read.rise_time + read.flat_time / 2
    preparations = min(32, max(0, int(spec["duration_s"] / tr) - ny - 2))
    for shot, y in enumerate(range(-preparations, ny)):
        yy = 0 if y < 0 else y - ny // 2
        pre_y = make_trapezoid("y", area=yy / fov[1], duration=gdur, system=system)
        phase = np.deg2rad(117) * shot * (shot + 1) / 2
        r = copy.copy(rf)
        r.phase_offset = phase
        a = copy.copy(adc)
        a.phase_offset = phase
        seq.add_block(r)
        elapsed = calc_duration(r)
        seq.add_block(pre_x, pre_y)
        elapsed += calc_duration(pre_x, pre_y)
        delay = te + rf_center - read_center - elapsed
        if delay < -1e-9:
            raise ValueError(f"water_selective_gre TE is infeasible: {delay}")
        if delay > 0:
            seq.add_block(make_delay(delay))
        seq.add_block(read, *([a] if y >= 0 else []))
        elapsed += max(delay, 0) + calc_duration(read, a)
        crusher = make_trapezoid(
            "x",
            area=spoil_area * spoiler_factors[shot % len(spoiler_factors)],
            duration=gdur,
            system=system,
        )
        rewind_y = make_trapezoid("y", area=-yy / fov[1], duration=gdur, system=system)
        seq.add_block(crusher, rewind_y)
        elapsed += calc_duration(crusher, rewind_y)
        if tr - elapsed > 0:
            seq.add_block(make_delay(tr - elapsed))
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq


def _make_dwi_epi(case, mutation=None):
    hardware = case["hardware"]
    spec = case["task"]["sequence"]
    nx, ny = spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    field = hardware["field_strength_T"]
    low = hardware["id"] == "low_field"
    system = Opts(
        B0=field,
        max_grad=hardware["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hardware["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hardware["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hardware["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hardware["adc_dead_time_us"] * 1e-6,
    )
    seq = Sequence(system)
    fat_hz = abs(case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"])
    rf = sparse_binomial_rf(
        np.pi / 2,
        1 if field == 0.55 else 2,
        fat_hz,
        system,
        "excitation",
        centre_raster=2 * system.block_duration_raster,
    )
    ref = make_block_pulse(
        np.pi,
        duration=0.0008,
        phase_offset=np.pi / 2,
        delay=system.rf_dead_time,
        system=system,
        use="refocusing",
    )
    read = spec["readout"]
    shots = int(read["shots"])
    etl = int(read["echo_train_length"])
    te = spec["te_ms"]["target"] / 1000
    tr = spec["tr_ms"]["target"] / 1000
    epi_spacing = 0.0023 if low else 0.002
    flat = 0.00064 if field < 3 else 0.00032
    rise = max(0.0001, np.ceil((nx / fov[0] / flat) / system.max_slew / 1e-5) * 1e-5)

    target_b = float(spec["application"]["b_value_s_per_mm2"])
    if mutation == "wrong_b_value":
        target_b *= 0.4
    lobe_dur = 0.018 if te <= 0.072 else 0.023
    lobe_ramp = 0.0008
    lobe_flat = lobe_dur - 2 * lobe_ramp
    delta_eff = lobe_flat + lobe_ramp
    centre_index = etl // 2
    first_echo_rel = te - centre_index * epi_spacing
    rf_half = rf.shape_dur / 2 + rf.ringdown_time
    ref_half = ref.delay + ref.shape_dur / 2
    t1_rel = np.ceil((rf_half + 0.0005) / 1e-5) * 1e-5
    t2_rel = np.ceil((te / 2 + ref.shape_dur / 2 + ref.ringdown_time + 0.0005) / 1e-5) * 1e-5
    if t2_rel + lobe_dur > first_echo_rel - 0.0015:
        raise ValueError("Diffusion lobe does not fit before EPI echo train")
    delta_sep = t2_rel - t1_rel
    # Exact trapezoidal Stejskal-Tanner b-value formula:
    # b = (2*pi)^2 * 1e-6 * g_amp^2 * [delta_eff^2 * (Delta - delta_eff/3) + ramp^3/30 - delta_eff*ramp^2/6]
    bracket = (
        (delta_eff**2) * (delta_sep - delta_eff / 3.0)
        + (lobe_ramp**3) / 30.0
        - delta_eff * (lobe_ramp**2) / 6.0
    )
    g_amp = np.sqrt(target_b / (4.0 * (np.pi**2) * 1e-6 * bracket))
    diff_lobe = make_trapezoid(
        "z",
        flat_area= g_amp * lobe_flat,
        flat_time=lobe_flat,
        rise_time=lobe_ramp,
        fall_time=lobe_ramp,
        system=system,
    )

    clock = 0.0
    moment = np.zeros(3)
    coherence = np.zeros(3)

    def add(*events):
        nonlocal clock, moment
        seq.add_block(*events)
        clock += calc_duration(*events)
        for event in events:
            if event.type in {"trap", "grad"}:
                moment["xyz".index(event.channel)] += event.area

    def wait(until):
        if until - clock > 1e-9:
            add(make_delay(np.ceil((until - clock - 1e-10) / 1e-5) * 1e-5))
        elif until - clock < -1e-8:
            raise ValueError(("dwi_epi", "infeasible timing", until - clock))

    def shift(target, duration=0.0006):
        area = np.asarray(target) - moment - coherence
        events = [
            trapezoid(axis, area=value, duration=duration, system=system)
            for axis, value in zip("xyz", area, strict=True)
            if abs(value) > 1e-10
        ]
        if events:
            add(*events)

    for j in range(shots):
        shot_start = clock
        phase = np.deg2rad(117) * j * (j + 1) / 2
        p_exc = copy.copy(rf)
        p_exc.phase_offset = phase
        exc_time = clock + p_exc.delay + p_exc.shape_dur / 2
        coherence = -moment.copy()
        add(p_exc)

        wait(exc_time + t1_rel)
        add(copy.copy(diff_lobe))

        wait(exc_time + te / 2 - ref_half)
        p_ref = copy.copy(ref)
        p_ref.phase_offset = phase + np.pi / 2
        coherence = -2 * moment - coherence
        add(p_ref)

        wait(exc_time + t2_rel)
        add(copy.copy(diff_lobe))

        ys = np.arange(j, ny, shots) - ny // 2
        c_idx = int(np.argmin(np.abs(ys)))
        adc_phase = phase + np.pi
        for e, y in enumerate(ys):
            pol = (-1) ** e
            when = exc_time + te + (e - c_idx) * epi_spacing
            gx = trapezoid(
                "x",
                flat_area=pol * nx / fov[0],
                flat_time=flat,
                rise_time=rise,
                fall_time=rise,
                system=system,
            )
            start_k = np.array(
                [
                    -pol * (nx / (2 * fov[0]) + nx / fov[0] / flat * (rise / 2)),
                    y / fov[1],
                    0.0,
                ]
            )
            shift(start_k, duration=0.0006)
            wait(when - rise - flat / 2)
            adc = make_adc(
                nx, duration=flat, delay=rise, phase_offset=adc_phase, system=system
            )
            add(gx, adc)
        spoil = trapezoid("x", area=4 * nx / fov[0], duration=0.002, system=system)
        add(spoil)
        wait(shot_start + tr)
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq


def _make_venc_flow_2d(case, mutation=None):
    hardware = case["hardware"]
    spec = case["task"]["sequence"]
    nx, ny = spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    field = hardware["field_strength_T"]
    system = Opts(
        B0=field,
        max_grad=hardware["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hardware["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hardware["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hardware["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hardware["adc_dead_time_us"] * 1e-6,
    )
    seq = Sequence(system)
    flip = np.deg2rad(spec.get("flip_angle_deg") or 15)
    te = spec["te_ms"]["target"] / 1000
    tr = spec["tr_ms"]["target"] / 1000
    venc = float(spec["application"]["venc_cm_per_s"])
    if mutation == "wrong_venc":
        venc *= 2.0
    rf = make_block_pulse(
        flip,
        duration=0.0004,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    readout_flat = 0.00032 if field == 3 else 0.00064
    read = make_trapezoid(
        "x", flat_area=nx / fov[0], flat_time=readout_flat, system=system
    )
    spoiler_factors = [1, 1.37, 1.91, 1.11, 1.73, 1.23, 1.59]
    spoil_area = nx / fov[0] * 1.3
    gdur = max(
        0.0006,
        np.ceil(
            calc_duration(
                make_trapezoid(
                    "x", area=spoil_area * max(spoiler_factors), system=system
                )
            )
            / 1e-5
        )
        * 1e-5
        + system.grad_raster_time,
    )
    bipolar_lobe_dur = 0.001
    # Toggle +/-A between frames 0 and 1 so |Delta m1| = 2 * A * bipolar_lobe_dur = 50 / venc.
    bipolar_area = 50.0 / (2.0 * venc * bipolar_lobe_dur)
    adc = make_adc(nx, duration=read.flat_time, delay=read.rise_time, system=system)
    pre_x = make_trapezoid("x", area=-read.area / 2, duration=gdur, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    read_center = read.rise_time + read.flat_time / 2
    preparations = 16
    shot = 0
    for frame_no in range(2):
        sign = -1.0 if frame_no == 0 else 1.0
        bip1 = make_trapezoid(
            "z", area=sign * bipolar_area, duration=bipolar_lobe_dur, system=system
        )
        bip2 = make_trapezoid(
            "z", area=-sign * bipolar_area, duration=bipolar_lobe_dur, system=system
        )
        start_y = -preparations if frame_no == 0 else 0
        for y in range(start_y, ny):
            yy = 0 if y < 0 else y - ny // 2
            pre_y = make_trapezoid("y", area=yy / fov[1], duration=gdur, system=system)
            phase = np.deg2rad(117) * shot * (shot + 1) / 2
            r = copy.copy(rf)
            r.phase_offset = phase
            a = copy.copy(adc)
            a.phase_offset = phase
            seq.add_block(r)
            seq.add_block(bip1)
            seq.add_block(bip2)
            seq.add_block(pre_x, pre_y)
            elapsed = (
                calc_duration(r)
                + calc_duration(bip1)
                + calc_duration(bip2)
                + calc_duration(pre_x, pre_y)
            )
            delay = te + rf_center - read_center - elapsed
            if delay < -1e-9:
                raise ValueError(f"venc_flow_2d TE is infeasible: {delay}")
            if delay > 0:
                seq.add_block(make_delay(delay))
            seq.add_block(read, *([a] if y >= 0 else []))
            elapsed += max(delay, 0) + calc_duration(read, a)
            crusher = make_trapezoid(
                "x",
                area=spoil_area * spoiler_factors[shot % len(spoiler_factors)],
                duration=gdur,
                system=system,
            )
            rewind_y = make_trapezoid(
                "y", area=-yy / fov[1], duration=gdur, system=system
            )
            seq.add_block(crusher, rewind_y)
            elapsed += calc_duration(crusher, rewind_y)
            if tr - elapsed > 0:
                seq.add_block(make_delay(tr - elapsed))
            shot += 1
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq


def make_sequence(
    case,
    mutation=None,
    *,
    dummy_count=None,
    ramp_count=None,
    tr_s=None,
    echo_time_s=None,
    catalyzation=False,
    preparation_dwell=None,
    segment_lines=None,
    readout_flat=None,
    gradient_duration=None,
    fat_saturation_duration=None,
    fat_saturation_before_inversion=False,
    readout_flip_deg=35,
    readout_dwell_s=None,
    interleaved_segments=False,
):
    """catalyzation adds a bSSFP-conditioning alpha/2 pulse TR/2 before each
    train. Optional fat saturation uses two spoiled pulses; its supplied
    duration is at 3 T and scales inversely with field."""
    hardware = case["hardware"]
    spec = case["task"]["sequence"]
    family = case["task"]["id"]
    if family == "water_selective_gre":
        return _make_water_selective_gre(case, mutation)
    if family == "dwi_epi":
        return _make_dwi_epi(case, mutation)
    if family == "venc_flow_2d":
        return _make_venc_flow_2d(case, mutation)
    nx, ny = spec["matrix"]
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
            128
            if field == 3 and family in {"t2_prepared", "t2_mapping"}
            else (32 if family == "cardiac_respiratory_bssfp" else 16)
        )
    tr = {0.55: 0.0035 if nx == 128 else 0.0025, 1.5: 0.0046, 3.0: 0.0023}[field]
    if field == 1.5 and family == "t1_mapping":
        tr = 0.0035
    if tr_s is not None:
        tr = tr_s
    te = tr / 2 if echo_time_s is None else echo_time_s
    flat = readout_flat or (
        0.00032 if (field == 3 and nx == 128) else (0.0004 if field == 3 else 0.00064)
    )
    gradient_duration = gradient_duration or (0.0004 if field == 3 else 0.0005)
    rf = make_block_pulse(
        np.deg2rad(readout_flip_deg),
        duration=0.0002,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    read_area = (
        nx / fov[0] if readout_dwell_s is None else flat / (fov[0] * readout_dwell_s)
    )
    read = make_trapezoid("x", flat_area=read_area, flat_time=flat, system=system)
    pre_x = trapezoid(
        "x", area=-read.area / 2, duration=gradient_duration, system=system
    )
    post_x = copy.copy(pre_x)
    if readout_dwell_s is None:
        adc = make_adc(nx, duration=flat, delay=read.rise_time, system=system)
    else:
        adc_duration = nx * readout_dwell_s
        adc_delay = read.rise_time + (flat - adc_duration) / 2
        adc_delay = (
            np.ceil(adc_delay / system.rf_raster_time - 1e-9) * system.rf_raster_time
        )
        adc = make_adc(nx, dwell=readout_dwell_s, delay=adc_delay, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    read_center = read.rise_time + flat / 2
    application = spec.get("application") or {}
    inversion_duration = (
        application.get("preparation_duration_ms", 10.0) / 1000
        if family == "adiabatic_inversion"
        else 0.01
    )
    inversion = make_adiabatic_pulse(
        "hypsec",
        duration=inversion_duration,
        dwell=preparation_dwell,
        beta=1000.0,
        mu=4.9,
        adiabaticity=4,
        delay=system.rf_dead_time,
        system=system,
        use="inversion",
    )
    refocusing = make_adiabatic_pulse(
        "hypsec",
        duration=0.008,
        dwell=preparation_dwell,
        beta=1100.0,
        mu=4.9,
        adiabaticity=4,
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
    values = {
        "adiabatic_inversion": [application.get("inversion_time_ms", 600.0)],
        "t1_mapping": application.get(
            "inversion_times_ms", [100.0, 400.0, 1200.0]
        ),
        "t2_prepared": [application.get("preparation_duration_ms", 40.0)],
        "t2_mapping": application.get(
            "preparation_durations_ms", [0.0, 25.0, 50.0]
        ),
        "cardiac_respiratory_bssfp": [0.0],
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
            trigger_time = (
                next_trigger
                if next_trigger == 0 or field != 0.55 or family != "t1_mapping"
                else next_trigger - 2 * system.block_duration_raster
            )
            if trigger_time < clock - 1e-8:
                raise ValueError(
                    "An ECG input trigger would intersect an RF/readout block"
                )
            if trigger_time - clock > 1e-9:
                add(make_delay(trigger_time - clock))
            add(
                make_trigger("physio1", duration=system.grad_raster_time, system=system)
            )
            next_trigger += 1.0
        if target - clock > 1e-9:
            add(make_delay(target - clock))

    def preparation_pulse(center, pulse):
        wait(center - pulse.delay - pulse.shape_dur / 2)
        add(pulse)

    fat = crushers = None
    if fat_saturation_duration is not None:
        duration = round(fat_saturation_duration * 3 / field / 20e-6) * 20e-6
        fat = make_gauss_pulse(
            np.pi / 2,
            duration=duration,
            dwell=preparation_dwell or 0,
            time_bw_product=2,
            freq_offset=case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"],
            delay=system.rf_dead_time,
            system=system,
            use="saturation",
        )
        crushers = []
        for areas in (
            {"x": 2 * nx / fov[0], "y": ny / fov[1]},
            {"x": 4 * nx / fov[0], "y": 3 * ny / fov[1]},
        ):
            duration = max(
                0.0012,
                *(
                    calc_duration(trapezoid(axis, area=area, system=system))
                    for axis, area in areas.items()
                ),
            )
            crushers.append(
                [
                    trapezoid(axis, area=area, duration=duration, system=system)
                    for axis, area in areas.items()
                ]
            )

    def fat_preparation(end):
        wait(end - 2 * calc_duration(fat) - sum(calc_duration(*g) for g in crushers))
        for gradients in crushers:
            add(copy.copy(fat))
            add(*gradients)

    # Centric ordering keeps the image's effective preparation at its first echo.
    ky = sorted(range(-ny // 2, ny - ny // 2), key=lambda value: (abs(value), value))
    if segment_lines is None:
        configured_segments = (spec.get("temporal") or {}).get("segments_per_frame")
        segment_lines = (
            int(np.ceil(ny / configured_segments)) if configured_segments else ny
        )
    segments = int(np.ceil(ny / (segment_lines or ny)))
    segment_width = int(np.ceil(ny / segments))
    if family == "cardiac_respiratory_bssfp":
        resp_s = (spec.get("temporal") or {}).get("respiratory_period_ms", 4000.0) / 1000
        step_s = 1.0 if mutation == "ungated_respiratory" else resp_s
        schedule = [
            (frame, segment, step_s * (frame * segments + segment), value)
            for frame, value in enumerate(values)
            for segment in range(segments)
        ]
    else:
        schedule = [
            (frame, segment, 1.0 + 5.0 * (frame * segments + segment), value)
            for frame, value in enumerate(values)
            for segment in range(segments)
        ]
    for frame, segment, trigger, value in schedule:
        frame_ramp_count = dummy_count if ramp_count is None else ramp_count
        frame_dummy_count = dummy_count
        if family == "t1_mapping" and field in {0.55, 3} and value >= 400.0:
            frame_dummy_count = max(frame_dummy_count, 64)
            frame_ramp_count = 16
        central_echo = (
            trigger
            + spec["temporal"]["trigger_delay_ms"] / 1000
            + 0.006
            + (0.20 if mutation == "late_adc" else 0.0)
        )
        first_center = central_echo - te
        catalyst_center = first_center - frame_dummy_count * tr
        actual_value = value
        if mutation == "wrong_preparation_time":
            actual_value += (
                25.0 if family in {"adiabatic_inversion", "t1_mapping"} else 30.0
            )
        if family in {"adiabatic_inversion", "t1_mapping"}:
            if fat is not None and fat_saturation_before_inversion:
                fat_preparation(
                    central_echo
                    - actual_value / 1000
                    - inversion.delay
                    - inversion.shape_dur / 2
                )
            preparation_pulse(central_echo - actual_value / 1000, inversion)
        elif actual_value > 0:
            # A symmetric, two-refocusing T2 preparation, then a separate catalyst.
            end = catalyst_center - 3 * tr
            start = end - actual_value / 1000
            preparation_pulse(start, tipdown)
            preparation_pulse(start + actual_value / 4000, refocusing)
            preparation_pulse(start + 3 * actual_value / 4000, refocusing)
            preparation_pulse(end, tipup)

        def half_alpha(center, phase):
            wait(center - rf_center)
            r = copy.copy(rf)
            r.signal = r.signal / 2
            r.phase_offset = phase
            add(r)

        if fat is not None and not fat_saturation_before_inversion:
            fat_preparation(
                first_center
                - frame_dummy_count * tr
                - rf_center
                - (tr / 2 if catalyzation else 0)
            )
        if catalyzation:
            # Opposite phase to the first train pulse, TR/2 earlier.
            half_alpha(
                first_center - frame_dummy_count * tr - tr / 2,
                np.pi * ((shot + 1) % 2),
            )
        segment_ky = (
            ky[segment::segments]
            if family in {"t1_mapping", "cardiac_respiratory_bssfp"}
            and interleaved_segments
            else ky[segment * segment_width : (segment + 1) * segment_width]
        )
        for line in range(-frame_dummy_count, len(segment_ky)):
            center = first_center + line * tr
            wait(center - rf_center)
            r = copy.copy(rf)
            phase = 0.0 if mutation == "no_phase_cycle" else np.pi * (shot % 2)
            r.phase_offset = phase
            if line < 0:
                # Smooth alpha ramp reduces preparation-to-bSSFP transients.
                step = line + frame_dummy_count + 1
                factor = (
                    1.0 if frame_ramp_count == 0 else min(1.0, step / frame_ramp_count)
                )
                r.signal = r.signal * factor
            a = copy.copy(adc)
            a.phase_offset = phase
            yy = 0 if line < 0 else segment_ky[line]
            pre_y = trapezoid(
                "y", area=yy / fov[1], duration=gradient_duration, system=system
            )
            post_y = trapezoid(
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
    recovery = 1 if family == "t2_prepared" and field == 3 and dummy_count > 16 else 0
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
