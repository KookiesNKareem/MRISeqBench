"""Independent Pulseq S2 controls. Geometry is authored before judge inspection."""

import copy

import numpy as np
import pypulseq as pp

from tests.fixtures.binomial_rf import sparse_binomial_rf
from tests.fixtures.gradients import trapezoid


def make_sequence(
    case,
    mutation=None,
    *,
    curved_duration=0.016,
    epi_spacing=0.0016,
    spiral_power=1.0,
    interleave_partitions=False,
    cartesian_flat=None,
    grase_offset=0.001,
    epi_tr=0.1,
    echo_time=None,
    radial_te=0.005,
    radial_tr_s=None,
    ref_duration=0.0008,
    overlap_blips=False,
    steady_state_preparations=None,
    ute_pre_duration=0.0001,
    water_selective_order=0,
    water_selective_refocusing_order=0,
    grase_spin_echo_spacing=None,
    centric_grase=False,
    split_grase_polarities=False,
    centric_tse=False,
    spiral_adc_oversampling=1,
    ute_te=0.0005,
    reverse_epi=False,
    adaptive_epi=False,
    fat_saturation_pulses=0,
    corpse_refocusing=False,
    refocusing_crushers=False,
):
    spec, h = case["task"]["sequence"], case["hardware"]
    fam = case["task"]["id"]
    if fat_saturation_pulses and fam not in {
        "grase",
        "tse_t2w",
        "propeller_tse",
        "epi_single_shot",
    }:
        raise ValueError("CHESS preparation is authored for refocused trains and EPI")
    n = spec["matrix"][0]
    fov = np.asarray(spec["fov_mm"]) / 1000
    read = spec.get("readout") or {}
    system = pp.Opts(
        B0=h["field_strength_T"],
        max_grad=h["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=h["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=h["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=h["adc_dead_time_us"] * 1e-6,
    )
    seq = pp.Sequence(system)
    refocused = fam in {"tse_t2w", "propeller_tse", "grase"}
    rf = pp.make_block_pulse(
        np.pi / 2 if refocused else np.deg2rad(5),
        duration=0.0001 if fam == "ute_radial" else 0.0006 if refocused else 0.0004,
        system=system,
        delay=system.rf_dead_time,
        use="excitation",
    )

    def binomial_rf(flip, order, use):
        if order not in {1, 2}:
            raise ValueError("Water-selective control supports first/second-order RF")
        fat_hz = abs(case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"])
        return sparse_binomial_rf(flip, order, fat_hz, system, use)

    if water_selective_order:
        rf = binomial_rf(
            np.pi / 2 if refocused else np.deg2rad(5),
            water_selective_order,
            "excitation",
        )
    ref = pp.make_block_pulse(
        np.pi,
        duration=ref_duration,
        phase_offset=np.pi / 2,
        delay=system.rf_dead_time,
        system=system,
        use="refocusing",
    )
    if water_selective_refocusing_order:
        if not refocused:
            raise ValueError("Water-selective refocusing requires a refocused family")
        ref = binomial_rf(np.pi, water_selective_refocusing_order, "refocusing")
    elif corpse_refocusing:
        # CORPSE180: 420y, -300y, 60y, one coherent RF event.
        lead = system.block_duration_raster
        edges = lead + ref_duration * np.array([0, 420 / 180, 720 / 180, 780 / 180])
        times = np.r_[
            0,
            np.column_stack((edges - 0.5e-6, edges + 0.5e-6)).ravel(),
            780 / 180 * ref_duration + 2 * lead,
        ]
        values = np.array([0, 0, 1, 1, -1, -1, 1, 1, 0, 0]) / (2 * ref_duration)
        ref = pp.make_arbitrary_rf(
            values,
            np.pi,
            no_signal_scaling=True,
            delay=system.rf_dead_time,
            system=system,
            phase_offset=np.pi / 2,
            use="refocusing",
        )
        ref.t, ref.shape_dur = times, times[-1]
        ref.center = ref.shape_dur / 2
    saturation = None
    if fat_saturation_pulses:
        if fat_saturation_pulses not in (1, 2, 3):
            raise ValueError("Use one to three CHESS fat-saturation pulses")
        duration = round(0.004 * 3 / h["field_strength_T"] / 20e-6) * 20e-6
        saturation = pp.make_gauss_pulse(
            np.pi / 2,
            duration=duration,
            dwell=20e-6,
            time_bw_product=2,
            freq_offset=case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"],
            delay=system.rf_dead_time,
            system=system,
            use="saturation",
        )
    clock, shot_start, exc_time, phase = 0.0, 0.0, 0.0, 0.0
    exc_block_start = 0.0
    # Effective k is explicitly carried through RF sign changes.
    moment = np.zeros(3)
    coherence = np.zeros(3)

    def add(*events):
        nonlocal clock, moment
        seq.add_block(*events)
        clock += pp.calc_duration(*events)
        for event in events:
            if event.type in {"trap", "grad"}:
                moment["xyz".index(event.channel)] += event.area

    def wait(until):
        if until - clock > 1e-9:
            add(pp.make_delay(np.ceil((until - clock - 1e-10) / 1e-5) * 1e-5))
        elif until - clock < -1e-8:
            raise ValueError((fam, "infeasible timing", until - clock))

    def excite(j):
        nonlocal coherence, shot_start, exc_time, phase, exc_block_start
        shot_start = clock
        phase = np.deg2rad(117) * j * (j + 1) / 2
        for index in range(fat_saturation_pulses):
            prep = copy.copy(saturation)
            prep.phase_offset = phase
            add(prep)
            # A low-flip gradient-echo excitation needs the saturated fat
            # coherence dephased by whole cycles across each in-plane voxel.
            spoil = (
                [("x", 400 + 200 * index), ("y", 300 + 100 * index)]
                if refocused
                else [("x", (4 + index) * n / fov[0]), ("y", (3 + index) * n / fov[1])]
            )
            add(
                *[
                    trapezoid(axis, area=area, duration=0.003, system=system)
                    for axis, area in spoil
                ]
            )
        pulse = copy.copy(rf)
        pulse.phase_offset = phase
        exc_block_start = clock
        exc_time = clock + pulse.delay + pulse.shape_dur / 2
        coherence = -moment.copy()
        add(pulse)

    crusher_number = 0

    def refocus(when):
        nonlocal coherence, crusher_number
        area = n / fov[0] * (1 + (crusher_number % 8) / 8)
        if refocusing_crushers:
            wait(when - ref.delay - ref.shape_dur / 2 - 0.0012)
            add(trapezoid("x", area=area, duration=0.0012, system=system))
        else:
            wait(when - ref.delay - ref.shape_dur / 2)
        r = copy.copy(ref)
        r.phase_offset = phase + np.pi / 2
        coherence = -2 * moment - coherence
        add(r)
        if refocusing_crushers:
            add(trapezoid("x", area=area, duration=0.0012, system=system))
            crusher_number += 1

    def shift(target, duration=0.0006):
        area = np.asarray(target) - moment - coherence
        events = [
            trapezoid(axis, area=value, duration=duration, system=system)
            for axis, value in zip("xyz", area, strict=True)
            if abs(value) > 1e-10
        ]
        if events:
            add(*events)

    def line(ky, angle, when, polarity=1, z=0, next_ky=None):
        # Constant-speed Cartesian readout, rotated for PROPELLER/radial.
        flat = cartesian_flat if cartesian_flat is not None else 0.00064
        gradient_flat = (
            np.ceil(flat / system.grad_raster_time - 1e-9) * system.grad_raster_time
        )
        if abs(gradient_flat - flat) < 1e-12:
            gradient_flat = flat
        padding = (gradient_flat - flat) / 2
        rise = max(0.0001, np.ceil((n / fov[0] / flat) / system.max_slew / 1e-5) * 1e-5)
        if overlap_blips and fam.startswith("epi"):
            rise = max(rise, 0.00016)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0]) * polarity
        normal = np.array([-np.sin(angle), np.cos(angle), 0.0])
        grads = [
            trapezoid(
                axis,
                flat_area=component * n / fov[0] * (gradient_flat / flat),
                flat_time=gradient_flat,
                rise_time=rise,
                fall_time=rise,
                system=system,
            )
            for axis, component in zip("xyz", direction, strict=True)
            if abs(component) > 1e-10
        ]
        start = (
            -direction * (n / (2 * fov[0]) + n / fov[0] / flat * (rise / 2 + padding))
            + normal * ky / fov[1]
        )
        start[2] = z / fov[-1] if len(fov) == 3 else 0
        blip = None if adaptive_epi and fam.startswith("epi") else 0.0006
        if (
            not adaptive_epi
            and (fam.startswith("epi") or fam == "grase")
            and flat < 0.00064
        ):
            area = float(np.max(np.abs(start - moment - coherence)))
            if area < 24:
                blip = max(
                    0.0001, np.ceil(2.2 * np.sqrt(area / system.max_slew) / 1e-5) * 1e-5
                )
        shift(start, duration=blip)
        if adaptive_epi and fam.startswith("epi"):
            when = max(when, clock + rise + padding + flat / 2)
        wait(when - rise - padding - flat / 2)
        adc = pp.make_adc(
            n, duration=flat, delay=rise + padding, phase_offset=phase, system=system
        )
        # The phase blip fits in the fall ramp, after the acquired flat top.
        if overlap_blips and next_ky is not None:
            grads.append(
                trapezoid(
                    "y",
                    area=(next_ky - ky) / fov[1],
                    duration=rise,
                    delay=rise + gradient_flat,
                    system=system,
                )
            )
        add(*grads, adc)

    def finish(tr):
        # Spoil within the registered object dimensions; a z spoiler cannot
        # destroy coherence on the benchmark's zero-thickness 2D plane.
        axis = "z" if len(fov) == 3 else "x"
        add(trapezoid(axis, area=4 * n / fov[0], duration=0.0012, system=system))
        wait(shot_start + tr)

    if fam in {"epi_single_shot", "epi_multishot", "tse_t2w", "grase", "propeller_tse"}:
        # Saturate long-T1 material before the first PROPELLER blade. These
        # non-acquired preparations fit its 20 s budget and are simulated in full.
        preparations = 0
        if fam == "propeller_tse":
            preparations = (
                3 if steady_state_preparations is None else steady_state_preparations
            )
        if fam in {"tse_t2w", "grase"}:
            preparations = (
                int(case["evaluation"]["version"] >= 3)
                if steady_state_preparations is None
                else steady_state_preparations
            )
        for dummy in range(preparations):
            excite(dummy)
            finish(
                1.05
                if fam == "propeller_tse"
                else (spec.get("tr_ms") or {}).get("target", 1200) / 1000
            )
        shots = 1 if fam == "epi_single_shot" else 16 if fam == "propeller_tse" else 4
        if n < 32 and fam != "propeller_tse":
            shots = 1 if fam == "epi_single_shot" else 2
        etl = n // shots if fam != "propeller_tse" else min(8, n)
        if spec.get("readout"):
            shots = read.get("shots", shots)
            etl = read.get("echo_train_length", n // shots)
        for j in range(shots):
            excite(j + preparations)
            if fam.startswith("epi"):
                ys = np.arange(j, n, shots) - n // 2
                sampling = spec.get("sampling") or {}
                acceleration = int(sampling.get("acceleration") or 1)
                if fam == "epi_single_shot" and acceleration > 1:
                    width = sampling.get("calibration_lines") or 0
                    centre = range(n // 2 - width // 2, n // 2 + (width + 1) // 2)
                    ys = (
                        np.array(
                            sorted(
                                set(range(n // 2 % acceleration, n, acceleration))
                                | set(centre)
                            )
                        )
                        - n // 2
                    )
                if reverse_epi:
                    ys = ys[::-1]
                # Constant-offset controls sample successive lines one field
                # cycle apart; the central line retains the prescribed TE.
                if mutation == "missing_line":
                    ys = ys[:-1]
                te = (
                    echo_time
                    if echo_time is not None
                    else (
                        (spec.get("te_ms") or {}).get("target")
                        or (30 if shots == 1 else 20)
                    )
                    / 1000
                )
                centre_index = int(np.argmin(abs(ys)))
                for e, y in enumerate(ys):
                    pol = 1 if mutation == "no_alternation" else (-1) ** e
                    line(
                        y,
                        0,
                        exc_time + te + (e - centre_index) * epi_spacing,
                        pol,
                        next_ky=ys[e + 1] if e + 1 < len(ys) else None,
                    )
                finish(epi_tr)
            else:
                angle = j * np.pi / shots if fam == "propeller_tse" else 0
                spacing = (read.get("echo_spacing_ms") or 10) / 1000
                # Place central ky at echo 4 (effective TE 40 ms).
                ys = (
                    np.roll(np.arange(n) - n // 2, 3 if fam == "tse_t2w" else 2)[
                        j * etl : (j + 1) * etl
                    ]
                    if fam != "propeller_tse"
                    else np.arange(etl) - etl // 2 + 1
                )
                if fam == "grase" and case["evaluation"]["version"] >= 3:
                    ordered = sorted(range(-n // 2, n // 2), key=lambda y: (abs(y), y))
                    if centric_grase:
                        # Low ky at the first spin echo leaves room for long RF.
                        ys = np.array(ordered)[j::shots]
                    else:
                        # Central ky at the effective-TE echo; adjacent GE pairs.
                        echoes = etl // 2
                        central = min(1, echoes - 1)
                        priority = sorted(
                            range(echoes), key=lambda e: (abs(e - central), e)
                        )
                        bins = {
                            echo_number: sorted(
                                ordered[rank * 2 * shots : (rank + 1) * 2 * shots]
                            )
                            for rank, echo_number in enumerate(priority)
                        }
                        ys = np.array(
                            [
                                y
                                for e in range(echoes)
                                for y in (
                                    [bins[e][j], bins[e][j + shots]]
                                    if split_grase_polarities
                                    else bins[e][2 * j : 2 * j + 2]
                                )
                            ]
                        )
                if fam == "tse_t2w" and centric_tse:
                    central = min(3, etl - 1)
                    priority = sorted(range(etl), key=lambda e: (abs(e - central), -e))
                    ordered = sorted(range(-n // 2, n // 2), key=lambda y: (abs(y), y))
                    bins = {
                        echo_number: sorted(ordered[rank * shots : (rank + 1) * shots])
                        for rank, echo_number in enumerate(priority)
                    }
                    ys = np.array([bins[e][j] for e in range(etl)])
                if fam == "grase":
                    previous_echo = exc_time
                    for e in range(etl // 2):
                        echo = (
                            exc_time
                            + (echo_time if echo_time is not None else 0.040)
                            + (e if centric_grase else e - 1) * grase_spin_echo_spacing
                            if grase_spin_echo_spacing is not None
                            else exc_time + (e + 1) * spacing * 2
                        )
                        refocus((previous_echo + echo) / 2)
                        previous_echo = echo
                        line(ys[2 * e], 0, echo - grase_offset, 1)
                        line(ys[2 * e + 1], 0, echo + grase_offset, -1)
                        shift(np.zeros(3))
                else:
                    for e, y in enumerate(ys):
                        echo = exc_time + (e + 1) * spacing
                        refocus(echo - spacing / 2)
                        line(y, angle, echo)
                        shift(np.zeros(3))
                finish(
                    (spec.get("tr_ms") or {}).get(
                        "target", 1000 if fam == "propeller_tse" else 1200
                    )
                    / 1000
                    + (
                        0.05
                        if fam == "propeller_tse" and not fat_saturation_pulses
                        else 0
                    )
                )
    elif fam in {"radial_readout", "ute_radial", "stack_of_stars"}:
        spokes = read.get("spokes") or (48 if n == 32 else 16)
        partitions = (
            range(-spec["matrix"][2] // 2, spec["matrix"][2] // 2)
            if fam == "stack_of_stars"
            else [0]
        )
        ordering = (
            [(z, s) for s in range(spokes) for z in partitions]
            if interleave_partitions
            else [(z, s) for z in partitions for s in range(spokes)]
        )
        radial_tr = (
            (spec.get("tr_ms") or {}).get("target", 12) / 1000
            if radial_tr_s is None
            else radial_tr_s
        )
        preparations = int(steady_state_preparations or 0)
        for dummy in range(preparations):
            excite(dummy)
            finish(radial_tr)
        for j, (z, s) in enumerate(ordering):
            excite(j + preparations)
            angle = s * (2 * np.pi if fam == "ute_radial" else np.pi) / spokes
            if fam == "ute_radial":
                # Centre-out readout with its first sample at the requested TE.
                direction = np.array([np.cos(angle), np.sin(angle)])
                amplitude = (n / 2 - 0.25) / fov[0] / 0.00064
                rise = max(
                    0.0001,
                    np.ceil(amplitude / system.max_slew / 1e-5 - 1e-9) * 1e-5,
                )
                grads = [
                    trapezoid(
                        axis,
                        amplitude=component * amplitude,
                        flat_time=0.00064,
                        rise_time=rise,
                        fall_time=rise,
                        system=system,
                    )
                    for axis, component in zip("xy", direction, strict=True)
                    if abs(component) > 1e-10
                ]
                # Precompensate half the rise area, then centre-out sampling.
                target = np.zeros(3)
                for g in grads:
                    target["xyz".index(g.channel)] = -g.amplitude * rise / 2
                shift(target, duration=ute_pre_duration)
                wait(exc_time + ute_te - rise - 0.00064 / n / 2)
                adc = pp.make_adc(
                    n,
                    duration=0.00064,
                    delay=rise,
                    phase_offset=phase,
                    system=system,
                )
                add(*grads, adc)
            else:
                line(0, angle, exc_time + radial_te, z=z)
            finish(radial_tr)
    else:
        count = read.get("interleaves") or 4
        for j in range(count):
            excite(j)
            duration = curved_duration
            dt = system.grad_raster_time
            t = (np.arange(round(duration / dt)) + 0.5) * dt
            u = t / duration
            angle = j * 2 * np.pi / count
            radius = n / 2 - 0.5
            turns = read.get("turns") or 8
            if spiral_power != 1.0:
                shift_u = 0.01
                u = ((u + shift_u) ** spiral_power - shift_u**spiral_power) / (
                    (1 + shift_u) ** spiral_power - shift_u**spiral_power
                )
            r, theta = radius * u, 2 * np.pi * turns * u + angle
            xy = np.c_[r * np.cos(theta), r * np.sin(theta)] / fov[:2]
            gradients = np.gradient(xy, dt, axis=0)
            # Explicit gradient ramp boundaries. ADC omits the ramp, and the
            # prephaser compensates its area rather than relying on definitions.
            ramp = round(0.0005 / dt)
            waveforms = [
                np.r_[
                    np.linspace(0, gradients[0, k], ramp, endpoint=False),
                    gradients[:, k],
                    np.linspace(gradients[-1, k], 0, ramp),
                ]
                for k in range(2)
            ]
            grads = [
                pp.make_arbitrary_grad(axis, waveform, first=0, last=0, system=system)
                for axis, waveform in zip("xy", waveforms, strict=True)
            ]
            area_before_first = np.array(
                [
                    np.trapezoid(
                        np.r_[0, w[: ramp + 1]],
                        x=np.r_[0, (np.arange(ramp + 1) + 0.5) * dt],
                    )
                    for w in waveforms
                ]
            )
            target = np.zeros(3)
            target[:2] = xy[0] - area_before_first
            shift(target)
            te = (spec.get("te_ms") or {}).get("target", 5) / 1000
            wait(exc_time + te - ramp * dt - dt / 2)
            # Oversampling keeps ADC sample spacing fine along long trajectories.
            adc = pp.make_adc(
                len(t) * spiral_adc_oversampling,
                dwell=dt / spiral_adc_oversampling,
                delay=ramp * dt,
                phase_offset=phase,
                system=system,
            )
            add(*grads, adc)
            finish((spec.get("tr_ms") or {}).get("target", 50) / 1000)
    if fat_saturation_pulses and (spec.get("tr_ms") or {}).get("target"):
        # Complete the final acquired TR, including the prep-to-excitation offset.
        wait(exc_block_start + spec["tr_ms"]["target"] / 1000)
    if mutation == "wrong_fov":
        # Scale physical gradients; metadata cannot repair the acquisition.
        for key, value in seq.grad_library.data.items():
            v = list(value)
            v[0] *= 0.5
            seq.grad_library.data[key] = tuple(v)
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq
