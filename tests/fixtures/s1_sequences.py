"""Small independently authored Cartesian control sequences and targeted mutations."""

import copy

import numpy as np
from pypulseq import (
    Opts,
    Sequence,
    calc_duration,
    make_adc,
    make_arbitrary_rf,
    make_block_pulse,
    make_delay,
    make_trapezoid,
)


def make_sequence(
    case,
    mutation=None,
    preparations=128,
    *,
    readout_flat=0.0016,
    echo_time=None,
    ref_duration=0.0008,
    water_selective_order=0,
    water_selective_refocusing_order=0,
):
    h, spec = case["hardware"], case["task"]["sequence"]
    fam = case["task"]["id"]
    system = Opts(
        B0=h["field_strength_T"],
        max_grad=h["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=h["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=h["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=h["adc_dead_time_us"] * 1e-6,
    )
    seq = Sequence(system)
    nx, ny, *nz = spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    se, balanced = fam == "spin_echo_t2w", fam == "bssfp"
    flip = spec.get("flip_angle_deg") or (90 if se else 30)
    te = (spec.get("te_ms") or {}).get("target", 5.0) / 1000
    tr = (
        (spec.get("tr_ms") or {}).get("target")
        or (spec.get("tr_ms") or {}).get("max")
        or 12.0
    ) / 1000
    if echo_time is not None:
        te = echo_time
    if mutation == "wrong_echo_time":
        te += 0.020 if se else 0.003
    rf = make_block_pulse(
        np.deg2rad(flip),
        duration=0.0008 if balanced else 0.0004,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )

    def binomial_rf(angle, order, use):
        if not se or order not in {1, 2}:
            raise ValueError(
                "Water-selective spin-echo control supports RF order 1 or 2"
            )
        fat_hz = abs(case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"])
        raster = system.rf_raster_time
        lag = round(1 / (2 * fat_hz) / raster)
        width = round(0.0004 / raster)
        coefficients = [1, 1] if order == 1 else [1, 2, 1]
        signal = np.zeros((len(coefficients) - 1) * lag + width)
        for index, coefficient in enumerate(coefficients):
            signal[index * lag : index * lag + width] = coefficient
        # An even number of block rasters keeps both pulse centres on that raster.
        padding = (-len(signal)) % (2 * round(system.block_duration_raster / raster))
        signal = np.pad(signal, (padding // 2, padding - padding // 2))
        return make_arbitrary_rf(
            signal, angle, delay=system.rf_dead_time, system=system, use=use
        )

    if water_selective_order:
        rf = binomial_rf(np.deg2rad(flip), water_selective_order, "excitation")
    ref = make_block_pulse(
        np.pi,
        duration=ref_duration,
        delay=system.rf_dead_time,
        system=system,
        use="refocusing",
    )
    if water_selective_refocusing_order:
        ref = binomial_rf(np.pi, water_selective_refocusing_order, "refocusing")
    read = make_trapezoid(
        "x", flat_area=nx / fov[0], flat_time=readout_flat, system=system
    )
    adc = make_adc(nx, duration=read.flat_time, delay=read.rise_time, system=system)
    pre_x = make_trapezoid("x", area=-read.area / 2, duration=0.0008, system=system)
    if mutation == "echo_outside_adc":
        pre_x = make_trapezoid("x", area=read.area / 2, duration=0.0008, system=system)
    spoil = make_trapezoid(
        "x", area=max(nx, 32) / fov[0] * 1.3, duration=0.0008, system=system
    )
    end_x = make_trapezoid("x", area=-read.area / 2, duration=0.0008, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    ref_center = ref.delay + ref.shape_dur / 2
    read_center = read.rise_time + read.flat_time / 2
    shot = 0
    for z in range(nz[0] if nz else 1):
        for y in range(-preparations if z == 0 and not se else 0, ny):
            if mutation == "missing_line" and y == ny - 1:
                continue
            yy = 0 if y < 0 or mutation == "repeated_lines" else y - ny // 2
            zz = z - (nz[0] // 2 if nz else 0)
            pre_y = make_trapezoid(
                "y", area=yy / fov[1], duration=0.0008, system=system
            )
            pre = [pre_x, pre_y]
            if nz:
                pre.append(
                    make_trapezoid(
                        "z", area=zz / fov[2], duration=0.0008, system=system
                    )
                )
            phase = (
                np.pi * (shot % 2)
                if balanced
                else np.deg2rad(117) * shot * (shot + 1) / 2
            )
            if mutation == "no_spoiling":
                phase = 0.0
            r = copy.copy(rf)
            r.phase_offset = phase
            if mutation == "tiny_rf":
                r.signal = r.signal * 1e-6
            a = copy.copy(adc)
            a.phase_offset = -phase if se else phase
            if mutation == "receiver_phase":
                a.phase_offset += np.pi * ((y + z) % 2)
            seq.add_block(r)
            elapsed = calc_duration(r)
            if se:
                shift = 0.005 if mutation == "refocus_late" else 0.0
                seq.add_block(
                    make_delay(te / 2 + rf_center - ref_center - elapsed + shift)
                )
                if mutation == "no_refocusing":
                    seq.add_block(make_delay(calc_duration(ref)))
                else:
                    rr = copy.copy(ref)
                    if mutation == "lying_refocusing_label":
                        rr.signal = rr.signal / 2
                    seq.add_block(rr)
                elapsed = te / 2 + rf_center - ref_center + calc_duration(ref) + shift
            seq.add_block(*pre)
            elapsed += calc_duration(*pre)
            delay = te + rf_center - read_center - elapsed
            if delay < -1e-9:
                raise ValueError(f"Control TE is infeasible: {delay}")
            if delay > 0:
                seq.add_block(make_delay(delay))
            seq.add_block(read, *([a] if y >= 0 else []))
            elapsed += max(delay, 0) + calc_duration(read, a)
            if balanced or mutation == "no_spoiling":
                post = [
                    end_x,
                    make_trapezoid(
                        "y", area=-yy / fov[1], duration=0.0008, system=system
                    ),
                ]
                if nz:
                    post.append(
                        make_trapezoid(
                            "z", area=-zz / fov[2], duration=0.0008, system=system
                        )
                    )
                if mutation == "unbalanced":
                    post = post[1:]
                seq.add_block(*post)
                elapsed += calc_duration(*post)
            else:
                seq.add_block(spoil)
                elapsed += calc_duration(spoil)
            if tr - elapsed < -1e-9:
                raise ValueError(f"Control TR is infeasible: {tr - elapsed}")
            last = z == (nz[0] - 1 if nz else 0) and y == ny - 1
            if tr - elapsed > 0 and not (mutation == "truncated_final_tr" and last):
                seq.add_block(make_delay(tr - elapsed))
            shot += 1
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq
