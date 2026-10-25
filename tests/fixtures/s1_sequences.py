"""Independently authored S1 Cartesian reference sequences."""

import copy

import numpy as np
from pypulseq import (
    Opts,
    Sequence,
    calc_duration,
    make_adc,
    make_block_pulse,
    make_delay,
    make_trapezoid,
)

from tests.fixtures.binomial_rf import sparse_binomial_rf


def make_sequence(
    case,
    *,
    preparations,
    readout_flat,
    echo_time=None,
    repetition_time=None,
    water_selective_order=0,
    water_selective_refocusing_order=0,
    refocusing_crushers=False,
    rf_duration=None,
    flip_angle=None,
    spoiler_factors=None,
    rewind_phase=False,
):
    """Spin echoes are 90x-180x; the receiver follows the physical echo in the
    excitation's FID convention: ADC phase = 2*refocusing phase - excitation
    phase + pi.
    """
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
    flip = (
        flip_angle
        if flip_angle is not None
        else spec.get("flip_angle_deg") or (90 if se else 30)
    )
    te = (spec.get("te_ms") or {}).get("target", 5.0) / 1000
    tr = (
        (spec.get("tr_ms") or {}).get("target")
        or (spec.get("tr_ms") or {}).get("max")
        or 12.0
    ) / 1000
    if echo_time is not None:
        te = echo_time
    if repetition_time is not None:
        tr = repetition_time
    rf = make_block_pulse(
        np.deg2rad(flip),
        duration=rf_duration or (0.0008 if balanced else 0.0004),
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
        return sparse_binomial_rf(
            angle,
            order,
            fat_hz,
            system,
            use,
            centre_raster=2 * system.block_duration_raster,
        )

    if water_selective_order:
        rf = binomial_rf(np.deg2rad(flip), water_selective_order, "excitation")
    ref = make_block_pulse(
        np.pi,
        duration=0.0008,
        delay=system.rf_dead_time,
        system=system,
        use="refocusing",
    )
    if water_selective_refocusing_order:
        ref = binomial_rf(np.pi, water_selective_refocusing_order, "refocusing")
    read = make_trapezoid(
        "x", flat_area=nx / fov[0], flat_time=readout_flat, system=system
    )
    # A balanced sequence has no spoiler. Size its prephasers from the moments
    # it actually plays, so the unused spoiler cannot inflate its minimum TR.
    spoil_area = max(nx, 32) / fov[0] * 1.3
    required_area = (
        max(read.area / 2, ny / (2 * fov[1]), nz[0] / (2 * fov[2]) if nz else 0)
        if balanced
        else spoil_area * (max(spoiler_factors) if spoiler_factors else 1)
    )
    gdur = max(
        0.0008,
        np.ceil(
            calc_duration(make_trapezoid("x", area=required_area, system=system)) / 1e-5
        )
        * 1e-5,
    )
    if spoiler_factors:
        gdur += system.grad_raster_time
    adc = make_adc(nx, duration=read.flat_time, delay=read.rise_time, system=system)
    pre_x = make_trapezoid("x", area=-read.area / 2, duration=gdur, system=system)
    spoil = (
        make_trapezoid("x", area=spoil_area, duration=gdur, system=system)
        if not balanced
        else None
    )
    end_x = make_trapezoid("x", area=-read.area / 2, duration=gdur, system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    ref_center = ref.delay + ref.shape_dur / 2
    read_center = read.rise_time + read.flat_time / 2
    shot = 0
    for z in range(nz[0] if nz else 1):
        for y in range(-preparations if z == 0 and not se else 0, ny):
            yy = 0 if y < 0 else y - ny // 2
            zz = z - (nz[0] // 2 if nz else 0)
            pre_y = make_trapezoid("y", area=yy / fov[1], duration=gdur, system=system)
            pre = [pre_x, pre_y]
            if nz:
                pre.append(
                    make_trapezoid("z", area=zz / fov[2], duration=gdur, system=system)
                )
            phase = (
                np.pi * (shot % 2)
                if balanced
                else np.deg2rad(117) * shot * (shot + 1) / 2
            )
            r = copy.copy(rf)
            r.phase_offset = phase
            a = copy.copy(adc)
            ref_phase = phase
            a.phase_offset = 2 * ref_phase - phase + np.pi if se else phase
            seq.add_block(r)
            elapsed = calc_duration(r)
            if se:
                seq.add_block(
                    make_delay(
                        te / 2
                        + rf_center
                        - ref_center
                        - elapsed
                        - (0.0012 if refocusing_crushers else 0)
                    )
                )
                if refocusing_crushers:
                    seq.add_block(
                        make_trapezoid(
                            "x", area=nx / fov[0], duration=0.0012, system=system
                        )
                    )
                rr = copy.copy(ref)
                rr.phase_offset = ref_phase
                seq.add_block(rr)
                elapsed = te / 2 + rf_center - ref_center + calc_duration(ref)
                if refocusing_crushers:
                    seq.add_block(
                        make_trapezoid(
                            "x", area=nx / fov[0], duration=0.0012, system=system
                        )
                    )
                    elapsed += 0.0012
            seq.add_block(*pre)
            elapsed += calc_duration(*pre)
            delay = te + rf_center - read_center - elapsed
            if delay < -1e-9:
                raise ValueError(f"Control TE is infeasible: {delay}")
            if delay > 0:
                seq.add_block(make_delay(delay))
            seq.add_block(read, *([a] if y >= 0 else []))
            elapsed += max(delay, 0) + calc_duration(read, a)
            if balanced:
                post = [
                    end_x,
                    make_trapezoid(
                        "y", area=-yy / fov[1], duration=gdur, system=system
                    ),
                ]
                if nz:
                    post.append(
                        make_trapezoid(
                            "z", area=-zz / fov[2], duration=gdur, system=system
                        )
                    )
                seq.add_block(*post)
                elapsed += calc_duration(*post)
            else:
                crusher = (
                    make_trapezoid(
                        "x",
                        area=spoil_area * spoiler_factors[shot % len(spoiler_factors)],
                        duration=gdur,
                        system=system,
                    )
                    if spoiler_factors
                    else spoil
                )
                post = [crusher]
                if rewind_phase:
                    post.append(
                        make_trapezoid(
                            "y", area=-yy / fov[1], duration=gdur, system=system
                        )
                    )
                    if nz:
                        post.append(
                            make_trapezoid(
                                "z", area=-zz / fov[2], duration=gdur, system=system
                            )
                        )
                seq.add_block(*post)
                elapsed += calc_duration(*post)
            if tr - elapsed < -1e-9:
                raise ValueError(f"Control TR is infeasible: {tr - elapsed}")
            if tr - elapsed > 0:
                seq.add_block(make_delay(tr - elapsed))
            shot += 1
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq
