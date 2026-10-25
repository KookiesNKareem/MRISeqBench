"""Independent Pulseq authors for S3 nominal acquisitions and corruptions."""

import copy

import numpy as np
import pypulseq as pp

from tests.fixtures.gradients import trapezoid


def mask_for(fam, matrix, sampling):
    """Author mask without importing any judge geometry or reconstruction code."""
    _nx, ny, *nz = matrix
    mask = np.zeros(matrix, bool)
    width = sampling.get("calibration_lines", 8)
    centre = list(range(ny // 2 - width // 2, ny // 2 + (width + 1) // 2))
    if fam in {"sense_sampling", "grappa_sampling"}:
        step = int(sampling.get("acceleration", 2))
        mask[:, sorted(set(range(0, ny, step)) | set(centre))] = True
    elif fam == "caipirinha":
        for j, z in enumerate(range(0, nz[0], 2)):
            for y in range(j % 2, ny, 2):
                mask[:, y, z] = True
        for y in centre:
            for z in range(nz[0] // 2 - width // 2, nz[0] // 2 + (width + 1) // 2):
                mask[:, y, z] = True
    elif fam in {"compressed_sensing", "variable_density"}:
        chosen = centre.copy()
        pool = np.array([y for y in range(ny) if y not in chosen])
        rng = np.random.Generator(np.random.PCG64(sampling.get("mask_seed", 42)))
        count = int(np.ceil(ny / sampling.get("acceleration", 3)))
        while len(chosen) < count:
            weights = np.array([1 - abs(y - ny // 2) / (ny / 2 + 1) for y in pool])
            value = int(rng.choice(pool, p=weights / weights.sum()))
            chosen.append(value)
            pool = np.array(
                [
                    y
                    for y in pool
                    if abs(y - value) >= (2 if fam == "compressed_sensing" else 1)
                ]
            )
        mask[:, chosen] = True
    elif fam == "partial_fourier":
        count = int(np.ceil(ny * sampling.get("partial_fourier_fraction", 0.75)))
        mask[:, ny - count :] = True
    elif fam == "reduced_fov":
        mask[:, np.arange(ny)[(np.arange(ny) - ny // 2) % 2 == 0]] = True
    else:
        for y in range(ny):
            for z in range(nz[0]):
                if ((y - ny // 2) / (ny / 2)) ** 2 + (
                    (z - nz[0] // 2) / (nz[0] / 2)
                ) ** 2 <= 1:
                    mask[:, y, z] = True
    return mask


def make_sequence(
    case,
    mutation=None,
    *,
    preparations=64,
    ordering="forward",
    readout_flat=0.0016,
    echo_time=None,
    repetition_time=None,
    selective_rf_duration=0.012,
    reduced_rf_dwell=None,
    reduced_refocusing="hard",
    reduced_refocusing_duration=0.0016,
    reduced_crusher_area=400,
    reduced_flip_deg=None,
    reduced_fat_saturation=False,
    reduced_fat_saturation_pulses=1,
    reduced_fat_saturation_duration=0.004,
    reduced_nonclosing_spoilers=False,
    spoiler_factors=None,
    rewind_phase=False,
    reduced_design="excitation",
    reduced_excitation_tbw=32,
    reduced_refocusing_tbw=16,
    reduced_refocusing_thickness=None,
    receiver_phase_offset=None,
    calibration_lines=None,
    rf_spoil_deg=117,
    spoiler_scale=1.0,
    gre_excitation="block",
    spoiler_axis="x",
):
    """Author an S3 control.

    ``reduced_design='inner_volume'`` is the textbook orthogonal design: a
    z-selective excitation and a y-selective sinc refocuser bracketed by x
    crushers, so unrefocused exterior signal is removed by intravoxel crusher
    dephasing rather than by y-selective excitation. ``calibration_lines``
    overrides the authored ACS width (a mask control). Valid alternatives
    for the positive ensemble: ``ordering`` (forward/reverse/centric),
    ``readout_flat`` (bandwidth), ``rf_spoil_deg`` and ``spoiler_scale``
    (spoiling), ``gre_excitation='sinc'`` (slice-selective sinc in 2D,
    nonselective sinc in 3D).
    """
    spec, h = case["task"]["sequence"], case["hardware"]
    fam = case["task"]["id"]
    nx, ny, *nz = spec["matrix"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    sampling = {k: v for k, v in (spec.get("sampling") or {}).items() if v is not None}
    if calibration_lines is not None:
        sampling["calibration_lines"] = calibration_lines
    mask = mask_for(fam, tuple(spec["matrix"]), sampling)
    if mutation == "full_sampling":
        mask[:] = True
    if mutation == "missing_line":
        coords = np.argwhere(mask.any(axis=0))[-1]
        mask[(slice(None), *coords)] = False
    if mutation == "no_calibration":
        if nz:
            mask[:, :, nz[0] // 2] = False
        else:
            mask[:, ny // 2] = False
    if mutation == "wrong_shift" and nz:
        mask[:] = False
        mask[:, ::2, ::2] = True
        w = sampling.get("calibration_lines", 8)
        mask[
            :,
            ny // 2 - w // 2 : ny // 2 + w // 2,
            nz[0] // 2 - w // 2 : nz[0] // 2 + w // 2,
        ] = True
    if mutation == "wrong_seed":
        mask[:, [0, 1]] = mask[:, [1, 0]]
        # Always perturb a peripheral retained line and a peripheral omitted one.
        ys = mask.any(axis=0)
        if ys.ndim == 1:
            retained = [y for y in np.flatnonzero(ys) if abs(y - ny // 2) > ny // 4]
            omitted = np.flatnonzero(~ys)
            mask[:, retained[0]] = False
            mask[:, omitted[0]] = True
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
    reduced = fam == "reduced_fov"
    flip = spec.get("flip_angle_deg") or (20 if reduced else 5)
    if reduced and reduced_flip_deg is not None:
        flip = reduced_flip_deg
    te = (spec.get("te_ms") or {}).get("target", 25 if reduced else 5) / 1000
    tr = (spec.get("tr_ms") or {}).get("target", 160 if reduced else 12) / 1000
    if echo_time is not None:
        te = echo_time
    if repetition_time is not None:
        tr = repetition_time
    if mutation == "all_refocusing":
        flip, tr = 180, 0.024
    if reduced:
        rf, selective, rephase = pp.make_sinc_pulse(
            np.deg2rad(flip),
            duration=selective_rf_duration,
            dwell=reduced_rf_dwell or 0,
            slice_thickness=fov[1] / 2,
            time_bw_product=reduced_excitation_tbw,
            apodization=0.5,
            return_gz=True,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
        if reduced_rf_dwell is None:
            # Preserve historical sample-hold pulses on the 1-us RF raster.
            rf.signal = np.repeat(rf.signal.reshape(-1, 20).mean(axis=1), 20)
        selective.channel = rephase.channel = (
            "z" if reduced_design == "inner_volume" else "y"
        )
        if mutation == "wrong_position" and reduced_design != "inner_volume":
            rf.freq_offset = selective.amplitude * fov[1] / 2
        preparations = min(preparations, 20)
    else:
        rf = None
    gre_gradient = gre_rephase = None
    if not reduced and gre_excitation == "sinc":
        rf, gre_gradient, gre_rephase = pp.make_sinc_pulse(
            np.deg2rad(flip),
            duration=0.0012,
            slice_thickness=0.005,
            time_bw_product=4,
            apodization=0.5,
            return_gz=True,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
        if nz:
            gre_gradient = gre_rephase = None
    elif not reduced and gre_excitation != "block":
        raise ValueError(f"Unsupported GRE excitation: {gre_excitation}")
    if not reduced and rf is None:
        rf = pp.make_block_pulse(
            np.deg2rad(flip),
            duration=0.005 if mutation == "all_refocusing" else 0.0004,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
    ref = pp.make_block_pulse(
        np.pi,
        duration=reduced_refocusing_duration if reduced else 0.0008,
        delay=system.rf_dead_time,
        phase_offset=np.pi / 2,
        system=system,
        use="refocusing",
    )
    ref_gradient = None
    if reduced and reduced_refocusing == "composite":
        # Standard 90x-180y-90x composite refocusing, represented as one RF
        # event so its full coherent envelope determines the nominal rotation.
        n = round(reduced_refocusing_duration / system.rf_raster_time)
        half = n // 2
        if (
            n % 2
            or abs(n * system.rf_raster_time - reduced_refocusing_duration) > 1e-12
        ):
            raise ValueError("Composite refocusing lobes must fit the RF raster")
        amplitude = 1 / (2 * reduced_refocusing_duration)
        signal = amplitude * np.r_[np.full(half, -1j), np.ones(n), np.full(half, -1j)]
        ref = pp.make_arbitrary_rf(
            signal,
            np.pi,
            no_signal_scaling=True,
            delay=system.rf_dead_time,
            phase_offset=np.pi / 2,
            system=system,
            use="refocusing",
            center=reduced_refocusing_duration,
        )
    elif reduced and reduced_refocusing == "sinc":
        ref, ref_gradient, _ = pp.make_sinc_pulse(
            np.pi,
            duration=reduced_refocusing_duration,
            slice_thickness=reduced_refocusing_thickness or fov[1] / 2,
            time_bw_product=reduced_refocusing_tbw,
            apodization=0.5,
            phase_offset=np.pi / 2,
            return_gz=True,
            delay=system.rf_dead_time,
            system=system,
            use="refocusing",
        )
        ref_gradient.channel = "y"
        if mutation == "no_selection":
            ref_gradient = None
        if mutation == "wrong_position":
            ref.freq_offset = ref_gradient.amplitude * fov[1] / 2
    elif reduced_refocusing != "hard":
        raise ValueError(f"Unsupported reduced-FOV refocusing: {reduced_refocusing}")
    read = trapezoid("x", flat_area=nx / fov[0], flat_time=readout_flat, system=system)
    pre_x = trapezoid("x", area=-read.area / 2, duration=0.0008, system=system)
    spoil_area = max(nx, 32) * 2 / fov[0] * spoiler_scale
    if spoiler_axis == "z":
        # Two cycles across the (thick) 3D partition voxel along z.
        spoil_area = 2 * nz[0] / fov[2] * spoiler_scale
    spoil_duration = 0.0012
    if spoiler_factors:
        spoil_duration = max(
            spoil_duration,
            np.ceil(
                pp.calc_duration(
                    trapezoid(
                        "x", area=spoil_area * max(spoiler_factors), system=system
                    )
                )
                / system.grad_raster_time
            )
            * system.grad_raster_time,
        )
    spoil = trapezoid(
        spoiler_axis, area=spoil_area, duration=spoil_duration, system=system
    )
    if reduced:
        saturation = pp.make_block_pulse(
            np.pi / 2,
            duration=0.001,
            delay=system.rf_dead_time,
            system=system,
            use="excitation",
        )
        seq.add_block(saturation)
        seq.add_block(
            trapezoid("x", area=1280, duration=0.003, system=system),
            trapezoid("y", area=960, duration=0.003, system=system),
        )
    fat_saturation = None
    if reduced and reduced_fat_saturation:
        # CHESS selective90+spoiling (Haase1985, doi:10.1088/0031-9155/30/4/008).
        # Preparation uses the scanner's intrinsic fat frequency.
        # Field-scaled duration preserves the spectral separation from water.
        duration = (
            round(reduced_fat_saturation_duration * 3 / h["field_strength_T"] / 20e-6)
            * 20e-6
        )
        fat_saturation = pp.make_gauss_pulse(
            np.pi / 2,
            duration=duration,
            dwell=reduced_rf_dwell or 0,
            time_bw_product=2,
            freq_offset=case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"],
            delay=system.rf_dead_time,
            system=system,
            use="saturation",
        )
        if reduced_rf_dwell is None:
            fat_saturation.signal = np.repeat(
                fat_saturation.signal.reshape(-1, 20).mean(axis=1), 20
            )
    pairs = [tuple(p) for p in np.argwhere(mask.any(axis=0))]
    if ordering == "reverse":
        pairs.reverse()
    elif ordering == "centric":
        centre = (ny // 2, nz[0] // 2) if nz else (ny // 2,)
        pairs.sort(key=lambda p: (sum((a - c) ** 2 for a, c in zip(p, centre)), p))
    elif ordering != "forward":
        raise ValueError(f"Unsupported control ordering: {ordering}")
    if mutation == "repeated_lines":
        pairs = [pairs[0]] * len(pairs)
    for shot, pair in enumerate([None] * preparations + pairs):
        phase = (
            0.0
            if mutation == "no_spoiling"
            # Wrapped: .seq phases are written to finite precision, and an
            # unwrapped quadratic phase would decorrelate RF and ADC rounding.
            else np.mod(np.deg2rad(rf_spoil_deg) * shot * (shot + 1) / 2, 2 * np.pi)
        )
        r = copy.copy(rf)
        r.phase_offset = phase
        rfc = r.delay + r.shape_dur / 2
        elapsed = 0.0
        if fat_saturation is not None:
            if reduced_fat_saturation_pulses not in (1, 2, 3):
                raise ValueError("Use one to three CHESS preparation pulses")
            if reduced_nonclosing_spoilers and reduced_fat_saturation_pulses != 2:
                raise ValueError("Nonclosing CHESS spoilers require two pulses")
            for fat_index in range(reduced_fat_saturation_pulses):
                fat = copy.copy(fat_saturation)
                fat.phase_offset = phase
                seq.add_block(fat)
                elapsed += pp.calc_duration(fat)
                areas = {"x": 400 + 200 * fat_index, "y": 300 + 100 * fat_index}
                if reduced_nonclosing_spoilers:
                    areas = [
                        {"x": 2 * nx / fov[0], "y": ny / fov[1]},
                        {"x": 4 * nx / fov[0], "y": 3 * ny / fov[1]},
                    ][fat_index]
                duration = max(
                    0.0012,
                    *(
                        pp.calc_duration(trapezoid(axis, area=area, system=system))
                        for axis, area in areas.items()
                    ),
                )
                gradients = [
                    trapezoid(axis, area=area, duration=duration, system=system)
                    for axis, area in areas.items()
                ]
                seq.add_block(*gradients)
                elapsed += pp.calc_duration(*gradients)
            rfc += elapsed
        if reduced and mutation == "no_selection" and reduced_design == "inner_volume":
            # Keep the slice-selective excitation; the refocuser loses selection.
            seq.add_block(r, selective)
            elapsed += pp.calc_duration(r, selective)
            seq.add_block(rephase)
            elapsed += pp.calc_duration(rephase)
        elif reduced and mutation != "no_selection":
            seq.add_block(r, selective)
            elapsed += pp.calc_duration(r, selective)
            seq.add_block(rephase)
            elapsed += pp.calc_duration(rephase)
        elif gre_gradient is not None:
            seq.add_block(r, gre_gradient)
            elapsed += pp.calc_duration(r, gre_gradient)
            seq.add_block(gre_rephase)
            elapsed += pp.calc_duration(gre_rephase)
        else:
            seq.add_block(r)
            elapsed += pp.calc_duration(r)
        if reduced:
            crusher_duration = max(
                0.0012,
                pp.calc_duration(
                    trapezoid("x", area=reduced_crusher_area, system=system)
                ),
            )
            crusher = trapezoid(
                "x", area=reduced_crusher_area, duration=crusher_duration, system=system
            )
            seq.add_block(crusher)
            elapsed += pp.calc_duration(crusher)
            rr = copy.copy(ref)
            rr.phase_offset = phase + np.pi / 2
            delay = te / 2 + rfc - rr.delay - rr.shape_dur / 2 - elapsed
            seq.add_block(pp.make_delay(delay))
            refocusing = [rr] + ([ref_gradient] if ref_gradient is not None else [])
            seq.add_block(*refocusing)
            elapsed += delay + pp.calc_duration(*refocusing)
            seq.add_block(crusher)
            elapsed += pp.calc_duration(crusher)
        y, z = (
            (0, 0)
            if pair is None
            else (pair[0] - ny // 2, pair[1] - (nz[0] // 2) if nz else 0)
        )
        gradients = [
            pre_x,
            trapezoid("y", area=y / fov[1], duration=0.0008, system=system),
        ]
        if nz:
            gradients.append(
                trapezoid("z", area=z / fov[2], duration=0.0008, system=system)
            )
        seq.add_block(*gradients)
        elapsed += pp.calc_duration(*gradients)
        delay = te + rfc - read.rise_time - read.flat_time / 2 - elapsed
        seq.add_block(pp.make_delay(delay))
        elapsed += delay
        events = [read]
        if pair is not None:
            xs = np.flatnonzero(mask[(slice(None), *pair)])
            adc = pp.make_adc(
                len(xs),
                dwell=read.flat_time / nx,
                delay=read.rise_time + xs[0] * read.flat_time / nx,
                phase_offset=phase,
                system=system,
            )
            if mutation == "receiver_phase":
                adc.phase_offset += (
                    np.pi / 2
                    if receiver_phase_offset is None
                    else receiver_phase_offset
                )
            events.append(adc)
        seq.add_block(*events)
        elapsed += pp.calc_duration(*events)
        if reduced:
            # Vary two in-plane spoiler moments to prevent long-T2 stimulated
            # pathways from returning at later selective spin echoes.
            post = [
                trapezoid(
                    "x",
                    area=1280 * (1 + (shot % 13) / 13),
                    duration=0.003,
                    system=system,
                ),
                trapezoid(
                    "y",
                    area=960 * (1 + ((shot * 7) % 11) / 11),
                    duration=0.003,
                    system=system,
                ),
            ]
            seq.add_block(*post)
            elapsed += pp.calc_duration(*post)
        else:
            post = [
                trapezoid(
                    "x",
                    area=spoil_area * spoiler_factors[shot % len(spoiler_factors)],
                    duration=spoil_duration,
                    system=system,
                )
                if spoiler_factors
                else spoil
            ]
            if mutation == "no_spoiling":
                # Unspoiled control: constant RF phase and fully rewound
                # gradients (a coherent steady state).
                post = [
                    trapezoid(
                        "x", area=-read.area / 2, duration=spoil_duration, system=system
                    )
                ]
            if rewind_phase or mutation == "no_spoiling":
                post.append(
                    trapezoid(
                        "y", area=-y / fov[1], duration=spoil_duration, system=system
                    )
                )
                if nz:
                    post.append(
                        trapezoid(
                            "z",
                            area=-z / fov[2],
                            duration=spoil_duration,
                            system=system,
                        )
                    )
            seq.add_block(*post)
            elapsed += pp.calc_duration(*post)
        seq.add_block(pp.make_delay(tr - elapsed))
    if mutation == "wrong_fov":
        for key, value in seq.grad_library.data.items():
            values = list(value)
            values[0] *= 0.5
            seq.grad_library.data[key] = tuple(values)
    ok, errors = seq.check_timing()
    if not ok:
        raise ValueError(errors)
    return seq
