"""Native 128-sample S1/S2 pulse authors.

The case supplies the acquisition requirements. These settings select physical
waveforms; they never rewrite a case, image target, or acceptance threshold.
"""

from tests.fixtures import s1_sequences, s2_sequences


def options_for(case):
    """Hardware-feasible pulse settings for the 128 in-plane acquisition."""
    spec, hardware = case["task"]["sequence"], case["hardware"]
    if spec["matrix"][:2] != [128, 128]:
        raise ValueError("This author requires a native 128 x 128 acquisition")
    family, field = case["task"]["id"], hardware["field_strength_T"]
    low = hardware["id"] == "low_field"
    if spec["level"] == "S1":
        options = {"readout_flat": 0.00096, "preparations": 128}
        if family in {"gre_t1w", "gre_t2star"}:
            options.update(
                spoiler_factors=[1, 1.37, 1.91, 1.11, 1.73, 1.23, 1.59],
                rewind_phase=True,
            )
            if field == 3:
                options["readout_flat"] = 0.00032
            if family == "gre_t1w":
                options["flip_angle"] = 20
            else:
                options["preparations"] = min(
                    128,
                    int(spec["duration_s"] / (spec["tr_ms"]["target"] / 1000))
                    - spec["matrix"][1]
                    - 1,
                )
        if family == "bssfp":
            options["rf_duration"] = 0.0004
            if field == 3:
                # A shorter allowed TR moves the fat stop band beyond the
                # left edge of the declared B0 map, retaining ordinary bSSFP.
                options.update(
                    repetition_time=0.00476,
                    echo_time=0.00238,
                    rf_duration=0.00068 if spec["flip_angle_deg"] == 45 else 0.00044,
                    readout_flat=0.00032,
                )
            elif low:
                options.update(
                    readout_flat=0.00064,
                    repetition_time=(
                        0.0052
                        if spec.get("slice_thickness_mm") is not None
                        else 0.00476
                    ),
                    echo_time=(
                        0.0026
                        if spec.get("slice_thickness_mm") is not None
                        else 0.00238
                    ),
                    rf_duration=(
                        0.0008 if spec.get("slice_thickness_mm") is not None else 0.0004
                    ),
                )
            elif spec["flip_angle_deg"] == 45:
                options["rf_duration"] = (
                    0.00072 if spec.get("slice_thickness_mm") is not None else 0.001
                )
        if family == "spin_echo_t2w" and spec.get("slice_thickness_mm") is None:
            if field == 3:
                options.update(
                    water_selective_order=2, water_selective_refocusing_order=2
                )
            elif field == 1.5:
                options.update(readout_flat=0.00064, corpse_refocusing=True)
        elif family == "spin_echo_t2w" and field == 1.5:
            options["readout_flat"] = 0.00064
        if family == "spin_echo_t2w":
            options["refocusing_crushers"] = True
        return options

    options = {"cartesian_flat": 0.00096}
    if family.startswith("epi"):
        single = family == "epi_single_shot"
        if single:
            flat, spacing, te = {
                "low_field": (0.000768, 0.00165, 0.043045),
                "standard_1_5t": (0.000512, 0.001, 0.024665),
                "high_performance": (0.000384, 0.00079, 0.01899),
            }[hardware["id"]]
            options.update(cartesian_flat=flat, epi_spacing=spacing, echo_time=te)
            requested_te = (spec.get("te_ms") or {}).get("target")
            if requested_te is not None:
                options["echo_time"] = requested_te / 1000
                if requested_te == 90:
                    fov_x = spec["fov_mm"][0]
                    spacing_by_fov = {
                        "low_field": {200.0: 0.00166, 220.0: 0.00158},
                        "standard_1_5t": {200.0: 0.00101, 220.0: 0.00097},
                        "high_performance": {200.0: 0.00079, 220.0: 0.00075},
                    }
                    options["epi_spacing"] = spacing_by_fov[hardware["id"]][fov_x]
                    options["echo_time"] += (
                        {
                            "low_field": 5e-6,
                            "standard_1_5t": 10e-6,
                            "high_performance": 10e-6,
                        }[hardware["id"]]
                        if fov_x == 200
                        else 0
                    )
        else:
            options["epi_spacing"] = 0.0023 if low else 0.002
        options.update(
            overlap_blips=single,
            water_selective_order=2,
            epi_tr=0.4 if single else 0.35,
        )
    elif family == "spiral_gre":
        options.update(
            curved_duration=0.022
            if low
            else (0.010 if hardware["id"] == "high_performance" else 0.014),
            spiral_power=2 / 3,
            spiral_adc_oversampling=4,
            water_selective_order=1 if low else 2,
        )
    elif family == "stack_of_stars":
        options.update(
            interleave_partitions=True,
            steady_state_preparations=min(
                256,
                int(spec["duration_s"] / (spec["tr_ms"]["target"] / 1000))
                - spec["readout"]["spokes"] * spec["matrix"][2]
                - 1,
            ),
        )
        if hardware["id"] == "high_performance":
            options["cartesian_flat"] = 0.00032
    elif family == "ute_radial":
        options.update(
            cartesian_flat=0.00064 if low else 0.00032,
            ute_pre_duration=0.0005 if low else 0.0003,
            ute_te=spec["te_ms"]["target"] / 1000,
            radial_tr_s=0.005,
            steady_state_preparations=700,
        )
        if hardware["id"] == "high_performance":
            options["cartesian_flat"] = 0.00016
    elif family == "grase":
        flat, offset, interval, centric = {
            "low_field": (0.00096, 0.0014, 0.020, True),
            "standard_1_5t": (0.00064, 0.0009, 0.012, True),
            "high_performance": (0.00032, 0.0005, 0.008, False),
        }[hardware["id"]]
        options.update(
            cartesian_flat=flat,
            grase_offset=offset,
            grase_spin_echo_spacing=interval,
            centric_grase=centric,
            ref_duration=0.0006,
            water_selective_order=2,
            water_selective_refocusing_order=2,
        )
        if field == 3:
            options.update(
                grase_offset=0.00075,
                echo_time=(
                    0.039255
                    if spec.get("te_ms", {}).get("target") == 40
                    else spec["te_ms"]["target"] / 1000
                    + (5e-6 if spec["fov_mm"][0] == 200 else 0)
                ),
                split_grase_polarities=True,
            )
        elif low:
            options.update(
                centric_grase=False,
                split_grase_polarities=True,
                echo_time=0.0386,
                water_selective_order=0,
                water_selective_refocusing_order=0,
                fat_saturation_pulses=2,
            )
        else:
            options.update(
                centric_grase=False,
                split_grase_polarities=True,
                echo_time=0.0391,
                water_selective_refocusing_order=0,
                fat_saturation_pulses=2,
                corpse_refocusing=True,
                refocusing_crushers=True,
            )
        requested_te = spec.get("te_ms", {}).get("target")
        if requested_te is not None and requested_te != 40:
            options["echo_time"] = requested_te / 1000 + (
                5e-6 if spec["fov_mm"][0] == 200 else 0
            )
    if family in {"tse_t2w", "propeller_tse"} and field == 3:
        options.update(
            water_selective_order=0,
            water_selective_refocusing_order=0,
            fat_saturation_pulses=2,
            ref_duration=0.0006,
            refocusing_crushers=True,
        )
    if family == "propeller_tse" and field == 1.5:
        options.update(
            cartesian_flat=0.00064,
            water_selective_order=0,
            water_selective_refocusing_order=0,
            ref_duration=0.0006,
            corpse_refocusing=True,
            refocusing_crushers=True,
        )
    if family == "tse_t2w":
        options["centric_tse"] = True
        if field == 1.5:
            options.update(
                ref_duration=0.0006,
                corpse_refocusing=True,
                refocusing_crushers=True,
            )
    if family == "propeller_tse" and spec.get("te_ms"):
        options["echo_time"] = spec["te_ms"]["target"] / 1000
    if family == "radial_readout" and spec.get("te_ms"):
        options["radial_te"] = spec["te_ms"]["target"] / 1000
    if family == "radial_readout":
        options["steady_state_preparations"] = 256
        if hardware["id"] == "high_performance":
            options["cartesian_flat"] = 0.00032
        elif spec.get("flip_angle_deg"):
            options["cartesian_flat"] = 0.00064
    return options


def build(case):
    """Return a timing-checked candidate; qualification requires GPU replay."""
    author = s1_sequences if case["task"]["sequence"]["level"] == "S1" else s2_sequences
    return author.make_sequence(case, **options_for(case))


def chess_epi_options(case):
    """Single-shot EPI with one CHESS + spoiler instead of binomial excitation.

    The 0.4 ms block excitation moves the echo centre half a 10 us raster; the
    small TE shift keeps each readout on the gradient raster.
    """
    if case["task"]["id"] != "epi_single_shot":
        raise ValueError("CHESS EPI options apply to epi_single_shot")
    options = options_for(case)
    shift = 0.0 if case["hardware"]["id"] == "low_field" else 5e-6
    options.update(
        water_selective_order=0,
        fat_saturation_pulses=1,
        echo_time=options["echo_time"] + shift,
    )
    return options
