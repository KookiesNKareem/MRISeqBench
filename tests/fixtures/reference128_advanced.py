"""Independent 128-grid Pulseq authors. Construction performs no MRI simulation."""

from tests.fixtures import (
    rf_excitation_sequences,
    rfov_epi_sequences,
    s3_sequences,
    s4_sequences,
)

# Textbook inner-volume reduced FOV: z-selective 90, y-selective sinc 180
# bracketed by in-plane x crushers; image qualification is separate.
INNER_VOLUME = {
    "reduced_design": "inner_volume",
    "reduced_refocusing": "sinc",
    "selective_rf_duration": 0.003,
    "reduced_excitation_tbw": 4,
    "reduced_refocusing_duration": 0.008,
    "reduced_refocusing_tbw": 8,
    "reduced_refocusing_thickness": 0.12,
}


def build(case, mutation=None, **overrides):
    """Build a candidate; only achieved-image GPU qualification makes a reference.

    ``mutation`` and ``overrides`` author controls from the same reference
    options (e.g. the inner-volume ``reduced_fov`` design or negative controls).
    """
    task = case["task"]
    family = task["id"]
    if family == "rf_excitation_2d":
        return rf_excitation_sequences.make_sequence(case)
    if family == "rfov_epi":
        return rfov_epi_sequences.make_sequence(case, mutation, **overrides)
    if task["sequence"]["level"] == "S4":
        segments_per_frame = (task["sequence"].get("temporal") or {}).get(
            "segments_per_frame", 4
        )
        options = {
            "segment_lines": task["sequence"]["matrix"][1] // segments_per_frame,
            "readout_flat": 0.00128,
            "gradient_duration": 0.001,
            "tr_s": 0.0046,
            "preparation_dwell": 10e-6,
        }
        if family == "t1_mapping" and case["hardware"]["field_strength_T"] == 3:
            options.update(
                tr_s=0.0024,
                readout_flat=0.00032,
                gradient_duration=0.0004,
                dummy_count=16,
                ramp_count=16,
                fat_saturation_duration=0.008,
                fat_saturation_before_inversion=True,
            )
        elif family == "t1_mapping" and case["hardware"]["field_strength_T"] == 0.55:
            options.update(
                tr_s=0.0044,
                readout_flat=0.00064,
                gradient_duration=0.0007,
                dummy_count=8,
                ramp_count=8,
                fat_saturation_duration=0.008,
                fat_saturation_before_inversion=True,
            )
        elif family == "t1_mapping" and case["hardware"]["field_strength_T"] == 1.5:
            options.update(interleaved_segments=True, readout_flat=0.00064)
        elif (
            family in {"t2_mapping", "t2_prepared", "adiabatic_inversion"}
            and case["hardware"]["field_strength_T"] == 0.55
        ):
            options.update(dummy_count=64, ramp_count=16)
        if (
            family in {"t2_mapping", "t2_prepared"}
            and case["hardware"]["field_strength_T"] == 3
        ):
            options.update(readout_flat=0.00032, readout_flip_deg=45)
        elif family == "t2_prepared" and case["hardware"]["field_strength_T"] == 1.5:
            options["readout_flat"] = 0.00064
        elif (
            family == "adiabatic_inversion"
            and case["hardware"]["field_strength_T"] == 3
        ):
            options.update(
                readout_flat=0.00032,
                gradient_duration=0.0004,
                tr_s=0.00216 if task["sequence"]["fov_mm"][0] == 220 else 0.0023,
            )
        elif (
            family == "adiabatic_inversion"
            and case["hardware"]["field_strength_T"] == 1.5
        ):
            options.update(
                readout_flat=0.00064,
                gradient_duration=0.0005,
                readout_flip_deg=60,
            )
            if task["sequence"]["application"]["inversion_time_ms"] == 650:
                larger_fov = task["sequence"]["fov_mm"][0] == 220
                options.update(
                    readout_flat=0.00046 if larger_fov else 0.00052,
                    readout_dwell_s=3.5e-6 if larger_fov else 4e-6,
                    gradient_duration=0.00064,
                    readout_flip_deg=35,
                    dummy_count=64,
                    ramp_count=16,
                )
        elif family == "cardiac_respiratory_bssfp":
            field = case["hardware"]["field_strength_T"]
            if field == 0.55:
                options.update(
                    tr_s=0.0044,
                    readout_flat=0.00064,
                    gradient_duration=0.0007,
                    readout_flip_deg=50,
                    dummy_count=64,
                    ramp_count=16,
                )
            elif field == 1.5:
                options.update(
                    readout_flat=0.00064,
                    dummy_count=32,
                    ramp_count=16,
                )
            elif field == 3:
                options.update(
                    tr_s=0.0023,
                    readout_flat=0.00032,
                    gradient_duration=0.0004,
                    readout_flip_deg=45,
                    dummy_count=64,
                    ramp_count=16,
                )
        options.update(overrides)
        return s4_sequences.make_sequence(case, mutation, **options)
    spec, field = task["sequence"], case["hardware"]["field_strength_T"]
    edited_flip = spec.get("flip_angle_deg")
    options = {"readout_flat": 0.00128}
    if field == 3 and (not edited_flip or family == "variable_density"):
        options["readout_flat"] = 0.00032
    if family in {"variable_density", "elliptical_kspace"} or edited_flip:
        options.update(
            spoiler_factors=[1, 1.37, 1.91, 1.11, 1.73, 1.23, 1.59],
            rewind_phase=True,
        )
    if edited_flip:
        options["spoiler_scale"] = 1.5
        if (
            family in {"compressed_sensing", "partial_fourier", "caipirinha"}
            or (
                field == 3
                and family
                in {
                    "variable_density",
                    "grappa_sampling",
                    "sense_sampling",
                }
            )
            or (family == "variable_density" and field == 1.5)
        ):
            options.update(
                readout_flat=0.00032 if field == 3 else 0.00064,
                spoiler_scale=3,
                repetition_time=0.035 if family == "partial_fourier" else 0.04,
            )
        elif family in {"grappa_sampling", "sense_sampling"} and field == 1.5:
            options["readout_flat"] = 0.00064
    if family == "elliptical_kspace" and (field >= 1.5 or edited_flip == 30):
        options.update(
            readout_flat=0.00032 if field == 3 else 0.00064,
            spoiler_axis="z",
            spoiler_scale=3,
        )
    if family == "reduced_fov":
        options.update(
            reduced_design="excitation",
            selective_rf_duration=0.0096,
            reduced_rf_dwell=10e-6,
            reduced_refocusing="composite",
            reduced_refocusing_duration=0.0008,
            reduced_excitation_tbw=48,
            reduced_flip_deg=60,
            preparations=1,
            repetition_time=0.352,
            reduced_fat_saturation=True,
            reduced_fat_saturation_pulses=2,
            reduced_fat_saturation_duration=0.008,
            reduced_crusher_area=1280 * (200 / spec["fov_mm"][0]),
            reduced_nonclosing_spoilers=True,
            readout_flat=0.00032 if field == 3 else 0.00064,
        )
        if field == 1.5:
            larger_fov = spec["fov_mm"][0] == 220
            options.update(
                readout_flat=0.00043 if larger_fov else 0.00047,
                reduced_readout_dwell=3.3e-6 if larger_fov else 3.6e-6,
            )
    options.update(overrides)
    return s3_sequences.make_sequence(case, mutation, **options)
