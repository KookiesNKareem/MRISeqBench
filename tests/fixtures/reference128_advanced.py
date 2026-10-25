"""Independent 128-grid Pulseq authors. Construction performs no MRI simulation."""

from tests.fixtures import (
    rf_excitation_sequences,
    rfov_epi_sequences,
    s3_sequences,
    s4_sequences,
)

# Textbook inner-volume reduced FOV: z-selective 90, y-selective sinc 180
# bracketed by in-plane x crushers (GPU-qualified on all three profiles).
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
        options = {
            "segment_lines": 32,
            "readout_flat": 0.00128,
            "gradient_duration": 0.001,
            "tr_s": 0.0046,
            "preparation_dwell": 10e-6,
        }
        if family == "t1_mapping" and case["hardware"]["field_strength_T"] == 3:
            options.update(
                tr_s=0.0026,
                readout_flat=0.00064,
                gradient_duration=0.0005,
                dummy_count=16,
                ramp_count=1,
                catalyzation=True,
                fat_saturation_duration=0.008,
            )
        elif family == "t1_mapping" and case["hardware"]["field_strength_T"] == 0.55:
            options.update(
                dummy_count=16,
                ramp_count=16,
                fat_saturation_duration=0.008,
                fat_saturation_before_inversion=True,
            )
        return s4_sequences.make_sequence(case, **options)
    options = {"readout_flat": 0.00128}
    if family in {"variable_density", "elliptical_kspace"}:
        options.update(
            spoiler_factors=[1, 1.37, 1.91, 1.11, 1.73, 1.23, 1.59],
            rewind_phase=True,
        )
    if family == "reduced_fov":
        options.update(
            reduced_design="excitation",
            selective_rf_duration=0.0096,
            reduced_rf_dwell=10e-6,
            reduced_refocusing="composite",
            reduced_refocusing_duration=0.0008,
            reduced_flip_deg=90,
            preparations=1,
            repetition_time=0.352,
            reduced_fat_saturation=True,
            reduced_fat_saturation_pulses=2,
            reduced_fat_saturation_duration=0.008,
            reduced_crusher_area=1280,
            reduced_nonclosing_spoilers=True,
        )
    options.update(overrides)
    return s3_sequences.make_sequence(case, mutation, **options)
