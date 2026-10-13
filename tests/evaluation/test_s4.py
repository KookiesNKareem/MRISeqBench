"""Independent cardiac preparation/trigger controls and adversarial waveforms."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pypulseq import Sequence

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.preflight import preflight
from mriseqbench.evaluation.s4_sequence import inspect_sequence
from mriseqbench.evaluation.sar import evaluate_sar
from tests.fixtures.qualify_s4 import applicable_mutations
from tests.fixtures.s4_rf_predictor import affine_rf_history, transition
from tests.fixtures.s4_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping")
HARDWARE = ("low_field", "standard_1_5t", "high_performance")


def case_for(family="adiabatic_inversion", hardware="high_performance"):
    catalog = Catalog(ROOT)
    return catalog.resolve(catalog.get("tasks", family), hardware_ref=hardware)


def test_independent_rf_predictor_matches_closed_form_free_precession():
    density, t1, t2, frequency, duration = 0.1, 0.955, 0.049, -217.145, 0.007
    state = np.array([0.03, -0.04, -0.07, 1.0])
    actual = transition((density, t1, t2, frequency), 0.0, 0.0, duration) @ state
    transverse = complex(*state[:2]) * np.exp(-duration / t2) * np.exp(
        -2j * np.pi * frequency * duration
    )
    expected = [
        transverse.real,
        transverse.imag,
        density + (state[2] - density) * np.exp(-duration / t1),
        1.0,
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_independent_rf_predictor_retains_transverse_history_between_pulses():
    # Two coherent 90-degree x pulses produce inversion. Resetting transverse
    # magnetization between them would instead create a second transverse echo.
    pulses = [
        SimpleNamespace(
            start=start,
            end=start + 0.001,
            edges=np.array([start, start + 0.001]),
            signal=np.array([250.0 + 0j]),
            phase=0.0,
            frequency=0.0,
        )
        for start in (0.0, 0.002)
    ]
    acquisition = SimpleNamespace(
        pulses=pulses, adc=np.array([0.001, 0.003]), adc_phase=np.zeros(2)
    )
    signal = affine_rf_history(acquisition, (1.0, 1e9, 1e9, 0.0))
    np.testing.assert_allclose(signal, [1j, 0j], atol=5e-12, rtol=0)


def test_s4_contract_requires_bart_without_changing_image_limits():
    case = case_for()
    assert case["evaluation"]["reconstruction"] == "bart_cartesian_fft@1"
    assert case["evaluation"]["calibrated"] is True
    assert case["evaluation"]["case_image_criteria"]["static"] == {
        "complex_nrmse_max": 0.45,
        "magnitude_nrmse_max": 0.35,
        "phase_rms_rad_max": 0.35,
        "tissue_signal_fraction_min": 0.5,
        "contrast_retention_min": 0.5,
        "outside_energy_fraction_max": 0.25,
    }


def test_bart_cardiac_target_matches_independent_inverse_dft():
    from mriseqbench.evaluation.s1_sampling import analytic_reference
    from mriseqbench.evaluation.s4 import reference_image

    case = case_for()
    case["task"]["sequence"]["matrix"] = [4, 4]
    response = np.array([[0.2j], [0.15], [0.1 - 0.05j]])
    acquired = SimpleNamespace(k=np.zeros((1, 3)))
    actual = reference_image(acquired, case, ROOT, response, None)
    indices = np.indices((4, 4)).reshape(2, -1).T
    k = np.column_stack(((indices - 2) / 0.2, np.zeros(16)))
    grid = SimpleNamespace(k=k, adc=np.arange(16))
    raw = analytic_reference(grid, case, ROOT, np.broadcast_to(response, (3, 16)))
    positions = ((indices + 0.5) / 4 - 0.5) * 0.2
    expected = (np.exp(2j * np.pi * (positions @ k[:, :2].T)) @ raw / 16).reshape(
        (4, 4)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


def physical_events(sequence):
    """Read physical file events directly; no evaluator or RF/ADC use labels."""
    clock = 0.0
    triggers, acquisitions = [], []
    for number in sequence.block_events:
        block = sequence.get_block(number)
        if getattr(block, "trigger", None) is not None:
            for event in block.trigger.values():
                if event.type == "trigger":
                    triggers.append((clock + event.delay, event.channel))
        if block.adc is not None:
            adc = block.adc
            acquisitions.append(
                clock + adc.delay + (np.arange(adc.num_samples) + 0.5) * adc.dwell
            )
        clock += sequence.block_durations[number]
    return triggers, acquisitions


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("hardware", HARDWARE)
def test_independent_positive_has_real_ecg_windows_and_valid_hardware(
    tmp_path, family, hardware
):
    case = case_for(family, hardware)
    sequence = make_sequence(case)
    path = tmp_path / "control.seq"
    sequence.write(str(path))
    readback = Sequence(sequence.system)
    readback.read(str(path))
    assert readback.check_timing()[0]
    assert all(item["passed"] for item in preflight(path, case))
    assert all(item["passed"] for item in evaluate_sar(path, case["hardware"]))
    assert all(item["passed"] for item in inspect_sequence(path, case).checks)
    triggers, acquisitions = physical_events(readback)
    assert triggers and all(channel == "physio1" for _, channel in triggers)
    assert all(abs(time - round(time)) < 1e-10 for time, _ in triggers)
    trigger_times = np.array([time for time, _ in triggers])
    for adc in acquisitions:
        trigger = trigger_times[
            np.searchsorted(trigger_times, adc[0], side="right") - 1
        ]
        assert np.all(adc - trigger >= 0.6 - 1e-10)
        assert np.all(adc - trigger <= 0.8 + 1e-10)
    frames = 3 if family in {"t1_mapping", "t2_mapping"} else 1
    assert len(acquisitions) == 32 * frames
    assert sum(len(adc) for adc in acquisitions) == 32 * 32 * frames
    assert readback.duration()[0] <= case["task"]["sequence"]["duration_s"]


def rotate_exact_sample_hold(rf, offsets_hz):
    """Exact lossless Rodrigues propagation of the authored RF, independently.

    Off-resonance is present during the pulse. This oracle does not use native
    RF labels, nominal pulse integrals or any benchmark simulation code.
    """
    offsets = np.asarray(offsets_hz)
    magnetization = np.zeros((len(offsets), 3))
    magnetization[:, 2] = 1
    dwell = rf.shape_dur / len(rf.signal)
    for sample in rf.signal * np.exp(1j * rf.phase_offset):
        axes = (
            2
            * np.pi
            * dwell
            * np.column_stack(
                (
                    np.full(len(offsets), sample.real),
                    np.full(len(offsets), sample.imag),
                    offsets,
                )
            )
        )
        angle = np.linalg.norm(axes, axis=1)
        unit = axes / angle[:, None]
        cosine = np.cos(angle)[:, None]
        magnetization = (
            magnetization * cosine
            - np.cross(unit, magnetization) * np.sin(angle)[:, None]
            + unit * np.sum(unit * magnetization, axis=1)[:, None] * (1 - cosine)
        )
    return magnetization


@pytest.mark.parametrize(
    "family,pulse_index", [("adiabatic_inversion", 0), ("t2_prepared", 1)]
)
def test_hyperbolic_secant_inverts_water_fat_and_declared_field_range(
    family, pulse_index
):
    sequence = make_sequence(case_for(family))
    pulse = [
        sequence.get_block(n).rf
        for n in sequence.block_events
        if sequence.get_block(n).rf is not None
    ][pulse_index]
    offsets = np.r_[
        np.linspace(-520, 70, 17), -434.2902819118434, -217.1451409559217, -79.62, 0.0
    ]
    final = rotate_exact_sample_hold(pulse, offsets)
    assert np.all(final[:, 2] < -0.94)
    assert np.rad2deg(np.arccos(final[-1, 2])) >= 178
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), 1, atol=1e-10)


@pytest.mark.parametrize(
    "mutation", ["missing_trigger", "wrong_trigger_channel", "late_adc"]
)
def test_trigger_corruptions_are_actual_events(mutation):
    sequence = make_sequence(case_for(), mutation)
    triggers, acquisitions = physical_events(sequence)
    if mutation == "missing_trigger":
        assert not triggers
    elif mutation == "wrong_trigger_channel":
        assert triggers and all(channel == "physio2" for _, channel in triggers)
    else:
        trigger_times = np.array([time for time, _ in triggers])
        assert all(
            adc[-1]
            - trigger_times[np.searchsorted(trigger_times, adc[0], side="right") - 1]
            > 0.8
            for adc in acquisitions
        )


def test_standard_matched_rf_receiver_carrier_is_coherent(tmp_path):
    case = case_for()
    sequence = make_sequence(case)
    for number in sequence.block_events:
        block = sequence.get_block(number)
        if block.rf is not None and block.rf.shape_dur <= 0.000201:
            block.rf.freq_offset = 100.0
        if block.adc is not None:
            block.adc.freq_offset = 100.0
        sequence.set_block(number, block)
    path = tmp_path / "matched_carrier.seq"
    sequence.write(str(path))
    result = inspect_sequence(path, case)
    assert all(item["passed"] for item in result.checks)


@pytest.mark.parametrize(
    "family,mutation",
    [
        (family, mutation)
        for family in FAMILIES
        for mutation in applicable_mutations(family, "high_performance")
        if mutation not in {"tiny_flip", "narrow_preparation"}
    ],
)
def test_exported_waveform_corruptions_fail_structural_judge(
    tmp_path, family, mutation
):
    case = case_for(family)
    path = tmp_path / "corrupted.seq"
    make_sequence(case, mutation).write(str(path))
    assert any(not item["passed"] for item in inspect_sequence(path, case).checks)


@pytest.mark.parametrize(
    "family,pulse_index,nominal_min",
    [("adiabatic_inversion", 0, 170), ("t2_prepared", 1, 178)],
)
def test_narrowband_preparation_cannot_be_certified_by_nominal_flip_alone(
    family, pulse_index, nominal_min
):
    sequence = make_sequence(case_for(family), "narrow_preparation")
    pulse = [
        sequence.get_block(n).rf
        for n in sequence.block_events
        if sequence.get_block(n).rf is not None
    ][pulse_index]
    result = rotate_exact_sample_hold(pulse, [0.0, -520.0])
    assert np.rad2deg(np.arccos(result[0, 2])) >= nominal_min
    assert -result[1, 2] < 0.9
