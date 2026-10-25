"""RF rotations checked against analytic area and independent 3D rotations."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mriseqbench.evaluation import s1_sequence


@pytest.mark.parametrize("shape", ["gaussian", "signed_sinc", "zero"])
def test_constant_axis_rotation_matches_analytic_pulse_area(shape):
    times = np.linspace(-3, 3, 1001)
    values = {
        "gaussian": (1 + 1j) * 250 * np.exp(-(times**2)),
        "signed_sinc": 500 * np.sinc(times).astype(complex),
        "zero": np.zeros(len(times), dtype=complex),
    }[shape]
    widths = np.linspace(0.5e-6, 1.5e-6, len(values))
    original_values, original_widths = values.copy(), widths.copy()
    area = 2 * np.pi * np.sum(values * widths)
    angle = abs(area)
    expected = (
        np.array([-area.imag, area.real, 0]) * np.sin(angle) / angle
        + np.array([0, 0, np.cos(angle)])
        if angle
        else np.array([0, 0, 1])
    )
    actual = s1_sequence.nominal_rotation(values, widths)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(widths, original_widths)


@pytest.mark.parametrize("phase_change", [np.pi / 2, 1e-10])
def test_phase_varying_rotation_matches_independent_rotations(phase_change):
    values = 250 * np.exp(1j * np.array([0.0, phase_change, 0.0]))
    widths = np.full(3, 0.001)
    expected = np.array([0.0, 0.0, 1.0])
    for value, width in zip(values, widths, strict=True):
        vector = -2 * np.pi * width * np.array([value.real, value.imag, 0])
        expected = Rotation.from_rotvec(vector).apply(expected)
    actual = s1_sequence.nominal_rotation(values, widths)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_repeated_rf_envelopes_preserve_each_pulses_metadata(tmp_path):
    import copy

    from pypulseq import Opts, Sequence, make_adc, make_delay, make_sinc_pulse

    from mriseqbench.catalog import Catalog

    system = Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    rf = make_sinc_pulse(
        0.4, duration=0.001, dwell=10e-6, system=system, use="excitation"
    )
    expected = []
    clock = 0.0
    for index in range(3):
        pulse = copy.deepcopy(rf)
        if index == 2:
            pulse.signal *= 1.01  # Distinct envelope must never reuse the memo.
        pulse.phase_offset = index * 0.07
        pulse.freq_offset = index * 3.0
        pulse.delay = (index % 2) * 10e-6
        expected.append(
            (clock + pulse.delay + pulse.t[0], pulse.phase_offset, pulse.freq_offset)
        )
        seq.add_block(pulse)
        seq.add_block(make_delay(0.001))
        seq.add_block(make_adc(4, dwell=10e-6, system=system))
        clock = seq.duration()[0]
    path = tmp_path / "repeated-sinc.seq"
    seq.write(str(path))
    catalog = Catalog(Path(__file__).resolve().parents[2])
    case = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    acquisition = s1_sequence.inspect_sequence(path, case)
    np.testing.assert_allclose(
        [(p.start, p.phase, p.frequency) for p in acquisition.pulses],
        expected,
        atol=1e-12,
    )
    assert acquisition.pulses[-1].flip != acquisition.pulses[0].flip
