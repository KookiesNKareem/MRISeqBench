"""CPU checks of the adiabatic-inversion B0 x B1 robustness assay (s4@3).

The GPU kernel (s1_koma.jl inversion_assay_kernel!) applies exact per-interval
rotations; ``rotation_assay`` below is the same maths in numpy, so the grid
logic and the physical discrimination are testable without CUDA.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pypulseq import Opts, make_adiabatic_pulse, make_block_pulse, make_sinc_pulse

from mriseqbench.evaluation.rf_waveform import rf_samples
from mriseqbench.evaluation.s4 import (
    ADIABATIC_B1_SCALES,
    INVERSION_MIN,
    assay_offsets,
    inversion_grid,
)


def rotation_assay(pulse, offsets_hz):
    detuning = np.asarray(offsets_hz, float) - pulse.frequency
    signal = np.asarray(pulse.signal) * np.exp(1j * pulse.phase)
    m = np.zeros((len(detuning), 3))
    m[:, 2] = 1
    for value, width in zip(signal, np.diff(pulse.edges), strict=True):
        axis = (
            2
            * np.pi
            * width
            * np.column_stack(
                (
                    np.full_like(detuning, value.real),
                    np.full_like(detuning, value.imag),
                    detuning,
                )
            )
        )
        angle = np.linalg.norm(axis, axis=1)[:, None]
        u = axis / np.maximum(angle, 1e-30)
        m = (
            m * np.cos(angle)
            - np.cross(u, m) * np.sin(angle)
            + u * np.sum(u * m, axis=1, keepdims=True) * (1 - np.cos(angle))
        )
    return -m[:, 2]


def as_pulse(rf):
    edges, values = rf_samples(rf, 1e-6, max_step=1e-5)
    return SimpleNamespace(
        edges=edges, signal=values, phase=rf.phase_offset, frequency=rf.freq_offset
    )


def chirp(amplitude_hz, bandwidth_hz, duration=0.01, samples=1000):
    edges = np.linspace(0, duration, samples + 1)
    t = (edges[:-1] + edges[1:]) / 2 - duration / 2
    phase = np.pi * bandwidth_hz / duration * t**2
    return SimpleNamespace(
        edges=edges,
        signal=amplitude_hz * np.exp(1j * phase),
        phase=0.0,
        frequency=0.0,
    )


SYSTEM = Opts(B0=3)


def grid_min(pulse, field):
    grid = inversion_grid(
        pulse, assay_offsets(field), ADIABATIC_B1_SCALES, assay=rotation_assay
    )
    assert grid.shape == (len(ADIABATIC_B1_SCALES), 17)
    return float(grid.min())


def test_b1_grid_spans_plus_minus_twenty_percent():
    assert ADIABATIC_B1_SCALES[0] == 0.8 and ADIABATIC_B1_SCALES[-1] == 1.2
    assert 1.0 in ADIABATIC_B1_SCALES
    assert INVERSION_MIN == 0.90


def test_grid_scales_rf_amplitude_only():
    seen = []

    def assay(pulse, offsets):
        seen.append(np.abs(pulse.signal).max())
        return np.zeros(len(offsets))

    pulse = SimpleNamespace(signal=np.array([100.0 + 0j, 50j]), phase=0.3)
    inversion_grid(pulse, [0.0, 1.0], (0.8, 1.2), assay=assay)
    assert seen == pytest.approx([80.0, 120.0])
    assert np.abs(pulse.signal).max() == 100.0


def test_cpu_rotation_matches_constant_field_closed_form():
    pulse = SimpleNamespace(
        signal=np.array([500.0 + 0j, 500.0 + 0j]),
        edges=np.array([0.0, 0.0005, 0.001]),
        phase=0.37,
        frequency=13.0,
    )
    detuning = np.array([-500.0, 0.0, 500.0])
    scale = 500.0**2 + detuning**2
    expected = (
        -(detuning**2 + 500.0**2 * np.cos(2 * np.pi * np.sqrt(scale) * 0.001)) / scale
    )
    np.testing.assert_allclose(
        rotation_assay(pulse, detuning + 13.0), expected, atol=1e-12
    )


@pytest.mark.parametrize("field", [3.0, 1.5, 0.55])
def test_reference_hyperbolic_secant_is_b1_robust(field):
    rf = make_adiabatic_pulse(
        "hypsec",
        duration=0.01,
        dwell=1e-5,
        beta=1000.0,
        mu=4.9,
        adiabaticity=4,
        system=SYSTEM,
    )
    assert grid_min(as_pulse(rf), field) >= INVERSION_MIN + 0.05


@pytest.mark.parametrize(
    "name",
    ["sinc_180", "hard_180", "chirp_850hz_8khz", "chirped_rectangle"],
)
@pytest.mark.parametrize("field", [3.0, 1.5, 0.55])
def test_non_adiabatic_or_truncated_pulses_fail_b1_grid(name, field):
    pulse = {
        "sinc_180": lambda: as_pulse(
            make_sinc_pulse(np.pi, duration=0.003, time_bw_product=4, system=SYSTEM)
        ),
        "hard_180": lambda: as_pulse(
            make_block_pulse(np.pi, duration=0.001, system=SYSTEM)
        ),
        # Constant-amplitude linear chirp that passes the nominal-B1 B0 assay.
        "chirp_850hz_8khz": lambda: chirp(850, 8000),
        # 180 deg rectangle with a 1e-3 Hz sweep.
        "chirped_rectangle": lambda: chirp(50, 1e-3),
    }[name]()
    assert grid_min(pulse, field) < INVERSION_MIN


def test_constant_amplitude_chirp_is_caught_only_by_b1_sweep():
    pulse = chirp(850, 8000)
    nominal = inversion_grid(pulse, assay_offsets(3.0), (1.0,), assay=rotation_assay)
    assert nominal.min() >= INVERSION_MIN
    assert grid_min(pulse, 3.0) < INVERSION_MIN
