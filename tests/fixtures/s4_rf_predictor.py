"""Independent sample-held affine Bloch gold for static cardiac RF design.

This predictor retains full transverse and longitudinal history. It has no
encoding gradients or spatial B0, so its Fourier images are design diagnostics.
The finite-RF propagators are exact matrix exponentials for each held sample.
"""

from functools import lru_cache
from itertools import pairwise

import numpy as np
from scipy.linalg import expm


@lru_cache(maxsize=262144)
def transition(material, real, imag, duration):
    density, t1, t2, frequency = material
    wx, wy, wz = 2 * np.pi * np.array([real, imag, frequency])
    generator = np.array(
        [
            [-1 / t2, wz, -wy, 0],
            [-wz, -1 / t2, wx, 0],
            [wy, -wx, -1 / t1, density / t1],
            [0, 0, 0, 0],
        ]
    )
    return expm(generator * duration)


def affine_rf_history(acquisition, material):
    material = tuple(float(v) for v in material)
    if any(pulse.frequency != 0 for pulse in acquisition.pulses):
        raise ValueError("Independent affine gold requires unmodulated carriers")
    events = np.unique(
        np.r_[0.0, acquisition.adc, *[pulse.edges for pulse in acquisition.pulses]]
    )
    state = np.array([0.0, 0.0, material[0], 1.0])
    output = np.zeros(len(acquisition.adc), complex)
    pulse_index = adc_index = 0
    pulses = acquisition.pulses
    for begin, end in pairwise(events):
        midpoint = (begin + end) / 2
        while pulse_index < len(pulses) and midpoint >= pulses[pulse_index].end:
            pulse_index += 1
        pulse = pulses[pulse_index] if pulse_index < len(pulses) else None
        signal = 0j
        if pulse is not None and pulse.start <= midpoint < pulse.end:
            index = np.searchsorted(pulse.edges, midpoint, side="right") - 1
            signal = pulse.signal[index] * np.exp(1j * pulse.phase)
        state = (
            transition(
                material, float(signal.real), float(signal.imag), round(end - begin, 12)
            )
            @ state
        )
        while (
            adc_index < len(acquisition.adc)
            and abs(acquisition.adc[adc_index] - end) < 1e-10
        ):
            output[adc_index] = (state[0] + 1j * state[1]) * np.exp(
                -1j * acquisition.adc_phase[adc_index]
            )
            adc_index += 1
    if adc_index != len(output):
        raise ValueError("Affine gold omitted an ADC sample")
    return output
