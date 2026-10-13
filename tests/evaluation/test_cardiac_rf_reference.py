"""Independent finite-RF chemical-shift gold for water-selective cardiac controls."""

from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_image import canonical_response
from mriseqbench.evaluation.s1_phantom import phantom
from mriseqbench.evaluation.s2_sequence import inspect_sequence
from tests.fixtures import s2_sequences
from tests.fixtures.calibrate_cardiac import options_for

ROOT = Path(__file__).resolve().parents[2]


def affine_rf_history(acq, material):
    """Piecewise-exact affine Bloch evolution, retaining Mz across ideal spoilers."""
    density, t1, t2, frequency = material

    @cache
    def transition(real, imag, duration):
        wx, wy, wz = 2 * np.pi * real, 2 * np.pi * imag, 2 * np.pi * frequency
        generator = np.array(
            [
                [-1 / t2, wz, -wy, 0],
                [-wz, -1 / t2, wx, 0],
                [wy, -wx, -1 / t1, density / t1],
                [0, 0, 0, 0],
            ]
        )
        return expm(generator * duration)

    events = np.unique(np.r_[0.0, acq.adc, *[pulse.edges for pulse in acq.pulses]])
    state = np.array([0.0, 0.0, density, 1.0])
    output = np.zeros(len(acq.adc), complex)
    adc_index = 0
    for begin, end in pairwise(events):
        midpoint = (begin + end) / 2
        pulse = next((p for p in acq.pulses if p.start <= midpoint < p.end), None)
        if any(abs(begin - p.start) < 1e-11 and not p.refocusing for p in acq.pulses):
            state[:2] = 0  # Preserve the longitudinal state and recovery history.
        signal = 0j
        if pulse is not None:
            assert pulse.frequency == 0, (
                "This independent gold uses unmodulated RF carriers"
            )
            index = np.searchsorted(pulse.edges, midpoint, side="right") - 1
            signal = pulse.signal[index] * np.exp(1j * pulse.phase)
        state = (
            transition(float(signal.real), float(signal.imag), round(end - begin, 12))
            @ state
        )
        while adc_index < len(acq.adc) and abs(acq.adc[adc_index] - end) < 1e-11:
            excitation = next(
                p for p in reversed(acq.pulses) if not p.refocusing and p.start < end
            )
            output[adc_index] = (state[0] + 1j * state[1]) * np.exp(
                -1j * excitation.phase
            )
            adc_index += 1
    assert adc_index == len(acq.adc)
    return output


@pytest.mark.parametrize("family", ["epi_single_shot", "epi_multishot"])
@pytest.mark.parametrize("hardware", ["low_field", "standard_1_5t", "high_performance"])
def test_binomial_native_reference_matches_affine_rf_history(
    tmp_path, family, hardware
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", family), hardware_ref=hardware)
    options = options_for(case, s2_sequences)
    options["water_selective_order"] = 2
    path = tmp_path / "sequence.seq"
    s2_sequences.make_sequence(case, **options).write(str(path))
    acq = inspect_sequence(path, case)
    geometry = phantom(case, ROOT, 4)
    actual = canonical_response(acq, geometry.properties, family, finite_rf=True)
    material = np.array(
        [
            [
                values[key]
                for key in ("proton_density", "T1_s", "T2_s", "chemical_shift_hz")
            ]
            for values in case["object"]["tissue_properties"].values()
        ]
    )
    assert list(case["object"]["tissue_properties"]) == geometry.materials
    expected = np.stack([affine_rf_history(acq, values) for values in material])
    relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    normalized_max = np.max(abs(actual - expected) / material[:, 0, None])
    print(f"{family}/{hardware}: NRMSE={relative:.6g}; max/PD={normalized_max:.6g}")
    assert relative < 0.003
    assert normalized_max < 0.0002
    fat = geometry.materials.index("fat")
    water = geometry.materials.index("myocardium")
    assert np.max(abs(expected[fat])) / np.max(abs(expected[water])) < 0.003
