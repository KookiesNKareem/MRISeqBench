"""Engine-boundary controls for KomaMRI physics, not copies of its implementation."""

from pathlib import Path

import numpy as np
import pytest
from pypulseq import (
    Opts,
    Sequence,
    make_adc,
    make_arbitrary_rf,
    make_block_pulse,
    make_delay,
)
from scipy.linalg import expm

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_image import canonical_response, nrmse
from mriseqbench.evaluation.s1_koma import simulate, versions
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s1_sequence import inspect_sequence

ROOT = Path(__file__).resolve().parents[2]


def probe(tmp_path, field=3.0, ppm=False, flip=45):
    system = Opts(B0=field, rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    rf = make_block_pulse(
        np.deg2rad(flip), duration=0.001, system=system, use="excitation"
    )
    if ppm:
        rf.freq_ppm, rf.phase_ppm = 1.0, 0.25
    else:
        rf.freq_offset = system.gamma * field * 1e-6
        rf.phase_offset = 0.25 * system.gamma * field * 1e-6
    seq.add_block(rf)
    seq.add_block(make_delay(0.02))
    seq.add_block(make_adc(8, duration=80e-6, system=system), make_delay(100e-6))
    path = tmp_path / "probe.seq"
    seq.write(str(path))
    cat = Catalog(ROOT)
    case = cat.resolve(cat.get("tasks", "gre_t1w"))
    case["hardware"]["field_strength_T"] = field
    acq = inspect_sequence(path, case)
    spins = Spins(
        np.array([[0, 0, 0], [0.01, -0.02, 0]]),
        np.zeros(2, dtype=np.int64),
        ["water"],
        np.array([[1.0, 0.8, 0.08]]),
        0.5,
        (2,),
        "engine-probe",
    )
    return acq, spins


def test_pinned_koma_and_bart_versions_are_reported():
    actual = versions()
    assert actual["KomaMRICore"] == "0.11.2"
    assert actual["KomaMRIBase"] == "0.11.3"
    assert actual["KomaMRIFiles"] == "0.10.3"
    from mriseqbench.evaluation.bart import provenance

    assert "MRIReco" not in actual
    identity = provenance()
    assert identity["engine"] == "BART"
    assert "v1.0.00" in identity["version"]
    assert len(identity["binary_sha256"]) == 64


def test_nominal_rf_reference_ignores_pulses_after_final_adc(tmp_path):
    acq, spins = probe(tmp_path)
    expected = canonical_response(acq, spins.properties, "gre_t1w", finite_rf=True)
    acq.seq.add_block(make_delay(0.01))
    acq.seq.add_block(
        make_block_pulse(np.pi / 2, duration=0.001, system=acq.seq.system)
    )
    path = tmp_path / "post-acquisition-rf.seq"
    acq.seq.write(str(path))
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    extended = inspect_sequence(path, case)
    assert len(extended.pulses) == len(acq.pulses) + 1
    actual = canonical_response(extended, spins.properties, "gre_t1w", finite_rf=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("integration_method", ["bloch", "magnus4"])
@pytest.mark.parametrize("offset", [0.0, 73.0])
@pytest.mark.parametrize("ideal_spoiling", [False, True])
def test_rf_train_has_no_endpoint_tail_across_long_delays(
    tmp_path, integration_method, offset, ideal_spoiling
):
    if ideal_spoiling and integration_method != "bloch":
        pytest.skip("Individual RF oracle uses the native BlochDict kernel")
    # Accumulated block times expose RF-edge roundoff missed by a single pulse.
    # Use exact affine Bloch propagation as an independent full-train oracle.
    system = Opts(rf_dead_time=100e-6, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    rf = make_block_pulse(
        np.deg2rad(20), duration=500e-6, delay=100e-6, system=system, use="excitation"
    )
    for _ in range(32):
        seq.add_block(rf)
        seq.add_block(make_delay(0.01668))
        seq.add_block(make_adc(32, dwell=10e-6, system=system))
        seq.add_block(make_delay(0.01228))
    path = tmp_path / "rf-train.seq"
    seq.write(str(path))
    cat = Catalog(ROOT)
    case = cat.resolve(cat.get("tasks", "gre_t1w"))
    acq = inspect_sequence(path, case)
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, dtype=np.int64),
        ["water"],
        np.array([[0.7, 0.8, 0.08]]),
        1.0,
        (1,),
        "exact-rf-train",
    )
    state = np.array([0.0, 0.0, 0.7, 1.0])
    expected = np.zeros(len(acq.adc), complex)
    time = 0.0

    def free_until(end):
        nonlocal time
        dt = end - time
        xy = (state[0] + 1j * state[1]) * np.exp((-1 / 0.08 - 2j * np.pi * offset) * dt)
        state[0], state[1] = xy.real, xy.imag
        state[2] = 0.7 + (state[2] - 0.7) * np.exp(-dt / 0.8)
        time = end

    for pulse in acq.pulses:
        free_until(pulse.start)
        if ideal_spoiling:
            state[:2] = 0
        wx = 2 * np.pi * pulse.signal[0].real
        wz = 2 * np.pi * offset
        ode = np.array(
            [
                [-1 / 0.08, wz, 0, 0],
                [-wz, -1 / 0.08, wx, 0],
                [0, -wx, -1 / 0.8, 0.7 / 0.8],
                [0, 0, 0, 0],
            ]
        )
        state = expm(ode * (pulse.end - pulse.start)) @ state
        time = pulse.end
        indices = np.flatnonzero((acq.adc > time) & (acq.adc < pulse.start + 0.02988))
        for i in indices:
            free_until(acq.adc[i])
            expected[i] = (state[0] + 1j * state[1]) * np.exp(-1j * acq.adc_phase[i])
    actual = simulate(
        acq,
        spins,
        frequency_offsets=np.array([offset]),
        rf_step_s=1e-6,
        integration_method=integration_method,
        individual=ideal_spoiling,
        zero_gradients=ideal_spoiling,
        ideal_spoiling=ideal_spoiling,
    )
    if ideal_spoiling:
        actual = actual[:, 0]
    assert nrmse(actual, expected) < 2e-5


@pytest.mark.parametrize("integration_method", ["bloch", "magnus4"])
def test_oscillating_rf_envelope_cannot_alias_both_temporal_steps(
    tmp_path, integration_method
):
    system = Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    seq.add_block(
        make_arbitrary_rf(
            1 + 0.8 * np.cos(2 * np.pi * np.arange(1000) / 5),
            np.deg2rad(20),
            dwell=1e-6,
            system=system,
            use="excitation",
        )
    )
    seq.add_block(make_adc(1, dwell=2e-6, system=system), make_delay(20e-6))
    path = tmp_path / "oscillating-rf.seq"
    seq.write(str(path))
    cat = Catalog(ROOT)
    case = cat.resolve(cat.get("tasks", "gre_t1w"))
    acq = inspect_sequence(path, case)
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, dtype=np.int64),
        ["water"],
        np.array([[0.7, 0.8, 0.08]]),
        1.0,
        (1,),
        "exact-oscillating-rf",
    )
    state = np.array([0.0, 0.0, 0.7, 1.0])
    pulse = acq.pulses[0]
    for amplitude, width in zip(pulse.signal, np.diff(pulse.edges), strict=True):
        wx = 2 * np.pi * amplitude.real
        ode = np.array(
            [
                [-1 / 0.08, 0, 0, 0],
                [0, -1 / 0.08, wx, 0],
                [0, -wx, -1 / 0.8, 0.7 / 0.8],
                [0, 0, 0, 0],
            ]
        )
        state = expm(ode * width) @ state
    expected = (state[0] + 1j * state[1]) * np.exp(-(acq.adc[0] - pulse.end) / 0.08)
    # Previously 20/10 us agreed within 1%, despite both exceeding 5% error.
    # The bound allows the pinned reader's sample-hold/linear-envelope mismatch.
    for step in [20e-6, 10e-6]:
        actual = simulate(
            acq, spins, integration_method=integration_method, rf_step_s=step
        )[0]
        assert abs(actual - expected) / abs(expected) < 0.002


@pytest.mark.parametrize("field", [0.55, 1.5, 3.0])
def test_rf_ppm_uses_selected_field_and_matches_explicit_hz_and_phase(tmp_path, field):
    plain, spins = probe(tmp_path, field=field)
    expected = simulate(plain, spins)
    ppm, spins = probe(tmp_path, field=field, ppm=True)
    actual = simulate(ppm, spins)
    assert nrmse(actual, expected) < 1e-10


@pytest.mark.parametrize("integration_method", ["bloch", "magnus4"])
def test_zero_dynamic_variation_matches_public_koma_simulation(
    tmp_path, integration_method
):
    acq, spins = probe(tmp_path)
    baseline = [{"kind": "b0", "model": "constant", "offset_hz": 73.0}]
    dynamic = baseline + [
        {
            "kind": "time_variation",
            "model": "sinusoidal",
            "target": "b0_offset",
            "amplitude_hz": 0.0,
            "period_s": 2.0,
            "phase_deg": 0.0,
        }
    ]
    expected = simulate(acq, spins, baseline, integration_method=integration_method)
    actual = simulate(acq, spins, dynamic, integration_method=integration_method)
    assert nrmse(actual, expected) < 1e-9


def test_sequence_cache_tracks_file_content_after_overwrite(tmp_path):
    first, spins = probe(tmp_path, flip=10)
    a = simulate(first, spins)
    second, spins = probe(tmp_path, flip=60)
    b = simulate(second, spins)
    assert np.linalg.norm(b) > 3 * np.linalg.norm(a)


def test_selected_adc_sample_preserves_fractional_microsecond_timing(tmp_path):
    from copy import copy

    acq, spins = probe(tmp_path)
    # Rebuild an original acquisition with 25 us dwell (half-sample at 12.5 us).
    seq = acq.seq
    for number in seq.block_events:
        block = seq.get_block(number)
        if block.adc is not None:
            adc = make_adc(8, duration=200e-6, system=seq.system)
            seq.set_block(number, adc, make_delay(220e-6))
    seq.write(acq.source_path)
    cat = Catalog(ROOT)
    case = cat.resolve(cat.get("tasks", "gre_t1w"))
    acq = inspect_sequence(acq.source_path, case)
    full = simulate(acq, spins)
    selected = copy(acq)
    selected.adc, selected.adc_phase, selected.k = (
        acq.adc[:1],
        acq.adc_phase[:1],
        acq.k[:1],
    )
    single = simulate(selected, spins)
    assert nrmse(single, full[:1]) < 1e-9
