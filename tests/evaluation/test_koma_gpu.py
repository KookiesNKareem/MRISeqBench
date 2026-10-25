"""Opt-in CUDA execution and decomposition controls; no CPU simulation."""

import os
from pathlib import Path

import numpy as np
import pytest
from pypulseq import (
    Opts,
    Sequence,
    make_adc,
    make_block_pulse,
    make_delay,
)

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1_case, s1_koma
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s1_sequence import inspect_sequence

ROOT = Path(__file__).resolve().parents[2]


def probe(tmp_path, field=3.0, flip=45):
    system = Opts(B0=field, rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    rf = make_block_pulse(
        np.deg2rad(flip), duration=0.001, system=system, use="excitation"
    )
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
    spins = Spins(
        np.array([[0, 0, 0], [0.01, -0.02, 0]]),
        np.zeros(2, dtype=np.int64),
        ["water"],
        np.array([[1.0, 0.8, 0.08]]),
        0.5,
        (2,),
        "engine-probe",
    )
    return inspect_sequence(path, case), spins


pytestmark = pytest.mark.skipif(
    os.environ.get("MRISEQBENCH_TEST_CUDA") != "1",
    reason="Set MRISEQBENCH_TEST_CUDA=1 on a CUDA host to run native GPU controls",
)


@pytest.fixture(scope="module")
def device_signals(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cuda-parity")
    acq, spins = probe(directory)
    cases = {
        "summed": {},
        "individual": {"individual": True},
        "receive": {"receive_weights": np.array([[1.0, 0.7], [0.2, 1.0]])},
    }
    prior_worker = s1_koma._WORKER
    s1_koma._WORKER = s1_koma.KomaWorker()
    try:
        identity = s1_koma.versions()
        assert identity["SimulationBackend"] == "cuda"
        assert identity["SimulationPrecision"] == "f64"
        signals = {
            name: s1_koma.simulate(acq, spins, **kwargs)
            for name, kwargs in cases.items()
        }
        assert s1_koma.worker().execution["cpu_fallback"] is False
        return signals
    finally:
        s1_koma._WORKER.close()
        s1_koma._WORKER = prior_worker


def test_cuda_native_spin_and_receive_decompositions(device_signals):
    individual = device_signals["individual"]
    np.testing.assert_allclose(
        individual.sum(axis=1), device_signals["summed"], rtol=1e-9, atol=1e-11
    )
    weights = np.array([[1.0, 0.7], [0.2, 1.0]])
    np.testing.assert_allclose(
        weights @ individual.T, device_signals["receive"], rtol=1e-9, atol=1e-11
    )
    for signal in device_signals.values():
        assert signal.dtype == np.complex128
        assert np.isfinite(signal).all()


def test_cuda_unavailable_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("MRISEQBENCH_KOMA_DEVICE", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    engine = s1_koma.KomaWorker()
    try:
        with pytest.raises(RuntimeError, match="functional CUDA GPU is required"):
            engine.call({"operation": "versions"})
    finally:
        engine.close()


def test_cuda_inversion_assay_matches_constant_field_rotation():
    from types import SimpleNamespace

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
    measured = s1_koma.inversion_assay(pulse, detuning + pulse.frequency)
    np.testing.assert_allclose(measured, expected, atol=1e-12, rtol=1e-12)
    assert s1_koma.worker().execution["cpu_fallback"] is False


def test_cuda_batched_phase_matches_separate_rf_history_prefixes(tmp_path, monkeypatch):
    from copy import copy

    from mriseqbench.evaluation import s4

    system = Opts(B0=3, rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    for phase in (0.0, 0.7, 1.9):
        rf = make_block_pulse(
            0.6, duration=0.0002, phase_offset=phase, system=system, use="excitation"
        )
        seq.add_block(rf)
        seq.add_block(make_delay(0.005))
        seq.add_block(make_adc(8, dwell=10e-6, system=system), make_delay(100e-6))
    path = tmp_path / "phase-history.seq"
    seq.write(str(path))
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), hardware_ref="high_performance@1"
    )
    case["task"]["sequence"].update(matrix=[2, 2], fov_mm=[200, 200])
    case["physics"]["effects"] = [
        {"kind": "b0", "model": "constant", "offset_hz": 13.0}
    ]
    source = inspect_sequence(path, case)
    geometry = Spins(
        np.array(
            [[-0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0]]
        ),
        np.array([0, 1, 0, 1]),
        ["water", "fat"],
        np.array([[1, 0.8, 0.08, 0], [0.9, 0.6, 0.05, -440]]),
        np.ones(4),
        (2, 2),
        "phase-test",
    )
    monkeypatch.setattr(s1_case, "phantom", lambda *_: geometry)
    indices = np.asarray([line[4] for line in source.lines])
    batch = {"acquisition": source, "indices": indices, "images": {}}
    for index in indices:
        frame = copy(source)
        frame.adc, frame.adc_phase, frame.k = (
            source.adc[[index]],
            source.adc_phase[[index]],
            source.k[[index]],
        )
        frame.lines = [np.array([0])]
        frame.pulses = [pulse for pulse in source.pulses if pulse.start < frame.adc[-1]]
        separate = s4.phase_reference(frame, case, ROOT, np.ones((2, 2)))
        batched = s4.phase_reference(frame, case, ROOT, np.ones((2, 2)), batch=batch)
        np.testing.assert_allclose(batched[0], separate[0], atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(batched[1], separate[1])
        assert batched[2]["echo_time_s"] == separate[2]["echo_time_s"]


@pytest.mark.parametrize("field", [1.5, 3.0])
@pytest.mark.parametrize("ideal_spoiling", [False, True])
def test_cuda_canonical_adc_compression_matches_full_rf_history(
    tmp_path, field, ideal_spoiling
):
    from mriseqbench.evaluation.s1_image import rf_only_response

    system = Opts(B0=field, rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    for number, phase in enumerate((0.0, 0.7, 1.9)):
        seq.add_block(
            make_block_pulse(
                0.6,
                duration=0.0002,
                phase_offset=phase,
                system=system,
                use="excitation",
            )
        )
        seq.add_block(make_delay(0.5 if number == 0 else 0.005))
        seq.add_block(make_adc(128, dwell=10e-6, system=system), make_delay(0.0013))
    path = tmp_path / "rf-free-adc-history.seq"
    seq.write(str(path))
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    case["hardware"]["field_strength_T"] = field
    source = inspect_sequence(path, case)
    properties = np.array([[1, 0.8, 0.08, 0], [0.9, 0.6, 0.005, -440]])
    full = rf_only_response(
        source, properties, ideal_spoiling=ideal_spoiling, compress=False
    )
    compressed = rf_only_response(source, properties, ideal_spoiling=ideal_spoiling)
    np.testing.assert_allclose(compressed, full, atol=1e-9, rtol=1e-9)
    assert s1_koma.worker().execution["cpu_fallback"] is False
