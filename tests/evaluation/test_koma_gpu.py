"""Opt-in CUDA execution and decomposition controls; no CPU simulation."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pypulseq import (
    Opts,
    Sequence,
    make_adc,
    make_block_pulse,
    make_delay,
    make_trapezoid,
)

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1_koma
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
    # Cross multiple GPU thread groups and the 256-step temporary-memory bound.
    seq = Sequence(acq.seq.system)
    seq.add_block(
        make_block_pulse(0.6, duration=0.001, system=seq.system, use="excitation")
    )
    seq.add_block(
        make_trapezoid(
            "x",
            amplitude=100,
            rise_time=200e-6,
            flat_time=0.006,
            fall_time=200e-6,
            system=seq.system,
        ),
        make_adc(600, dwell=10e-6, delay=200e-6, system=seq.system),
    )
    path = directory / "multiwarp.seq"
    seq.write(str(path))
    catalog = Catalog(Path(__file__).resolve().parents[2])
    long_acq = inspect_sequence(path, catalog.resolve(catalog.get("tasks", "gre_t1w")))
    many_spins = replace(
        spins,
        positions=np.linspace(-0.01, 0.01, 1025)[:, None]
        * np.array([[1.0, -0.5, 0.2]]),
        labels=np.zeros(1025, dtype=np.int64),
        weight=1 / 1025,
        grid=(1025,),
    )
    cases = {
        "summed": {},
        "magnus4": {"integration_method": "magnus4"},
        "individual": {"individual": True},
        "receive": {"receive_weights": np.array([[1.0, 0.7], [0.2, 1.0]])},
        "demodulated": {"zero_gradients": True, "demodulate_encoding": True},
        "spoiled": {"individual": True, "zero_gradients": True, "ideal_spoiling": True},
        "phase_windows": {
            "individual": True,
            "zero_gradients": True,
            "phase_windows": [(0.0, float(acq.adc[-1]) + 1e-5)],
        },
        "dynamic": {
            "effects": [
                {
                    "kind": "time_variation",
                    "model": "sinusoid",
                    "target": "b0_offset",
                    "amplitude_hz": 5.0,
                    "period_s": 0.5,
                    "phase_deg": 0.0,
                }
            ]
        },
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
        signals["multiwarp"] = s1_koma.simulate(long_acq, many_spins)
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
