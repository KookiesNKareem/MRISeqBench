"""Exact encoded ADC reuse; all physics inputs remain part of cache identity."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s4
from mriseqbench.evaluation.s1_phantom import Spins


@pytest.fixture
def replay(tmp_path, monkeypatch):
    source = tmp_path / "sequence.seq"
    source.write_bytes(b"fixture-only native replay cache identity")
    acquisition = SimpleNamespace(
        source_path=source,
        seq=SimpleNamespace(system=SimpleNamespace(B0=3.0)),
        adc=np.array([0.1, 0.2, 0.6, 0.7, 1.1, 1.2]),
        k=np.arange(18).reshape(6, 3),
        adc_phase=np.arange(6) * 0.1,
        pulses=[SimpleNamespace(start=t) for t in [0.0, 0.5, 1.0]],
    )
    spins = Spins(
        np.zeros((2, 3)),
        np.array([0, 1]),
        ["water", "fat"],
        np.array([[1.0, 1.0, 0.1, 0.0], [1.0, 0.4, 0.07, -434.0]]),
        np.ones(2),
        (2,),
        "fixture",
    )
    calls = []

    def simulate(acq, nodes, effects=(), **steps):
        calls.append((acq, nodes, effects, steps))
        return np.arange(len(acq.adc), dtype=complex) + 1j * np.sum(nodes.properties)

    monkeypatch.setattr(s4, "simulate", simulate)
    cache = s4.FullFrameReplay(acquisition, tmp_path / "cache")

    def frame(number):
        result = copy(acquisition)
        indices = np.arange(2 * number, 2 * number + 2)
        result.adc, result.k, result.adc_phase = (
            getattr(acquisition, name)[indices] for name in ["adc", "k", "adc_phase"]
        )
        result.pulses = acquisition.pulses[: number + 1]
        return result

    return acquisition, spins, cache, frame, calls


def test_cache_simulates_all_frames_once_and_preserves_complex_adc_values(replay):
    acquisition, spins, cache, frame, calls = replay
    for number in range(3):
        value = cache(frame(number), spins, step_s=0.001, rf_step_s=20e-6)
        np.testing.assert_array_equal(value.real, [2 * number, 2 * number + 1])
        assert np.all(value.imag == np.sum(spins.properties))
    assert len(calls) == cache.calls == 1
    assert calls[0][0] is acquisition
    assert cache.hits == 2


@pytest.mark.parametrize(
    "changed", ["positions", "labels", "weight", "properties", "effects", "steps"]
)
def test_cache_keys_every_spin_physics_input_and_step(replay, changed):
    _, spins, cache, frame, calls = replay
    cache(frame(0), spins, step_s=0.001)
    effects, steps = (), {"step_s": 0.001}
    if changed in {"positions", "labels", "weight", "properties"}:
        values = getattr(spins, changed).copy()
        values.flat[0] += 1
        spins = replace(spins, **{changed: values})
    elif changed == "effects":
        effects = [{"kind": "b0", "model": "constant", "offset_hz": 50.0}]
    else:
        steps["step_s"] /= 2
    cache(frame(1), spins, effects, **steps)
    assert len(calls) == cache.calls == 2


@pytest.mark.parametrize("changed", ["phase", "encoding", "rf", "rf_only", "motion"])
def test_unsupported_or_modified_frame_never_uses_full_encoded_cache(replay, changed):
    acquisition, spins, cache, frame, calls = replay
    subset = frame(1)
    effects, steps = (), {}
    if changed == "phase":
        subset.adc_phase += 0.1
    elif changed == "encoding":
        subset.k = subset.k + 0.1
    elif changed == "rf":
        subset.pulses = [copy(pulse) for pulse in subset.pulses]
    elif changed == "rf_only":
        steps["zero_gradients"] = True
    else:
        effects = [{"kind": "motion", "model": "fixture"}]
    cache(subset, spins, effects, **steps)
    assert calls[0][0] is subset
    assert calls[0][0] is not acquisition
    assert cache.calls == cache.hits == 0
