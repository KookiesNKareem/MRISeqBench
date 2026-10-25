"""Exact RF-free output expansion, without running signal simulations."""

from types import SimpleNamespace

import numpy as np
import pytest
from pypulseq import Opts, Sequence, calc_duration, make_block_pulse, make_trapezoid

from mriseqbench.evaluation import s1_image, s1_koma, s4
from mriseqbench.evaluation.s4_sequence import spoiled_preparations

SPEC = {"matrix": [128, 128], "fov_mm": [200, 200]}  # 1.5625 mm voxels


@pytest.mark.parametrize(
    "axis,area,spoiled",
    [
        ("x", 1280, True),  # 2.0 cycles/voxel: the minimum
        ("y", 2560, True),
        ("x", 1200, False),  # 1.875 cycles/voxel
        ("x", 640, False),
        ("x", 1e-7, False),  # numerically nonzero, physically no dephasing
        ("z", 3000, False),  # through-plane moment cannot dephase a 2D slice
        (None, 0, False),
    ],
)
def test_additional_preparation_spoiling_needs_two_cycles_per_voxel(
    axis, area, spoiled
):
    system = Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    seq = Sequence(system)
    rf = make_block_pulse(np.pi / 2, duration=0.001, system=system)
    seq.add_block(rf)
    if axis:
        seq.add_block(make_trapezoid(axis, area=area, duration=0.003, system=system))
    following = sum(seq.block_durations.values())
    seq.add_block(rf)
    acq = SimpleNamespace(
        seq=seq,
        pulses=[
            SimpleNamespace(end=calc_duration(rf)),
            SimpleNamespace(start=following, end=following + calc_duration(rf)),
        ],
    )
    ids, times = spoiled_preparations(acq, [0], SPEC)
    assert ids == ([0] if spoiled else [])
    assert times == ([following] if spoiled else [])


def test_s4_oracle_spoils_only_verified_preparations_not_readout_rf(monkeypatch):
    acq = SimpleNamespace(
        adc=np.array([0.03, 0.06]),
        measurements={"preparation_spoil_times_s": [0.02, 0.05, 0.99]},
    )
    calls = []
    monkeypatch.setattr(s4, "rf_only_response", lambda *a, **kw: calls.append(kw))
    s4.canonical_response(acq, np.array([[1, 1, 0.1]]))
    assert calls == [{"spoil_times_s": [0.02, 0.05]}]
    acq.measurements = {}
    s4.canonical_response(acq, np.array([[1, 1, 0.1]]))
    assert calls[-1] == {"spoil_times_s": []}


def test_explicit_preparation_spoiling_cannot_enter_encoded_replay():
    with pytest.raises(ValueError, match="RF-only replay"):
        s1_koma.simulate(None, None, spoil_times_s=[0.1])


@pytest.mark.parametrize(
    "field,columns,ideal_spoiling", [(1.5, 4, True), (3.0, 4, False), (3.0, 3, False)]
)
def test_canonical_adc_expansion_preserves_decay_frequency_and_receiver(
    monkeypatch, field, columns, ideal_spoiling
):
    pulses = [SimpleNamespace(start=t, end=t + 0.001) for t in (0.0, 0.02, 0.04)]
    adc = np.concatenate([p.end + np.arange(1, 129) * 10e-6 for p in pulses])
    source = SimpleNamespace(
        pulses=pulses,
        adc=adc,
        k=np.zeros((len(adc), 3)),
        seq=SimpleNamespace(system=SimpleNamespace(B0=field)),
    )
    properties = np.array([[1, 0.8, 0.08, 0], [0.9, 0.6, 0.005, -440]])[:, :columns]
    monkeypatch.setattr(s1_image, "expected_receiver_phase", lambda a: 31 * a.adc)
    calls = []

    def exact_outputs(acquisition, spins, **kwargs):
        assert acquisition.pulses is pulses  # Never truncate intervening RF history.
        assert kwargs == {
            "zero_gradients": True,
            "individual": True,
            "ideal_spoiling": ideal_spoiling,
        }
        np.testing.assert_array_equal(spins.properties, properties)
        calls.append(len(acquisition.adc))
        groups = np.searchsorted([p.end for p in pulses], acquisition.adc) - 1
        elapsed = acquisition.adc - np.array([p.end for p in pulses])[groups]
        frequency = properties[:, 3] * field / 3 if columns == 4 else np.zeros(2)
        return (groups[:, None] + 1 + 0.3j) * np.exp(
            -elapsed[:, None] / properties[:, 2]
            - 2j * np.pi * elapsed[:, None] * frequency
            - 1j * acquisition.adc_phase[:, None]
        )

    monkeypatch.setattr(s1_image, "simulate", exact_outputs)
    compressed = s1_image.rf_only_response(
        source, properties, ideal_spoiling=ideal_spoiling
    )
    full = s1_image.rf_only_response(
        source, properties, ideal_spoiling=ideal_spoiling, compress=False
    )
    assert calls == [3, 384]
    np.testing.assert_allclose(compressed, full, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(source.adc, adc)


@pytest.mark.parametrize("mutation", ["during_rf", "overlap", "unordered", "before_rf"])
def test_unproven_rf_free_intervals_keep_all_adc_outputs(monkeypatch, mutation):
    pulses = [
        SimpleNamespace(start=0.0, end=0.001),
        SimpleNamespace(start=0.02, end=0.021),
    ]
    adc = np.array([0.002, 0.003, 0.022, 0.023])
    if mutation == "during_rf":
        adc[2] = 0.0205
    elif mutation == "overlap":
        pulses[1].start = 0.0005
    elif mutation == "unordered":
        adc = adc[::-1]
    else:
        adc[0] = -0.001
    source = SimpleNamespace(pulses=pulses, adc=adc)
    monkeypatch.setattr(
        s1_image, "expected_receiver_phase", lambda a: np.zeros(len(a.adc))
    )

    def full_outputs(acquisition, *_args, **_kwargs):
        np.testing.assert_array_equal(acquisition.adc, adc)
        return np.ones((4, 1), dtype=complex)

    monkeypatch.setattr(s1_image, "simulate", full_outputs)
    np.testing.assert_array_equal(
        s4.canonical_response(source, np.array([[1, 1, 0.1]])), np.ones((1, 4))
    )


@pytest.mark.parametrize(
    "family,spoiled",
    [("gre_t1w", True), ("bssfp", False), ("rfov_epi", False), ("spin_echo_t2w", True)],
)
def test_finite_rf_oracle_preserves_spoiling_and_external_refocusing_sign(
    monkeypatch, family, spoiled
):
    pulse = SimpleNamespace(
        start=0.0, center=0.0005, phase=0.3, frequency=0.0, refocusing=True
    )
    source = SimpleNamespace(pulses=[pulse], adc=np.array([0.002, 0.003]))
    properties = np.array([[1.0, 0.8, 0.08]])

    def response(acquisition, actual_properties, *, ideal_spoiling):
        assert acquisition.pulses == source.pulses
        assert ideal_spoiling is spoiled
        np.testing.assert_array_equal(actual_properties, properties)
        return np.full((1, 2), 1 + 0.3j)

    monkeypatch.setattr(s1_image, "rf_only_response", response)
    np.testing.assert_array_equal(
        s1_image.canonical_response(source, properties, family, finite_rf=True),
        np.full((1, 2), -1 - 0.3j),
    )
