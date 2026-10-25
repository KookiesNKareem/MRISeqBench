"""Batched RF probes retain every causal echo and each shot's actual phase."""

from types import SimpleNamespace

import numpy as np

from mriseqbench.evaluation import s2


def test_selective_preparation_signal_and_t2_probes_use_finite_rf(monkeypatch):
    # An off-resonance CHESS envelope is nominally 90 degrees, but must not be
    # interpreted as an instantaneous water excitation by either diagnostic.
    prep = SimpleNamespace(start=0.0, center=0.002, refocusing=False, frequency=-440)
    exc = SimpleNamespace(start=0.01, center=0.0101, refocusing=False, frequency=0)
    ref = SimpleNamespace(start=0.03, center=0.0301, refocusing=True, frequency=0)
    acq = SimpleNamespace(
        pulses=[prep, exc, ref],
        excitation=np.array([1]),
        adc=np.array([0.0501]),
        lines=[np.array([0])],
        k=np.zeros((1, 3)),
        measurements={"centre_sample_indices": [0]},
    )
    mode = "t2"

    def response(_acq, properties, _family, *, finite_rf=False):
        # A T2-decaying echo after water-preserving preparation; the old
        # instantaneous model instead has almost no surviving water signal.
        if not finite_rf:
            return np.full((len(properties), 1), 0.001)
        if mode == "weak":
            return np.full((len(properties), 1), 0.001)
        if mode == "t1":
            return 1 - np.exp(-0.04 / properties[:, 1, None])
        return np.exp(-0.04 / properties[:, 2, None])

    monkeypatch.setattr(s2, "canonical_response", response)
    monkeypatch.setattr(s2, "simulate", lambda *_a, **_k: np.ones(1))
    checks, metrics, reference = s2.mechanism_probes(
        acq,
        {
            "task": {
                "id": "tse_t2w",
                "sequence": {"matrix": [128, 128], "fov_mm": [240, 240]},
            }
        },
        ["water"],
        np.array([[1.0, 0.8, 0.08]]),
    )
    assert all(c["passed"] for c in checks), checks
    np.testing.assert_allclose(metrics["diagnostic_signal_fraction"], np.exp(-0.5))
    np.testing.assert_allclose(reference, [[np.exp(-0.5)]])
    weighting = next(c for c in checks if c["name"] == "t2_weighting")
    assert weighting["detail"]["t1"] == 0 and weighting["detail"]["t2"] > 0

    for mode, failed_name in (("weak", "usable_echo_signal"), ("t1", "t2_weighting")):
        checks, _, _ = s2.mechanism_probes(
            acq,
            {
                "task": {
                    "id": "tse_t2w",
                    "sequence": {"matrix": [128, 128], "fov_mm": [240, 240]},
                }
            },
            ["water"],
            np.array([[1.0, 0.8, 0.08]]),
        )
        assert any(c["name"] == failed_name and not c["passed"] for c in checks)


def test_batched_refocusing_keeps_later_echo_and_serialized_phase_failures(monkeypatch):
    def pulse(start, phase, refocusing):
        return SimpleNamespace(
            start=start, center=start + 0.0001, phase=phase, refocusing=refocusing
        )

    pulses = []
    for shot, phase in enumerate((0.0, 0.3, 0.6)):
        pulses += [
            pulse(shot, phase, False),
            pulse(shot + 0.01, phase + np.pi / 2, True),
            pulse(shot + 0.03, phase + np.pi / 2 + (0.1 if shot == 2 else 0), True),
        ]
    acq = SimpleNamespace(
        pulses=pulses,
        excitation=np.repeat(np.arange(3), 2),
        adc=np.array(
            [shot + offset for shot in range(3) for offset in (0.0201, 0.0401)]
        ),
        lines=[np.array([i]) for i in range(6)],
        measurements={"centre_sample_indices": list(range(6))},
    )
    monkeypatch.setattr(s2, "canonical_response", lambda *_a, **_k: np.ones((1, 6)))

    def replay(actual, _spins, *, zero_gradients, frequency_offsets=None):
        assert zero_gradients
        values = []
        for echo in actual.adc:
            prefix = [p for p in actual.pulses if p.start < echo]
            refs = [p for p in prefix if p.refocusing]
            retention = 0.99 - 0.01 * (len(refs) - 1)
            if any(abs(p.phase - prefix[0].phase - np.pi / 2) > 0.05 for p in refs):
                retention = 0.4
            baseline = 0.8 ** len(refs)
            values.append(
                baseline if frequency_offsets is None else baseline * retention
            )
        return np.asarray(values)

    monkeypatch.setattr(s2, "simulate", replay)
    checks, metrics, _response = s2.mechanism_probes(
        acq, {"task": {"id": "grase"}}, ["water"], np.array([[1.0, 0.8, 0.08]])
    )
    response = next(c for c in checks if c["name"] == "refocusing_response")
    np.testing.assert_allclose(
        response["detail"]["retentions"], [0.99, 0.98, 0.99, 0.98, 0.99, 0.4]
    )
    assert not response["passed"] and metrics["min_refocusing_retention"] == 0.4
