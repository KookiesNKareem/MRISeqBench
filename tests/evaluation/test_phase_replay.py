"""Exact input sharing for RF phase targets; native dynamics are stubbed here."""

from types import SimpleNamespace

import numpy as np

from mriseqbench.evaluation import s1_case
from mriseqbench.evaluation.s1_phantom import Spins


def inputs():
    positions = np.array(
        [
            [0, 0, 0],
            [0, 0.02, 0],
            [0.01, 0.01, 0],
            [0, 0.02, 0],
            [0.01, -0.03, 0],
            [np.nextafter(0.01, 1), 0.01, 0],
        ]
    )
    spins = Spins(
        positions,
        np.array([0, 0, 0, 1, 1, 0]),
        ["water", "fat"],
        np.array([[1, 0.8, 0.08, 0], [0.6, 1.1, 0.09, -433]]),
        np.array([0.1, 0.3, 0.2, 0.4, 0.5, 0.6]),
        (6,),
        "test",
    )
    acq = SimpleNamespace(adc=np.array([0.01, 0.03, 0.07]))
    return acq, spins


def response(acq, spins, effects, **_options):
    offsets = np.zeros(len(spins.labels))
    for effect in effects:
        if effect["kind"] == "b0" and effect["model"] in {"constant", "linear"}:
            offsets += effect.get("offset_hz", 0)
            if effect["model"] == "linear":
                offsets += spins.positions @ np.array(effect["gradient_hz_per_m"])
    props = spins.properties[spins.labels]
    phase = acq.adc[:, None] * (offsets + props[:, 3]) + np.arange(3)[:, None] / 7
    return (
        props[:, 0]
        * spins.weight
        * (1 - np.exp(-0.4 / props[:, 1]))
        * np.exp(-acq.adc[:, None] / props[:, 2] + 2j * np.pi * phase)
    )


def test_phase_input_sharing_preserves_weights_materials_offsets_and_contexts(
    monkeypatch,
):
    acq, spins = inputs()
    seen = []

    def replay(a, s, effects, **options):
        seen.append(s)
        return response(a, s, effects, **options)

    monkeypatch.setattr(s1_case, "simulate", replay)
    effects = [
        {
            "kind": "b0",
            "model": "linear",
            "offset_hz": 0.2,
            "gradient_hz_per_m": [100, 0, 0],
        },
        {"kind": "b0", "model": "constant", "offset_hz": -0.1},
    ]
    for field in ((), effects):
        actual = s1_case.phase_replay(
            acq, spins, field, zero_gradients=True, individual=True
        )
        np.testing.assert_allclose(
            actual, response(acq, spins, field), rtol=2e-15, atol=1e-16
        )
        assert len(seen[-1].labels) == (2 if not field else 5)
        assert seen[-1].weight == 1
    # Even cancelling fields retain every input axis used by either Julia field.
    cancelling = effects + [
        {
            "kind": "b0",
            "model": "linear",
            "offset_hz": 0,
            "gradient_hz_per_m": [-100, 0, 0],
        }
    ]
    actual = s1_case.phase_replay(
        acq, spins, cancelling, zero_gradients=True, individual=True
    )
    np.testing.assert_allclose(
        actual, response(acq, spins, cancelling), rtol=2e-15, atol=1e-16
    )
    assert len(seen[-1].labels) == 5
    np.testing.assert_array_equal(spins.weight, [0.1, 0.3, 0.2, 0.4, 0.5, 0.6])


def test_spatial_excitation_motion_and_other_fields_keep_original_inputs(monkeypatch):
    acq, spins = inputs()
    seen = []

    def replay(a, s, effects, **options):
        seen.append((s, effects, options))
        return response(a, s, effects)

    monkeypatch.setattr(s1_case, "simulate", replay)
    for effects, zero_gradients in [
        ((), False),
        ([{"kind": "motion", "model": "rigid_sinusoidal"}], True),
        ([{"kind": "b0", "model": "gaussian"}], True),
        ([{"kind": "time_variation", "model": "sinusoidal"}], True),
    ]:
        s1_case.phase_replay(
            acq, spins, effects, zero_gradients=zero_gradients, individual=True
        )
        assert seen[-1][0] is spins and seen[-1][1] is effects
        assert seen[-1][2]["zero_gradients"] is zero_gradients
