"""An unavailable cardiac phase oracle has no image or mapping verdict."""

from types import SimpleNamespace

import numpy as np

from mriseqbench.evaluation import s1_case, s4
from mriseqbench.evaluation.s1_phantom import Spins


def test_cardiac_phase_oracle_needs_half_the_reference_energy(monkeypatch):
    geometry = Spins(
        np.array(
            [[-0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0]]
        ),
        np.zeros(4, int),
        ["myocardium"],
        np.ones((1, 4)),
        np.ones(4),
        (2, 2),
        "test",
    )
    acquisition = SimpleNamespace(
        adc=np.array([0.01]), k=np.zeros((1, 3)), lines=[np.array([0])], pulses=[]
    )
    case = {
        "task": {"sequence": {"matrix": [2, 2], "fov_mm": [200, 200]}},
        "physics": {"effects": [{"kind": "b0"}]},
        "object": {"phantom": "cardiac_2d@1"},
    }
    monkeypatch.setattr(s1_case, "phantom", lambda *_: geometry)
    monkeypatch.setattr(s4, "expected_receiver_phase", lambda *_: np.zeros(1))
    monkeypatch.setattr(
        s4,
        "simulate",
        lambda _a, _s, effects, **_kw: (
            np.array([[1, 0, 0, 0]]) if effects else np.ones((1, 4))
        ),
    )
    _, _, info = s4.phase_reference(acquisition, case, None, np.ones((2, 2)))
    assert info["temporal_phase_chord_rms"] == 0
    assert info["reference_energy_coverage"] == 0.25
    assert not info["converged"]


def test_early_unresolved_frame_skips_mapping_and_encoded_replay(monkeypatch, tmp_path):
    acquisition = SimpleNamespace(
        checks=[{"name": "sequence", "passed": True}],
        adc=np.array([0.01, 0.02]),
        measurements={"frames": [{"central_adc_index": 0}, {"central_adc_index": 1}]},
    )
    case = {"task": {"id": "t1_mapping"}, "evaluation": {"version": 2}}
    monkeypatch.setattr(s4, "inspect_sequence", lambda *_: acquisition)
    monkeypatch.setattr(s4, "frame_acquisition", lambda *_: acquisition)

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Unresolved phase must not reach encoded replay or mapping"
        )

    monkeypatch.setattr(s4, "FullFrameReplay", lambda *_: forbidden)
    monkeypatch.setattr(s4, "_mapping", forbidden)
    monkeypatch.setattr(s4, "_mapping_model", forbidden)
    monkeypatch.setattr(
        s4,
        "evaluate_s1",
        lambda *_args, **_kwargs: {
            "status": "numerically_unresolved",
            "sequence_valid": True,
            "numerically_converged": False,
            "checks": [],
            "metrics": {},
            "image_checks": [],
            "artifacts": {},
        },
    )
    result = s4.evaluate_s4(None, case, None, tmp_path, physio_resolution={})
    assert result["status"] == "numerically_unresolved"
    assert result["image_verdict"] == "numerically_unresolved"
    assert result["sequence_valid"]
    assert result["imaging_success_under_case_physics"] is None
    assert result["image_checks"] == []
    assert not result["application_checks"]
