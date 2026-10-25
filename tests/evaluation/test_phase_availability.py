"""An unavailable independent phase target cannot become a participant failure."""

from types import SimpleNamespace

import numpy as np

from mriseqbench.evaluation import s1, s1_case
from mriseqbench.evaluation.s1_phantom import Spins


def test_null_phase_oracle_is_unresolved_before_encoded_replay(monkeypatch, tmp_path):
    case = {
        "task": {
            "id": "gre_t1w",
            "sequence": {"matrix": [1, 1], "fov_mm": [200, 200]},
        },
        "image_policy": "static",
        "evaluation": {
            "id": "s1",
            "version": 4,
            "metrics": [],
            "case_image_criteria": {
                "static": {
                    "complex_nrmse_max": 0.45,
                    "magnitude_nrmse_max": 0.35,
                    "phase_rms_rad_max": 0.35,
                    "tissue_signal_fraction_min": 0.5,
                    "contrast_retention_min": 0.5,
                    "outside_energy_fraction_max": 0.25,
                }
            },
        },
        "physics": {"effects": [{"kind": "b0"}]},
        "object": {"phantom": "synthetic"},
    }
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, int),
        ["water"],
        np.array([[1, 0.8, 0.08]]),
        1.0,
        (1, 1),
        "test",
    )
    acq = SimpleNamespace(
        checks=[],
        measurements={},
        adc=np.array([0.01]),
        k=np.zeros((1, 3)),
        lines=[np.array([0])],
        pulses=[],
    )
    monkeypatch.setattr(s1, "phantom", lambda *_: spins)
    monkeypatch.setattr(s1_case, "phantom", lambda *_: spins)
    monkeypatch.setattr(s1_case, "expected_receiver_phase", lambda *_: np.zeros(1))
    monkeypatch.setattr(
        s1_case,
        "phase_replay",
        lambda _acq, _spins, effects, **_kwargs: np.array([[0j if effects else 1j]]),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Cannot grade an image against an unavailable phase oracle"
        )

    pulse = tmp_path / "sequence.seq"
    pulse.write_text("waveforms stubbed")
    report = s1.evaluate_s1(
        pulse,
        case,
        tmp_path,
        tmp_path / "evaluation",
        inspector=lambda *_: acq,
        probes=lambda *_: ([], {}, np.ones((1, 1))),
        analytic_sampler=lambda *_: np.ones(1),
        reference_builder=lambda *_: np.ones((1, 1)),
        simulator=forbidden,
        control_builder=forbidden,
    )
    assert report["sequence_valid"] is True
    assert report["status"] == "numerically_unresolved"
    assert report["numerically_converged"] is False
    assert report["failure_kind"] == "phase_reference_not_resolved"
    assert report["case_phase_reference"]["reference_energy_coverage"] == 0
    assert not any(check["passed"] is False for check in report["checks"])
