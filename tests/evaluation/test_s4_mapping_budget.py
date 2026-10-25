"""Mapping uncertainty must respect the same public sample cap as image scoring."""

from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s4


def frame(samples):
    return {
        "sequence_valid": True,
        "image_checks": [{"name": "image", "passed": True}],
        "spatial_sampling": {"samples_per_replicate": samples},
    }


def test_frame_override_cannot_exceed_public_cap(monkeypatch, tmp_path):
    acquisition = SimpleNamespace(measurements={"frames": [{"central_adc_index": 0}]})
    seen = []
    monkeypatch.setattr(s4, "frame_acquisition", lambda *_: acquisition)

    def evaluate(_submission, case, *_args, **_kwargs):
        seen.append(case["evaluation"]["spatial_samples"])
        return frame(seen[-1])

    monkeypatch.setattr(s4, "evaluate_s1", evaluate)
    case = {"evaluation": {"spatial_samples": 32768, "spatial_max_samples": 100000}}
    s4._evaluate_frames(acquisition, None, case, None, tmp_path, samples=262144)
    assert seen == [100000]
    assert case["evaluation"]["spatial_samples"] == 32768


@pytest.mark.parametrize("outcome_at_cap", [None, True])
@pytest.mark.parametrize("initial_samples", [[32768], [32768, 100000]])
def test_mapping_clips_growth_and_stops_at_cap(
    monkeypatch, tmp_path, outcome_at_cap, initial_samples
):
    seen = []
    case = {
        "evaluation": {
            "spatial_samples": 32768,
            "spatial_max_samples": 100000,
            "mapping_relative_error_precision": 0.001,
        }
    }

    def mapping(_acquisitions, results, *_args, **_kwargs):
        outcome = (
            outcome_at_cap
            if all(
                result["spatial_sampling"]["samples_per_replicate"] == 100000
                for result in results
            )
            else None
        )
        return [{"name": "mapping", "passed": outcome}], {"error": 0.02}

    def evaluate(*_args, samples, **_kwargs):
        seen.append(samples)
        assert samples <= 100000
        return [None] * len(initial_samples), [frame(samples)] * len(initial_samples)

    monkeypatch.setattr(s4, "_mapping", mapping)
    monkeypatch.setattr(s4, "_evaluate_frames", evaluate)
    result = s4._mapping_refinement(
        None,
        None,
        [None] * len(initial_samples),
        [frame(n) for n in initial_samples],
        case,
        None,
        tmp_path,
    )
    assert seen == ([65536, 100000] if max(initial_samples) < 100000 else [100000])
    assert result[2][0]["passed"] is outcome_at_cap
    assert result[4]["max_samples_per_replicate"] == 100000
    if outcome_at_cap is None:
        assert result[4]["stop_reason"] == "spatial_max_samples"
    else:
        assert "stop_reason" not in result[4]


def certified(passed, converged=True):
    return {
        "status": "failed" if False in passed else "passed",
        "sequence_valid": True,
        "numerically_converged": converged,
        "checks": [],
        "metrics": {},
        "image_checks": [
            {"name": f"m{i}", "passed": value} for i, value in enumerate(passed)
        ],
        "ideal_image_checks": [],
        "spatial_sampling": {"samples_per_replicate": 32768},
        "artifacts": {},
    }


def test_certified_frame_failure_stops_frame_loop(monkeypatch, tmp_path):
    acquisition = SimpleNamespace(
        measurements={"frames": [{"central_adc_index": i} for i in range(3)]}
    )
    monkeypatch.setattr(s4, "frame_acquisition", lambda *_: acquisition)
    seen = []

    def evaluate(*_args, **_kwargs):
        seen.append(len(seen))
        return certified([False, None, True])

    monkeypatch.setattr(s4, "evaluate_s1", evaluate)
    case = {"evaluation": {"spatial_samples": 32768}}
    _, results = s4._evaluate_frames(acquisition, None, case, None, tmp_path)
    assert seen == [0] and len(results) == 1


def test_refinement_stage_can_certify_a_failure(monkeypatch, tmp_path):
    case = {
        "evaluation": {
            "spatial_samples": 32768,
            "spatial_max_samples": 100000,
            "mapping_relative_error_precision": 0.001,
        }
    }
    mapped = []

    def mapping(_acquisitions, results, *_args, **_kwargs):
        mapped.append(len(results))
        return [{"name": "mapping", "passed": None}], {"error": 0.02}

    monkeypatch.setattr(s4, "_mapping", mapping)
    monkeypatch.setattr(
        s4,
        "_evaluate_frames",
        lambda *_a, samples, **_k: ([None], [certified([False, None])]),
    )
    result = s4._mapping_refinement(
        None,
        None,
        [None, None],
        [certified([True]), certified([True])],
        case,
        None,
        tmp_path,
    )
    assert mapped == [2]  # one stage at the initial sampling, then certified
    assert len(result[1]) == 1 and result[2] == []
    assert result[4]["stop_reason"] == "certified_image_failure"
    assert result[4]["certified_failed_frames"] == [
        {"frame": 0, "failed_checks": ["m0"]}
    ]


def test_decisive_failure_outranks_unresolved_metric(monkeypatch, tmp_path):
    acquisition = SimpleNamespace(
        checks=[{"name": "sequence", "passed": True}],
        adc=np.array([0.01]),
        measurements={"frames": [{"central_adc_index": 0}]},
    )
    case = {
        "task": {"id": "adiabatic_inversion"},
        "evaluation": {"version": 2, "calibrated": True},
    }
    monkeypatch.setattr(s4, "inspect_sequence", lambda *_: acquisition)
    monkeypatch.setattr(s4, "frame_acquisition", lambda *_: acquisition)
    monkeypatch.setattr(s4, "family", lambda *_: "adiabatic_inversion")
    monkeypatch.setattr(s4, "bart_provenance", dict)
    monkeypatch.setattr(
        s4, "evaluate_s1", lambda *_a, **_k: certified([False, None, True])
    )
    result = s4.evaluate_s4(None, case, None, tmp_path, physio_resolution={})
    assert result["status"] == "failed" and result["image_verdict"] == "failed"
    assert result["numerically_converged"] is True
    assert result["imaging_success_under_case_physics"] is False
    monkeypatch.setattr(s4, "evaluate_s1", lambda *_a, **_k: certified([None, True]))
    result = s4.evaluate_s4(None, case, None, tmp_path / "u", physio_resolution={})
    assert result["status"] == "numerically_unresolved"
