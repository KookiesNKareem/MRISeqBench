"""Published integration budgets govern 2D and 3D adaptive replay; no MRI replay."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s1
from mriseqbench.evaluation.s1_phantom import Spins


@pytest.mark.parametrize(
    "matrix,maximum,resume",
    [
        ([128, 128], 131072, False),
        ([128, 128, 32], 100000, False),
        ([128, 128], 100000, True),
        ([128, 128, 32], 131072, True),
    ],
)
def test_unresolved_scores_refine_only_within_declared_budget(
    monkeypatch, tmp_path, matrix, maximum, resume
):
    initial, replicates = 32768, 8
    case = {
        "task": {
            "id": "caipirinha",
            "sequence": {"matrix": matrix, "fov_mm": [200] * len(matrix)},
        },
        "evaluation": {
            "id": "s3",
            "version": 3,
            "metrics": [],
            "calibrated": True,
            "spatial_samples": initial,
            "spatial_max_samples": maximum,
            "spatial_replicates": replicates,
        },
        "physics": {"effects": []},
        "object": {"phantom": "synthetic"},
    }
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, int),
        ["water"],
        np.ones((1, 3)),
        1.0,
        tuple(matrix),
        "test",
    )
    acquisition = SimpleNamespace(checks=[], measurements={}, adc=np.zeros(1))
    requested = []

    def sample(_case, _root, samples, replicate, **_kwargs):
        assert samples <= maximum, "A numerical budget must not be silently exceeded"
        requested.append((samples, replicate))
        return spins

    class Replay:
        hits = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return np.ones(1, complex)

    monkeypatch.setattr(s1, "phantom", lambda *_: spins)
    monkeypatch.setattr(s1, "sampled_phantom", sample)
    monkeypatch.setattr(s1, "ReplayCache", Replay)
    monkeypatch.setattr(s1, "image_scores", lambda *_: ({}, {}))
    monkeypatch.setattr(s1, "score_precision", lambda *_: (False, {}))
    monkeypatch.setattr(
        s1, "resolve_decision", lambda *_args, **_kwargs: (None, "unresolved")
    )
    submission = tmp_path / "sequence.seq"
    submission.write_text("mocked sequence")
    if resume:
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "case_sha256": hashlib.sha256(
                        json.dumps(case, sort_keys=True).encode()
                    ).hexdigest(),
                    "sequence_sha256": hashlib.sha256(
                        submission.read_bytes()
                    ).hexdigest(),
                    "progress": {
                        "spatial_refinement": [
                            {
                                "samples": 2 * initial,
                                "decision_basis": {"case": "unresolved"},
                            }
                        ],
                        "temporal_refinement": [
                            {"step_s": 0.001, "rf_step_s": 0.00002, "nrmse": 0.0}
                        ],
                    },
                }
            )
        )
        monkeypatch.setenv("MRISEQBENCH_NUMERICAL_CHECKPOINT", str(checkpoint))
    report = s1.evaluate_s1(
        submission,
        case,
        tmp_path,
        tmp_path / "evaluation",
        inspector=lambda *_: acquisition,
        probes=lambda *_: ([], {}, np.ones((1, 1))),
        analytic_sampler=lambda *_: np.ones(1),
        sampler=lambda *_: np.ones(1),
        # Linear like the real reconstructions: a zero timestep difference
        # gives a zero reconstructed temporal error.
        reconstructor=lambda raw, *_: np.reshape(raw, (1, 1)),
        reference_builder=lambda *_: np.ones((1, 1)),
        control_builder=None,
    )
    assert requested == [
        (count, replicate)
        for count in ([maximum] if resume else [initial, 2 * initial, maximum])
        for replicate in range(replicates)
    ]
    assert report["status"] == "numerically_unresolved"
    assert report["numerically_converged"] is False
    assert report["failure_kind"] == "numerical_precision_not_reached"


def test_never_converging_replay_stops_at_the_timestep_cap(monkeypatch, tmp_path):
    from mriseqbench.evaluation.evaluate import normalize_verdict
    from mriseqbench.evaluation.s1_case import MAX_TEMPORAL_HALVINGS

    matrix = [128, 128]
    case = {
        "case_id": "never-converges",
        "task": {
            "id": "caipirinha",
            "sequence": {"matrix": matrix, "fov_mm": [200, 200]},
        },
        "evaluation": {
            "id": "s3",
            "version": 3,
            "metrics": [],
            "calibrated": True,
            "spatial_samples": 32768,
            "spatial_replicates": 2,
        },
        "physics": {"effects": []},
        "object": {"phantom": "synthetic"},
    }
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, int),
        ["water"],
        np.ones((1, 3)),
        1.0,
        tuple(matrix),
        "test",
    )
    acquisition = SimpleNamespace(checks=[], measurements={}, adc=np.zeros(1))
    steps = []

    class Replay:
        hits = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, step_s, **_kwargs):
            steps.append(step_s)
            return np.array([1.0 + step_s * 1e6], complex)  # 10% change per halving

    monkeypatch.setattr(s1, "phantom", lambda *_: spins)
    monkeypatch.setattr(s1, "sampled_phantom", lambda *_a, **_k: spins)
    monkeypatch.setattr(s1, "ReplayCache", Replay)
    submission = tmp_path / "sequence.seq"
    submission.write_text("mocked sequence")
    report = s1.evaluate_s1(
        submission,
        case,
        tmp_path,
        tmp_path / "evaluation",
        inspector=lambda *_: acquisition,
        probes=lambda *_: ([], {}, np.ones((1, 1))),
        analytic_sampler=lambda *_: np.ones(1),
        sampler=lambda *_: np.ones(1),
        reconstructor=lambda raw, *_: np.reshape(raw, (1, 1)),
        reference_builder=lambda *_: np.ones((1, 1)),
        control_builder=None,
    )
    assert report["status"] == "numerically_unresolved"
    assert report["failure_kind"] == "unresolved_at_timestep_cap"
    assert len(report["temporal_refinement"]) == MAX_TEMPORAL_HALVINGS
    assert min(steps) == pytest.approx(0.001 / 2**MAX_TEMPORAL_HALVINGS)
    verdict = normalize_verdict(report, case)
    assert verdict["status"] == "failed"
    assert verdict["failure_kind"] == "unresolved_at_timestep_cap"


def test_phase_oracle_refinement_is_capped():
    from mriseqbench.evaluation.s1_case import MAX_TEMPORAL_HALVINGS, converge_phase

    calls = []

    def replay(refinement=1):
        calls.append(refinement)
        # Phase keeps moving by a fixed amount however fine the step.
        return np.full((2, 2), np.exp(1j * 0.1 * refinement)), np.ones((2, 2), bool)

    case = {"object": {"phantom": "synthetic"}}
    _, _, summary = converge_phase(replay, np.ones((2, 2)), 1, case)
    assert not summary["converged"] and summary["timestep_cap_reached"]
    assert max(calls) == 2**MAX_TEMPORAL_HALVINGS


def test_s4_keeps_a_frame_timestep_cap_cause(monkeypatch, tmp_path):
    from mriseqbench.catalog import Catalog
    from mriseqbench.evaluation import s4
    from mriseqbench.evaluation.evaluate import normalize_verdict
    from mriseqbench.evaluation.s1 import failed_report
    from tests.fixtures.reference128_advanced import build

    root = Path(__file__).resolve().parents[2]
    catalog = Catalog(root)
    case = catalog.resolve(
        catalog.get("tasks", "t2_prepared"), hardware_ref="standard_1_5t"
    )
    path = tmp_path / "t2_prepared.seq"
    build(case).write(str(path))

    def capped_frames(acq, *args, **kwargs):
        frame = failed_report(case, [])
        frame.update(
            status="numerically_unresolved",
            sequence_valid=True,
            failure_kind="unresolved_at_timestep_cap",
        )
        return [acq], [frame]

    monkeypatch.setattr(s4, "_evaluate_frames", capped_frames)
    report = s4.evaluate_s4(path, case, root, tmp_path / "work", physio_resolution={})
    assert report["unresolved_frames"] == [0]
    verdict = normalize_verdict(report, case)
    assert verdict["status"] == "failed"
    assert verdict["failure_kind"] == "unresolved_at_timestep_cap"
