"""Mapping precision is distinct from the unchanged physical acceptance limit."""

from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s4


@pytest.mark.parametrize(
    "value,bounds,expected,basis",
    [
        (0.0995, (0.0988, 0.1002), True, "precision_target"),
        (0.1005, (0.0998, 0.1012), False, "precision_target"),
        (0.10, (0.0995, 0.1005), True, "precision_target"),
        (0.08, (0.04, 0.095), True, "interval_certified"),
        (0.12, (0.105, 0.15), False, "interval_certified"),
        (0.10, (0.08, 0.12), None, "unresolved"),
    ],
)
def test_mapping_point_and_interval_decisions(value, bounds, expected, basis):
    actual, decision_basis, radius = s4._mapping_decision(value, bounds, 0.10, 0.001)
    assert actual is expected
    assert decision_basis == basis
    assert radius == pytest.approx(max(abs(value - bounds[0]), abs(bounds[1] - value)))


def test_mapping_reuses_fixed_dictionary_and_reports_precision(tmp_path, monkeypatch):
    grid = np.linspace(0.5, 1.5, 101)
    cache = {
        "model": {
            "dictionary": np.array([grid, 2 - grid]),
            "grid": grid,
            "mask": np.ones((1, 1), bool),
            "truth": 1.0,
            "axis": 1,
        }
    }
    reports = []
    for number, value in enumerate([1.0995, 0.9005]):
        path = tmp_path / f"frame-{number}.npz"
        np.savez(path, physical_replicates=np.full((4, 1, 1), value))
        reports.append({"artifacts": {"images": str(path)}})
    monkeypatch.setattr(
        s4, "_mapping_model", lambda *args: pytest.fail("Dictionary rebuilt")
    )
    monkeypatch.setattr(s4, "interval", lambda *args: (0.0988, 0.1002))
    case = {
        "evaluation": {
            "mapping_relative_error_max": 0.10,
            "mapping_relative_error_precision": 0.001,
        }
    }
    checks, metrics = s4._mapping([], reports, case, tmp_path, cache=cache)
    assert checks[0]["passed"] is True
    assert checks[0]["detail"]["decision_basis"] == "precision_target"
    assert checks[0]["detail"]["max"] == 0.10
    assert metrics["myocardial_t1_relative_error"] == pytest.approx(0.0995, abs=1e-7)


def test_mapping_doubles_independent_spatial_stages_and_keeps_artifacts(
    tmp_path, monkeypatch
):
    stages, caches = [], []
    case = {
        "evaluation": {
            "spatial_samples": 4096,
            "spatial_max_samples": 8192,
            "mapping_relative_error_precision": 0.001,
        }
    }

    def frames(samples, directory):
        return [
            {
                "sequence_valid": True,
                "spatial_sampling": {"samples_per_replicate": samples},
                "artifacts": {"images": str(directory / f"frame-{n}.npz")},
            }
            for n in range(3)
        ]

    def mapping(acquisitions, results, case, root, *, cache):
        caches.append(cache)
        cache["unchanged_dictionary"] = True
        return [
            {
                "name": "myocardial_t1_relative_error",
                "passed": True if len(caches) == 3 else None,
            }
        ], {}

    def evaluate_frames(
        acq, submission, case, root, workspace, *, deadline, samples, simulator
    ):
        assert deadline == 123
        assert simulator is None
        stages.append((samples, workspace))
        return [], frames(samples, workspace)

    monkeypatch.setattr(s4, "_mapping", mapping)
    monkeypatch.setattr(s4, "_evaluate_frames", evaluate_frames)
    *_, numerical = s4._mapping_refinement(
        SimpleNamespace(),
        tmp_path / "sequence.seq",
        [],
        frames(4096, tmp_path),
        case,
        tmp_path,
        tmp_path,
        deadline=123,
    )
    assert [n for n, _ in stages] == [8192, 16384]
    assert stages[0][1] != stages[1][1]
    assert all(cache is caches[0] for cache in caches)
    assert numerical["exceeded_initial_budget"] is True
    assert [
        row["samples_per_replicate"][0] for row in numerical["refinement_history"]
    ] == [4096, 8192, 16384]
    assert numerical["refinement_history"][0]["frame_artifacts"][0]["images"] == str(
        tmp_path / "frame-0.npz"
    )
    assert (tmp_path / "mapping-progress.json").exists()
