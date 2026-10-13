"""Precision must be independent of the distance to an acceptance threshold."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1
from mriseqbench.models import Evaluation, ImagePrecision
from tests.fixtures.legacy_catalog import resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value,passed", [(0.1999, True), (0.2, True), (0.2001, False)])
def test_precise_scores_crossing_threshold_receive_point_verdict(value, passed):
    metrics = {"complex_image_nrmse": value}
    metrics["normalized_squared_error"] = value**2
    intervals = {"normalized_squared_error": (value**2 - 0.0004, value**2 + 0.0004)}
    precise, _ = s1.score_precision(metrics, intervals, ImagePrecision().model_dump())
    assert precise
    assert (
        s1.decision(
            [{"name": "ideal_complex_image_nrmse", "min": None, "max": 0.2}],
            metrics,
            "ideal_",
        )
        is passed
    )


def test_precision_target_is_cutoff_independent_and_zero_clipping_is_honest():
    for value, limits in [(0.25, (0.24, 0.26)), (-0.0015, (-0.002, 0.001))]:
        precise, errors = s1.score_precision(
            {"normalized_squared_error": value},
            {"normalized_squared_error": limits},
            ImagePrecision().model_dump(),
        )
        assert not precise
        assert errors["normalized_squared_error"] > 0.0008


@pytest.mark.parametrize(
    "value,limits,expected",
    [(0.1, (0.05, 0.15), True), (0.4, (0.3, 0.5), False), (0.2, (0.19, 0.21), None)],
)
def test_decisive_intervals_avoid_unnecessary_refinement(value, limits, expected):
    bounds = [{"name": "ideal_complex_image_nrmse", "min": None, "max": 0.2}]
    verdict, basis = s1.resolve_decision(
        bounds,
        {"complex_image_nrmse": value},
        {"complex_image_nrmse": limits},
        False,
        "ideal_",
    )
    assert verdict is expected
    assert basis == ("interval_certified" if expected is not None else "unresolved")


def test_precision_and_sampling_contract_is_validated():
    case = Catalog(ROOT).resolve(
        Catalog(ROOT).get("tasks", "bssfp"), Catalog(ROOT).get("physics", "ideal")
    )
    for patch in [
        {"spatial_max_samples": 65535},
        {"spatial_max_samples": 4096},
        {"spatial_max_samples": 2097152},
        {"image_precision": {"normalized_squared_error": 0}},
        {"image_precision": {"normalized_squared_error": float("nan")}},
        {"spatial_seed": -1},
    ]:
        with pytest.raises(ValidationError):
            Evaluation.model_validate({**case["evaluation"], **patch})


@pytest.mark.parametrize(
    "resolve_at,deadline,temporal_steps",
    [(8192, None, 0), (16384, None, 0), (16384, 0, 0), (8192, None, 3)],
)
def test_pipeline_refines_on_precision_and_preserves_unresolved_checks(
    monkeypatch, tmp_path, resolve_at, deadline, temporal_steps
):
    # Engine correctness has independent real KomaMRI tests. Here controlled
    # replays isolate the stopping/verdict/report logic, including a straddling CI.
    case = resolve_legacy_case(Catalog(ROOT), "bssfp", physics="ideal")
    case["task"]["sequence"]["matrix"] = [8, 8]
    case["evaluation"].update(spatial_samples=4096, spatial_max_samples=8192)
    loader = importlib.util.spec_from_file_location(
        "controls", ROOT / "tests/fixtures/s1_sequences.py"
    )
    controls = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(controls)
    path = tmp_path / "sequence.seq"
    controls.make_sequence(case).write(str(path))
    seen = []
    original_sampler = s1.sampled_phantom

    def sampler(case, root, samples, replicate):
        seen.append(samples)
        return original_sampler(case, root, samples, replicate)

    def scores(images, reference, support):
        radius = 0.001 if seen[-1] >= resolve_at else 0.01
        values = {
            "complex_image_nrmse": 0.2001,
            "normalized_squared_error": 0.2001**2,
            "signal_fraction": 1.0,
            "outside_object_energy_fraction": 0.01,
        }
        intervals = {
            name: (value - radius, value + radius) for name, value in values.items()
        }
        intervals["normalized_squared_error"] = (
            values["normalized_squared_error"] - radius * 0.4,
            values["normalized_squared_error"] + radius * 0.4,
        )
        return values, intervals

    monkeypatch.setattr(s1, "sampled_phantom", sampler)
    monkeypatch.setattr(
        s1,
        "mechanism_probes",
        lambda acq, case, materials, props: (
            [],
            {},
            np.ones((len(materials), len(acq.adc)), complex),
        ),
    )
    monkeypatch.setattr(
        s1,
        "simulate",
        lambda acq, spins, *args, **kwargs: (
            np.ones(len(acq.adc), complex)
            * np.sum(spins.weight)
            * (
                1 + (kwargs.get("rf_step_s", 20e-6) / 20e-6) ** 2
                if temporal_steps
                else 1
            )
        ),
    )
    monkeypatch.setattr(
        s1, "reconstruct", lambda raw, acq, matrix: np.asarray(raw).reshape(matrix)
    )
    monkeypatch.setattr(s1, "image_scores", scores)
    monkeypatch.setattr(s1, "versions", dict)
    result = s1.evaluate_s1(
        path, deepcopy(case), ROOT, tmp_path / "evaluation", deadline=deadline
    )
    assert seen == [n for n in (4096, 8192, 16384) if n <= resolve_at for _ in range(8)]
    assert result["spatial_sampling"]["stop_reason"] == "precision_target"
    assert result["spatial_sampling"]["exceeded_initial_budget"] is (resolve_at > 8192)
    assert result["sequence_valid"]
    assert result["status"] == "failed"
    assert result["ideal_imaging_success"] is False
    assert result["image_checks"][0]["passed"] is False
    assert result["numerically_converged"] is True
    assert result["metrics"]["temporal_nrmse"] <= 0.01
    if temporal_steps:
        import json

        progress = json.loads((tmp_path / "evaluation/progress.json").read_text())
        assert len(progress["temporal_refinement"]) > temporal_steps
