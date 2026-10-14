"""Phase, local tissue quality, and actual-physics acceptance regressions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1
from mriseqbench.evaluation.s1_case import (
    case_bounds,
    case_image_scores,
    phase_reference,
)
from mriseqbench.evaluation.s1_image import material_masks
from mriseqbench.evaluation.s1_koma import simulate
from mriseqbench.evaluation.s1_phantom import Spins, phantom
from mriseqbench.evaluation.s1_sequence import inspect_sequence
from mriseqbench.harness.prepare import evaluation_summary
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import legacy_evaluator, resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]
LEGACY_EVALUATOR = legacy_evaluator("s1")
loader = importlib.util.spec_from_file_location(
    "controls", ROOT / "tests/fixtures/s1_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)


def fixture_images():
    reference = np.zeros((32, 32), complex)
    reference[4:28, 4:28] = 1j
    reference[10:20, 10:20] = 2j
    support = reference != 0
    masks = {"a": support & (np.abs(reference) == 1), "b": np.abs(reference) == 2}
    x = np.arange(32)[:, None]
    phasor = np.exp(1j * np.broadcast_to(0.4 + 0.1 * x, reference.shape))
    return reference, support, masks, phasor


@pytest.mark.parametrize(
    "mutation",
    [None, "phase", "wrong_field_phase", "dropout", "contrast", "ghost", "blank"],
)
def test_case_quality_preserves_expected_phase_and_rejects_corruptions(mutation):
    reference, support, masks, phasor = fixture_images()
    actual = reference * phasor
    if mutation == "phase":
        actual *= 1j  # Identical magnitude, incorrect receiver phase.
    elif mutation == "wrong_field_phase":
        actual = reference.copy()  # Discarding genuine phase must fail too.
    elif mutation == "dropout":
        actual[masks["b"]] *= 0.05
    elif mutation == "contrast":
        actual[support] = 1.1j * phasor[support]
    elif mutation == "ghost":
        actual = np.roll(actual, 12, axis=0)
    elif mutation == "blank":
        actual[:] = 0
    images = np.repeat(actual[None], 8, axis=0)
    metrics, intervals, _ = case_image_scores(
        images, reference, support, phasor, support, masks
    )
    catalog = Catalog(ROOT)
    policies = [
        catalog.get("evaluators", name).case_image_criteria["static"].model_dump()
        for name in ("s1", "s2", "s3", "s4")
    ]
    assert all(policy == policies[0] for policy in policies)
    policy = policies[0]
    precise, _ = s1.score_precision(
        metrics, intervals, s1.ImagePrecision().model_dump()
    )
    assert precise
    assert s1.decision(case_bounds(policy), metrics) is (mutation is None)
    if mutation == "phase":
        assert metrics["magnitude_image_nrmse"] < 1e-7
        assert metrics["phase_error_rad"] == pytest.approx(np.pi / 2)
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize(
    "case_pass,ideal_unresolved,coverage",
    [(True, False, 1.0), (False, False, 1.0), (True, True, 1.0), (True, False, 0.2)],
)
def test_phase_aware_gate_uses_case_image_not_ideal(
    monkeypatch, tmp_path, case_pass, ideal_unresolved, coverage
):
    case = resolve_legacy_case(Catalog(ROOT), "bssfp", physics="b0_offset")
    case["task"]["sequence"]["matrix"] = [8, 8]
    case["evaluation"].update(spatial_samples=4096, spatial_max_samples=16384)
    path = tmp_path / "sequence.seq"
    controls.make_sequence(case).write(str(path))
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
        lambda acq, spins, *args, **kwargs: np.ones(len(acq.adc), complex),
    )
    monkeypatch.setattr(
        s1, "reconstruct", lambda raw, acq, matrix: np.asarray(raw).reshape(matrix)
    )
    monkeypatch.setattr(
        s1,
        "phase_reference",
        lambda *args, **kwargs: (
            np.ones((8, 8), complex),
            np.ones((8, 8), bool),
            {"converged": True, "reference_energy_coverage": coverage},
        ),
    )
    monkeypatch.setattr(s1, "versions", dict)

    def scores(error, case_score=False):
        values = {
            "complex_image_nrmse": error,
            "normalized_squared_error": error**2,
            "signal_fraction": 1.0,
            "outside_object_energy_fraction": 0.01,
        }
        if case_score:
            values.update(
                magnitude_image_nrmse=0.05,
                normalized_squared_magnitude_error=0.0025,
                phase_error_rad=0.05,
                normalized_squared_phase_error=0.0025,
                min_tissue_signal_fraction=1.0,
                contrast_retention=1.0,
            )
        ci = {key: [value - 1e-6, value + 1e-6] for key, value in values.items()}
        return values, ci

    def ideal(*args):
        values, ci = scores(0.5 if case_pass else 0.05)
        if ideal_unresolved:
            ci["normalized_squared_error"] = [0.0, 0.5]
            ci["complex_image_nrmse"] = [0.0, np.sqrt(0.5)]
        return values, ci

    monkeypatch.setattr(s1, "image_scores", ideal)
    monkeypatch.setattr(
        s1,
        "case_image_scores",
        lambda *args: (*scores(0.1 if case_pass else 0.9, True), {}),
    )
    result = s1.evaluate_s1(path, case, ROOT, tmp_path / "evaluation")
    accepted = case_pass and coverage >= 0.5
    assert result["status"] == ("passed" if accepted else "failed")
    assert result["imaging_success_under_case_physics"] is accepted
    assert result["numerically_converged"]
    if coverage < 0.5:
        assert (
            result["case_image_checks"][-1]["name"] == "case_phase_reference_coverage"
        )
        assert result["case_image_checks"][-1]["passed"] is False
    assert result["ideal_imaging_success"] is (
        None if ideal_unresolved else not case_pass
    )
    assert result["image_acceptance_basis"] == "case_physics"
    assert all(check["name"].startswith("case_") for check in result["image_checks"])


def test_contract_is_public_versioned_and_validated():
    contract = Catalog(ROOT).get("evaluators", "s1").model_dump(mode="json")
    assert contract["version"] == 3
    assert contract["calibrated"] is True
    assert set(contract["case_image_criteria"]) == {"static"}
    assert (
        contract["case_image_criteria"]["static"]
        == LEGACY_EVALUATOR["case_image_criteria"]["P2"]
    )
    brief = evaluation_summary(contract, "static")
    assert "ideal-control imaging is diagnostic only" in brief
    assert "wrapped phase RMS" in brief
    assert "never submitted ADC phase" in brief
    for patch in (
        {"case_image_criteria": None},
        {"case_image_criteria": {"P2": contract["case_image_criteria"]["static"]}},
        {"version": 1},
    ):
        with pytest.raises(ValidationError):
            Evaluation.model_validate({**contract, **patch})


@pytest.mark.parametrize("profile", ["b0_offset"])
def test_koma_individual_responses_sum_to_full_signal(tmp_path, profile):
    case = resolve_legacy_case(Catalog(ROOT), "gre_t1w", physics=profile)
    case["task"]["sequence"]["matrix"] = [8, 8]
    path = tmp_path / "sequence.seq"
    controls.make_sequence(case, preparations=0).write(str(path))
    acq = inspect_sequence(path, case)
    spins = Spins(
        np.array([[0.0, 0, 0], [0.02, 0.01, 0], [-0.03, 0, 0]]),
        np.zeros(3, dtype=np.int64),
        ["diagnostic"],
        np.array([[1.0, 0.8, 0.08]]),
        1 / 3,
        (3,),
        "test",
    )
    individual = simulate(acq, spins, case["physics"]["effects"], individual=True)
    summed = simulate(acq, spins, case["physics"]["effects"])
    assert individual.shape == (len(acq.adc), 3)
    assert np.allclose(individual.sum(axis=1), summed, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("fam", ["gre_t1w", "spin_echo_t2w", "bssfp"])
def test_case_phase_reference_ignores_submitted_receiver_phase(tmp_path, fam):
    case = resolve_legacy_case(Catalog(ROOT), fam, physics="b0_offset")
    case["task"]["sequence"]["matrix"] = [8, 8]
    path = tmp_path / "sequence.seq"
    controls.make_sequence(case, preparations=0).write(str(path))
    acq = inspect_sequence(path, case)
    masks = material_masks(phantom(case, ROOT, 4), case["task"]["sequence"])
    reference = np.logical_or.reduce(list(masks.values())).astype(complex)
    original, original_valid, info = phase_reference(acq, case, ROOT, reference)
    acq.adc_phase = acq.adc_phase + np.pi / 2
    changed, changed_valid, changed_info = phase_reference(acq, case, ROOT, reference)
    assert info["converged"] and changed_info["converged"]
    assert np.array_equal(original_valid, changed_valid)
    assert np.allclose(original, changed, atol=1e-12, rtol=0)
