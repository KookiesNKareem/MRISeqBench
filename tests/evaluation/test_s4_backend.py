"""External protocol fixtures only; synthetic metrics are not physics evidence."""

import json
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.evaluate import s4_backend_contract
from mriseqbench.harness.prepare import evaluation_overview, evaluation_summary
from mriseqbench.models import Backend

ROOT = Path(__file__).resolve().parents[2]


def contract_fixture():
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "t1_mapping"))
    required, bounds = s4_backend_contract(case)
    required |= set(case["task"]["sequence"]["capabilities"]) | {
        "matrix",
        "fov_mm",
        "temporal",
        "application",
    }
    result = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "reconstruction": "bart_cartesian_fft@1",
        "reconstruction_provenance": {
            "engine": "BART",
            "version": "v1.0.00",
            "binary_sha256": "b" * 64,
            "source_commit": None,
            "commands": [{"argv": ["fft", "-i", "3", "input", "output"], "calls": 1}],
        },
        "requirements": {name: True for name in required},
        "effects": ["b0"],
        "metrics": {bound["name"]: max(bound["min"] or 0, 0.001) for bound in bounds},
    }
    return case, result


def backend_fixture(result):
    return Backend(
        command=[
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            "{result}",
            json.dumps(result),
        ],
        timeout_s=5,
    )


@pytest.mark.parametrize(
    "missing",
    [
        "frame_1_case_magnitude_image_nrmse",
        "frame_2_preparation_effect_fraction",
        "frame_0_min_preparation_rf_inversion_efficiency",
        "myocardial_t1_relative_error",
    ],
)
def test_external_s4_cannot_omit_frame_or_application_metrics(
    submission, tmp_path, missing
):
    case, result = contract_fixture()
    result["metrics"].pop(missing)
    with pytest.raises(RuntimeError, match=f"backend omitted metric: {missing}"):
        evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))


def test_external_s4_rejects_one_bad_frame_even_if_other_frames_pass(
    submission, tmp_path
):
    case, result = contract_fixture()
    case["evaluation"]["calibrated"] = True
    result["metrics"]["frame_1_case_magnitude_image_nrmse"] = 0.351
    report = evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
    assert report["status"] == "failed"
    assert report["imaging_success_under_case_physics"] is False


def test_external_s4_requires_numerical_convergence_and_preserves_uncalibrated(
    submission, tmp_path
):
    case, result = contract_fixture()
    case["evaluation"]["calibrated"] = False
    report = evaluate(
        submission, case, ROOT, tmp_path / "uncalibrated", backend_fixture(result)
    )
    assert report["status"] == "uncalibrated"
    assert report["imaging_success_under_case_physics"] is None
    result["requirements"]["numerical_convergence"] = False
    report = evaluate(
        submission, case, ROOT, tmp_path / "unresolved", backend_fixture(result)
    )
    assert report["status"] == "numerically_unresolved"
    assert report["imaging_success_under_case_physics"] is None


def test_external_s4_requires_real_trigger_window_requirement(submission, tmp_path):
    case, result = contract_fixture()
    result["requirements"].pop("cardiac_adc_window")
    with pytest.raises(RuntimeError, match="backend omitted requirements"):
        evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))


def test_s4_public_summary_discloses_application_limits_without_bloating_prompt():
    case, _ = contract_fixture()
    summary = evaluation_summary(case["evaluation"], "static")
    assert "BART Cartesian FFT" in summary
    assert "numerical relative-error precision 0.001" in summary
    assert "relative error <= 0.1" in summary
    assert "Preparation effect >= 0.02" in summary
    assert "[-520,70] Hz" in summary
    assert len(evaluation_overview(case)) < 400
