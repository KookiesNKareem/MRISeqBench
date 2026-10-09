import json
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.models import Backend

ROOT = Path(__file__).resolve().parents[2]


def gre_case():
    catalog = Catalog(ROOT)
    return catalog.resolve(
        catalog.get("tasks", "gre_t1w"), catalog.get("physics", "ideal")
    )


def fake_backend(case, **changes):
    # Contract fixtures only; no synthetic metric is a real benchmark result.
    result = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "reconstruction": "cartesian_fft@1",
        "requirements": {
            k: True
            for k in ["spoiled_gre", "cartesian", "matrix", "fov_mm", "te_ms", "tr_ms"]
        },
        "effects": [],
        "metrics": {"wm_gm_ratio": 1.3, "outside_object_energy_fraction": 0.1},
    }
    result.update(changes)
    script = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    return Backend(
        command=[sys.executable, "-c", script, "{result}", json.dumps(result)],
        timeout_s=5,
    )


def test_real_pulseq_smoke(submission, tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.cases(catalog.get("suites", "smoke"))[0]
    result = evaluate(submission, case, ROOT, tmp_path)
    assert result["status"] == "smoke_passed"
    assert all(c["passed"] for c in result["checks"])


def test_no_backend_cannot_pass(submission, tmp_path):
    assert evaluate(submission, gre_case(), ROOT, tmp_path)["status"] == "incomplete"


def test_uncalibrated_cannot_pass(submission, tmp_path):
    case = gre_case()
    assert (
        evaluate(submission, case, ROOT, tmp_path, fake_backend(case))["status"]
        == "uncalibrated"
    )


def test_calibrated_contract_pass_and_fail(submission, tmp_path):
    case = gre_case()
    case["evaluation"]["calibrated"] = True  # test fixture, not a released calibration
    assert (
        evaluate(submission, case, ROOT, tmp_path / "pass", fake_backend(case))[
            "status"
        ]
        == "passed"
    )
    backend = fake_backend(
        case, metrics={"wm_gm_ratio": 0.9, "outside_object_energy_fraction": 0.1}
    )
    assert (
        evaluate(submission, case, ROOT, tmp_path / "fail", backend)["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"case_id": "wrong"},
        {"reconstruction": "wrong"},
        {"requirements": {}},
        {"requirements": {"spoiled_gre": "true"}},
        {"metrics": {"wm_gm_ratio": 1.3}},
        {
            "metrics": {
                "wm_gm_ratio": float("nan"),
                "outside_object_energy_fraction": 0.1,
            }
        },
        {"effects": ["motion"]},
    ],
)
def test_backend_cannot_silently_omit_contract(submission, tmp_path, changes):
    case = gre_case()
    assert (
        evaluate(submission, case, ROOT, tmp_path, fake_backend(case, **changes))[
            "status"
        ]
        == "evaluator_error"
    )


def test_invalid_and_missing_submissions(tmp_path):
    path = tmp_path / "bad.seq"
    path.write_text("invalid sequence")
    assert evaluate(path, gre_case(), ROOT, tmp_path / "bad")["status"] == "failed"
    assert (
        evaluate(tmp_path / "missing.seq", gre_case(), ROOT, tmp_path / "missing")[
            "status"
        ]
        == "failed"
    )


def test_hardware_failure(submission, tmp_path):
    case = gre_case()
    case["hardware"]["max_b1_uT"] = 0.01
    result = evaluate(submission, case, ROOT, tmp_path)
    assert result["status"] == "failed"
    assert any(c["name"] == "peak_b1" and not c["passed"] for c in result["checks"])


def test_raster_header_cannot_relax_hardware(submission, tmp_path):
    path = tmp_path / "modified.seq"
    path.write_text(
        submission.read_text().replace("AdcRasterTime 1e-07", "AdcRasterTime 1e-06")
    )
    assert path.read_text() != submission.read_text()
    result = evaluate(path, gre_case(), ROOT, tmp_path / "evaluation")
    assert result["status"] == "failed"
