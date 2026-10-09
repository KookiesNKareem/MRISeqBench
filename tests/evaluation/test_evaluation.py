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


def test_selected_hardware_enforces_real_gradient_limit(tmp_path):
    from pypulseq import Opts, Sequence, make_adc, make_block_pulse, make_trapezoid

    from mriseqbench.evaluation.preflight import preflight

    catalog = Catalog(ROOT)
    system = Opts(
        max_grad=80,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )
    seq = Sequence(system)
    seq.add_block(
        make_block_pulse(
            0.5, duration=1e-3, delay=100e-6, system=system, use="excitation"
        )
    )
    # 30 mT/m at 30 T/m/s clears the slew limit of both profiles,
    # but exceeds Free.Max's 26 mT/m amplitude limit.
    grad = make_trapezoid(
        "x",
        amplitude=0.03 * system.gamma,
        rise_time=1e-3,
        flat_time=1e-3,
        system=system,
    )
    adc = make_adc(32, duration=960e-6, delay=1e-3, system=system)
    seq.add_block(grad, adc)
    path = tmp_path / "hardware.seq"
    seq.write(str(path))
    for hardware, passes in [("high_performance@1", True), ("low_field@1", False)]:
        case = catalog.resolve(
            catalog.get("tasks", "gre_t1w"),
            catalog.get("physics", "ideal"),
            hardware_ref=hardware,
        )
        checks = preflight(path, case)
        assert next(c for c in checks if c["name"] == "gradient_x")["passed"] is passes
        assert next(c for c in checks if c["name"] == "slew_x")["passed"]
        assert all(c["passed"] for c in checks) is passes


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
