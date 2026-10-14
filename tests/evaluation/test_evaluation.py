import json
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.models import Backend
from tests.fixtures.legacy_catalog import resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]


def gre_case():
    catalog = Catalog(ROOT)
    return resolve_legacy_case(catalog, "gre_t1w")


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
        "reconstruction": case["evaluation"]["reconstruction"],
        "requirements": {
            k: True
            for k in ["spoiled_gre", "cartesian", "matrix", "fov_mm", "te_ms", "tr_ms"]
        },
        "effects": [],
        "metrics": {
            "ideal_complex_image_nrmse": 0.05,
            "ideal_signal_fraction": 1.0,
            "ideal_outside_object_energy_fraction": 0.1,
        },
    }
    if case["evaluation"]["reconstruction"].startswith("bart_"):
        result["reconstruction_provenance"] = {
            "engine": "BART",
            "version": "v1.0.00",
            "binary_sha256": "a" * 64,
            "source_commit": None,
            "commands": [{"argv": ["fft", "-i", "3", "input", "output"], "calls": 1}],
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


@pytest.mark.parametrize("provenance", [None, {}, {"engine": "MRIReco"}])
def test_current_cardiac_backend_requires_bart_evidence(
    submission, tmp_path, provenance
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    backend = fake_backend(case, reconstruction_provenance=provenance)
    with pytest.raises(RuntimeError, match="BART reconstruction provenance"):
        evaluate(submission, case, ROOT, tmp_path, backend)


def test_reduced_fov_public_brief_requires_a_spin_echo():
    from mriseqbench.harness.prepare import task_brief

    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "reduced_fov"))
    assert task_brief(case)["sequence"]["capabilities"] == [
        "spin_echo",
        "cartesian",
        "reduced_fov",
    ]


@pytest.mark.parametrize("spin_echo", [None, False, True])
def test_reduced_fov_external_backend_requires_spin_echo(
    submission, tmp_path, spin_echo
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "reduced_fov"))
    requirements = {
        name: True
        for name in (
            "cartesian",
            "reduced_fov",
            "matrix",
            "fov_mm",
            "sampling",
            "spatial_selection",
            "numerical_convergence",
        )
    }
    if spin_echo is None:
        requirements["spoiled_gre"] = True
    else:
        requirements["spin_echo"] = spin_echo
    backend = fake_backend(
        case,
        requirements=requirements,
        effects=sorted({effect["kind"] for effect in case["physics"]["effects"]}),
        metrics={
            "case_complex_image_nrmse": 0.1,
            "case_magnitude_image_nrmse": 0.1,
            "case_phase_error_rad": 0.1,
            "case_min_tissue_signal_fraction": 0.9,
            "case_contrast_retention": 0.9,
            "case_outside_object_energy_fraction": 0.1,
            "case_phase_reference_coverage": 1.0,
            "selection_leakage_ratio": 0.05,
            "diagnostic_signal_fraction": 0.1,
        },
    )
    if spin_echo is None:
        with pytest.raises(RuntimeError, match="omitted requirements.*spin_echo"):
            evaluate(submission, case, ROOT, tmp_path, backend)
    else:
        result = evaluate(submission, case, ROOT, tmp_path, backend)
        expected = "passed" if case["evaluation"]["calibrated"] else "uncalibrated"
        assert result["status"] == (expected if spin_echo else "failed")


def test_native_s4_rejects_submission_without_ecg_trigger(submission, tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "adiabatic_inversion"))
    result = evaluate(submission, case, ROOT, tmp_path)
    assert result["status"] == "failed"
    assert any(
        check["name"] == "ecg_trigger_events" and not check["passed"]
        for check in result["checks"]
    )


def test_external_s3_backend_cannot_replace_case_metrics_with_ideal_metrics(
    submission, tmp_path
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "partial_fourier"))
    requirements = {
        name: True
        for name in [
            *case["task"]["sequence"]["capabilities"],
            "matrix",
            "fov_mm",
            "sampling",
            "numerical_convergence",
        ]
    }
    backend = fake_backend(
        case,
        reconstruction=case["evaluation"]["reconstruction"],
        requirements=requirements,
        effects=["b0"],
    )
    with pytest.raises(
        RuntimeError, match="backend omitted metric: case_complex_image_nrmse"
    ):
        evaluate(submission, case, ROOT, tmp_path, backend)


def test_uncalibrated_cannot_pass(submission, tmp_path):
    case = gre_case()
    case["evaluation"]["calibrated"] = False
    assert (
        evaluate(submission, case, ROOT, tmp_path, fake_backend(case))["status"]
        == "uncalibrated"
    )


def test_calibrated_contract_pass_and_fail(submission, tmp_path):
    case = gre_case()
    case["evaluation"]["calibrated"] = True
    assert (
        evaluate(submission, case, ROOT, tmp_path / "pass", fake_backend(case))[
            "status"
        ]
        == "passed"
    )
    backend = fake_backend(
        case,
        metrics={
            "ideal_complex_image_nrmse": 0.9,
            "ideal_signal_fraction": 1.0,
            "ideal_outside_object_energy_fraction": 0.1,
        },
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
    with pytest.raises(RuntimeError, match="Physical backend returned invalid output"):
        evaluate(submission, case, ROOT, tmp_path, fake_backend(case, **changes))


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


@pytest.mark.parametrize("level", ["P2"])
@pytest.mark.parametrize("case_error", [0.1, 0.9, None])
def test_external_s1_backend_must_supply_case_metrics(
    submission, tmp_path, level, case_error
):
    catalog = Catalog(ROOT)
    case = resolve_legacy_case(catalog, "gre_t1w", physics="b0_offset")
    metrics = {"ideal_complex_image_nrmse": 0.9}
    if case_error is not None:
        metrics.update(
            case_complex_image_nrmse=case_error,
            case_magnitude_image_nrmse=0.1,
            case_phase_error_rad=0.1,
            case_min_tissue_signal_fraction=0.8,
            case_contrast_retention=0.8,
            case_outside_object_energy_fraction=0.1,
        )
    backend = fake_backend(
        case,
        metrics=metrics,
        effects=sorted({e["kind"] for e in case["physics"]["effects"]}),
    )
    if case_error is None:
        with pytest.raises(RuntimeError, match="backend omitted metric"):
            evaluate(submission, case, ROOT, tmp_path, backend)
    else:
        result = evaluate(submission, case, ROOT, tmp_path, backend)
        assert result["status"] == ("passed" if case_error == 0.1 else "failed")


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
