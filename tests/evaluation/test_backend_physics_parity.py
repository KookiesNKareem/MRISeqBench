"""External protocol fixtures; synthetic metrics are not physics evidence."""

from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from tests.evaluation.test_s4_backend import backend_fixture, contract_fixture

ROOT = Path(__file__).resolve().parents[2]


def cardiac_fixture(family):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", family))
    # A protocol-local calibrated flag makes successful acceptance observable.
    # It does not modify a catalog, evaluator policy or qualification record.
    case["evaluation"]["calibrated"] = True
    required = set(case["task"]["sequence"]["capabilities"]) | {
        "matrix",
        "fov_mm",
        "numerical_convergence",
    }
    required |= {
        name
        for name in (
            "flip_angle_deg",
            "te_ms",
            "tr_ms",
            "readout",
            "sampling",
            "temporal",
            "application",
        )
        if case["task"]["sequence"][name] is not None
    }
    metrics = {
        "case_complex_image_nrmse": 0.1,
        "case_magnitude_image_nrmse": 0.1,
        "case_phase_error_rad": 0.1,
        "case_min_tissue_signal_fraction": 0.8,
        "case_contrast_retention": 0.8,
        "case_outside_object_energy_fraction": 0.1,
        "case_phase_reference_coverage": 0.5,
    }
    if family == "reduced_fov":
        required.add("spatial_selection")
        metrics.update(selection_leakage_ratio=0.1, diagnostic_signal_fraction=0.01)
    result = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "reconstruction": case["evaluation"]["reconstruction"],
        "reconstruction_provenance": {
            "engine": "BART",
            "version": "protocol-fixture-only",
            "binary_sha256": "b" * 64,
            "commands": [{"argv": ["fft", "-i", "3", "input", "output"], "calls": 1}],
        },
        "requirements": dict.fromkeys(required, True),
        "effects": ["b0"],
        "metrics": metrics,
    }
    return case, result


@pytest.mark.parametrize("family", ["gre_t1w", "radial_readout", "partial_fourier"])
@pytest.mark.parametrize("coverage", [None, 0.499, 0.5])
def test_current_cardiac_backend_phase_reference_coverage(
    submission, tmp_path, family, coverage
):
    case, result = cardiac_fixture(family)
    if coverage is None:
        result["metrics"].pop("case_phase_reference_coverage")
        with pytest.raises(
            RuntimeError, match="backend omitted metric: case_phase_reference_coverage"
        ):
            evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
        return
    result["metrics"]["case_phase_reference_coverage"] = coverage
    report = evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
    assert report["status"] == ("passed" if coverage == 0.5 else "failed")


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_external_s4_requires_each_frame_phase_reference_coverage(
    submission, tmp_path, frame
):
    case, result = contract_fixture()
    metric = f"frame_{frame}_case_phase_reference_coverage"
    result["metrics"].pop(metric)
    with pytest.raises(RuntimeError, match=f"backend omitted metric: {metric}"):
        evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))


@pytest.mark.parametrize("coverage", [0.499, 0.5])
def test_external_s4_phase_reference_coverage_cannot_hide_a_bad_frame(
    submission, tmp_path, coverage
):
    case, result = contract_fixture()
    case["evaluation"]["calibrated"] = True
    result["metrics"]["frame_1_case_phase_reference_coverage"] = coverage
    report = evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
    assert report["status"] == ("passed" if coverage == 0.5 else "failed")
    assert report["imaging_success_under_case_physics"] is (coverage == 0.5)


@pytest.mark.parametrize(
    "missing",
    ["spatial_selection", "selection_leakage_ratio", "diagnostic_signal_fraction"],
)
def test_external_reduced_fov_cannot_omit_physical_selection(
    submission, tmp_path, missing
):
    case, result = cardiac_fixture("reduced_fov")
    target = "requirements" if missing == "spatial_selection" else "metrics"
    result[target].pop(missing)
    with pytest.raises(RuntimeError, match="backend omitted"):
        evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))


@pytest.mark.parametrize(
    "name,value",
    [
        ("spatial_selection", False),
        ("selection_leakage_ratio", 0.100001),
        ("diagnostic_signal_fraction", 0.009999),
    ],
)
def test_external_reduced_fov_rejects_bad_selection_even_with_good_images(
    submission, tmp_path, name, value
):
    case, result = cardiac_fixture("reduced_fov")
    target = "requirements" if name == "spatial_selection" else "metrics"
    result[target][name] = value
    report = evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
    assert report["status"] == "failed"
    assert (
        next(check for check in report["checks"] if check["name"] == name)["passed"]
        is False
    )


def test_external_reduced_fov_accepts_exact_native_selection_boundaries(
    submission, tmp_path
):
    case, result = cardiac_fixture("reduced_fov")
    assert (
        evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))["status"]
        == "passed"
    )


@pytest.mark.parametrize("family", ["gre_t1w", "radial_readout", "partial_fourier"])
@pytest.mark.parametrize("converged", [None, False, True])
def test_current_cardiac_backend_requires_resolved_numerics(
    submission, tmp_path, family, converged
):
    case, result = cardiac_fixture(family)
    if converged is None:
        result["requirements"].pop("numerical_convergence")
        with pytest.raises(
            RuntimeError, match="backend omitted requirements.*numerical_convergence"
        ):
            evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
        return
    result["requirements"]["numerical_convergence"] = converged
    report = evaluate(submission, case, ROOT, tmp_path, backend_fixture(result))
    assert report["status"] == ("passed" if converged else "numerically_unresolved")
    assert report["numerically_converged"] is converged
    if not converged:
        assert report["imaging_success_under_case_physics"] is None
