"""Physical controls, adversarial submissions, and every registered object family."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.linalg import expm

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.s1_image import canonical_response, nrmse, reference_signal
from mriseqbench.evaluation.s1_koma import simulate
from mriseqbench.evaluation.s1_phantom import Spins, environment, phantom
from mriseqbench.evaluation.s1_sequence import inspect_sequence
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import legacy_task

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s1_controls", ROOT / "tests/fixtures/s1_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)

# Historical analytic controls exercise the original decision policy independently
# of the unqualified, scanner-specific cardiac benchmark.
LEGACY_EVALUATOR = Evaluation.model_validate(
    load_yaml(ROOT / "tests/fixtures/legacy_evaluators/s1.yaml")
).model_dump(mode="json")


def case_for(
    fam="bssfp", obj=None, physics="ideal", research=False, hardware=None, matrix=8
):
    cat = Catalog(ROOT)
    task = legacy_task(cat, fam + ("_research" if research else ""))
    obj = obj or (
        "volume_ellipsoids@1"
        if len(task.sequence.matrix) == 3
        else "nested_ellipses@1"
        if fam == "rich_geometry_gre"
        else "tissue_discs@1"
    )
    case = cat.resolve(
        task,
        cat.get("physics", physics),
        obj,
        hardware_ref=hardware,
    )
    case["evaluation"] = deepcopy(LEGACY_EVALUATOR)
    case["image_policy"] = case["object_level"]
    case["task"]["evaluation"] = f"s1@{case['evaluation']['version']}"
    case["task"]["sequence"]["matrix"] = [matrix] * len(
        case["task"]["sequence"]["matrix"]
    )
    if matrix < 32:
        case["evaluation"]["spatial_samples"] = 4096
        case["evaluation"]["spatial_max_samples"] = 16384
    return case


def submit(case, directory, mutation=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sequence.seq"
    preparations = 256 if case["physics"]["id"] == "rigid_steps" else 128
    controls.make_sequence(case, mutation, preparations=preparations).write(str(path))
    return path


OBJECT_CASES = [
    (f, o)
    for f in ("gre_t1w", "gre_t2star", "spin_echo_t2w", "bssfp", "rich_geometry_gre")
    for o in ("uniform_disc", "tissue_discs", "nested_ellipses")
] + [("volume_gre", o) for o in ("uniform_sphere", "volume_ellipsoids")]


@pytest.mark.parametrize("fam,obj", OBJECT_CASES)
@pytest.mark.parametrize("research", [False, True])
def test_every_s1_family_and_registered_phantom_has_real_images(
    tmp_path, fam, obj, research
):
    case = case_for(fam, obj, research=research)
    path = submit(case, tmp_path)
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    # Reduced 8x8 controls establish coverage, not guaranteed image quality:
    # long-lived tissue coherences can legitimately fail the fixed image bounds.
    assert result["status"] in {"passed", "failed"}, json.dumps(result, indent=2)
    assert result["sequence_valid"] is True
    assert isinstance(result["ideal_imaging_success"], bool)
    assert (
        result["ideal_imaging_success"] is result["imaging_success_under_case_physics"]
    )
    assert "reason" not in result, result
    assert result["metrics"]  # Physical replay, not a successful preflight proxy.
    assert all(np.isfinite(v) for v in result["metrics"].values())
    artifacts = tmp_path / "evaluation/s1"
    images = np.load(artifacts / "images.npz")
    assert images["physical"].shape == tuple(case["task"]["sequence"]["matrix"])
    assert images["ideal"].shape == images["reference"].shape
    assert np.linalg.norm(images["physical_raw"]) > 0
    assert result["image_verdict"] in {"passed", "failed"}
    provenance = json.loads((artifacts / "provenance.json").read_text())
    assert provenance["phantom"] == obj + "@1"
    assert provenance["calibrated"] is True
    assert len(provenance["submission_sha256"]) == 64


@pytest.mark.parametrize(
    "profile",
    [
        "b0_offset",
        "b0_smooth",
        "b0_localized",
        "b0_drift",
        "rigid_motion",
        "rigid_steps",
        "deformation",
        "dynamic_tissue",
        "fields_and_motion",
    ],
)
def test_all_registered_2d_physics_are_replayed(tmp_path, profile):
    case = case_for(physics=profile)
    path = submit(case, tmp_path)
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["status"] in {"passed", "failed"}, json.dumps(result, indent=2)
    assert result["sequence_valid"] is True
    assert result["effects"] == sorted({e["kind"] for e in case["physics"]["effects"]})
    images = np.load(tmp_path / "evaluation/s1/images.npz")
    assert nrmse(images["physical_raw"], images["ideal_raw"]) > 1e-5
    assert result["metrics"]["temporal_nrmse"] < 0.01


@pytest.mark.parametrize(
    "profile", ["b0_smooth_3d", "rigid_motion_3d", "fields_and_motion_3d"]
)
@pytest.mark.parametrize("obj", ["uniform_sphere", "volume_ellipsoids"])
def test_registered_3d_physics_and_objects_are_replayed(tmp_path, profile, obj):
    case = case_for("volume_gre", obj, physics=profile)
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result["status"] in {"passed", "failed"}, json.dumps(result, indent=2)
    assert result["sequence_valid"] is True
    assert result["numerically_converged"] is True
    # The diagnostic ideal replay can remain unresolved after the required
    # case-image decision has already settled; it must not force more sampling.
    assert result["ideal_imaging_success"] is None or isinstance(
        result["ideal_imaging_success"], bool
    )
    if result["ideal_imaging_success"] is None:
        assert result["decision_basis"]["ideal"] == "unresolved"
    assert isinstance(result["imaging_success_under_case_physics"], bool)
    assert result["metrics"], result
    assert result["metrics"]["case_vs_ideal_image_nrmse"] > 1e-5
    assert result["metrics"]["temporal_nrmse"] < 0.01


@pytest.mark.parametrize(
    "fam,mutation,failed_check",
    [
        ("gre_t1w", "missing_line", "matrix"),
        ("gre_t1w", "repeated_lines", "matrix"),
        ("volume_gre", "flat_volume", "matrix"),
        ("bssfp", "unbalanced", "bssfp"),
        ("spin_echo_t2w", "no_refocusing", "spin_echo"),
        ("spin_echo_t2w", "lying_refocusing_label", "spin_echo"),
        ("gre_t1w", "tiny_rf", "gre"),
        ("gre_t1w", "no_spoiling", "spoiled_gre"),
        ("gre_t1w", "echo_outside_adc", "echo_in_adc"),
        ("gre_t1w", "wrong_echo_time", "te_ms"),
        ("spin_echo_t2w", "wrong_echo_time", "te_ms"),
        ("spin_echo_t2w", "refocus_late", "spin_echo"),
        ("bssfp", "truncated_final_tr", "tr_ms"),
    ],
)
def test_broken_sequences_fail_for_the_physical_reason(
    tmp_path, fam, mutation, failed_check
):
    case = case_for(fam)
    result = evaluate(
        submit(case, tmp_path, mutation), case, ROOT, tmp_path / "evaluation"
    )
    assert result["status"] == "failed", result
    assert any(
        c["name"] == failed_check and not c["passed"] for c in result["checks"]
    ), result


@pytest.mark.parametrize(
    "hardware", ["standard", "low_field", "standard_1_5t", "high_performance"]
)
def test_built_in_judge_uses_selected_hardware(tmp_path, hardware):
    case = case_for(hardware=hardware)
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result["status"] == "passed", json.dumps(result, indent=2)
    assert all(c["passed"] for c in result["checks"])


def test_final_verdicts_are_booleans_and_omit_provisional_fields(tmp_path):
    case = case_for()
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result["sequence_valid"]
    assert result["status"] == "passed"
    assert result["numerically_converged"]
    assert result["ideal_imaging_success"] is True
    assert result["imaging_success_under_case_physics"] is True
    assert result["image_verdict"] == "passed"
    assert not any("provisional" in k for k in result)


def test_excess_long_t2_coherence_fails_image_with_valid_sequence(tmp_path):
    case = case_for("rich_geometry_gre", "tissue_discs", matrix=32)
    result = evaluate(
        submit(case, tmp_path, "short_tr_coherence"),
        case,
        ROOT,
        tmp_path / "evaluation",
    )
    assert result["sequence_valid"] is True
    assert result["status"] == "failed"
    assert result["ideal_imaging_success"] is False
    assert result["image_verdict"] == "failed"
    assert result["numerical_intervals"]["ideal"]["complex_image_nrmse"][0] > 0.2


def test_fourier_oracle_matches_direct_sum(tmp_path):
    case = case_for("gre_t2star")
    acq = inspect_sequence(submit(case, tmp_path), case)
    spins = phantom(case, ROOT, 2)
    response = canonical_response(acq, spins.properties, "gre_t2star")
    actual = reference_signal(acq, spins, case["task"]["sequence"], response)
    expected = np.array(
        [
            np.sum(
                np.exp(-2j * np.pi * (spins.positions @ k))
                * response[spins.labels, i]
                * spins.weight
            )
            for i, k in enumerate(acq.k)
        ]
    )
    assert (
        nrmse(actual, expected) < 1e-4
    )  # Serialized gradients introduce tiny lattice residuals.


@pytest.mark.parametrize("integration_method", ["bloch", "magnus4"])
def test_koma_bloch_matches_independent_matrix_exponential(
    tmp_path, integration_method
):
    from pypulseq import Opts, Sequence, make_adc, make_block_pulse, make_delay

    # A real Pulseq acquisition, KomaMRI dynamics, independent affine ODE oracle.
    duration, rf_hz, offset = 0.001, 220.0, 73.0
    properties = np.array([[0.7, 0.8, 0.08]])
    system = Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    seq.add_block(
        make_block_pulse(
            2 * np.pi * rf_hz * duration,
            duration=duration,
            system=system,
            use="excitation",
        )
    )
    seq.add_block(make_adc(1, dwell=2e-6, system=system), make_delay(10e-6))
    path = tmp_path / "ode.seq"
    seq.write(str(path))
    case = case_for()
    acq = inspect_sequence(path, case)
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, dtype=np.int64),
        ["test"],
        properties,
        1.0,
        (1,),
        "test",
    )
    signal = simulate(
        acq,
        spins,
        frequency_offsets=np.array([offset]),
        rf_step_s=1e-6,
        integration_method=integration_method,
    )[0]
    wx, wz = 2 * np.pi * rf_hz, 2 * np.pi * offset
    ode = np.array(
        [
            [-1 / 0.08, wz, 0, 0],
            [-wz, -1 / 0.08, wx, 0],
            [0, -wx, -1 / 0.8, 0.7 / 0.8],
            [0, 0, 0, 0.0],
        ]
    )
    expected = expm(ode * duration) @ np.array([0, 0, 0.7, 1.0])
    xy = (expected[0] + 1j * expected[1]) * np.exp(
        (-1 / 0.08 - 1j * wz) * (acq.adc[0] - duration)
    )
    assert abs(signal - xy) < 1e-5


def test_motion_fields_and_tissue_equilibrium_use_exact_midpoint_coordinates():
    effects = case_for(physics="fields_and_motion")["physics"]["effects"]
    transforms, shifts, b0, gradient, _gaussian, pd = environment(
        np.array([0.25]), effects, ["wm", "gm"]
    )
    point = np.array([0.02, 0.01, 0.0])
    moved = transforms[0] @ point + shifts[0]
    assert shifts[0, 0] == pytest.approx(0.003)
    assert gradient @ moved + b0[0] == pytest.approx(moved[0] * 1000)
    tissue = case_for(physics="dynamic_tissue")["physics"]["effects"]
    *_, pd = environment(np.array([0.5]), tissue, ["wm", "gm"])
    assert pd[0].tolist() == pytest.approx([1.1, 1.0])


def test_native_phantom_sampling_stays_available_for_provenance():
    for fam, obj in OBJECT_CASES:
        case = case_for(fam, obj)
        spins = phantom(case, ROOT, 2, gauss_legendre=False)
        assert {spins.materials[i] for i in np.unique(spins.labels)} <= set(
            case["object"]["definition"]["materials"]
        )
        assert np.all(spins.properties[:, 1] >= spins.properties[:, 2])
        repeat = phantom(case, ROOT, 2, gauss_legendre=False)
        assert np.array_equal(spins.positions, repeat.positions)


def test_research_timing_is_chosen_by_the_submission(tmp_path):
    case = case_for("gre_t1w", research=True)
    path = submit(case, tmp_path)
    acq = inspect_sequence(path, case)
    assert all(
        c["name"] not in {"te_ms", "tr_ms", "flip_angle_deg"} for c in acq.checks
    )
    wrong = deepcopy(case)
    wrong["task"]["prompt_mode"] = "detailed"
    wrong["task"]["sequence"]["te_ms"] = {"target": 40.0, "tolerance": 1.0, "max": None}
    inspected = inspect_sequence(path, wrong)
    assert any(c["name"] == "te_ms" and not c["passed"] for c in inspected.checks)


@pytest.mark.parametrize("profile", ["ideal", "b0_offset"])
def test_calibrated_fixture_requires_convergence_and_separates_robustness(
    tmp_path, profile
):
    case = case_for("bssfp", "uniform_disc", physics=profile, matrix=32)
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result["sequence_valid"] is True, result
    assert result["numerically_converged"] is True, result
    assert result["status"] == ("passed" if profile == "ideal" else "failed"), (
        json.dumps(result, indent=2)
    )
    assert result["imaging_success_under_case_physics"] is (profile == "ideal")


def test_sampling_settings_are_versioned_and_bounded():
    contract = case_for(matrix=32)["evaluation"]
    contract["spatial_samples"] = 65536
    assert Evaluation.model_validate(contract).spatial_samples == 65536
    for count in (1024, 65535, 2097152):
        contract["spatial_samples"] = count
        with pytest.raises(ValidationError):
            Evaluation.model_validate(contract)


@pytest.mark.parametrize("fam", ["bssfp", "gre_t1w"])
def test_wrong_receiver_phase_cannot_define_its_own_reference(tmp_path, fam):
    case = case_for(fam, "uniform_disc")
    result = evaluate(
        submit(case, tmp_path, "receiver_phase"), case, ROOT, tmp_path / "evaluation"
    )
    assert result["status"] == "failed", result
    assert (
        result.get("ideal_imaging_success") is False
        or result["sequence_valid"] is False
    )


@pytest.mark.parametrize("fam", ["gre_t1w", "spin_echo_t2w", "bssfp"])
def test_koma_echo_peak_matches_waveform_te(tmp_path, fam):
    case = case_for(fam)
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    echo = next(c for c in result["checks"] if c["name"] == "echo_formation")
    assert echo["passed"], result
    assert echo["detail"]["engine"] == "KomaMRICore"
    assert echo["detail"]["peak_time_error_s"] <= echo["detail"]["tolerance_s"]


def test_analytic_shape_encoding_matches_independent_dense_quadrature(tmp_path):
    from mriseqbench.evaluation.s1_sampling import analytic_reference

    for fam, obj in OBJECT_CASES:
        case = case_for(fam, obj)
        path = submit(case, tmp_path / (fam + obj))
        acq = inspect_sequence(path, case)
        spins = phantom(case, ROOT, 12)
        response = np.ones((len(spins.materials), len(acq.adc)), dtype=complex)
        exact = analytic_reference(acq, case, ROOT, response)
        sampled = reference_signal(acq, spins, case["task"]["sequence"], response)
        assert nrmse(sampled, exact) < 0.02, (fam, obj)


def test_quadratic_scores_remove_sampling_noise_without_erasing_real_error():
    from mriseqbench.evaluation.s1_sampling import image_scores

    rng = np.random.default_rng(123)
    reference = np.ones((64, 64), dtype=complex)
    support = np.ones(reference.shape, dtype=bool)
    for error in (0.0, 0.1, 0.4):
        noise = 0.3 * (
            rng.normal(size=(16, 64, 64)) + 1j * rng.normal(size=(16, 64, 64))
        )
        images = reference + error + noise
        metrics, ci = image_scores(images, reference, support)
        assert abs(metrics["complex_image_nrmse"] - error) < 0.02
        assert ci["complex_image_nrmse"][0] <= error <= ci["complex_image_nrmse"][1]
        assert nrmse(images[0], reference) > error + 0.1


@pytest.mark.parametrize("matrix", [(8, 8), (8, 8, 8)])
def test_bart_matches_fixed_direct_fourier_convention(tmp_path, matrix):
    from types import SimpleNamespace

    from mriseqbench.evaluation.s1_koma import reconstruct

    indices = np.indices(matrix).reshape(len(matrix), -1).T
    acq = SimpleNamespace(indices=indices, offset=np.zeros(len(matrix)))
    rng = np.random.default_rng(123)
    raw = rng.normal(size=len(indices)) + 1j * rng.normal(size=len(indices))
    q = indices - np.asarray(matrix) // 2
    grid = (raw * np.exp(1j * np.pi * np.sum(q / np.asarray(matrix), axis=1))).reshape(
        matrix
    )
    expected = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(grid)))
    assert nrmse(reconstruct(raw, acq, matrix), expected) < 2e-6
