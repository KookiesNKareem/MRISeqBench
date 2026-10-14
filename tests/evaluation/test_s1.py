"""Physical controls, adversarial submissions, and independent numerical oracles."""

import importlib.util
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
from mriseqbench.evaluation.s1_phantom import Spins, phantom
from mriseqbench.evaluation.s1_sequence import inspect_sequence
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import legacy_task

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s1_controls", ROOT / "tests/fixtures/s1_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)

# Small analytic controls isolate numerical and decision-policy regressions.
LEGACY_EVALUATOR = Evaluation.model_validate(
    load_yaml(ROOT / "tests/fixtures/legacy_evaluators/s1.yaml")
).model_dump(mode="json")


def case_for(fam="bssfp", obj=None, physics="ideal", hardware=None, matrix=8):
    cat = Catalog(ROOT)
    task = legacy_task(cat, fam)
    obj = obj or (
        "volume_ellipsoids@1" if len(task.sequence.matrix) == 3 else "tissue_discs@1"
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
    preparations = 128
    controls.make_sequence(case, mutation, preparations=preparations).write(str(path))
    return path


@pytest.mark.parametrize(
    "fam,mutation,failed_check",
    [
        ("gre_t1w", "missing_line", "matrix"),
        ("gre_t1w", "repeated_lines", "matrix"),
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


def test_native_phantom_sampling_stays_available_for_provenance():
    for fam, obj in [
        ("gre_t2star", "nested_ellipses"),
        ("elliptical_kspace", "volume_ellipsoids"),
    ]:
        case = case_for(fam, obj)
        spins = phantom(case, ROOT, 2, gauss_legendre=False)
        assert {spins.materials[i] for i in np.unique(spins.labels)} <= set(
            case["object"]["definition"]["materials"]
        )
        assert np.all(spins.properties[:, 1] >= spins.properties[:, 2])
        repeat = phantom(case, ROOT, 2, gauss_legendre=False)
        assert np.array_equal(spins.positions, repeat.positions)


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
    from mriseqbench.evaluation.s1_probes import echo_formation_probe

    acq = inspect_sequence(submit(case, tmp_path), case)
    echo, _ = echo_formation_probe(acq, case, np.array([[1.0, 0.8, 0.08]]))
    assert echo["passed"], echo
    assert echo["detail"]["engine"] == "KomaMRICore"
    assert echo["detail"]["peak_time_error_s"] <= echo["detail"]["tolerance_s"]


def test_analytic_shape_encoding_matches_independent_dense_quadrature(tmp_path):
    from mriseqbench.evaluation.s1_sampling import analytic_reference

    for fam, obj in [
        ("gre_t2star", "nested_ellipses"),
        ("elliptical_kspace", "volume_ellipsoids"),
    ]:
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
