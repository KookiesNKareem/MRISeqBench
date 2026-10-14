"""S2 waveform controls and independent arbitrary-node Fourier/BART oracles."""

import importlib.util
import json
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import finufft
import numpy as np
import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.preflight import preflight
from mriseqbench.evaluation.s1_image import nrmse
from mriseqbench.evaluation.s1_koma import simulate
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s2 import (
    canonical_response,
    reconstruct,
    reference_index,
    reference_signal,
)
from mriseqbench.evaluation.s2_sequence import FAMILIES, inspect_sequence
from mriseqbench.evaluation.sar import evaluate_sar
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import legacy_task

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s2_controls", ROOT / "tests/fixtures/s2_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)

# Small analytic controls isolate numerical and decision-policy regressions.
LEGACY_EVALUATOR = Evaluation.model_validate(
    load_yaml(ROOT / "tests/fixtures/legacy_evaluators/s2.yaml")
).model_dump(mode="json")


def test_refocusing_only_submission_is_rejected_before_replay(tmp_path, monkeypatch):
    import pypulseq as pp

    case = case_for("radial_readout")
    system = pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=30e-6, adc_dead_time=10e-6)
    seq = pp.Sequence(system)
    # A misleading use label cannot turn a measured 180-degree pulse into an excitation.
    seq.add_block(
        pp.make_block_pulse(
            np.pi, duration=2e-3, delay=100e-6, use="excitation", system=system
        )
    )
    seq.add_block(pp.make_delay(1.0))
    seq.add_block(
        pp.make_adc(32, duration=640e-6, delay=10e-6, system=system),
        pp.make_delay(660e-6),
    )
    path = tmp_path / "only-refocusing.seq"
    seq.write(str(path))
    assert all(c["passed"] for c in preflight(path, case))
    assert all(c["passed"] for c in evaluate_sar(path, case["hardware"]))

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("Missing excitation must be rejected before Bloch replay")

    monkeypatch.setattr("mriseqbench.evaluation.s2.mechanism_probes", unexpected_replay)
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["status"] == "failed"
    assert result["sequence_valid"] is False
    assert result["image_verdict"] == "not_evaluated"
    assert any(
        c["name"] == "excitation_before_adc" and not c["passed"]
        for c in result["checks"]
    )


def case_for(fam, obj=None, physics="ideal", small=False):
    cat = Catalog(ROOT)
    task = legacy_task(cat, fam)
    obj = obj or (
        "volume_ellipsoids@1" if len(task.sequence.matrix) == 3 else "tissue_discs@1"
    )
    case = cat.resolve(
        task,
        cat.get("physics", physics),
        obj,
    )
    case["evaluation"] = deepcopy(LEGACY_EVALUATOR)
    case["image_policy"] = case["object_level"]
    case["task"]["evaluation"] = f"s2@{case['evaluation']['version']}"
    if small:
        case["task"]["sequence"]["matrix"] = [8] * len(
            case["task"]["sequence"]["matrix"]
        )
        read = case["task"]["sequence"].get("readout")
        if read:
            if read.get("spokes"):
                read["spokes"] = 16
            if read.get("rings"):
                read["rings"] = 4
            if read.get("turns"):
                read["turns"] = 2
            if fam in {"tse_t2w", "grase"}:
                read.update(shots=2, echo_train_length=4)
        case["evaluation"].update(
            spatial_samples=4096, spatial_max_samples=4096, spatial_replicates=4
        )
    return case


def submit(case, directory, mutation=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sequence.seq"
    controls.make_sequence(case, mutation).write(str(path))
    return path


def test_centre_selection_uses_earliest_numerically_tied_sample():
    acq = SimpleNamespace(k=np.array([[2.5, 0, 0], [2.499999, 0, 0], [2.5, 0, 0]]))
    assert reference_index(acq, {"matrix": [32, 32], "fov_mm": [200, 200]}) == 0


def test_ideal_s2_stopping_uses_its_own_quality_limits(monkeypatch, tmp_path):
    from mriseqbench.evaluation import s1

    case = case_for("epi_single_shot", "uniform_disc", small=True)
    case["evaluation"]["spatial_max_samples"] = 8192
    seen = []
    original_sampler = s1.sampled_phantom

    def sampler(case, root, samples, replicate):
        seen.append(samples)
        return original_sampler(case, root, samples, replicate)

    def scores(*args):
        # This interval straddles the inherited S1 diagnostic limit of 0.20,
        # but is fully inside S2's 0.30. Precision alone is intentionally unmet.
        return {
            "complex_image_nrmse": 0.20,
            "normalized_squared_error": 0.04,
            "signal_fraction": 1.0,
            "outside_object_energy_fraction": 0.01,
        }, {
            "complex_image_nrmse": [0.19, 0.21],
            "normalized_squared_error": [0.035, 0.045],
            "signal_fraction": [0.95, 1.05],
            "outside_object_energy_fraction": [0.005, 0.015],
        }

    monkeypatch.setattr(s1, "sampled_phantom", sampler)
    monkeypatch.setattr(s1, "simulate", lambda acq, *a, **k: np.ones(len(acq.adc)))
    monkeypatch.setattr(s1, "image_scores", scores)
    monkeypatch.setattr(s1, "versions", dict)
    result = s1.evaluate_s1(
        submit(case, tmp_path),
        case,
        ROOT,
        tmp_path / "evaluation",
        inspector=inspect_sequence,
        probes=lambda acq, case, materials, props: (
            [],
            {},
            np.ones((len(materials), len(acq.adc)), complex),
        ),
        sampler=lambda acq, *args: np.zeros(len(acq.adc)),
        reconstructor=lambda raw, acq, matrix: np.zeros(matrix, complex),
        reference_builder=lambda acq, case, *args: np.ones(
            case["task"]["sequence"]["matrix"], complex
        ),
    )
    assert seen == [4096] * 4
    assert result["spatial_sampling"]["stop_reason"] == "interval_certified"
    assert result["status"] == "passed"
    assert result["ideal_imaging_success"] is True
    assert result["imaging_success_under_case_physics"] is True
    assert result["numerically_converged"] is True


@pytest.mark.parametrize("fam", ["tse_t2w", "grase"])
def test_cpmg_train_reference_matches_independent_finite_rf_bloch(tmp_path, fam):
    case = case_for(fam, small=True)
    case["evaluation"]["version"] = 3
    acquisition = inspect_sequence(submit(case, tmp_path), case)
    a = copy(acquisition)
    first_acquired = int(np.min(acquisition.excitation))
    ids = np.concatenate(
        [
            acquisition.lines[j]
            for j in np.flatnonzero(acquisition.excitation == first_acquired)
        ]
    )
    next_excitation = [p for p in a.pulses if not p.refocusing][
        first_acquired + 1
    ].start
    a.pulses = [p for p in a.pulses if p.start < next_excitation]
    a.adc, a.adc_phase, a.k = a.adc[ids], a.adc_phase[ids], a.k[ids]
    properties = np.array([[1.0, 0.8, 0.08]])
    spins = Spins(
        np.zeros((1, 3)),
        np.zeros(1, dtype=int),
        ["diagnostic"],
        properties,
        1.0,
        (1,),
        "test",
    )
    actual = simulate(a, spins, zero_gradients=True)
    expected = canonical_response(a, properties, fam)[0]
    assert nrmse(actual, expected) < 0.02


@pytest.mark.parametrize("fam", sorted(FAMILIES - {"rosette_gre", "concentric_rings"}))
def test_all_s2_waveform_identities(tmp_path, fam):
    case = case_for(fam)
    path = submit(case, tmp_path)
    basic = preflight(path, case) + evaluate_sar(path, case["hardware"])
    assert all(c["passed"] for c in basic), basic
    acq = inspect_sequence(path, case)
    assert all(c["passed"] for c in acq.checks), json.dumps(acq.checks, indent=2)


@pytest.mark.parametrize(
    "fam,mutation,check",
    [
        ("epi_single_shot", "no_alternation", "epi"),
        ("epi_multishot", "missing_line", "matrix"),
        ("tse_t2w", "no_refocusing", "refocused_echo_train"),
        ("tse_t2w", "lying_refocusing_label", "refocused_echo_train"),
        ("radial_readout", "repeated_spokes", "radial"),
        ("propeller_tse", "repeated_blades", "propeller"),
        ("radial_readout", "wrong_fov", "radial"),
    ],
)
def test_structural_corruptions_fail(tmp_path, fam, mutation, check):
    case = case_for(fam)
    acq = inspect_sequence(submit(case, tmp_path, mutation), case)
    assert any(c["name"] == check and not c["passed"] for c in acq.checks), acq.checks


def test_arbitrary_node_control_variate_matches_direct_sum():
    rng = np.random.default_rng(71)
    for dims in (2, 3):
        xyz = np.zeros((17, 3))
        xyz[:, :dims] = rng.uniform(-0.1, 0.1, (17, dims))
        labels = np.arange(17) % 2
        weights = rng.uniform(0.1, 0.3, 17)
        spins = Spins(
            xyz, labels, ["a", "b"], np.ones((2, 3)), weights, (8,) * dims, "test"
        )
        k = np.zeros((31, 3))
        k[:, :dims] = rng.uniform(-20, 20, (31, dims))
        acq = SimpleNamespace(k=k, adc=np.arange(31))
        response = rng.normal(size=(2, 31)) + 1j * rng.normal(size=(2, 31))
        actual = reference_signal(acq, spins, {"matrix": [8] * dims}, response)
        expected = np.sum(
            weights[:, None] * np.exp(-2j * np.pi * (xyz @ k.T)) * response[labels],
            axis=0,
        )
        assert nrmse(actual, expected) < 1e-8


@pytest.mark.parametrize("dims", [2, 3])
def test_noncartesian_bart_has_fixed_scale_phase_and_linearity(dims):
    matrix = (8,) * dims
    inds = np.indices(matrix).reshape(dims, -1).T
    k = np.zeros((len(inds), 3))
    k[:, :dims] = (inds - 4) / 0.2
    acq = SimpleNamespace(
        k=k, measurements={"family": "radial_readout", "fov_mm": [200] * dims}
    )
    # Independent voxel-centred direct Fourier sum, including complex phase.
    positions = ((inds + 0.5) / 8 - 0.5) * 0.2
    centre = np.array([0.017, -0.026, 0.011])[:dims]
    image = (
        np.exp(-np.sum((positions - centre) ** 2, axis=1) / 0.003)
        * (1 + 0.3j)
        * np.exp(5j * positions[:, 0])
    )
    raw = np.exp(-2j * np.pi * (k[:, :dims] @ positions.T)) @ image
    actual = reconstruct(raw, acq, matrix)
    assert nrmse(actual, image.reshape(matrix)) < 0.005
    # BART's CFL/FFTW operator is complex64; this is a precision check, separate
    # from the unchanged reconstruction-quality bound above.
    assert nrmse(reconstruct((2 - 1j) * raw, acq, matrix), (2 - 1j) * actual) < 2e-6


@pytest.mark.parametrize(
    "fam",
    ["radial_readout", "ute_radial", "propeller_tse", "stack_of_stars", "spiral_gre"],
)
def test_nominal_noncartesian_operator_on_withheld_asymmetric_images(tmp_path, fam):
    case = case_for(fam)
    acq = inspect_sequence(submit(case, tmp_path), case)
    matrix = tuple(case["task"]["sequence"]["matrix"])
    dims = len(matrix)
    inds = np.indices(matrix).reshape(dims, -1).T
    positions = ((inds + 0.5) / np.asarray(matrix) - 0.5) * 0.2
    first = np.array([0.022, -0.015, 0.011])[:dims]
    second = np.array([-0.035, 0.04, -0.022])[:dims]
    image = (
        np.exp(-np.sum((positions - first) ** 2, axis=1) / (2 * 0.018**2))
        + 0.4j * np.exp(-np.sum((positions - second) ** 2, axis=1) / (2 * 0.01**2))
    ).reshape(matrix)
    nodes = acq.k[:, :dims] * 0.2 / np.asarray(matrix)
    transform = finufft.nufft2d2 if dims == 2 else finufft.nufft3d2
    raw = transform(
        *[np.ascontiguousarray(2 * np.pi * nodes[:, j]) for j in range(dims)],
        image.astype(complex),
        eps=1e-12,
        isign=-1,
        nthreads=1,
    )
    raw *= np.exp(-1j * np.pi * nodes.sum(axis=1))
    # Independent complex profiles test the fixed regularized inverse without
    # fitting gain, phase or object-dependent reconstruction parameters.
    assert nrmse(reconstruct(raw, acq, matrix), image) < 0.15


def test_all_s2_full_suite_cases_route_to_versioned_contract():
    cat = Catalog(ROOT)
    cases = [
        c for c in cat.cases(cat.get("suites", "full")) if c["sequence_level"] == "S2"
    ]
    assert len(cases) == 27
    assert {case["task"]["id"] for case in cases} == FAMILIES - {
        "rosette_gre",
        "concentric_rings",
    }
    assert {c["image_policy"] for c in cases} == {"static"}
    assert all("object_level" not in c for c in cases)
    assert {c["hardware"]["id"] for c in cases} == {
        "low_field",
        "standard_1_5t",
        "high_performance",
    }
    for case in cases:
        assert Evaluation.model_validate(case["evaluation"]).id == "s2"
        assert case["evaluation"]["version"] == 3
        assert case["evaluation"]["calibrated"] is True


def test_noncartesian_reconstruction_handles_rotated_blade_overscan():
    matrix = (8, 8)
    inds = np.indices(matrix).reshape(2, -1).T
    nodes = (inds - 4) / 8
    # Additional acquired nodes lie outside the reconstruction band.
    extra = np.array([[0.6, 0.0], [-0.7, 0.1]])
    k = np.zeros((len(nodes) + len(extra), 3))
    k[:, :2] = np.r_[nodes, extra] * 40
    acq = SimpleNamespace(
        k=k, measurements={"family": "propeller_tse", "fov_mm": [200, 200]}
    )
    raw = np.r_[np.ones(len(nodes), complex), [10000j, 10000j]]
    actual = reconstruct(raw, acq, matrix)
    cropped = SimpleNamespace(k=k[:-2], measurements=acq.measurements)
    assert nrmse(actual, reconstruct(raw[:-2], cropped, matrix)) < 1e-10


def test_repeated_centres_choose_readout_temporal_centroid():
    from mriseqbench.evaluation.s2 import reference_indices

    acq = SimpleNamespace(
        k=np.array([[0.0, 0, 0], [4, 0, 0], [0, 0, 0], [4, 0, 0], [0, 0, 0]]),
        adc=np.array([1, 2, 3, 4, 5.0]),
        lines=[np.arange(5)],
    )
    assert reference_indices(
        acq, {"matrix": [32, 32], "fov_mm": [200, 200]}
    ).tolist() == [2]
