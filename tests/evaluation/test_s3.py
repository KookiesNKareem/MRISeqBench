"""S3 sampling controls, independent encoding assays and registered-physics replay."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.preflight import preflight
from mriseqbench.evaluation.s1 import decision
from mriseqbench.evaluation.s1_case import (
    case_bounds,
    case_image_scores,
    uses_case_image,
)
from mriseqbench.evaluation.s1_image import nrmse
from mriseqbench.evaluation.s1_koma import reconstruct as bart_reconstruct
from mriseqbench.evaluation.s1_numerics import ReplayCache
from mriseqbench.evaluation.s1_sampling import sampled_phantom
from mriseqbench.evaluation.s3 import (
    REGULARIZATION,
    analytic_channels,
    forward,
    inverse,
    mechanism_probes,
    receive_maps,
    reconstruct,
    reference_channels,
    reference_image,
)
from mriseqbench.evaluation.s3_sequence import (
    FAMILIES,
    PARALLEL,
    inspect_sequence,
    prescribed_mask,
    research_pattern,
)
from mriseqbench.evaluation.sar import evaluate_sar
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import legacy_task

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s3_controls", ROOT / "tests/fixtures/s3_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)

# Historical analytic controls exercise the original decision policy independently
# of the unqualified, scanner-specific cardiac benchmark.
LEGACY_EVALUATOR = Evaluation.model_validate(
    load_yaml(ROOT / "tests/fixtures/legacy_evaluators/s3.yaml")
).model_dump(mode="json")


def case_for(fam, obj=None, physics="ideal", research=False, small=False):
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
    )
    case["evaluation"] = deepcopy(LEGACY_EVALUATOR)
    case["image_policy"] = case["object_level"]
    case["task"]["evaluation"] = f"s3@{case['evaluation']['version']}"
    # Synthetic test qualification exercises pass/fail reporting; active S3 stays false.
    case["evaluation"]["calibrated"] = True
    if small:
        dims = len(case["task"]["sequence"]["matrix"])
        case["task"]["sequence"]["matrix"] = [8] * dims
        sampling = case["task"]["sequence"].get("sampling")
        if sampling and sampling.get("calibration_lines"):
            sampling["calibration_lines"] = 2
        case["evaluation"].update(
            spatial_samples=4096, spatial_max_samples=32768, spatial_replicates=8
        )
    return case


def submit(case, directory, mutation=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sequence.seq"
    controls.make_sequence(case, mutation).write(str(path))
    return path


@pytest.mark.parametrize("fam", sorted(FAMILIES))
@pytest.mark.parametrize("research", [False, True])
def test_all_waveform_identities_on_nominal_grid(tmp_path, fam, research):
    case = case_for(fam, research=research)
    path = submit(case, tmp_path)
    basic = preflight(path, case) + evaluate_sar(path, case["hardware"])
    assert all(c["passed"] for c in basic), basic
    acq = inspect_sequence(path, case)
    assert all(c["passed"] for c in acq.checks), json.dumps(acq.checks, indent=2)
    if research:
        assert not {"te_ms", "tr_ms", "flip_angle_deg"} & {
            c["name"] for c in acq.checks
        }


@pytest.mark.parametrize("fam", sorted(FAMILIES))
@pytest.mark.parametrize(
    "hardware", ["standard", "high_performance", "low_field", "standard_1_5t"]
)
def test_hardware_lattice_branch_cut_preserves_nominal_support(tmp_path, fam, hardware):
    cat = Catalog(ROOT)
    case = cat.resolve(
        legacy_task(cat, fam), cat.get("physics", "ideal"), hardware_ref=hardware
    )
    path = submit(case, tmp_path)
    acq = inspect_sequence(path, case)
    assert all(c["passed"] for c in acq.checks), acq.checks
    mask = prescribed_mask(
        fam,
        tuple(case["task"]["sequence"]["matrix"]),
        case["task"]["sequence"].get("sampling") or {},
    )
    np.testing.assert_array_equal(np.asarray(acq.measurements["sampling_mask"]), mask)
    q = acq.k[:, : len(acq.offset)] * (np.asarray(acq.measurements["fov_mm"]) / 1000)
    assert (
        np.max(abs(q - (acq.indices - np.asarray(mask.shape) // 2 + acq.offset)))
        <= 0.01
    )


@pytest.mark.parametrize(
    "fam,mutation,name",
    [
        ("sense_sampling", "full_sampling", "sampling_pattern"),
        ("sense_sampling", "repeated_lines", "matrix"),
        ("grappa_sampling", "no_calibration", "sampling_pattern"),
        ("caipirinha", "wrong_shift", "sampling_pattern"),
        ("caipirinha", "no_calibration", "sampling_pattern"),
        ("compressed_sensing", "wrong_seed", "sampling_pattern"),
        ("variable_density", "wrong_seed", "sampling_pattern"),
        ("partial_fourier", "full_sampling", "sampling_pattern"),
        ("asymmetric_echo", "full_sampling", "sampling_pattern"),
        ("reduced_fov", "full_sampling", "sampling_pattern"),
        ("elliptical_kspace", "full_sampling", "sampling_pattern"),
        ("elliptical_kspace", "missing_line", "sampling_pattern"),
        ("sense_sampling", "wrong_fov", "cartesian"),
    ],
)
def test_structural_corruptions_fail(tmp_path, fam, mutation, name):
    case = case_for(fam)
    acq = inspect_sequence(submit(case, tmp_path, mutation), case)
    assert any(c["name"] == name and not c["passed"] for c in acq.checks), acq.checks


@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_research_rejects_full_sampling(tmp_path, fam):
    case = case_for(fam, research=True)
    acq = inspect_sequence(submit(case, tmp_path, "full_sampling"), case)
    assert any(c["name"] == "sampling_pattern" and not c["passed"] for c in acq.checks)


def test_seeded_masks_have_frozen_public_indices():
    cat = Catalog(ROOT)
    expected = {
        "compressed_sensing": [10, 12, 13, 14, 15, 16, 17, 18, 19, 24, 27],
        "variable_density": [11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 26],
    }
    for fam, ys in expected.items():
        spec = legacy_task(cat, fam).sequence.model_dump()
        mask = prescribed_mask(fam, tuple(spec["matrix"]), spec["sampling"])
        assert np.flatnonzero(mask.all(axis=0)).tolist() == ys


@pytest.mark.parametrize(
    "step,truncated,valid",
    [(2, False, True), (4, False, True), (3, False, False), (2, True, False)],
)
def test_reduced_fov_research_support_fits_prescribed_voxel_grid(
    step, truncated, valid
):
    mask = np.zeros((32, 32), bool)
    residue = 16 % step
    mask[:, residue::step] = True
    if truncated:
        mask[:, :12] = False
        mask[:, 22:] = False
    assert bool(research_pattern("reduced_fov", mask)) is valid


@pytest.mark.parametrize("dims", [2, 3])
def test_voxel_fourier_matches_direct_sum_and_bart(dims):
    matrix = (8,) * dims
    rng = np.random.default_rng(93)
    image = rng.normal(size=matrix) + 1j * rng.normal(size=matrix)
    offset = np.array([0.5, 0.0, -0.25][:dims])
    inds = np.indices(matrix).reshape(dims, -1).T
    coordinates = (inds + 0.5) / 8 - 0.5
    q = inds - 4 + offset
    expected = np.exp(-2j * np.pi * (q @ coordinates.T)) @ image.ravel()
    actual = forward(image, offset).ravel()
    assert nrmse(actual, expected) < 2e-6
    assert nrmse(inverse(actual.reshape(matrix), offset), image) < 2e-6
    acq = SimpleNamespace(indices=inds, offset=offset)
    assert nrmse(bart_reconstruct(expected, acq, matrix), image) < 2e-6


@pytest.mark.parametrize("fam", sorted(PARALLEL))
def test_fixed_sense_recovers_withheld_complex_images_and_is_linear(tmp_path, fam):
    case = case_for(fam, small=True)
    acq = inspect_sequence(submit(case, tmp_path), case)
    matrix = tuple(case["task"]["sequence"]["matrix"])
    dims = len(matrix)
    inds = np.indices(matrix).reshape(dims, -1).T
    fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    xyz = np.zeros((len(inds), 3))
    xyz[:, :dims] = ((inds + 0.5) / matrix - 0.5) * fov
    # Smooth asymmetric test image, neither a registered phantom nor a density probe.
    first, second = np.array([0.02, -0.03, 0.01]), np.array([-0.04, 0.01, -0.02])
    image = (
        np.exp(-np.sum((xyz - first) ** 2, axis=1) / 0.001)
        + 0.4j * np.exp(-np.sum((xyz - second) ** 2, axis=1) / 0.0005)
    ).reshape(matrix)
    maps = receive_maps(xyz, fov).reshape((-1, *matrix))
    np.testing.assert_allclose(np.sum(maps**2, axis=0), 1, atol=1e-14)
    raw = np.array(
        [
            forward(image * sensitivity, acq.offset)[tuple(acq.indices.T)]
            for sensitivity in maps
        ]
    )
    actual = reconstruct(raw, acq, matrix)
    assert nrmse(actual, image) < (
        0.15 if fam in {"compressed_sensing", "variable_density"} else 0.06
    )
    # BART uses complex64. The fixed regularizer bounds amplification for
    # irregular masks; independent dense-normal-equation tests check accuracy.
    assert nrmse(reconstruct((2 - 1j) * raw, acq, matrix), (2 - 1j) * actual) < 1e-4
    # Zero-fill has aliasing on an independent object; exact SENSE maps improve it.
    if fam in {"sense_sampling", "caipirinha"}:
        grid = np.zeros((len(maps), *matrix), complex)
        grid[(slice(None), *acq.indices.T)] = raw
        zero_filled = np.sum(
            [s * inverse(g, acq.offset) for s, g in zip(maps, grid, strict=True)],
            axis=0,
        )
        assert nrmse(actual, image) < nrmse(zero_filled, image) / 5
    assert REGULARIZATION > 0


@pytest.mark.parametrize("fam", ["sense_sampling", "caipirinha"])
def test_coil_control_variate_matches_independent_direct_sum(tmp_path, fam):
    case = case_for(fam, small=True)
    acq = inspect_sequence(submit(case, tmp_path), case)
    spins = sampled_phantom(case, ROOT, 4096, 0)
    rng = np.random.default_rng(12)
    response = rng.normal(size=(len(spins.materials), len(acq.adc))) + 1j * rng.normal(
        size=(len(spins.materials), len(acq.adc))
    )
    actual = reference_channels(acq, spins, case["task"]["sequence"], response)
    maps = receive_maps(
        spins.positions, np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    )
    encoded = np.exp(-2j * np.pi * (spins.positions @ acq.k.T)) * response[spins.labels]
    expected = (maps * spins.weight) @ encoded
    assert nrmse(actual, expected) < 1e-8
    uniform = np.ones_like(response)
    analytic = analytic_channels(acq, case, ROOT, uniform)
    sampled = reference_channels(acq, spins, case["task"]["sequence"], uniform)
    assert nrmse(sampled, analytic) < 0.05


def test_image_target_is_complete_and_independent_of_mask(tmp_path):
    case = case_for("partial_fourier", small=True)
    acq = inspect_sequence(submit(case, tmp_path), case)
    response = np.full(
        (
            len(
                json.loads((ROOT / "benchmark/objects/shapes.json").read_text())[
                    "materials"
                ]
            ),
            len(acq.adc),
        ),
        -0.2j,
    )
    reference = reference_image(acq, case, ROOT, response, np.zeros((1, len(acq.adc))))
    changed = deepcopy(acq)
    changed.k, changed.adc = acq.k[:8], acq.adc[:8]
    # Time-independent tissue response: dropping acquired data cannot change target.
    other = reference_image(changed, case, ROOT, response[:, :8], np.zeros((1, 8)))
    np.testing.assert_allclose(reference, other, rtol=0, atol=0)
    zero_filled = reconstruct(analytic_channels(acq, case, ROOT, response), acq, (8, 8))
    assert nrmse(zero_filled, reference) > 0.01


@pytest.mark.parametrize("mutation", [None, "no_selection"])
def test_reduced_fov_requires_physical_spatial_selection(tmp_path, mutation):
    case = case_for("reduced_fov", small=True)
    acq = inspect_sequence(submit(case, tmp_path, mutation), case)
    checks, metrics, _ = mechanism_probes(
        acq, case, ["water"], np.array([[1.0, 0.8, 0.08]])
    )
    assert all(c["passed"] for c in checks) is (mutation is None), metrics


@pytest.mark.parametrize("level", ["P2", "P3"])
@pytest.mark.parametrize(
    "corruption", [None, "phase", "dropout", "contrast", "ghost", "blank"]
)
def test_quality_policy_rejects_corruptions(level, corruption):
    reference = np.zeros((32, 32), complex)
    reference[4:28, 4:28] = 1j
    reference[10:20, 10:20] = 2j
    support = reference != 0
    masks = {"a": support & (abs(reference) == 1), "b": abs(reference) == 2}
    actual = reference.copy()
    if corruption == "phase":
        actual *= 1j
    elif corruption == "dropout":
        actual[masks["b"]] *= 0.1
    elif corruption == "contrast":
        actual[support] = 1.1j
    elif corruption == "ghost":
        actual = np.roll(actual, 12, axis=0)
    elif corruption == "blank":
        actual[:] = 0
    metrics, _, _ = case_image_scores(
        np.repeat(actual[None], 8, axis=0),
        reference,
        support,
        np.ones(reference.shape, complex),
        support,
        masks,
    )
    policy = LEGACY_EVALUATOR["case_image_criteria"][level]
    assert decision(case_bounds(policy), metrics) is (corruption is None)


def test_multichannel_replay_cache_rejects_wrong_shape_and_resumes(tmp_path):
    case = case_for("sense_sampling", small=True)
    spins = sampled_phantom(case, ROOT, 4096, 0)
    acq = SimpleNamespace(adc=np.arange(3))
    calls = []

    def simulator(acq, part, effects, **steps):
        calls.append(1)
        return np.full((4, 3), sum(part.weight), complex)

    replay = ReplayCache(
        tmp_path, "s3-test", simulator, chunk_size=4096, signal_shape=(4, 3)
    )
    first = replay(acq, spins)
    assert len(calls) == 1
    file = next(tmp_path.glob("*.npy"))
    np.save(file, np.ones(3))
    second = replay(acq, spins)
    assert len(calls) == 2
    np.testing.assert_array_equal(first, second)
    replay(acq, spins)
    assert len(calls) == 2 and replay.hits == 1


# Every family, every object, and every registered compatible physics profile.
# Reduced FOV uses the nominal-grid acceptance and temporal regression below.
REPLAY_CASES = [(f, None, "ideal") for f in sorted(FAMILIES) if f != "reduced_fov"]
REPLAY_CASES += [
    ("sense_sampling", o, "ideal") for o in ("uniform_disc", "nested_ellipses")
]
REPLAY_CASES += [("elliptical_kspace", "uniform_sphere", "ideal")]
REPLAY_CASES += [
    ("partial_fourier", None, p)
    for p in (
        "b0_offset",
        "b0_smooth",
        "b0_localized",
        "b0_drift",
        "rigid_motion",
        "rigid_steps",
        "deformation",
        "dynamic_tissue",
        "fields_and_motion",
    )
]
REPLAY_CASES += [
    ("elliptical_kspace", None, p)
    for p in ("b0_smooth_3d", "rigid_motion_3d", "fields_and_motion_3d")
]


@pytest.mark.parametrize("fam,obj,profile", REPLAY_CASES)
def test_real_bloch_images_and_artifacts(tmp_path, fam, obj, profile):
    case = case_for(fam, obj, profile, small=True)
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result.get("sequence_valid") is True, json.dumps(result, indent=2)
    assert result["status"] in {"passed", "failed"}, result
    if case["object_level"] in {"P0", "P1"}:
        assert isinstance(result["ideal_imaging_success"], bool)
    assert result["numerically_converged"] is True
    assert result["evaluation_contract"] == "s3@1"
    assert result["effects"] == sorted({e["kind"] for e in case["physics"]["effects"]})
    assert uses_case_image(case) is (case["object_level"] in {"P2", "P3"})
    images = np.load(tmp_path / "evaluation/s3/images.npz")
    assert images["physical"].shape == tuple(case["task"]["sequence"]["matrix"])
    assert np.linalg.norm(images["physical_raw"]) > 0
    assert images["physical_raw"].shape[0] == (
        16 if fam == "caipirinha" else 4 if fam in PARALLEL else 1
    )
    provenance = json.loads((tmp_path / "evaluation/s3/provenance.json").read_text())
    assert provenance["phantom"] == case["object"]["phantom"]
    assert provenance["encoded_integrator"] == "KomaMRICore.Bloch"
    assert provenance["calibrated"] is True
    assert provenance["receive_encoding"]["dynamics"] == (
        "KomaMRICore.BlochDict" if fam in PARALLEL else "KomaMRICore.Bloch"
    )
    assert (tmp_path / "evaluation/s3/policy.json").is_file()


def test_full_suite_routes_all_s3_cases_to_current_cardiac_contract():
    cat = Catalog(ROOT)
    cases = [
        c for c in cat.cases(cat.get("suites", "full")) if c["sequence_level"] == "S3"
    ]
    assert len(cases) == 24
    assert {c["image_policy"] for c in cases} == {"static"}
    assert all("object_level" not in c for c in cases)
    assert {c["hardware"]["id"] for c in cases} == {
        "low_field",
        "standard_1_5t",
        "high_performance",
    }
    assert {c["task"]["id"] for c in cases} == FAMILIES - {"asymmetric_echo"}
    for case in cases:
        assert Evaluation.model_validate(case["evaluation"]).id == "s3"
        assert case["evaluation"]["version"] == 2
        assert case["evaluation"]["calibrated"] is True


def test_s3_contract_requires_case_policies_and_power_of_two_sampling():
    from pydantic import ValidationError

    definition = Catalog(ROOT).get("evaluators", "s3").model_dump()
    for change in ({"case_image_criteria": None}, {"spatial_samples": 6000}):
        with pytest.raises(ValidationError):
            Evaluation.model_validate({**definition, **change})


def test_receiver_phase_corruption_survives_encoded_replay(tmp_path):
    case = case_for("sense_sampling", "uniform_disc", small=True)
    result = evaluate(
        submit(case, tmp_path, "receiver_phase"), case, ROOT, tmp_path / "evaluation"
    )
    assert result["sequence_valid"] is True, result
    assert result["status"] == "failed", result
    assert any(
        c["name"] == "ideal_complex_image_nrmse" and c["passed"] is False
        for c in result["ideal_image_checks"]
    )
    assert result["ideal_imaging_success"] is False
    assert result["metrics"]["ideal_complex_image_nrmse"] > 1


@pytest.mark.parametrize(
    "fam,profile",
    [
        ("caipirinha", "ideal"),
        ("caipirinha", "b0_smooth_3d"),
        ("caipirinha", "rigid_motion_3d"),
        ("caipirinha", "fields_and_motion_3d"),
        ("sense_sampling", "dynamic_tissue"),
    ],
)
def test_streamed_receive_matches_separate_native_bloch_density_replays(
    tmp_path, fam, profile
):
    from dataclasses import replace

    from mriseqbench.evaluation.s1_koma import simulate

    case = case_for(fam, physics=profile, small=True)
    sequence = controls.make_sequence(case, preparations=2)
    path = tmp_path / "sequence.seq"
    sequence.write(str(path))
    acq = inspect_sequence(path, case)
    spins = sampled_phantom(case, ROOT, 4096, 0)
    ids = np.linspace(0, len(spins.positions) - 1, 37).astype(int)
    spins = replace(
        spins,
        positions=spins.positions[ids],
        labels=spins.labels[ids],
        weight=spins.weight[ids],
    )
    maps = receive_maps(
        spins.positions, np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
    )
    expected = np.asarray(
        [
            simulate(
                acq,
                replace(spins, weight=spins.weight * sensitivity),
                case["physics"]["effects"],
            )
            for sensitivity in maps
        ]
    )
    actual = simulate(acq, spins, case["physics"]["effects"], receive_weights=maps)
    assert nrmse(actual, expected) < 1e-10


def test_clipped_reference_has_independent_volume_and_quadrature_accuracy():
    from mriseqbench.evaluation.s3 import clipped_reference

    case = case_for("reduced_fov", "uniform_disc")
    nodes = np.array(
        [[0.0, 0.0, 0.0], [20.0, -30.0, 0.0], [-55.0, 40.0, 0.0], [75.0, 70.0, 0.0]]
    )
    acq = SimpleNamespace(adc=np.arange(len(nodes)), k=nodes)
    response = np.ones((7, len(nodes)), complex)
    actual = clipped_reference(acq, case, ROOT, response, 0.5)
    fine = clipped_reference(acq, case, ROOT, response, 0.5, order=1024)
    voxel = (0.2 / 32) ** 2
    radius, half = 0.06, 0.05
    u = half / radius
    area = 2 * radius**2 * (u * np.sqrt(1 - u * u) + np.arcsin(u))
    assert actual[0].real == pytest.approx(area / voxel, rel=1e-10)
    assert nrmse(actual, fine) < 1e-9
    spins = sampled_phantom(case, ROOT, 65536, 198)
    keep = abs(spins.positions[:, 1]) <= half
    direct = (
        np.exp(-2j * np.pi * (nodes @ spins.positions[keep].T)) @ spins.weight[keep]
    )
    assert nrmse(actual, direct) < 0.002


def test_no_excitation_is_candidate_failure_without_backend_interruption(tmp_path):
    case = case_for("sense_sampling")
    path = submit(case, tmp_path, "all_refocusing")
    basic = preflight(path, case) + evaluate_sar(path, case["hardware"])
    assert all(c["passed"] for c in basic), basic
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["status"] == "failed"
    assert result["sequence_valid"] is False
    assert any(
        c["name"] == "rf_before_adc" and c["passed"] is False for c in result["checks"]
    )


def test_explicit_uncalibrated_case_never_claims_official_image_success(tmp_path):
    case = case_for("sense_sampling", "uniform_disc", small=True)
    case["evaluation"]["calibrated"] = False
    result = evaluate(submit(case, tmp_path), case, ROOT, tmp_path / "evaluation")
    assert result["sequence_valid"] is True, result
    assert result["numerically_converged"] is True
    assert result["status"] == "uncalibrated"
    assert result["ideal_imaging_success"] is None
    assert result["imaging_success_under_case_physics"] is None


@pytest.mark.parametrize("corruption", ["rf_frequency", "adc_phase"])
def test_nonfinite_carrier_metadata_is_a_candidate_failure(tmp_path, corruption):
    case = case_for("sense_sampling", small=True)
    sequence = controls.make_sequence(case, preparations=2)
    path = tmp_path / "sequence.seq"
    sequence.write(str(path))
    lines = path.read_text().split("[SIGNATURE]")[0].splitlines()
    section = "[RF]" if corruption == "rf_frequency" else "[ADC]"
    index = lines.index(section) + 1
    while not lines[index].strip() or lines[index].startswith("#"):
        index += 1
    fields = lines[index].split()
    # Pulseq 1.5 numeric columns: RF frequency offset, or ADC receiver phase.
    fields[9 if corruption == "rf_frequency" else 7] = "nan"
    lines[index] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n")
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["status"] == "failed"
    assert any(
        not c["passed"] and c["name"] in {"finite_rf", "finite_adc", "pulseq_preflight"}
        for c in result["checks"]
    )


def test_nominal_reduced_fov_checks_reconstruction_amplified_timestep_error(tmp_path):
    case = case_for("reduced_fov", "tissue_discs")
    case["evaluation"].update(
        spatial_samples=4096, spatial_replicates=8, spatial_seed=106458
    )
    path = submit(case, tmp_path)
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["status"] == "passed"
    assert result["evaluation_contract"] == "s3@1"
    assert result["numerically_converged"] is True
    assert all(c["passed"] is True for c in result["image_checks"])
    assert result["metrics"]["temporal_raw_nrmse"] <= 0.01
    assert result["metrics"]["temporal_image_nrmse"] <= 0.01
    provenance = json.loads(Path(result["artifacts"]["provenance"]).read_text())
    history = provenance["temporal_refinement_history"]
    assert history
    assert any(h["image_nrmse"] > h["raw_nrmse"] for h in history)
    for item in history:
        assert item["nrmse"] == pytest.approx(
            max(item["raw_nrmse"], item["image_nrmse"]), abs=1e-15
        )
    assert history[-1]["nrmse"] <= 0.01
    assert result["metrics"]["temporal_raw_nrmse"] == history[-1]["raw_nrmse"]
    assert result["metrics"]["temporal_image_nrmse"] == history[-1]["image_nrmse"]
    assert provenance["rf_step_s"][0] <= 20e-6
