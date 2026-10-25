"""Native 128 sampling and segmented timing checks; no signal simulation."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1_case
from mriseqbench.evaluation.evaluate import s4_backend_contract
from mriseqbench.evaluation.excitation import target_spins
from mriseqbench.evaluation.excitation_target import target_arrays, target_at
from mriseqbench.evaluation.s1_case import case_bounds, case_policy
from mriseqbench.evaluation.s1_numerics import ReplayCache
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s4_sequence import inspect_sequence
from mriseqbench.harness.prepare import case_evaluation_summary
from tests.fixtures.s4_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "family", ["adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping"]
)
def test_mapping_uses_direct_parameter_fit_instead_of_single_frame_fat_contrast(family):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", family), hardware_ref="high_performance@1"
    )
    mapping = family in {"t1_mapping", "t2_mapping"}
    full = case_bounds(case_policy(case))
    native = case_bounds(case_policy(case), contrast=not mapping)
    assert native == [
        bound
        for bound in full
        if not (mapping and bound["name"] == "contrast_retention")
    ]
    _, external = s4_backend_contract(case)
    frame_zero = {
        bound["name"] for bound in external if bound["name"].startswith("frame_0_case_")
    }
    assert frame_zero == {"frame_0_case_" + bound["name"] for bound in native}
    if mapping:
        assert any(
            bound["name"] == f"myocardial_{family[:2]}_relative_error"
            for bound in external
        )
    assert ("contrast retention" in case_evaluation_summary(case)) is not mapping


def test_excitation_target_is_native_128_and_uses_physical_coordinates():
    arrays = target_arrays()
    spins, target = target_spins()
    assert arrays["target_mxy_magnitude"].shape == spins.grid == (128, 128)
    np.testing.assert_allclose(target, target_at(spins.positions[:, :2] * 1000))
    assert np.max(np.linalg.norm(spins.positions, axis=1)) <= 0.06


def test_expired_replay_budget_prevents_simulation(tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("An expired grading budget must not launch simulation")

    replay = ReplayCache(
        tmp_path, "deadline-test", forbidden, deadline=time.monotonic() - 1
    )
    spins = Spins(
        np.zeros((1, 3)), np.zeros(1, int), ["water"], np.ones((1, 3)), 1, (1,), "test"
    )
    with pytest.raises(TimeoutError, match="grading budget"):
        replay(SimpleNamespace(adc=np.zeros(1)), spins)


@pytest.mark.parametrize(
    "family", ["adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping"]
)
def test_segmented_cardiac_frames_preserve_full_grid_and_window(tmp_path, family):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", family), hardware_ref="high_performance@1"
    )
    assert case["task"]["sequence"]["temporal"]["segments_per_frame"] == 4
    sequence = make_sequence(
        case,
        segment_lines=32,
        readout_flat=0.00128,
        gradient_duration=0.001,
        tr_s=0.0046,
    )
    path = tmp_path / f"{family}.seq"
    sequence.write(str(path))
    acquisition = inspect_sequence(path, case)
    failed = [item for item in acquisition.checks if not item["passed"]]
    assert not failed, failed
    # Aggregated segment diagnostics must survive the same strict JSON writer
    # used after GPU grading; nested NumPy booleans otherwise discard a verdict.
    json.dumps(acquisition.checks, allow_nan=False)
    frames = acquisition.measurements["frames"]
    assert len(frames) == case["task"]["sequence"]["temporal"]["frames"]
    for frame in frames:
        assert len(frame["segments"]) == 4
        assert len(frame["adc_indices"]) == 128 * 128
    assert next(
        item for item in acquisition.checks if item["name"] == "cardiac_adc_window"
    )["detail"]["required_ms"] == [600, 800]


@pytest.mark.parametrize(
    "mutation,rejected",
    [
        ("late_adc", "cardiac_adc_window"),
        ("wrong_preparation_time", "frame_0_inversion"),
        ("unbalanced", "frame_0_balanced_bssfp"),
        ("no_phase_cycle", "frame_0_balanced_bssfp"),
    ],
)
def test_segmented_cardiac_rejects_wrong_waveforms(tmp_path, mutation, rejected):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "t1_mapping"), hardware_ref="high_performance@1"
    )
    sequence = make_sequence(
        case,
        mutation=mutation,
        segment_lines=32,
        readout_flat=0.00128,
        gradient_duration=0.001,
        tr_s=0.0046,
    )
    path = tmp_path / f"{mutation}.seq"
    sequence.write(str(path))
    acquisition = inspect_sequence(path, case)
    assert any(
        item["name"] == rejected and not item["passed"] for item in acquisition.checks
    )


def test_sparse_anatomy_preserves_native_tissue_volumes_and_fourier_integrals(
    monkeypatch,
):
    from mriseqbench.evaluation import s1_anatomy

    labels = np.ones((12, 10, 8), dtype=int)
    labels[6:] = 2
    anatomy = s1_anatomy.Anatomy(
        labels,
        np.full(3, 0.001),
        np.zeros(3),
        ["water", "fat"],
        np.array([[1.0, 0.8, 0.08], [0.9, 0.9, 0.09]]),
        "synthetic-geometry",
    )
    monkeypatch.setattr(s1_anatomy, "load_anatomy", lambda *_: anatomy)
    case = {
        "object": {"phantom": "test"},
        "evaluation": {"spatial_seed": 1729},
        "task": {"sequence": {"matrix": [12, 10, 8], "fov_mm": [12, 10, 8]}},
    }
    frequency = np.array([31.0, 18.0, 11.0])
    measurements = []
    for replicate in range(32):
        spins = s1_anatomy.sampled_anatomy(case, ROOT, 257, replicate)
        assert len(spins.positions) == 257
        assert np.all(np.abs(spins.positions) <= np.array([0.006, 0.005, 0.004]))
        integrals = []
        for label in range(2):
            keep = spins.labels == label
            np.testing.assert_allclose(spins.weight[keep].sum(), 480, rtol=1e-14)
            integrals.append(
                np.sum(
                    spins.weight[keep]
                    * np.exp(-2j * np.pi * (spins.positions[keep] @ frequency))
                )
            )
        measurements.append(integrals)
    measured = np.asarray(measurements)
    centers = np.array([[-0.003, 0, 0], [0.003, 0, 0]])
    expected = (
        480
        * np.exp(-2j * np.pi * (centers @ frequency))
        * np.prod(np.sinc(frequency * np.array([0.006, 0.010, 0.008])))
    )
    for component in (np.real, np.imag):
        values = component(measured)
        error = np.abs(values.mean(axis=0) - component(expected))
        assert np.all(error < 5 * values.std(axis=0, ddof=1) / np.sqrt(len(values)))
    first = s1_anatomy.sampled_anatomy(case, ROOT, 257, 0)
    repeated = s1_anatomy.sampled_anatomy(case, ROOT, 257, 0)
    independent = s1_anatomy.sampled_anatomy(case, ROOT, 257, 1)
    np.testing.assert_array_equal(first.positions, repeated.positions)
    assert not np.array_equal(first.positions, independent.positions)


def test_batched_cardiac_phase_preserves_each_echo(monkeypatch):
    from copy import copy

    from mriseqbench.evaluation import s4

    geometry = Spins(
        np.array(
            [[-0.05, -0.05, 0], [-0.05, 0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0]]
        ),
        np.array([0, 1, 0, 1]),
        ["water", "fat"],
        np.array([[1, 0.8, 0.08, 0], [0.9, 0.6, 0.05, -440]]),
        np.ones(4),
        (2, 2),
        "phase-test",
    )
    monkeypatch.setattr(s1_case, "phantom", lambda *_: geometry)
    pulse = lambda t: SimpleNamespace(
        start=t, center=t + 0.001, phase=t, frequency=0, refocusing=False
    )
    source = SimpleNamespace(
        adc=np.array([0.01, 0.02, 0.03]),
        adc_phase=np.zeros(3),
        k=np.zeros((3, 3)),
        lines=[np.array([i]) for i in range(3)],
        pulses=[pulse(0), pulse(0.011), pulse(0.021)],
    )
    case = {
        "task": {"sequence": {"matrix": [2, 2], "fov_mm": [200, 200]}},
        "object": {"phantom": "cardiac_test"},
        "physics": {"effects": ["test-field"]},
    }

    def response(acquisition, spins, effects, **_):
        phase = acquisition.adc[:, None] * (7 if effects else 3)
        return (
            np.broadcast_to(
                np.exp(1j * phase), (len(acquisition.adc), len(spins.positions))
            )
            * spins.weight
        )

    monkeypatch.setattr(s4, "simulate", response)
    batch = {"acquisition": source, "indices": np.arange(3), "images": {}}
    frames = []
    for i in range(3):
        frame = copy(source)
        frame.adc, frame.adc_phase, frame.k = (
            source.adc[[i]],
            source.adc_phase[[i]],
            source.k[[i]],
        )
        frame.lines = [np.array([0])]
        frame.pulses = [p for p in source.pulses if p.start < frame.adc[0]]
        frames.append(frame)
    batched = [
        s4.phase_reference(a, case, ROOT, np.ones((2, 2)), batch=batch) for a in frames
    ]
    separate = [s4.phase_reference(a, case, ROOT, np.ones((2, 2))) for a in frames]
    for a, b in zip(batched, separate, strict=True):
        np.testing.assert_allclose(a[0], b[0], atol=1e-15)
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]


def test_cardiac_probes_reuse_only_unchanged_acquisition_and_case(
    monkeypatch, tmp_path
):
    from copy import copy, deepcopy

    from mriseqbench.evaluation import s4

    source = SimpleNamespace(
        measurements={"frames": [{"central_adc_index": i} for i in range(3)]}
    )
    case = {"evaluation": {"spatial_samples": 32768}, "physics": {"offset_hz": 0}}
    calls = []
    monkeypatch.setattr(
        s4,
        "frame_acquisition",
        lambda _, frame: SimpleNamespace(number=frame["central_adc_index"]),
    )

    def probe(acquisition, local_case, materials, properties):
        calls.append((acquisition.number, local_case["physics"]["offset_hz"]))
        return (
            [{"passed": True}],
            {"signal": acquisition.number + 1},
            np.array([[acquisition.number + 1j]]),
        )

    def evaluate(_submission, local_case, _root, _workspace, *, inspector, probes, **_):
        acquisition = inspector(None, None)
        checks, metrics, response = probes(
            acquisition, local_case, [], np.zeros((1, 3))
        )
        assert np.array_equal(response, [[acquisition.number + 1j]])
        # Changing a returned response must not corrupt the cached original.
        response[:] = 999
        return {"sequence_valid": True, "checks": checks, "metrics": metrics}

    monkeypatch.setattr(s4, "mechanism_probes", probe)
    monkeypatch.setattr(s4, "evaluate_s1", evaluate)
    _, initial = s4._evaluate_frames(
        source, "unused", case, ROOT, tmp_path, samples=32768
    )
    _, refined = s4._evaluate_frames(
        source, "unused", case, ROOT, tmp_path, samples=65536
    )
    assert initial == refined
    assert calls == [(0, 0), (1, 0), (2, 0)]
    changed_case = deepcopy(case)
    changed_case["physics"]["offset_hz"] = 13
    s4._evaluate_frames(source, "unused", changed_case, ROOT, tmp_path)
    assert calls[-3:] == [(0, 13), (1, 13), (2, 13)]
    s4._evaluate_frames(copy(source), "unused", changed_case, ROOT, tmp_path)
    assert len(calls) == 9
