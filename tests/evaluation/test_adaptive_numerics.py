"""Convergence, quadrature correctness, and interrupted-work recovery controls."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s1_case
from mriseqbench.evaluation.s1_numerics import ReplayCache
from mriseqbench.evaluation.s1_sampling import (
    primitive_fourier,
    sampled_phantom,
    source,
)
from tests.fixtures.legacy_catalog import resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "obj",
    [
        "nested_ellipses",
        "volume_ellipsoids",
    ],
)
def test_stratified_quadrature_matches_independent_analytic_encoding(obj):
    family = (
        "elliptical_kspace"
        if obj in {"uniform_sphere", "volume_ellipsoids"}
        else "gre_t2star"
    )
    case = resolve_legacy_case(Catalog(ROOT), family, object_ref=obj, physics="ideal")
    spins = sampled_phantom(case, ROOT, 32768, 0)
    dims = len(case["task"]["sequence"]["matrix"])
    k = np.array([[0, 0, 0], [3, -2, 1], [-6, 5, -2]], dtype=float)
    shapes, primitives, _ = source(case, ROOT)
    voxel = np.prod(
        np.asarray(case["task"]["sequence"]["fov_mm"])
        / 1000
        / case["task"]["sequence"]["matrix"]
    )
    expected = np.zeros(len(k), complex)
    for index, shape in enumerate(primitives):
        parent = None
        for earlier in primitives[:index]:
            q = (np.asarray(shape["center_mm"]) - earlier["center_mm"]) / earlier[
                "radii_mm"
            ]
            if (np.abs(q) <= 1).all() if earlier["kind"] == "box" else q @ q <= 1:
                parent = earlier["material"]
        rho = shapes["materials"][shape["material"]][0]
        rho -= shapes["materials"][parent][0] if parent else 0
        expected += primitive_fourier(shape, k) / voxel * rho
    actual = np.exp(-2j * np.pi * k[:, :dims] @ spins.positions[:, :dims].T) @ (
        spins.weight * spins.properties[spins.labels, 0]
    )
    np.testing.assert_allclose(actual, expected, atol=abs(expected[0]) * 3e-4)
    assert actual[0] == pytest.approx(expected[0], rel=1e-13)


def test_replay_cache_resumes_completed_chunks_and_invalidates_changed_input(tmp_path):
    case = resolve_legacy_case(Catalog(ROOT), "gre_t2star", physics="ideal")
    spins = sampled_phantom(case, ROOT, 4096, 0)
    acq = SimpleNamespace(adc=np.arange(4))
    calls = []

    def simulate(acq, part, effects, **steps):
        calls.append(len(part.positions))
        if len(calls) == 2:
            raise RuntimeError("worker interrupted")
        return np.full(len(acq.adc), np.sum(part.weight), complex)

    cache = ReplayCache(tmp_path, "fixed input", simulate, chunk_size=1024)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache(acq, spins, step_s=0.001)
    result = cache(acq, spins, step_s=0.001)
    assert cache.hits == 1
    assert len(calls) == 5  # One completed chunk survives interruption.
    np.testing.assert_allclose(result, np.sum(spins.weight))
    assert max(calls) <= 1024
    again = ReplayCache(tmp_path, "fixed input", simulate, chunk_size=1024)
    np.testing.assert_allclose(again(acq, spins, step_s=0.001), result)
    assert len(calls) == 5
    again(acq, spins, step_s=0.0005)
    assert len(calls) == 9
    changed = ReplayCache(tmp_path, "changed sequence", simulate, chunk_size=1024)
    changed(acq, spins, step_s=0.001)
    assert len(calls) == 13


def test_replay_cache_repairs_corruption_and_tracks_material_names_and_shape_source(
    tmp_path,
):
    case = resolve_legacy_case(Catalog(ROOT), "gre_t2star", physics="ideal")
    spins = sampled_phantom(case, ROOT, 4096, 0)
    acq = SimpleNamespace(adc=np.arange(4))
    calls = []

    def simulate(acq, part, effects, **steps):
        calls.append(part)
        return np.full(len(acq.adc), np.sum(part.weight), complex)

    cache = ReplayCache(tmp_path, "fixed input", simulate, chunk_size=8192)
    expected = cache(acq, spins)
    next(tmp_path.glob("*.npy")).write_bytes(b"")
    np.testing.assert_allclose(cache(acq, spins), expected)
    assert len(calls) == 2
    cache(acq, replace(spins, materials=["changed-" + m for m in spins.materials]))
    assert len(calls) == 3
    cache(acq, replace(spins, source_sha256="changed shape registry"))
    assert len(calls) == 4


@pytest.mark.parametrize("signal_fraction", [1.0, 0.001])
def test_phase_oracle_refines_time_and_separates_physical_dropout(
    monkeypatch, signal_fraction
):
    case = resolve_legacy_case(Catalog(ROOT), "bssfp", physics="b0_offset")
    case["task"]["sequence"]["matrix"] = [8, 8]
    acq = SimpleNamespace(
        adc=np.array([0.01]),
        k=np.zeros((1, 3)),
        lines=[np.array([0])],
        pulses=[],
        measurements={"family": "bssfp"},
    )
    monkeypatch.setattr(s1_case, "expected_receiver_phase", lambda acq: np.zeros(1))
    steps = []

    def simulate(acq, spins, effects, **options):
        step = options["rf_step_s"]
        steps.append(step)
        # A known smooth step-size error; physical dropout is independent.
        phase = np.exp(1j * 20000 * step) if effects else 1
        fraction = signal_fraction if effects else 1
        return (np.asarray(spins.weight) * fraction * phase)[None, :]

    monkeypatch.setattr(s1_case, "simulate", simulate)
    reference = np.zeros((8, 8), complex)
    reference[2:6, 2:6] = 1
    _, _, info = s1_case.phase_reference(acq, case, ROOT, reference)
    assert info["converged"]
    assert info["temporal_phase_chord_rms"] <= 0.01
    assert info["coverage_adequate"] is (signal_fraction == 1.0)
    if signal_fraction == 1.0:
        assert min(steps) < 2.5e-6
        assert len(info["temporal_refinement_history"]) > 3
