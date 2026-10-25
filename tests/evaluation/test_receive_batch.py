"""Batched plans preserve the streamed, physical-coordinate Fourier operator."""

from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s2, s3
from mriseqbench.evaluation.s1_phantom import Spins


@pytest.mark.parametrize(
    "dims,parallel,gaussian",
    [(2, False, False), (2, True, False), (3, True, False), (2, True, True)],
)
def test_receive_batch_matches_streaming_across_chunks(
    monkeypatch, dims, parallel, gaussian
):
    rng = np.random.default_rng(71)
    xyz = rng.uniform(-0.04, 0.04, (19, 3))
    labels = np.arange(len(xyz)) % 2
    spins = Spins(
        xyz,
        labels,
        ["a", "b"],
        np.ones((2, 3)),
        rng.uniform(0.01, 0.02, len(xyz)),
        (19,),
        "batch-test",
    )
    acq = SimpleNamespace(
        adc=np.arange(11),
        k=rng.uniform(-40, 40, (11, 3)),
        offset=np.zeros(dims),
        measurements={
            "family": "variable_density"
            if gaussian
            else "caipirinha"
            if parallel
            else "partial_fourier"
        },
    )
    spec = {"fov_mm": [200] * dims, "matrix": [8] * dims}
    response = rng.normal(size=(2, 11)) + 1j * rng.normal(size=(2, 11))
    monkeypatch.setattr(s3, "REFERENCE_CHUNK_SPINS", 7)
    angles = (
        list(product(np.arange(4) * np.pi / 2, repeat=dims - 1)) if parallel else [()]
    )
    if gaussian:
        angles = list(range(8))
    expected = np.zeros((len(angles), len(acq.adc)), complex)
    for start in range(0, len(xyz), 7):
        part = replace(
            spins,
            positions=xyz[start : start + 7],
            labels=labels[start : start + 7],
            weight=spins.weight[start : start + 7],
        )
        for channel, phases in enumerate(angles):
            weights = part.weight * (
                s3.gaussian_receive_maps(part.positions, np.full(dims, 0.2))[channel]
                if gaussian
                else s3._receive_map(part.positions, np.full(dims, 0.2), phases)
            )
            expected[channel] += s2.reference_signal(
                acq, replace(part, weight=weights), spec, response
            )
    np.testing.assert_allclose(
        s3.reference_channels(acq, spins, spec, response),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_variable_density_receive_encoding_has_no_nullspace():
    from mriseqbench.evaluation.s3_sequence import prescribed_mask

    catalog = Catalog(Path(__file__).resolve().parents[2])
    spec = catalog.get("tasks", "variable_density").sequence.model_dump()
    matrix = np.asarray(spec["matrix"])
    mask = prescribed_mask("variable_density", matrix, spec["sampling"])
    y = (np.arange(matrix[1]) + 0.5) / matrix[1] - 0.5
    xyz = np.zeros((len(y), 3))
    xyz[:, 1] = y * spec["fov_mm"][1] / 1000
    acq = SimpleNamespace(measurements={"family": "variable_density"})
    maps = s3.acquisition_maps(acq)(xyz, np.asarray(spec["fov_mm"]) / 1000)
    ky = np.flatnonzero(mask.any(axis=0)) - matrix[1] // 2
    fourier = np.exp(-2j * np.pi * ky[:, None] * y) / np.sqrt(matrix[1])
    operator = np.concatenate([fourier * coil for coil in maps])
    singular = np.linalg.svd(operator, compute_uv=False)
    assert np.count_nonzero(singular > singular[0] * 1e-12) == matrix[1]
