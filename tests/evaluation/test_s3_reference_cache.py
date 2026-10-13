"""Exact Fourier receive encoding against independent physical point integrals."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import s3


@pytest.mark.parametrize("dims", [2, 3])
def test_harmonic_reference_matches_independent_coil_weighted_integral(
    monkeypatch, dims
):
    rng = np.random.default_rng(20261031)
    fov = np.array([0.20, 0.18, 0.16][:dims])
    points = rng.uniform(-0.45, 0.45, (17, dims)) * fov
    labels = np.arange(len(points)) % 3
    weights = rng.uniform(0.1, 1.0, len(points))
    k = np.zeros((7, 3))
    k[:, :dims] = rng.uniform(-3, 3, (7, dims)) / fov
    response = rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
    acq = SimpleNamespace(k=k, adc=np.arange(7))
    case = {
        "task": {
            "id": "caipirinha" if dims == 3 else "sense_sampling",
            "sequence": {"fov_mm": (fov * 1000).tolist()},
        }
    }
    calls = []

    def exact_points(shifted, case, root, signal):
        calls.append(shifted.k.copy())
        encoder = np.exp(-2j * np.pi * (shifted.k[:, :dims] @ points.T))
        return np.sum(encoder * weights * signal[labels].T, axis=1)

    monkeypatch.setattr(s3, "analytic_reference", exact_points)
    normalization = np.sqrt(4 * (1 + 0.8**2 / 2))
    maps = np.stack(
        [
            np.prod(
                [
                    (1 + 0.8 * np.cos(2 * np.pi * points[:, axis] / fov[axis] - angle))
                    / normalization
                    for axis, angle in enumerate(angles, start=1)
                ],
                axis=0,
            )
            for angles in product(np.arange(4) * np.pi / 2, repeat=dims - 1)
        ]
    )
    encoder = np.exp(-2j * np.pi * (k[:, :dims] @ points.T))
    expected = np.asarray(
        [np.sum(encoder * weights * coil * response[labels].T, axis=1) for coil in maps]
    )
    actual = s3.analytic_channels(acq, case, None, response)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-14)
    assert len(calls) == 3 ** (dims - 1)
    calls.clear()
    # Memoization is local to this response; changing material phase cannot reuse
    # an earlier case's Fourier amplitudes.
    rotated = s3.analytic_channels(acq, case, None, 1j * response)
    np.testing.assert_allclose(rotated, 1j * expected, rtol=5e-14, atol=5e-14)
    assert len(calls) == 3 ** (dims - 1)
