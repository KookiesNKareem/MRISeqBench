"""Independent reference Fourier operators match direct complex sums."""

import numpy as np
import pytest

from mriseqbench.evaluation.s1_control import cell_fourier, fourier


def test_widely_separated_queries_match_direct_complex_fourier_sums():
    rng = np.random.default_rng(73)
    points = rng.uniform(-0.1, 0.1, (19, 2))
    weights = rng.normal(size=(2, 19)) + 1j * rng.normal(size=(2, 19))
    frequencies = np.array(
        [[0, 0], [2100, 0], [-2300, 100], [2500, 3000], [-5200, -9000], [10000, 6500]]
    )
    direct = weights @ np.exp(-2j * np.pi * points @ frequencies.T)
    np.testing.assert_allclose(
        fourier(points, weights, frequencies), direct, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_regular_cells_at_large_frequencies_match_direct_integrals(dims):
    rng = np.random.default_rng(47)
    sizes = np.array([0.001, 0.002, 0.00625])[:dims]
    centres = np.indices((5,) * dims).reshape(dims, -1).T * sizes - 0.073
    weights = rng.normal(size=len(centres)) + 1j * rng.normal(size=len(centres))
    frequencies = rng.uniform(-50000, 50000, (31, dims))
    direct = (weights @ np.exp(-2j * np.pi * centres @ frequencies.T)) * np.prod(
        np.sinc(frequencies * sizes), axis=1
    )
    np.testing.assert_allclose(
        cell_fourier(centres, weights, sizes, frequencies), direct, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("bend", [0, 1e-10, 1])
def test_3d_flat_roundoff_and_noncollinear_queries_match_direct_sums(bend):
    rng = np.random.default_rng(17)
    points = rng.uniform(-0.1, 0.1, (23, 3))
    from mriseqbench.evaluation.s3 import receive_maps

    weights = receive_maps(points, np.full(3, 0.2), True) * rng.uniform(
        0.1, 1, (1, len(points))
    )
    # Actual declared z-B0 makes the readout line oblique in three dimensions.
    frequencies = np.array([17.0, 25.0, 6.0]) + np.linspace(-0.00064, 0.00064, 128)[
        :, None
    ] * np.array([201000.0, 0.0, 300.0])
    frequencies[64, 1] += bend
    direct = weights @ np.exp(-2j * np.pi * (points @ frequencies.T))
    np.testing.assert_allclose(
        fourier(points, weights, frequencies), direct, rtol=1e-8, atol=1e-8
    )
