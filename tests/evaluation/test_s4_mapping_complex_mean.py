"""Mapping fits the complex replicate mean, not mean replicate magnitudes (CPU)."""

import numpy as np
import pytest

from mriseqbench.evaluation.s1_sampling import interval
from mriseqbench.evaluation.s4 import _mapping_estimates, _roi_magnitudes

TIS = np.array([0.1, 0.4, 1.2])
GRID = np.geomspace(0.1, 4.0, 241)
DICTIONARY = np.abs(1 - 2 * np.exp(-TIS[:, None] / GRID[None]))


def ensembles(truth=1.0, replicates=8, pixels=40, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for ti in TIS:
        clean = (1 - 2 * np.exp(-ti / truth)) * np.exp(0.7j)
        jitter = noise * (
            rng.standard_normal((replicates, pixels))
            + 1j * rng.standard_normal((replicates, pixels))
        )
        jitter -= jitter.mean(axis=0)  # exact complex ensemble mean = clean
        result.append(clean + jitter)
    return result


def test_complex_mean_is_unbiased_where_mean_magnitude_is_not():
    data = ensembles()
    point, loo = _mapping_estimates(data, DICTIONARY, GRID)
    assert point == pytest.approx(1.0, rel=1e-4)
    # The former estimator (mean of per-replicate magnitudes) is biased.
    biased = np.array([np.mean(np.abs(r)) for r in data])
    assert np.max(np.abs(biased - _roi_magnitudes(data))) > 0.1
    # Each leave-one-out ensemble is refitted end to end: nonzero spread.
    assert np.ptp(loo) > 0
    lo, hi = interval(abs(point - 1.0), np.abs(loo - 1.0))
    assert hi > lo


def test_loo_drops_exactly_one_replicate():
    data = ensembles(replicates=4)
    expected = [np.mean(np.abs(r[1:].mean(axis=0))) for r in data]
    np.testing.assert_allclose(_roi_magnitudes(data, exclude=0), expected)


def test_mapping_requires_equal_replicate_ensembles():
    data = ensembles(replicates=4)
    data[1] = data[1][:3]
    with pytest.raises(ValueError):
        _mapping_estimates(data, DICTIONARY, GRID)
