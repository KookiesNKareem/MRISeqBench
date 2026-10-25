"""Analytic phantom encoding and randomized, variance-controlled Bloch integration.

The residual control variate preserves every simulated coherence pathway. Independent
scrambles permit unbiased quadratic image scores without squaring sampling noise.
"""

import numpy as np
from scipy.stats import t

from .s1_anatomy import anatomical_reference as analytic_reference
from .s1_anatomy import sampled_anatomy as sampled_phantom

__all__ = ["analytic_reference", "sampled_phantom"]


def quadratic(values):
    """Unbiased squared norm of the integral, and delete-one estimates.

    Each row is an independent randomized quadrature, not an ADC sample. Cross
    products remove the positive image-energy bias caused by quadrature noise.
    """
    flat = np.asarray(values).reshape(len(values), -1)
    n = len(flat)
    summed = flat.sum(axis=0)
    norms = np.sum(np.abs(flat) ** 2, axis=1)
    total = float(np.vdot(summed, summed).real)
    value = (total - norms.sum()) / (n * (n - 1))
    loo = (
        total - 2 * np.real(flat.conj() @ summed) + norms - (norms.sum() - norms)
    ) / ((n - 1) * (n - 2))
    return float(value), loo


def interval(value, loo):
    n = len(loo)
    se = np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    radius = float(t.ppf(0.995, n - 1) * se)
    return float(value - radius), float(value + radius)


def image_scores(images, reference, support):
    denom = max(float(np.linalg.norm(reference) ** 2), 1e-30)
    error, error_loo = quadratic(images - reference)
    energy, energy_loo = quadratic(images)
    outside, outside_loo = quadratic(images[:, ~support])
    error_ci = interval(error / denom, error_loo / denom)
    signal_ci = interval(energy / denom, energy_loo / denom)
    fraction = outside / max(energy, 1e-30)
    fraction_ci = interval(fraction, outside_loo / np.maximum(energy_loo, 1e-30))
    return {
        "complex_image_nrmse": np.sqrt(max(0.0, error / denom)),
        "normalized_squared_error": error / denom,
        "signal_fraction": np.sqrt(max(0.0, energy / denom)),
        "outside_object_energy_fraction": max(0.0, fraction),
    }, {
        "complex_image_nrmse": [np.sqrt(max(0.0, x)) for x in error_ci],
        "normalized_squared_error": list(error_ci),
        "signal_fraction": [np.sqrt(max(0.0, x)) for x in signal_ci],
        "outside_object_energy_fraction": [max(0.0, x) for x in fraction_ci],
    }
