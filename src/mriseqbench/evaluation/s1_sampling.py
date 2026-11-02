"""Analytic phantom encoding and randomized, variance-controlled Bloch integration.

Candidate images are reconstructed from native Koma signals. Independent scrambles
estimate sampling uncertainty; spatial refinement adds spins without changing signals.
"""

import numpy as np
from scipy.stats import t

from .s1_anatomy import anatomical_reference as analytic_reference
from .s1_anatomy import sampled_anatomy as sampled_phantom

__all__ = ["analytic_reference", "sampled_phantom"]


def quadratic(values, *, bias_correct=True):
    """Unbiased squared norm of the integral, and delete-one estimates.

    Each row is an independent randomized quadrature, not an ADC sample. Cross
    products remove the positive image-energy bias caused by quadrature noise.
    """
    flat = np.asarray(values).reshape(len(values), -1)
    n = len(flat)
    summed = flat.sum(axis=0)
    norms = np.sum(np.abs(flat) ** 2, axis=1)
    total = float(np.vdot(summed, summed).real)
    if not bias_correct:
        return total / n**2, (total - 2 * np.real(flat.conj() @ summed) + norms) / (
            n - 1
        ) ** 2
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


def sampling_image_nrmse(images, reference):
    """Estimated L2 sampling error of the displayed mean, relative to target norm."""
    n = len(images)
    variance = np.sum(np.abs(images - images.mean(axis=0)) ** 2) / (n * (n - 1))
    return float(np.sqrt(variance) / max(np.linalg.norm(reference), 1e-30))


def image_scores(images, reference, support, *, bias_correct=True):
    denom = max(float(np.linalg.norm(reference) ** 2), 1e-30)
    error, error_loo = quadratic(images - reference, bias_correct=bias_correct)
    energy, energy_loo = quadratic(images, bias_correct=bias_correct)
    outside, outside_loo = quadratic(images[:, ~support], bias_correct=bias_correct)
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
