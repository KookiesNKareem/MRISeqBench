"""Validated Pulseq RF: legacy raster holds and explicit complex-linear vertices.

Explicit time shapes follow KomaMRIFiles.get_RF: support begins at the first
declared vertex and ends at the last. Nothing is extrapolated before/after it.
Existing constant/default-raster waveforms retain their exact old convention.
"""

import numpy as np


def rf_waveform(rf, raster):
    y = np.asarray(rf.signal, dtype=complex)
    t = np.asarray(rf.t, dtype=float)
    duration = float(rf.shape_dur)
    if (
        y.ndim != 1
        or not y.size
        or t.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(t).all()
        or not np.isfinite(duration)
        or duration <= 0
        or not np.isfinite(raster)
        or raster <= 0
        or t[0] < 0
        or t[-1] > duration + 1e-12
        or (np.diff(t) <= 0).any()
    ):
        raise ValueError(
            "Invalid RF vertices: require finite signal, increasing times within positive shape duration"
        )
    if np.all(y == y[0]):
        return np.array([0.0, duration]), y[:1], False
    expected = (np.arange(len(y)) + 0.5) * raster
    if np.allclose(t, expected, rtol=0, atol=1e-12) and np.isclose(
        duration, len(y) * raster, rtol=0, atol=1e-12
    ):
        change = np.r_[0, np.flatnonzero(np.diff(y) != 0) + 1, len(y)]
        return change * raster, y[change[:-1]], False
    return t, y, True


def rf_samples(rf, raster, *, max_step=0.25e-6):
    """Midpoint propagation of explicit linear RF; old holds remain exact.

    Each original segment is subdivided independently (no shifted endpoints).
    This is only the structural/RF-assay quadrature. Native simulation always
    reads the original, unchanged Pulseq waveform.
    """
    edges, y, linear = rf_waveform(rf, raster)
    if not linear:
        return edges, y
    counts = np.maximum(1, np.ceil(np.diff(edges) / max_step).astype(int))
    if counts.sum() > 4_000_000:
        raise ValueError("Explicit RF quadrature exceeds 4000000 intervals")
    segment = np.repeat(np.arange(len(counts)), counts)
    offset = np.arange(counts.sum()) - np.repeat(
        np.r_[0, np.cumsum(counts)[:-1]], counts
    )
    step = np.diff(edges)[segment] / counts[segment]
    left = edges[segment] + offset * step
    values = y[segment] + (y[segment + 1] - y[segment]) * (
        (offset + 0.5) / counts[segment]
    )
    return np.r_[left, edges[-1]], values
