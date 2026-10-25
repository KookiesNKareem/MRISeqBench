"""Validated Pulseq RF: legacy raster holds and explicit complex-linear vertices.

Explicit time shapes follow KomaMRIFiles.get_RF: support begins at the first
declared vertex and ends at the last. Nothing is extrapolated before/after it.
Existing constant/default-raster waveforms retain their exact old convention.
"""

import numpy as np


class RFVertexError(ValueError):
    """Submitted RF vertices do not define a waveform."""


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
        raise RFVertexError(
            "Invalid RF vertices: require finite signal, increasing times within positive shape duration"
        )
    expected = (np.arange(len(y)) + 0.5) * raster
    default_raster = np.allclose(t, expected, rtol=0, atol=1e-12)
    if np.all(y == y[0]):
        # Default-raster holds fill the shape duration; explicit time shapes
        # keep their declared support [t0, t_end], as KomaMRI decodes them.
        support = [0.0, duration] if default_raster or len(y) < 2 else [t[0], t[-1]]
        return np.array(support), y[:1], False
    if default_raster and np.isclose(duration, len(y) * raster, rtol=0, atol=1e-12):
        change = np.r_[0, np.flatnonzero(np.diff(y) != 0) + 1, len(y)]
        return change * raster, y[change[:-1]], False
    return t, y, True


def _interval_counts(edges, y, max_step):
    counts = np.maximum(1, np.ceil(np.diff(edges) / max_step).astype(int))
    # Constant complex RF rotates about one fixed axis exactly over the interval.
    counts[y[1:] == y[:-1]] = 1
    return counts


def quadrature_intervals(rf, raster, *, max_step=0.25e-6):
    """Number of structural quadrature intervals rf_samples would expand to."""
    edges, y, linear = rf_waveform(rf, raster)
    if not linear:
        return len(y)
    return int(_interval_counts(edges, y, max_step).sum())


class RFQuadratureLimit(ValueError):
    """A submitted RF waveform needs more structural quadrature than allowed."""


def rf_samples(rf, raster, *, max_step=0.25e-6):
    """Midpoint propagation of explicit linear RF; old holds remain exact.

    Each original segment is subdivided independently (no shifted endpoints).
    This is only the structural/RF-assay quadrature. Native simulation always
    reads the original, unchanged Pulseq waveform.
    """
    edges, y, linear = rf_waveform(rf, raster)
    if not linear:
        return edges, y
    counts = _interval_counts(edges, y, max_step)
    if counts.sum() > 4_000_000:
        raise RFQuadratureLimit("Explicit RF quadrature exceeds 4000000 intervals")
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
