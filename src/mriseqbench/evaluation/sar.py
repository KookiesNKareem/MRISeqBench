"""Quadrature (CP) body-coil, homogeneous-sphere benchmark SAR. See README.md for assumptions and sources."""

import numpy as np

from ..models import SARModel
from .preflight import MALFORMED_SEQUENCE
from .rf_waveform import rf_waveform


def evaluate_sar(submission, hardware):
    """Judge-only submission checks, never included in agent feedback."""
    from pypulseq import Opts, Sequence

    try:
        seq = Sequence(Opts(B0=hardware["field_strength_T"]))
        seq.read(str(submission))
        # Preflight validates the file rasters first; the judge owns integration.
        seq.rf_raster_time = hardware["rf_raster_us"] * 1e-6
        return sar_checks(seq, hardware)
    except MALFORMED_SEQUENCE as exc:  # Unsupported RF fails closed.
        return [
            {
                "name": "sar_model",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]


def rf_intervals(rf, raster_s, gamma_hz_per_T):
    """Return hold power or exact quadratic power of complex-linear RF vertices.

    Quadratic coefficients are in ascending powers of time since each left edge.
    """
    edges, signal, linear = rf_waveform(rf, raster_s)
    if not np.isfinite(gamma_hz_per_T) or gamma_hz_per_T <= 0:
        raise ValueError("RF power requires finite positive gamma")
    signal = signal / gamma_hz_per_T * 1e6
    if linear:
        slope = np.diff(signal) / np.diff(edges)
        return edges, np.column_stack(
            (
                np.abs(signal[:-1]) ** 2,
                2 * np.real(signal[:-1] * np.conj(slope)),
                np.abs(slope) ** 2,
            )
        )
    power = np.abs(signal) ** 2
    changes = np.r_[0, np.flatnonzero(np.diff(power) != 0) + 1, len(power)]
    return edges[changes], power[changes[:-1]]


def peak_window(edges, power, window_s):
    """Exact maximum integral for piecewise-constant power, with no binning.

    For short acquisitions use the actual duration, without zero padding or
    implicit repetition. A maximum starts or ends at a power discontinuity.
    """
    if np.ndim(power) == 2:
        return _peak_linear_rf_window(edges, power, window_s)
    width = min(window_s, edges[-1])
    energy = np.concatenate(([0.0], np.cumsum(np.diff(edges) * power)))
    starts = np.unique(
        np.clip(np.concatenate((edges, edges - width)), 0, edges[-1] - width)
    )
    averages = (
        np.interp(starts + width, edges, energy) - np.interp(starts, edges, energy)
    ) / width
    index = int(np.argmax(averages))
    return float(averages[index]), float(starts[index]), float(width)


def _peak_linear_rf_window(edges, coefficients, window_s):
    """Exact polynomial integration and all stationary sliding-window starts.

    A window maximum may be between vertices. Its derivative is P(t+w)-P(t),
    quadratic between the union of original and shifted segment boundaries.
    """
    width = min(window_s, edges[-1])
    spans = np.diff(edges)
    c = coefficients
    cumulative = np.r_[
        0, np.cumsum(c[:, 0] * spans + c[:, 1] * spans**2 / 2 + c[:, 2] * spans**3 / 3)
    ]

    def integral(t):
        i = np.minimum(np.searchsorted(edges, t, side="right") - 1, len(c) - 1)
        x = t - edges[i]
        return cumulative[i] + x * (c[i, 0] + x * (c[i, 1] / 2 + x * c[i, 2] / 3))

    boundaries = np.unique(np.clip(np.r_[edges, edges - width], 0, edges[-1] - width))
    if len(boundaries) == 1:
        return float(cumulative[-1] / width), 0.0, float(width)
    left, span = boundaries[:-1], np.diff(boundaries)
    mid = left + span / 2
    i = np.minimum(np.searchsorted(edges, mid, side="right") - 1, len(c) - 1)
    j = np.minimum(np.searchsorted(edges, mid + width, side="right") - 1, len(c) - 1)
    x, z = left - edges[i], left + width - edges[j]
    # Normalize each interval to [0,1] for stable quadratic roots.
    a = (c[j, 2] - c[i, 2]) * span**2
    b = (c[j, 1] + 2 * c[j, 2] * z - c[i, 1] - 2 * c[i, 2] * x) * span
    d = c[j, 0] + z * (c[j, 1] + z * c[j, 2]) - c[i, 0] - x * (c[i, 1] + x * c[i, 2])
    roots = []
    linear = (a == 0) & (b != 0)
    roots.append((np.flatnonzero(linear), -d[linear] / b[linear]))
    discriminant = b * b - 4 * a * d
    quadratic = (a != 0) & (discriminant >= 0)
    indices = np.flatnonzero(quadratic)
    # Stable quadratic formula, including a repeated zero root.
    q = -0.5 * (b[indices] + np.copysign(np.sqrt(discriminant[indices]), b[indices]))
    roots.append((indices, q / a[indices]))
    nonzero = q != 0
    roots.append((indices[nonzero], d[indices[nonzero]] / q[nonzero]))
    starts = [boundaries]
    for indices, root in roots:
        keep = (root > 0) & (root < 1)
        starts.append(left[indices[keep]] + root[keep] * span[indices[keep]])
    starts = np.concatenate(starts)
    averages = (integral(starts + width) - integral(starts)) / width
    index = int(np.argmax(averages))
    return float(averages[index]), float(starts[index]), float(width)


def sar_checks(seq, hardware):
    """Return required SAR checks using only the judge's hardware contract."""
    model = SARModel.model_validate(hardware["sar"])
    field = float(hardware["field_strength_T"])
    gamma = float(seq.system.gamma)
    if not np.isfinite(field) or field <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("SAR requires finite positive field strength and gamma")
    # Quadrature (CP) body coil: the Pulseq amplitude is the co-rotating B1+;
    # the counter-rotating B1- component is zero, halving linear-coil SAR.
    coefficient = (
        model.conductivity_S_per_m
        * (2 * np.pi * gamma * field) ** 2
        * model.radius_m**2
        / (10 * model.density_kg_per_m3)
        * 1e-12
    )
    edges, powers = [0.0], []
    clock = 0.0
    cache = {}
    for number, event_ids in seq.block_events.items():
        block_duration = float(seq.block_durations[number])
        if not np.isfinite(block_duration) or block_duration < 0:
            raise ValueError(
                f"Block {number}: invalid duration {block_duration!r} s; require finite nonnegative duration"
            )
        if block_duration == 0:
            if any(event_ids[1:6]):
                raise ValueError(
                    f"Block {number}: zero duration contains RF, gradient or ADC events"
                )
            continue
        end = clock + block_duration
        rf_id = int(event_ids[1])
        if rf_id:
            if rf_id not in cache:
                rf = seq.get_block(number).rf
                try:
                    relative, power = rf_intervals(rf, seq.rf_raster_time, gamma)
                except ValueError as exc:
                    raise ValueError(f"Block {number}, RF {rf_id}: {exc}") from exc
                cache[rf_id] = (float(rf.delay), relative, power)
            delay, relative, power = cache[rf_id]
            absolute = clock + delay + relative
            if delay < 0 or absolute[-1] > end + 1e-12:
                raise ValueError(
                    f"Block {number}, RF {rf_id}: RF support [{absolute[0]}, {absolute[-1]}] s extends outside block [{clock}, {end}] s"
                )
            absolute[-1] = min(absolute[-1], end)
            if absolute[0] > edges[-1]:
                edges.append(float(absolute[0]))
                powers.append([0.0, 0.0, 0.0])
            edges.extend(absolute[1:].tolist())
            if power.ndim == 1:
                power = np.column_stack((power, np.zeros((len(power), 2))))
            powers.extend(power.tolist())
        if end > edges[-1]:
            edges.append(end)
            powers.append([0.0, 0.0, 0.0])
        clock = end
    edges, powers = np.asarray(edges), np.asarray(powers)
    if (
        edges.size < 2
        or not np.isfinite(edges).all()
        or (np.diff(edges) <= 0).any()
        or not np.isfinite(powers).all()
        or not np.isfinite(coefficient)
    ):
        raise ValueError("Invalid SAR integration timeline")
    if not np.any(powers[:, 1:]):
        powers = powers[:, 0]
    checks = []
    for window, multiplier in ((360, 1), (10, 2)):
        mean_power, start, width = peak_window(edges, powers, window)
        value = coefficient * mean_power
        limit = model.whole_body_limit_W_per_kg * multiplier
        checks.append(
            {
                "name": f"sar_{window}s",
                "passed": bool(np.isfinite(value) and value <= limit * (1 + 1e-6)),
                "detail": {
                    "model": model.model,
                    "field_strength_T": field,
                    "peak_W_per_kg": value,
                    "limit_W_per_kg": limit,
                    "window_start_s": start,
                    "averaging_duration_s": width,
                    "b1_rms_uT": float(np.sqrt(mean_power)),
                    "coefficient_W_per_kg_per_uT2": float(coefficient),
                    "load": model.model_dump(mode="json"),
                    "scope": "synthetic global SAR; no local SAR or scanner certification",
                },
            }
        )
    return checks
