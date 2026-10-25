"""Sparse linear vertices for rectangular binomial RF subpulses."""

import numpy as np
import pypulseq as pp


def sparse_binomial_rf(angle, order, fat_hz, system, use, *, centre_raster=None):
    """Preserve subpulse area/centre with symmetric one-raster edge ramps."""
    if order not in {1, 2}:
        raise ValueError("Binomial RF requires order 1 or 2")
    raster = system.rf_raster_time
    width = round(0.0004 / raster) * raster
    lag = round(1 / (2 * abs(fat_hz)) / raster) * raster
    coefficients = [1, 1] if order == 1 else [1, 2, 1]
    quantum = centre_raster or system.block_duration_raster
    original_samples = round((order * lag + width) / raster)
    padding = (-original_samples) % round(quantum / raster)
    lead = (padding // 2) * raster + system.block_duration_raster
    duration = (original_samples + padding) * raster + 2 * system.block_duration_raster
    amplitude = angle / (2 * np.pi * width * sum(coefficients))
    times, values = [0.0], [0.0]
    for index, coefficient in enumerate(coefficients):
        start, end = lead + index * lag, lead + index * lag + width
        times.extend(
            [start - raster / 2, start + raster / 2, end - raster / 2, end + raster / 2]
        )
        values.extend([0.0, coefficient * amplitude, coefficient * amplitude, 0.0])
    times.append(duration)
    values.append(0.0)
    rf = pp.make_arbitrary_rf(
        np.asarray(values),
        angle,
        no_signal_scaling=True,
        delay=system.rf_dead_time,
        system=system,
        use=use,
    )
    rf.t = np.asarray(times)
    rf.shape_dur = duration
    rf.center = duration / 2
    return rf
