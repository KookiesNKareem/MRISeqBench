"""Single-transmit, homogeneous-sphere benchmark SAR. See README.md for assumptions and sources."""

import numpy as np

from ..models import SARModel


def evaluate_sar(submission, hardware):
    """Judge-only submission checks, never included in agent feedback."""
    from pypulseq import Opts, Sequence

    try:
        seq = Sequence(Opts(B0=hardware["field_strength_T"]))
        seq.read(str(submission))
        # Preflight validates the file rasters first; the judge owns integration.
        seq.rf_raster_time = hardware["rf_raster_us"] * 1e-6
        return sar_checks(seq, hardware)
    except Exception as exc:  # noqa: BLE001 -- unsupported RF must fail closed
        return [
            {
                "name": "sar_model",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]


def rf_intervals(rf, raster_s, gamma_hz_per_T):
    """Return RF sample-hold edges and B1+ squared in uT².

    Pulseq's regular raster uses sample centres. Constant two-point block
    pulses also have an exact integral. Other explicit time shapes are rejected
    rather than assigning them an unverified interpolation convention.
    """
    signal = np.asarray(rf.signal)
    times = np.asarray(rf.t, dtype=float)
    duration = float(rf.shape_dur)
    if (
        signal.ndim != 1
        or signal.size == 0
        or times.shape != signal.shape
        or not np.isfinite(signal).all()
        or not np.isfinite(times).all()
        or not np.isfinite(duration)
        or duration <= 0
        or times[0] < 0
        or times[-1] > duration
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Invalid RF waveform for SAR integration")
    power = np.abs(signal / gamma_hz_per_T * 1e6) ** 2
    if not np.isfinite(power).all():
        raise ValueError("Non-finite RF power")
    if np.all(signal == signal[0]):
        return np.array([0.0, duration]), power[:1]
    expected = (np.arange(signal.size) + 0.5) * raster_s
    if not (
        np.allclose(times, expected, rtol=0, atol=1e-12)
        and np.isclose(duration, signal.size * raster_s, rtol=0, atol=1e-12)
    ):
        raise ValueError("SAR model requires regular-raster RF or constant RF")
    # Merge equal-power samples, preserving the exact sample-hold integral.
    changes = np.flatnonzero(np.diff(power) != 0) + 1
    indices = np.concatenate(([0], changes, [signal.size]))
    return indices * raster_s, power[indices[:-1]]


def peak_window(edges, power, window_s):
    """Exact maximum integral for piecewise-constant power, with no binning.

    For short acquisitions use the actual duration, without zero padding or
    implicit repetition. A maximum starts or ends at a power discontinuity.
    """
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


def sar_checks(seq, hardware):
    """Return required SAR checks using only the judge's hardware contract."""
    model = SARModel.model_validate(hardware["sar"])
    field = float(hardware["field_strength_T"])
    gamma = float(seq.system.gamma)
    if not np.isfinite(field) or field <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("SAR requires finite positive field strength and gamma")
    # Circularly polarized field amplitude is the Pulseq B1+ amplitude.
    coefficient = (
        model.conductivity_S_per_m
        * (2 * np.pi * gamma * field) ** 2
        * model.radius_m**2
        / (5 * model.density_kg_per_m3)
        * 1e-12
    )
    edges, powers = [0.0], []
    clock = 0.0
    cache = {}
    for number, event_ids in seq.block_events.items():
        block_duration = float(seq.block_durations[number])
        if not np.isfinite(block_duration) or block_duration <= 0:
            raise ValueError("SAR requires finite positive block durations")
        end = clock + block_duration
        rf_id = int(event_ids[1])
        if rf_id:
            if rf_id not in cache:
                rf = seq.get_block(number).rf
                relative, power = rf_intervals(rf, seq.rf_raster_time, gamma)
                cache[rf_id] = (float(rf.delay), relative, power)
            delay, relative, power = cache[rf_id]
            absolute = clock + delay + relative
            if delay < 0 or absolute[-1] > end + 1e-12:
                raise ValueError("RF extends outside its block")
            absolute[-1] = min(absolute[-1], end)
            if absolute[0] > edges[-1]:
                edges.append(float(absolute[0]))
                powers.append(0.0)
            edges.extend(absolute[1:].tolist())
            powers.extend(power.tolist())
        if end > edges[-1]:
            edges.append(end)
            powers.append(0.0)
        clock = end
    edges, powers = np.asarray(edges), np.asarray(powers)
    if (
        edges.size < 2
        or not np.isfinite(edges).all()
        or (np.diff(edges) <= 0).any()
        or not np.isfinite(coefficient)
    ):
        raise ValueError("Invalid SAR integration timeline")
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
