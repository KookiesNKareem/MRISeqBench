"""Pulseq hardware and timing checks."""

import numpy as np


def preflight(submission, case, *, hardware_limits=True):
    """Check file/timing validity; optionally also enforce public hardware limits."""
    from pypulseq import Opts, Sequence

    hw = case["hardware"]
    system = Opts(
        B0=hw["field_strength_T"],
        max_grad=hw["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hw["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hw["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hw["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hw["adc_dead_time_us"] * 1e-6,
        grad_raster_time=hw["gradient_raster_us"] * 1e-6,
        rf_raster_time=hw["rf_raster_us"] * 1e-6,
        adc_raster_time=hw["adc_raster_us"] * 1e-6,
        block_duration_raster=hw["block_raster_us"] * 1e-6,
    )
    seq = Sequence(system)
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    try:
        seq.read(str(submission))
        for name in (
            "grad_raster_time",
            "rf_raster_time",
            "adc_raster_time",
            "block_duration_raster",
        ):
            check(
                name,
                np.isclose(
                    getattr(seq, name), getattr(system, name), rtol=1e-9, atol=1e-12
                ),
                "Submitted raster must match the hardware profile",
            )
        # File headers cannot relax the task's raster or hardware limits.
        seq.grad_raster_time = system.grad_raster_time
        seq.rf_raster_time = system.rf_raster_time
        seq.adc_raster_time = system.adc_raster_time
        seq.block_duration_raster = system.block_duration_raster
        check("pulseq_parse", True, "Pulseq file parsed")
        timing_ok, errors = seq.check_timing()
        check("timing", timing_ok, str(errors[:10]))
        duration = float(seq.duration()[0])
        check(
            "duration",
            np.isfinite(duration)
            and 0 < duration <= case["task"]["sequence"]["duration_s"],
            {
                "duration_s": duration if np.isfinite(duration) else None,
                "max_s": case["task"]["sequence"]["duration_s"],
            },
        )
        adc_count, rf_count, rf_peak_hz = 0, 0, 0.0
        finite_events = True
        unsupported_events = []
        for number in seq.block_events:
            block = seq.get_block(number)
            if block.adc is not None:
                adc_count += block.adc.num_samples
            if block.rf is not None:
                rf_count += 1
                finite_events &= bool(np.isfinite(block.rf.signal).all())
                rf_peak_hz = max(rf_peak_hz, float(np.max(np.abs(block.rf.signal))))
            # These extensions require a scanner/platform model, absent here.
            for name in ("soft_delay", "rf_shim"):
                if getattr(block, name, None) is not None:
                    unsupported_events.append(name)
        check(
            "supported_events", not unsupported_events, sorted(set(unsupported_events))
        )
        check(
            "has_rf_and_adc",
            adc_count > 0 and rf_count > 0,
            {"adc_samples": adc_count, "rf_events": rf_count},
        )
        check("finite_rf", finite_events, "RF samples must be finite")
        b1 = rf_peak_hz / system.gamma * 1e6
        if hardware_limits:
            check(
                "peak_b1",
                np.isfinite(b1) and b1 <= hw["max_b1_uT"] * (1 + 1e-6),
                {"peak_uT": b1 if np.isfinite(b1) else None, "max_uT": hw["max_b1_uT"]},
            )
        for axis, waveform in zip("xyz", seq.waveforms()):
            if waveform.size == 0:
                continue
            t, g = waveform
            valid = np.isfinite(waveform).all() and (np.diff(t) > 0).all()
            check(
                f"finite_gradient_{axis}", valid, "finite samples with increasing time"
            )
            if not valid or not hardware_limits:
                continue
            peak = float(np.max(np.abs(g)) / system.gamma * 1e3)
            slew = (
                float(np.max(np.abs(np.diff(g) / np.diff(t))) / system.gamma)
                if len(t) > 1
                else 0.0
            )
            check(
                f"gradient_{axis}",
                peak <= hw["max_gradient_mT_per_m"] * (1 + 1e-6),
                {"peak_mT_per_m": peak, "max": hw["max_gradient_mT_per_m"]},
            )
            check(
                f"slew_{axis}",
                slew <= hw["max_slew_T_per_m_per_s"] * (1 + 1e-6),
                {"peak_T_per_m_per_s": slew, "max": hw["max_slew_T_per_m_per_s"]},
            )
    except Exception as exc:  # noqa: BLE001 -- malformed external sequences become failed checks
        check("pulseq_preflight", False, f"{type(exc).__name__}: {exc}")
    return checks
