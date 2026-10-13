"""Disclosed, deterministic mechanism probes, independent of object composition."""

from copy import copy

import numpy as np

from .s1_image import canonical_response, nrmse
from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_sequence import check


def radical_inverse(n, base):
    value, factor = 0.0, 1.0 / base
    while n:
        value += (n % base) * factor
        n //= base
        factor /= base
    return value


def echo_formation_probe(acq, case, probe_props):
    """Locate the actual gradient-rephased echo with KomaMRI, not an RF label."""
    ids = acq.lines[0]
    axis = int(np.argmax(np.ptp(acq.k[ids], axis=0)))
    exc = [p for p in acq.pulses if not p.refocusing][acq.excitation[0]]
    echo = exc.center + acq.te[0]
    span = min(echo - acq.adc[ids[0]], acq.adc[ids[-1]] - echo)
    count = int(np.floor(span / 10e-6))
    times = np.round(echo / 1e-6) * 1e-6 + np.arange(-count, count + 1) * 10e-6
    if count < 2:
        return check(
            "echo_formation", False, "Insufficient ADC window around the rephasing time"
        ), {}
    a = copy(acq)
    a.pulses = [p for p in acq.pulses if exc.start <= p.start < times[-1]]
    a.adc, a.adc_phase, a.k = times, np.zeros(len(times)), np.zeros((len(times), 3))
    points = np.zeros((257, 3))
    points[:, axis] = (
        np.linspace(-0.5, 0.5, 257) * case["task"]["sequence"]["fov_mm"][axis] / 1000
    )
    spins = Spins(
        points,
        np.zeros(257, dtype=np.int64),
        ["diagnostic"],
        probe_props,
        1.0 / 257,
        (257,),
        "echo-formation",
    )
    signal = simulate(a, spins)
    baseline = simulate(a, spins, zero_gradients=True)
    envelope = np.abs(signal) / np.maximum(np.abs(baseline), 1e-15)
    peak = int(np.argmax(envelope))
    error = abs(times[peak] - echo)
    tolerance = max(20e-6, float(np.min(np.diff(acq.adc[ids]))) / 2)
    # A genuine echo has a peak in the acquired window, not a monotone FID tail.
    formed = 0 < peak < len(times) - 1 and error <= tolerance
    detail = {
        "predicted_echo_time_s": float(echo),
        "measured_echo_peak_time_s": float(times[peak]),
        "peak_time_error_s": float(error),
        "tolerance_s": tolerance,
        "peak_retention": float(envelope[peak]),
        "engine": "KomaMRICore",
    }
    return check("echo_formation", formed, detail), {
        "echo_peak_time_error_s": float(error)
    }


def mechanism_probes(acq, case, materials, properties):
    fam = case["task"]["id"].removesuffix("_research")
    center_ids = np.array(
        [
            ids[
                np.argmin(
                    np.abs(acq.k[ids, int(np.argmax(np.ptp(acq.k[ids], axis=0)))])
                )
            ]
            for ids in acq.lines
        ]
    )
    diagnostic = canonical_response(
        acq,
        properties,
        fam,
        finite_rf=case["evaluation"].get("intrinsic_tissue_response", False),
    )
    metrics, checks = {}, []
    if "wm" in materials and "gm" in materials:
        weighting_response = diagnostic
        wm, gm = materials.index("wm"), materials.index("gm")
    else:
        # Weighting is a disclosed common diagnostic independent of anatomy.
        weighting_response = canonical_response(
            acq, np.array([[0.7, 0.8, 0.08], [0.8, 1.3, 0.1]]), fam
        )
        wm, gm = 0, 1
    # One disclosed common diagnostic material, not a hidden task timing target.
    probe_props = np.array([[1.0, 0.8, 0.08]])
    echo_check, echo_metrics = echo_formation_probe(acq, case, probe_props)
    checks.append(echo_check)
    metrics.update(echo_metrics)
    ideal = canonical_response(acq, probe_props, fam)[0]
    metrics["diagnostic_wm_gm_ratio"] = float(
        np.mean(np.abs(weighting_response[wm, center_ids]))
        / max(np.mean(np.abs(weighting_response[gm, center_ids])), 1e-15)
    )
    mean_signal = float(np.mean(np.abs(ideal[center_ids])))
    metrics["diagnostic_signal_fraction"] = mean_signal
    checks.append(
        check(
            "usable_echo_signal",
            mean_signal >= 0.01,
            {"mean_echo_signal": mean_signal, "min": 0.01},
        )
    )
    for axis, name in ((1, "t1"), (2, "t2")):
        varied = probe_props.copy()
        varied[:, axis] *= 1.1
        altered = canonical_response(acq, varied, fam)[0]
        metrics[f"diagnostic_{name}_sensitivity"] = float(
            np.mean(np.abs(np.abs(altered[center_ids]) - np.abs(ideal[center_ids])))
            / max(0.1 * mean_signal, 1e-15)
        )
    if fam not in {"bssfp", "spin_echo_t2w"}:
        width = np.r_[
            np.asarray(case["task"]["sequence"]["fov_mm"])
            / np.asarray(case["task"]["sequence"]["matrix"])
            / 1000,
            0.0,
        ]
        width = width[:3]
        width[width == 0] = width[0]
        errors = []
        for count in (1021, 2053):
            points = (
                np.array(
                    [
                        [radical_inverse(i, b) - 0.5 for b in (2, 3, 5)]
                        for i in range(1, count + 1)
                    ]
                )
                * width
            )
            spins = Spins(
                points,
                np.zeros(count, dtype=np.int64),
                ["diagnostic"],
                probe_props,
                1.0 / count,
                (count,),
                "diagnostic",
            )
            centered = copy(acq)
            centered.adc, centered.adc_phase, centered.k = (
                acq.adc[center_ids],
                acq.adc_phase[center_ids],
                acq.k[center_ids],
            )
            actual = simulate(centered, spins, demodulate_encoding=True)
            errors.append(nrmse(actual, ideal[center_ids]))
        metrics["spoiling_relative_error"] = float(max(errors))
        checks.append(
            check(
                "spoiled_gre",
                max(errors) <= 0.2,
                {
                    "relative_errors": errors,
                    "max": 0.2,
                    "ensemble": "1021/2053 Halton isochromats across one voxel, including z; encoding phase removed",
                },
            )
        )
    if fam in {"gre_t2star", "spin_echo_t2w"}:
        # Isolate the first acquired echo to avoid confounding dephasing with steady state.
        a = copy(acq)
        idx = center_ids[0]
        excitation = [p for p in acq.pulses if not p.refocusing][acq.excitation[0]]
        a.pulses = [p for p in acq.pulses if excitation.start <= p.start < acq.adc[idx]]
        a.adc, a.adc_phase, a.k = (
            acq.adc[idx : idx + 1],
            acq.adc_phase[idx : idx + 1],
            acq.k[idx : idx + 1],
        )
        offsets = np.linspace(-50.0, 50.0, 65)
        spins = Spins(
            np.zeros((65, 3)),
            np.zeros(65, dtype=np.int64),
            ["diagnostic"],
            probe_props,
            1.0 / 65,
            (65,),
            "diagnostic",
        )
        dispersed = simulate(a, spins, zero_gradients=True, frequency_offsets=offsets)
        baseline = simulate(a, spins, zero_gradients=True)
        retention = float(abs(dispersed[0]) / max(abs(baseline[0]), 1e-15))
        metrics["off_resonance_ensemble_retention"] = retention
        if fam == "spin_echo_t2w":
            checks.append(
                check(
                    "refocusing_response",
                    retention >= 0.9,
                    {
                        "retention": retention,
                        "min": 0.9,
                        "frequency_range_hz": [-50, 50],
                    },
                )
            )
    if fam == "gre_t1w":
        checks.append(
            check(
                "t1_weighting",
                metrics["diagnostic_wm_gm_ratio"] >= 1.15
                and metrics["diagnostic_t1_sensitivity"]
                > metrics["diagnostic_t2_sensitivity"],
                {k: v for k, v in metrics.items() if "diagnostic" in k},
            )
        )
    if case["task"].get("prompt_mode") == "research":
        suitable = mean_signal >= 0.01
        if fam == "gre_t2star":
            suitable &= metrics["off_resonance_ensemble_retention"] <= 0.9
        if fam == "spin_echo_t2w":
            suitable &= (
                metrics["diagnostic_t2_sensitivity"]
                > metrics["diagnostic_t1_sensitivity"]
            )
        checks.append(
            check(
                "reasonable_parameters",
                suitable,
                {"signal_min": 0.01, "diagnostics": metrics},
            )
        )
    return checks, metrics, diagnostic
