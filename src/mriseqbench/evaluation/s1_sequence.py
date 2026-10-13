"""Waveform-derived S1 structure. RF use labels and file definitions are not proof."""

from dataclasses import dataclass

import numpy as np
from pypulseq import Opts, Sequence

FAMILIES = {
    "gre_t1w",
    "gre_t2star",
    "spin_echo_t2w",
    "bssfp",
    "rich_geometry_gre",
    "volume_gre",
}


class MissingExcitation(ValueError):
    """A candidate has RF but no waveform-derived excitation."""


def family(case):
    return case["task"]["id"].removesuffix("_research")


def check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def rf_samples(rf, raster):
    """Exact sample-hold convention already enforced by the SAR integrator."""
    t = np.asarray(rf.t)
    y = np.asarray(rf.signal, dtype=complex)
    if np.all(y == y[0]):
        return np.array([0.0, float(rf.shape_dur)]), y[:1]
    expected = (np.arange(len(y)) + 0.5) * raster
    if not np.allclose(t, expected, rtol=0, atol=1e-12):
        raise ValueError("S1 requires regular-raster or constant RF")
    change = np.r_[0, np.flatnonzero(np.diff(y) != 0) + 1, len(y)]
    return change * raster, y[change[:-1]]


def rotate(m, axis):
    angle = np.linalg.norm(axis)
    if angle == 0:
        return m
    u = axis / angle
    # Bloch convention: dM/dt = gamma M cross B.
    return (
        m * np.cos(angle)
        - np.cross(u, m) * np.sin(angle)
        + u * np.dot(u, m) * (1 - np.cos(angle))
    )


@dataclass
class Pulse:
    start: float
    end: float
    center: float
    edges: np.ndarray
    signal: np.ndarray
    frequency: float
    phase: float
    flip: float
    refocusing: bool
    phase_correction: float = 0.0
    block_start: float = 0.0


@dataclass
class Acquisition:
    seq: object
    pulses: list
    adc: np.ndarray
    adc_phase: np.ndarray
    lines: list
    k: np.ndarray
    indices: np.ndarray
    offset: np.ndarray
    excitation: np.ndarray
    te: np.ndarray
    tr: np.ndarray
    flips: np.ndarray
    checks: list
    measurements: dict
    source_path: str | None = None


def gradient_integrals(seq, times):
    """Physical integrated gradient area in cycles/m, without RF bookkeeping."""
    out = np.zeros((len(times), 3))
    for axis, grad in enumerate(seq.get_gradients()):
        if grad is not None:
            integral = grad.antiderivative()
            q = np.clip(times, integral.x[0], integral.x[-1])
            out[:, axis] = integral(q) - integral(
                np.clip(0.0, integral.x[0], integral.x[-1])
            )
    return out


def inspect_sequence(path, case):
    spec = case["task"]["sequence"]
    seq = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    seq.read(str(path))
    pulses, lines, adc, phases = [], [], [], []
    clock = 0.0
    for number in seq.block_events:
        block = seq.get_block(number)
        if block.rf is not None:
            rf = block.rf
            edges, values = rf_samples(rf, seq.rf_raster_time)
            freq = float(
                rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            phase = float(
                rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            mids, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
            # Nominal rotation comes from the envelope, not the supplied use/center.
            m = np.array([0.0, 0.0, 1.0])
            for y, dt in zip(values, widths, strict=True):
                m = rotate(m, 2 * np.pi * dt * np.array([y.real, y.imag, 0.0]))
            flip = float(np.rad2deg(np.arccos(np.clip(m[2], -1, 1))))
            center = float(
                np.sum(mids * np.abs(values) * widths)
                / max(np.sum(np.abs(values) * widths), 1e-30)
            )
            start = clock + float(rf.delay)
            pulses.append(
                Pulse(
                    start,
                    start + edges[-1],
                    start + center,
                    edges + start,
                    values,
                    freq,
                    phase,
                    flip,
                    flip >= 135,
                    phase - float(rf.phase_offset),
                    clock,
                )
            )
        if block.adc is not None:
            a = block.adc
            local = (np.arange(a.num_samples) + 0.5) * a.dwell
            t = clock + a.delay + local
            freq = a.freq_offset + a.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            phase = (
                a.phase_offset + a.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            extra = getattr(a, "phase_modulation", None)
            phase = (
                phase
                + 2 * np.pi * freq * local
                + (np.asarray(extra) if extra is not None and np.size(extra) else 0.0)
            )
            begin = len(adc)
            adc.extend(t)
            phases.extend(np.broadcast_to(phase, t.shape))
            lines.append(np.arange(begin, len(adc)))
        clock += seq.block_durations[number]
    adc, phases = np.asarray(adc), np.asarray(phases)
    if not pulses or not lines:
        raise ValueError("S1 needs RF and acquired readout lines")
    pt = np.array([p.center for p in pulses])
    moments = gradient_integrals(seq, np.r_[pt, adc])
    pm, am = moments[: len(pt)], moments[len(pt) :]
    delta, baseline = np.zeros(3), []
    for p, moment in zip(pulses, pm, strict=True):
        delta = -2 * moment - delta if p.refocusing else -moment
        baseline.append(delta.copy())
    preceding = np.searchsorted(pt, adc, side="right") - 1
    k = am + np.asarray(baseline)[np.maximum(preceding, 0)]
    exc = [p for p in pulses if not p.refocusing]
    if not exc:
        raise MissingExcitation("No waveform-derived excitation before acquisition")
    exct = np.array([p.center for p in exc])
    exc_end = np.array([p.end for p in exc])
    centers, excitation, te, ref_counts, ref_errors = [], [], [], [], []
    echoes_inside, nonoverlap, ref_counts_tr = [], [], []
    for ids in lines:
        line = k[ids]
        axis = int(np.argmax(np.ptp(line, axis=0)))
        if len(ids) > 1 and abs(line[-1, axis] - line[0, axis]) > 1e-12:
            fraction = -line[0, axis] / (line[-1, axis] - line[0, axis])
            echoes_inside.append(bool(0 <= fraction <= 1))
            center = adc[ids[0]] + fraction * (adc[ids[-1]] - adc[ids[0]])
        else:
            center = float(np.mean(adc[ids]))
            echoes_inside.append(False)
        centers.append(center)
        j = int(np.searchsorted(exct, adc[ids[0]], side="right") - 1)
        excitation.append(j)
        next_exc = exct[j + 1] if j + 1 < len(exct) else clock
        nonoverlap.append(
            j >= 0
            and adc[ids[-1]] < next_exc
            and all(
                not (p.start < adc[ids[-1]] and p.end > adc[ids[0]]) for p in pulses
            )
        )
        te.append(center - exct[max(j, 0)])
        ref_counts_tr.append(
            int(
                sum(
                    p.refocusing and exct[max(j, 0)] < p.center < next_exc
                    for p in pulses
                )
            )
        )
        refs = [
            p for p in pulses if p.refocusing and exct[max(j, 0)] < p.center < center
        ]
        ref_counts.append(len(refs))
        ref_errors.append(
            abs(refs[0].center - (exct[max(j, 0)] + center) / 2)
            if len(refs) == 1
            else 1e6
        )
    excitation, te = np.asarray(excitation), np.asarray(te)
    # The final acquired TR ends at the actual sequence end, not at a copied
    # preceding spacing. Otherwise a truncated final balanced/TR segment passes.
    tr_all = np.r_[np.diff(exct), clock - exc[-1].block_start]
    tr = tr_all[np.maximum(excitation, 0)]
    flips = np.array([p.flip for p in exc])[np.maximum(excitation, 0)]
    acquired = excitation >= 0
    if acquired.all():
        acquired &= np.array(
            [
                adc[ids[0]] >= exc_end[j] - 1e-12
                for ids, j in zip(lines, excitation, strict=True)
            ]
        )
    checks = [
        check(
            "rf_before_adc",
            acquired.all() and all(nonoverlap),
            "Every ADC line follows a completed excitation",
        )
    ]
    checks.append(
        check(
            "echo_in_adc",
            all(echoes_inside),
            {"echo_times_s": centers, "all_readouts_bracket_zero": all(echoes_inside)},
        )
    )
    matrix, fov = np.asarray(spec["matrix"]), np.asarray(spec["fov_mm"]) / 1000
    dims = len(matrix)
    q = k[:, :dims] * fov
    # A Cartesian lattice can be integer or half-sample shifted. Preserve offset.
    offset = np.angle(np.mean(np.exp(2j * np.pi * q), axis=0)) / (2 * np.pi)
    # +/- half a sample are equivalent modulo one, but only one origin places
    # a given readout inside the prescribed centred acquisition grid.
    offset += np.rint((np.max(q, axis=0) + np.min(q, axis=0)) / 2 + 0.5 - offset)
    integer = np.rint(q - offset).astype(int)
    indices = integer + matrix // 2
    residual = float(np.max(np.abs(q - offset - integer)))
    in_bounds = bool(((indices >= 0) & (indices < matrix)).all())
    through_plane = bool(dims == 3 or np.max(np.abs(k[:, 2])) <= 0.01 / min(fov))
    cartesian = residual <= 0.01 and through_plane and np.max(np.abs(offset)) <= 0.51
    unique = len(np.unique(indices, axis=0)) if in_bounds else 0
    coverage = unique / int(np.prod(matrix))
    steps = []
    for ids in lines:
        dq = np.diff(q[ids], axis=0)
        axis = int(np.argmax(np.ptp(q[ids], axis=0)))
        other = [j for j in range(dims) if j != axis]
        steps.append(
            bool(
                len(ids) == matrix[axis]
                and dq.shape[0] > 0
                and np.all(np.abs(np.abs(dq[:, axis]) - 1) <= 0.01)
                and np.all(np.abs(dq[:, other]) <= 0.01)
            )
        )
    checks += [
        check(
            "cartesian",
            cartesian and all(steps),
            {"max_lattice_error_samples": residual, "one_readout_axis": all(steps)},
        ),
        check(
            "matrix",
            in_bounds and coverage == 1.0,
            {
                "unique_samples": unique,
                "required_samples": int(np.prod(matrix)),
                "coverage": coverage,
            },
        ),
        check(
            "fov_mm",
            cartesian and all(steps) and in_bounds and coverage == 1.0,
            {"fov_mm": spec["fov_mm"], "spacing_cycles_per_m": (1 / fov).tolist()},
        ),
        check(
            "single_line_per_excitation",
            len(np.unique(excitation)) == len(lines),
            "Each S1 excitation acquires one line; unacquired preparation pulses are allowed",
        ),
    ]
    for name, measured in (("te_ms", te * 1000), ("tr_ms", tr * 1000)):
        limit = spec.get(name)
        if limit:
            ok = measured > 0
            if limit.get("target") is not None:
                ok &= np.abs(measured - limit["target"]) <= limit["tolerance"] + 1e-6
            if limit.get("max") is not None:
                ok &= measured <= limit["max"] + 1e-6
            checks.append(
                check(
                    name,
                    ok.all(),
                    {
                        "min": float(measured.min()),
                        "max": float(measured.max()),
                        "contract": limit,
                    },
                )
            )
    if spec.get("flip_angle_deg") is not None:
        checks.append(
            check(
                "flip_angle_deg",
                np.all(np.abs(flips - spec["flip_angle_deg"]) <= 1.0),
                {
                    "min": float(flips.min()),
                    "max": float(flips.max()),
                    "tolerance_deg": 1.0,
                },
            )
        )
    fam = family(case)
    starts = np.array([p.start for p in exc])
    moments_tr = gradient_integrals(seq, np.r_[starts, clock])
    areas = np.diff(moments_tr, axis=0)[np.maximum(excitation, 0)]
    balanced = bool(
        len(exct) > 1
        and np.max(np.abs(areas) * np.r_[fov, np.ones(3 - dims) * min(fov)]) <= 0.01
    )
    if fam == "spin_echo_t2w":
        detailed = case["task"].get("prompt_mode") != "research"
        valid = (
            all(n == 1 for n in ref_counts)
            and all(n == 1 for n in ref_counts_tr)
            and (
                not detailed
                or (
                    np.all(np.abs(flips - 90) <= 2)
                    and all(abs(p.flip - 180) <= 2 for p in pulses if p.refocusing)
                )
            )
            and np.max(ref_errors) <= max(seq.rf_raster_time, 1e-6)
        )
        checks.append(
            check(
                "spin_echo",
                valid,
                {
                    "refocusing_counts": ref_counts,
                    "refocusing_counts_per_tr": ref_counts_tr,
                    "max_midpoint_error_s": float(max(ref_errors)),
                },
            )
        )
    else:
        checks.append(
            check(
                "no_refocusing",
                not any(p.refocusing for p in pulses),
                "Gradient echoes contain no nominal 180-degree pulses",
            )
        )
    if fam == "bssfp":
        phase = np.array(
            [p.phase + 2 * np.pi * p.frequency * (p.center - p.start) for p in exc]
        )[np.unique(np.maximum(excitation, 0))]
        dphase = np.angle(np.exp(1j * np.diff(phase)))
        phase_ok = (
            len(dphase) > 0
            and np.max(np.abs(np.angle(np.exp(1j * (dphase - dphase[0]))))) < 0.01
        )
        if case["task"].get("prompt_mode") != "research":
            phase_ok &= bool(np.all(np.abs(np.abs(dphase) - np.pi) < 0.01))
        checks.append(
            check(
                "bssfp",
                balanced
                and phase_ok
                and np.all(np.abs(te - tr / 2) <= np.maximum(1e-5, tr * 0.05)),
                {
                    "balanced": balanced,
                    "coherent_phase_cycle": bool(phase_ok),
                    "echo_at_half_tr": bool(
                        np.all(np.abs(te - tr / 2) <= np.maximum(1e-5, tr * 0.05))
                    ),
                },
            )
        )
    elif fam != "spin_echo_t2w":
        # Effective spoiling is assessed with physical ensemble probes by the backend.
        checks.append(
            check(
                "gre",
                bool(
                    np.all((flips > 0.1) & (flips < 135))
                    and all(n == 0 for n in ref_counts)
                ),
                "Non-refocused gradient echoes",
            )
        )
    measurements = {
        "family": fam,
        "rf_pulses": len(pulses),
        "readouts": len(lines),
        "te_ms": (te * 1000).tolist(),
        "echo_times_s": centers,
        "echo_in_adc": echoes_inside,
        "refocusing_midpoint_errors_s": ref_errors if fam == "spin_echo_t2w" else [],
        "tr_ms": (tr * 1000).tolist(),
        "flip_angle_deg": flips.tolist(),
        "lattice_offset": offset.tolist(),
        "coverage": coverage,
        "nominal_rotation_scope": "on-resonance envelope rotation; RF selectivity evaluated by Bloch replay",
    }
    return Acquisition(
        seq,
        pulses,
        adc,
        phases,
        lines,
        k,
        indices,
        offset,
        excitation,
        te,
        tr,
        flips,
        checks,
        measurements,
        str(path),
    )
