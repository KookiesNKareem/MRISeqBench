"""Waveform-derived S1 structure. RF use labels and file definitions are not proof."""

from dataclasses import dataclass

import numpy as np
from pypulseq import Opts, Sequence

from .rf_waveform import (
    RFQuadratureLimit,
    RFVertexError,
    quadrature_intervals,
    rf_samples,
)

FAMILIES = {
    "gre_t1w",
    "gre_t2star",
    "spin_echo_t2w",
    "bssfp",
}


class MissingExcitation(ValueError):
    """A candidate has RF but no waveform-derived excitation."""


class MissingAcquisition(MissingExcitation):
    """No ADC sample; handled wherever a missing excitation fails structure."""


class NonFiniteDerived(MissingExcitation):
    """Finite inputs whose derived frequency/phase/timing/k overflow."""

    check = "finite_derived_values"


class InvalidRFVertices(MissingExcitation):
    """Explicit RF vertices that define no waveform; fails structure."""

    check = "rf_vertices"


class UnsupportedRF(MissingExcitation):
    """Candidate RF exceeds the structural quadrature limit; fails structure."""

    check = "rf_quadrature"


def midpoint_tolerance(seq):
    """Refocuser placement tolerance: two of the larger block/RF raster."""
    return 2 * max(seq.block_duration_raster, seq.rf_raster_time)


def family(case):
    return case["task"]["id"]


def check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


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


def nominal_rotation(values, widths):
    """Envelope rotation; rotations about the same (possibly signed) axis commute."""
    m = np.array([0.0, 0.0, 1.0])
    axis = values[np.argmax(np.abs(values))]
    if np.all(values.real * axis.imag == values.imag * axis.real):
        area = 2 * np.pi * np.sum(values * widths)
        return rotate(m, np.array([area.real, area.imag, 0.0]))
    for value, width in zip(values, widths, strict=True):
        m = rotate(m, 2 * np.pi * width * np.array([value.real, value.imag, 0.0]))
    return m


# Largest reference needs 302,228 intervals (rfov_epi); ~26x headroom.
RF_QUADRATURE_TOTAL = 8_000_000

BANDWIDTH_FLOOR_HZ_PER_PIXEL = 500.0
BANDWIDTH_LEVELS = {"S1", "S2", "S3"}


def readout_bandwidth(spec, k, lines, dwells):
    """S1-S3 imaging ADCs: dwell <= 1/(500 Hz x N) (bandwidth >= 500 Hz/pixel).

    N is the in-plane matrix size along the readout: the in-plane axis of
    largest k travel within the ADC (square matrices make it axis-independent).
    """
    if spec.get("level") not in BANDWIDTH_LEVELS:
        return []
    matrix = np.asarray(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    plane = min(2, len(matrix))
    bandwidths, failing = [], []
    for index, (ids, dwell) in enumerate(zip(lines, dwells, strict=True)):
        travel = np.ptp(k[ids, :plane] * fov[:plane], axis=0)
        n = int(matrix[int(np.argmax(travel))])
        bandwidth = 1 / (n * dwell)
        bandwidths.append(bandwidth)
        if dwell > (1 + 1e-9) / (BANDWIDTH_FLOOR_HZ_PER_PIXEL * n):
            failing.append(index)
    return [
        check(
            "readout_bandwidth",
            not failing,
            {
                "min_hz_per_pixel": BANDWIDTH_FLOOR_HZ_PER_PIXEL,
                "measured_min_hz_per_pixel": float(min(bandwidths, default=0.0)),
                "rule": "every imaging ADC dwell <= 1/(500 Hz x N), N = in-plane "
                "matrix size along the readout",
                "failing_readouts": failing[:20],
                "failing_count": len(failing),
            },
        )
    ]


def valid_adc(adc):
    """Positive finite dwell; phase modulation empty or one value per sample."""
    dwell = float(adc.dwell)
    modulation = getattr(adc, "phase_modulation", None)
    return bool(
        np.isfinite(dwell)
        and dwell > 0
        and (modulation is None or np.size(modulation) in (0, int(adc.num_samples)))
    )


ENVELOPE_COLLINEAR_TOL = 1e-3


def envelope_phase(values, widths):
    """Phase carried by a constant-phase complex envelope (its rotation axis).

    A co-linear envelope signal*exp(i phi) rotates about phi (or phi + pi for
    negative net area). Phase-modulated envelopes have no single axis and keep
    the per-block RF phase only.
    """
    values = np.asarray(values, dtype=complex)
    peak = float(np.max(np.abs(values), initial=0.0))
    if peak == 0:
        return 0.0
    axis = values[np.argmax(np.abs(values))] / peak
    # Collinear up to file roundoff: the component perpendicular to the peak
    # axis stays below 1e-3 of the peak (Pulseq stores magnitude/phase shapes
    # with finite precision). Genuine phase modulation exceeds this by orders.
    if np.max(np.abs((values * np.conj(axis)).imag)) > ENVELOPE_COLLINEAR_TOL * peak:
        return 0.0
    area = np.sum(values * widths)
    return float(np.angle(area)) if abs(area) > 0 else 0.0


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
    block_number: int | None = None


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
    empty_adcs = invalid_adcs = 0
    dwells = []
    envelopes = {}
    quadrature = 0
    clock = 0.0
    for number in seq.block_events:
        block = seq.get_block(number)
        # An all-zero RF event is no RF: KomaMRI drops it as well, so both
        # engines see the same RF train.
        if block.rf is not None and np.any(np.asarray(block.rf.signal) != 0):
            rf = block.rf
            key = (
                np.asarray(rf.signal, dtype=complex).tobytes(),
                np.asarray(rf.t, dtype=float).tobytes(),
                float(rf.shape_dur),
                seq.rf_raster_time,
            )
            if key not in envelopes:
                # Cap cumulative structural quadrature over distinct envelopes
                # before expanding any of them (per-envelope cap in rf_samples).
                try:
                    quadrature += quadrature_intervals(rf, seq.rf_raster_time)
                except RFVertexError as exc:
                    raise InvalidRFVertices(str(exc)) from exc
                if quadrature > RF_QUADRATURE_TOTAL:
                    raise UnsupportedRF(
                        f"Distinct RF envelopes need more than {RF_QUADRATURE_TOTAL} "
                        "structural quadrature intervals in total"
                    )
                try:
                    edges, values = rf_samples(rf, seq.rf_raster_time)
                except RFQuadratureLimit as exc:
                    raise UnsupportedRF(str(exc)) from exc
                mids, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
                # Envelope only: phase/frequency/delay remain per-block below.
                m = nominal_rotation(values, widths)
                flip = float(np.rad2deg(np.arccos(np.clip(m[2], -1, 1))))
                center = float(
                    np.sum(mids * np.abs(values) * widths)
                    / max(np.sum(np.abs(values) * widths), 1e-30)
                )
                envelopes[key] = (
                    edges,
                    values,
                    flip,
                    center,
                    envelope_phase(values, widths),
                )
            edges, values, flip, center, carried = envelopes[key]
            freq = float(
                rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            phase = float(
                rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            origin = clock + float(rf.delay)
            pulses.append(
                Pulse(
                    origin + edges[0],
                    origin + edges[-1],
                    origin + center,
                    edges + origin,
                    values,
                    freq,
                    # Effective RF phase includes a constant envelope phase;
                    # KomaMRI already replays it from the waveform itself.
                    # Wrapped so coherence propagation cannot overflow.
                    float(np.remainder(phase + carried, 2 * np.pi)),
                    flip,
                    flip >= 135,
                    phase - float(rf.phase_offset),
                    clock,
                    number,
                )
            )
        if block.adc is not None and int(block.adc.num_samples) < 1:
            # A zero-sample ADC acquires nothing; it fails readout_samples.
            empty_adcs += 1
        elif block.adc is not None and not valid_adc(block.adc):
            # Non-positive dwell or per-sample phase of the wrong length has no
            # sample timeline; it fails readout_samples instead of raising.
            invalid_adcs += 1
        elif block.adc is not None:
            a = block.adc
            dwells.append(float(a.dwell))
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
    if not lines:
        raise MissingAcquisition("No ADC event acquires a sample")
    if not pulses:
        raise MissingExcitation("No nonzero RF before acquisition")
    pt = np.array([p.center for p in pulses])
    moments = gradient_integrals(seq, np.r_[pt, adc])
    pm, am = moments[: len(pt)], moments[len(pt) :]
    delta, baseline = np.zeros(3), []
    for p, moment in zip(pulses, pm, strict=True):
        delta = -2 * moment - delta if p.refocusing else -moment
        baseline.append(delta.copy())
    preceding = np.searchsorted(pt, adc, side="right") - 1
    k = am + np.asarray(baseline)[np.maximum(preceding, 0)]
    derived = {
        "rf_frequency_hz": [p.frequency for p in pulses],
        "rf_phase_rad": [p.phase for p in pulses],
        "rf_phase_correction_rad": [p.phase_correction for p in pulses],
        # Effective RF phase at the centre: frequency x elapsed time can
        # overflow even when both factors are finite.
        "rf_effective_phase_rad": [
            p.phase + 2 * np.pi * p.frequency * (p.center - p.start) for p in pulses
        ],
        "rf_times_s": [t for p in pulses for t in (p.start, p.center, p.end)],
        "adc_times_s": adc,
        "adc_phase_rad": phases,
        "k_cycles_per_m": k,
    }
    overflow = sorted(
        name
        for name, values in derived.items()
        if not np.isfinite(np.asarray(values, dtype=complex)).all()
    )
    if overflow:
        # Finite file tokens (e.g. ppm 1e308) can overflow once scaled by
        # gamma*B0, accumulated or integrated; fail before any simulation.
        raise NonFiniteDerived(f"Non-finite derived RF/ADC/gradient values: {overflow}")
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
    # TR is measured between acquired excitation centres. Preparation RF does
    # not start a repetition, and a terminal recovery tail has no following RF.
    # That tail remains part of the authored duration and SAR average.
    acquired_excitation = np.unique(excitation[excitation >= 0])
    tr = np.diff(exct[acquired_excitation])
    flips = np.array([p.flip for p in exc])[np.maximum(excitation, 0)]
    acquired = excitation >= 0
    if acquired.all():
        acquired &= np.array(
            [
                adc[ids[0]] >= exc_end[j] - 1e-12
                for ids, j in zip(lines, excitation, strict=True)
            ]
        )
    short = [i for i, ids in enumerate(lines) if len(ids) < 2]
    fam = family(case)
    checks = [
        check(
            "readout_samples",
            not short and not empty_adcs and not invalid_adcs,
            {
                "min_samples_per_readout": 2,
                "short_readouts": short[:20],
                "zero_sample_adc_events": empty_adcs,
                "invalid_adc_events": invalid_adcs,
            },
        ),
        *readout_bandwidth(spec, k, lines, dwells),
        check(
            "rf_before_adc",
            acquired.all() and all(nonoverlap),
            "Every ADC line follows a completed excitation",
        ),
        check(
            "echo_in_adc",
            all(echoes_inside),
            {"echo_times_s": centers, "all_readouts_bracket_zero": all(echoes_inside)},
        ),
    ]
    # Documented S1 TR rule: no unacquired excitation between acquired ones.
    interleaved = 0
    if fam in FAMILIES and len(acquired_excitation):
        between = np.arange(acquired_excitation[0] + 1, acquired_excitation[-1])
        interleaved = len(np.setdiff1d(between, acquired_excitation))
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
    # A band shifted by one whole sample (-N/2+1..N/2, e.g. a negated textbook
    # PE table) is the same N-point Nyquist grid: +N/2 aliases to -N/2. The
    # offset phase reconstructs it exactly as that one-line wrap.
    on_grid = (np.abs(offset) <= 0.51) | (np.abs(offset - 1) <= 0.01)
    cartesian = residual <= 0.01 and through_plane and bool(on_grid.all())
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
            {
                "max_lattice_error_samples": residual,
                "one_readout_axis": all(steps),
                "lattice_offset_samples": offset.tolist(),
                "lattice_offset_limit": "|offset| <= 0.51, or +1 (+N/2 edge line)",
                "acquired_k_index_range": [
                    np.round(np.min(q, axis=0), 3).tolist(),
                    np.round(np.max(q, axis=0), 3).tolist(),
                ],
            },
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
            {
                "fov_mm": spec["fov_mm"],
                "spacing_cycles_per_m": (1 / fov).tolist(),
                "interpretation": "requested FOV; acceptance also requires the complete physical Cartesian grid",
                "max_lattice_error_samples": residual,
                "lattice_error_limit_samples": 0.01,
                "observed_k_min_cycles_per_m": k.min(axis=0).tolist(),
                "observed_k_max_cycles_per_m": k.max(axis=0).tolist(),
                "unique_samples": unique,
                "required_samples": int(np.prod(matrix)),
                "coverage": coverage,
                "conditions": {
                    "cartesian_lattice_and_plane": bool(cartesian),
                    "one_readout_axis_and_unit_steps": all(steps),
                    "indices_in_bounds": in_bounds,
                    "complete_grid": coverage == 1.0,
                },
            },
        ),
        check(
            "single_line_per_excitation",
            len(np.unique(excitation)) == len(lines) and not interleaved,
            {
                "rule": "Each S1 excitation acquires one line; unacquired "
                "preparation pulses may precede the first acquired excitation "
                "but must not sit between acquired excitations",
                "unacquired_excitations_between_acquired": interleaved,
            },
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
                    measured.size > 0 and ok.all(),
                    {
                        "min": float(measured.min()) if measured.size else None,
                        "max": float(measured.max()) if measured.size else None,
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
    # Balance intervals start at acquired excitations only; the last runs to the
    # end of the sequence, so unacquired RF cannot hide a terminal spoiler.
    starts = np.array([p.start for p in exc])[acquired_excitation]
    moments_tr = gradient_integrals(seq, np.r_[starts, clock])
    interval = np.searchsorted(acquired_excitation, np.maximum(excitation, 0))
    areas = (
        np.diff(moments_tr, axis=0)[np.minimum(interval, len(starts) - 1)]
        if len(starts)
        else np.full((1, 3), np.inf)
    )
    balanced = bool(
        len(starts) > 1
        and np.max(np.abs(areas) * np.r_[fov, np.ones(3 - dims) * min(fov)]) <= 0.01
    )
    if fam == "spin_echo_t2w":
        valid = (
            all(n == 1 for n in ref_counts)
            and all(n == 1 for n in ref_counts_tr)
            and np.all(np.abs(flips - 90) <= 2)
            and all(abs(p.flip - 180) <= 2 for p in pulses if p.refocusing)
            and np.max(ref_errors) <= midpoint_tolerance(seq)
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
        phase_ok &= bool(np.all(np.abs(np.abs(dphase) - np.pi) < 0.01))
        checks.append(
            check(
                "bssfp",
                balanced and phase_ok,
                {
                    "balanced": balanced,
                    "coherent_phase_cycle": bool(phase_ok),
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
