"""ECG windows, preparation identity and bSSFP derived from Pulseq events."""

from copy import copy
from types import SimpleNamespace

import numpy as np
from pypulseq import Sequence
from scipy.optimize import least_squares

from .s1_sequence import (
    check,
    family,
    gradient_integrals,
    midpoint_tolerance,
)
from .s1_sequence import inspect_sequence as inspect_cartesian

APPLICATION_S4_FAMILIES = {"water_selective_gre", "dwi_epi", "venc_flow_2d"}


def _bloch_rf_group_response(pulses, offsets_hz):
    """Integrate 1D Bloch response M(0)=[0,0,1] across a pulse group at offsets_hz."""
    offsets = np.asarray(offsets_hz, dtype=float)
    mx = np.zeros_like(offsets)
    my = np.zeros_like(offsets)
    mz = np.ones_like(offsets)
    for idx, p in enumerate(pulses):
        if idx > 0:
            gap = float(p.start - pulses[idx - 1].end)
            if gap > 0:
                phi = -2.0 * np.pi * offsets * gap
                ca, sa = np.cos(phi), np.sin(phi)
                mx, my = mx * ca - my * sa, mx * sa + my * ca
        widths = np.diff(p.edges)
        signal = np.asarray(p.signal, dtype=complex) * np.exp(1j * p.phase)
        bz = float(p.frequency) + offsets
        for val, w in zip(signal, widths, strict=True):
            bx = np.full_like(offsets, float(val.real))
            by = np.full_like(offsets, float(val.imag))
            norm = np.sqrt(bx * bx + by * by + bz * bz)
            angle = 2.0 * np.pi * float(w) * norm
            denom = np.maximum(norm, 1e-30)
            ux, uy, uz = bx / denom, by / denom, bz / denom
            ca, sa = np.cos(angle), np.sin(angle)
            dot = ux * mx + uy * my + uz * mz
            cx = uy * mz - uz * my
            cy = uz * mx - ux * mz
            cz = ux * my - uy * mx
            mx = mx * ca - cx * sa + ux * dot * (1.0 - ca)
            my = my * ca - cy * sa + uy * dot * (1.0 - ca)
            mz = mz * ca - cz * sa + uz * dot * (1.0 - ca)
    return mx, my, mz


def _inspect_water_selective_gre(acq, case):
    spec = case["task"]["sequence"]
    exc = [p for p in acq.pulses if not p.refocusing]
    n_lines = len(acq.lines)
    if not exc or n_lines == 0:
        return _failed(acq, [check("water_selective_suppression", False, "No excitation")])
    line_groups = []
    for line_idx, ids in enumerate(acq.lines):
        t_start = float(acq.adc[acq.lines[line_idx - 1][-1]]) if line_idx > 0 else -1.0
        t_end = float(acq.adc[ids[0]])
        candidates = [
            i for i, p in enumerate(exc) if p.start > t_start - 1e-12 and p.end <= t_end + 1e-12
        ]
        line_groups.append(candidates)
    group_size = len(line_groups[-1]) if line_groups and line_groups[-1] else 1
    if line_groups and len(line_groups[0]) > group_size and len(line_groups[0]) % group_size == 0:
        line_groups[0] = line_groups[0][-group_size:]
    valid_grouping = all(len(g) == group_size and len(g) >= 1 for g in line_groups)
    all_groups = []
    if valid_grouping and group_size >= 1:
        first_acq_pulse = line_groups[0][0]
        for start_i in range(0, first_acq_pulse, group_size):
            chunk = list(range(start_i, min(start_i + group_size, first_acq_pulse)))
            if len(chunk) == group_size:
                all_groups.append([exc[i] for i in chunk])
        for g in line_groups:
            all_groups.append([exc[i] for i in g])
    else:
        all_groups = [[p] for p in exc]

    fat_hz = float(case["object"]["tissue_properties"]["fat"]["chemical_shift_hz"])
    water_mxy, fat_mxy, ratios, group_flips, group_centers = [], [], [], [], []
    internal_crusher_cycles = []
    for g in (all_groups[-n_lines:] if len(all_groups) >= n_lines else all_groups):
        mx, my, mz = _bloch_rf_group_response(g, [0.0, fat_hz])
        mxy = np.hypot(mx, my)
        w_val, f_val = float(mxy[0]), float(mxy[1])
        water_mxy.append(w_val)
        fat_mxy.append(f_val)
        ratios.append(f_val / max(w_val, 1e-15))
        group_flips.append(float(np.rad2deg(np.arccos(np.clip(mz[0], -1.0, 1.0)))))
        weights = [float(np.sum(np.abs(p.signal) * np.diff(p.edges))) for p in g]
        total_w = max(sum(weights), 1e-30)
        group_centers.append(
            float(sum(p.center * w for p, w in zip(g, weights, strict=True)) / total_w)
        )
        if len(g) > 1:
            span = gradient_integrals(acq.seq, [g[0].start, g[-1].end])
            internal_crusher_cycles.append(
                float(np.max(crusher_cycles_per_voxel(span[1] - span[0], spec)))
            )
        else:
            internal_crusher_cycles.append(0.0)

    if group_size > 1 and valid_grouping and len(group_centers) == n_lines:
        centers = np.asarray(acq.measurements["echo_times_s"], dtype=float)
        acq.te = centers - np.asarray(group_centers, dtype=float)
        acq.tr = np.diff(np.asarray(group_centers, dtype=float))
        acq.flips = np.asarray(group_flips, dtype=float)
        acq.measurements["te_ms"] = (acq.te * 1000).tolist()
        acq.measurements["tr_ms"] = (acq.tr * 1000).tolist()
        acq.measurements["flip_angle_deg"] = acq.flips.tolist()

    checks = [
        c
        for c in acq.checks
        if c["name"]
        in {
            "readout_samples",
            "rf_before_adc",
            "echo_in_adc",
            "cartesian",
            "matrix",
            "fov_mm",
            "slice_thickness_mm",
            "no_refocusing",
        }
    ]
    checks.append(
        check(
            "single_line_per_excitation",
            valid_grouping and len(line_groups) == n_lines,
            {"subpulses_per_excitation": group_size, "lines": n_lines},
        )
    )
    for name, measured in (("te_ms", acq.te * 1000), ("tr_ms", acq.tr * 1000)):
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
                    measured.size > 0 and bool(ok.all()),
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
                bool(np.all(np.abs(acq.flips - spec["flip_angle_deg"]) <= 1.0)),
                {
                    "min": float(acq.flips.min()),
                    "max": float(acq.flips.max()),
                    "tolerance_deg": 1.0,
                },
            )
        )
    checks.append(
        check(
            "gre",
            bool(
                np.all((acq.flips > 0.1) & (acq.flips < 135))
                and not any(p.refocusing for p in acq.pulses)
            ),
            "Non-refocused water-selective gradient echoes",
        )
    )
    max_ratio = float(spec["application"]["fat_suppression_max_ratio"])
    max_measured_ratio = float(max(ratios, default=1.0))
    min_water = float(min(water_mxy, default=0.0))
    max_crusher = float(max(internal_crusher_cycles, default=0.0))
    suppression_conditions = {
        "water_excitation_present": min_water >= 0.05,
        "fat_suppression_within_limit": max_measured_ratio <= max_ratio,
        "no_inter_subpulse_crusher": max_crusher <= 0.5,
        "no_refocusing_or_inversion_prep": not any(p.refocusing for p in acq.pulses),
    }
    checks.append(
        check(
            "water_selective_suppression",
            all(suppression_conditions.values()),
            {
                "fat_chemical_shift_hz": fat_hz,
                "min_water_transverse_mxy": min_water,
                "max_fat_transverse_mxy": float(max(fat_mxy, default=0.0)),
                "max_fat_to_water_ratio": max_measured_ratio,
                "fat_suppression_max_ratio": max_ratio,
                "subpulses_per_excitation": group_size,
                "max_internal_crusher_cycles_per_voxel": max_crusher,
                "conditions": suppression_conditions,
            },
        )
    )
    all_ids = np.concatenate(acq.lines).astype(int)
    central = int(all_ids[np.argmin(np.linalg.norm(acq.k[all_ids], axis=1))])
    unique = len(np.unique(acq.indices[all_ids], axis=0))
    checks.append(
        check(
            "frame_0_full_cartesian_grid",
            unique == int(np.prod(spec["matrix"])) and len(all_ids) == int(np.prod(spec["matrix"])),
            {"unique_samples": unique, "samples": len(all_ids), "segments": 1},
        )
    )
    frame = {
        "adc_indices": all_ids.tolist(),
        "line_indices": list(range(n_lines)),
        "central_adc_index": central,
        "preparation_pulse_indices": [],
        "conditioning_pulse_indices": [],
        "readout_pulse_indices": [g[-1] for g in line_groups] if valid_grouping else [],
        "train_start_s": float(all_groups[0][0].start),
        "trigger_time_s": 0.0,
        "segment_index": 0,
    }
    spoil_times = [float(g[0].start) for g in all_groups]
    acq.checks = checks
    acq.measurements.update(
        frames=[frame],
        family="water_selective_gre",
        reconstruction="bart_cartesian_fft@1",
        preparation_spoil_times_s=spoil_times,
        water_selective={
            "fat_to_water_ratio": max_measured_ratio,
            "fat_suppression_max_ratio": max_ratio,
            "subpulses_per_excitation": group_size,
        },
    )
    return acq


def _shot_b_value_s_per_mm2(seq, t_exc, t_ref, t_ref_end, t_adc0):
    """Exact Karlicek-Lowe / Stejskal-Tanner b-value (s/mm^2) from PPoly gradients."""
    grads = seq.get_gradients()
    integrals = [g.antiderivative() if g is not None else None for g in grads]

    def g_int(times):
        times = np.asarray(times, dtype=float)
        out = np.zeros((len(times), 3))
        for axis, integ in enumerate(integrals):
            if integ is not None:
                q = np.clip(times, integ.x[0], integ.x[-1])
                out[:, axis] = integ(q) - integ(np.clip(0.0, integ.x[0], integ.x[-1]))
        return out

    breakpoints = [t_ref_end, t_adc0]
    for g in grads:
        if g is not None:
            bp = g.x[(g.x >= t_ref_end - 1e-12) & (g.x <= t_adc0 + 1e-12)]
            breakpoints.extend(bp.tolist())
    probe = np.unique(np.clip(np.r_[breakpoints, np.linspace(t_ref_end, t_adc0, 257)], t_ref_end, t_adc0))
    m_exc = g_int([t_exc])[0]
    m_ref = g_int([t_ref])[0]
    k_after = g_int(probe) - 2.0 * m_ref + m_exc
    norms = np.linalg.norm(k_after, axis=1)
    best = int(np.argmin(norms))
    t_end = float(probe[best])
    residual_k = float(norms[best])

    knots = [t_exc, t_ref, t_end]
    for g in grads:
        if g is not None:
            bp = g.x[(g.x > t_exc + 1e-12) & (g.x < t_end - 1e-12)]
            knots.extend(bp.tolist())
    knots = np.unique(np.clip(knots, t_exc, t_end))
    if len(knots) < 2:
        return 0.0, residual_k, 0.0
    # 3-point Gauss-Legendre is exact for degree <= 5 polynomials (||k(t)||^2 is degree 4).
    gl_nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gl_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    a_seg, b_seg = knots[:-1], knots[1:]
    mid = 0.5 * (a_seg + b_seg)
    half = 0.5 * (b_seg - a_seg)
    quad_t = (mid[:, None] + half[:, None] * gl_nodes[None, :]).ravel()
    quad_w = (half[:, None] * gl_weights[None, :]).ravel()
    m_quad = g_int(quad_t)
    before_ref = quad_t < t_ref
    k_quad = np.where(
        before_ref[:, None],
        m_quad - m_exc[None, :],
        m_quad - 2.0 * m_ref[None, :] + m_exc[None, :],
    )
    k_sq = np.sum(k_quad * k_quad, axis=1)
    b_val = float(4.0 * (np.pi**2) * 1e-6 * np.sum(quad_w * k_sq))
    peak_k = float(np.max(np.sqrt(k_sq), initial=0.0))
    return b_val, residual_k, peak_k


def _inspect_dwi_epi(acq, case):
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    q = acq.k[:, : len(matrix)] * fov
    read = spec.get("readout") or {}
    checks = [
        c
        for c in acq.checks
        if c["name"]
        in {
            "readout_samples",
            "rf_before_adc",
            "echo_in_adc",
            "cartesian",
            "matrix",
            "fov_mm",
            "slice_thickness_mm",
            "tr_ms",
        }
    ]
    exc = [p for p in acq.pulses if not p.refocusing]
    refs = [p for p in acq.pulses if p.refocusing]
    groups = [np.flatnonzero(acq.excitation == j) for j in np.unique(acq.excitation)]
    shots = len(groups)
    centre_ids = np.array(
        [ids[np.argmin(np.linalg.norm(q[ids, :2], axis=1))] for ids in acq.lines]
    )
    centres = np.asarray(acq.measurements["echo_times_s"])
    acq.te = centres - np.array([exc[max(j, 0)].center for j in acq.excitation])
    effective_line = int(np.argmin(np.linalg.norm(q[centre_ids], axis=1)))
    te_ms = float(acq.te[effective_line] * 1000)
    if spec.get("te_ms"):
        limit = spec["te_ms"]
        ok = te_ms > 0
        if limit.get("target") is not None:
            ok &= abs(te_ms - limit["target"]) <= limit["tolerance"] + 1e-3
        if limit.get("max") is not None:
            ok &= te_ms <= limit["max"] + 1e-3
        checks.append(check("te_ms", ok, {"effective_te_ms": te_ms, "contract": limit}))
    if read.get("shots"):
        checks.append(
            check(
                "shots",
                shots == read["shots"],
                {"measured": shots, "required": read["shots"]},
            )
        )
    if read.get("echo_train_length"):
        counts = [len(g) for g in groups]
        checks.append(
            check(
                "echo_train_length",
                all(n == read["echo_train_length"] for n in counts),
                counts,
            )
        )
    x_steps = [np.abs(np.abs(np.diff(q[ids, 0])) - 1) for ids in acq.lines]
    transverse = [np.ptp(q[ids, 1:], axis=0) for ids in acq.lines]
    x_readout = all(
        len(ids) == matrix[0] and np.all(step <= 0.01) and np.all(motion <= 0.01)
        for ids, step, motion in zip(acq.lines, x_steps, transverse, strict=True)
    )
    checks.append(
        check(
            "readout_axis",
            x_readout,
            {"required_axis": "x", "samples_per_readout": int(matrix[0])},
        )
    )
    polarity = np.array([np.sign(q[ids[-1], 0] - q[ids[0], 0]) for ids in acq.lines])
    alternating = all(
        np.all(polarity[g][1:] * polarity[g][:-1] == -1) for g in groups
    )
    epi_valid = alternating and shots >= 2 and all(len(g) >= 2 for g in groups)
    for g in groups:
        ky = q[[centre_ids[i] for i in g], 1]
        steps = np.diff(ky)
        epi_valid &= (
            len(ky) > 1
            and bool(np.all(np.abs(np.abs(steps) - shots) <= 0.02))
            and bool(np.all(steps > 0) or np.all(steps < 0))
        )
    checks.append(
        check("epi", epi_valid, {"shots": shots, "alternating_polarity": alternating})
    )

    shot_refs, midpoint_errors, b_values, residuals, peaks = [], [], [], [], []
    for j, g in zip(np.unique(acq.excitation), groups, strict=True):
        p_exc = exc[int(j)]
        t_adc0 = float(acq.adc[acq.lines[g[0]][0]])
        r_list = [p for p in refs if p_exc.center < p.center < t_adc0]
        shot_refs.append(len(r_list))
        if len(r_list) == 1:
            p_ref = r_list[0]
            spin_echo_time = 2.0 * p_ref.center - p_exc.center
            shot_centres = centres[g]
            midpoint_errors.append(float(np.min(np.abs(shot_centres - spin_echo_time))))
            b_val, res_k, peak_k = _shot_b_value_s_per_mm2(
                acq.seq, p_exc.center, p_ref.center, p_ref.end, t_adc0
            )
            b_values.append(b_val)
            residuals.append(res_k)
            peaks.append(peak_k)
        else:
            midpoint_errors.append(1e6)
            b_values.append(0.0)
            residuals.append(1e6)
            peaks.append(0.0)

    tol_se = max(midpoint_tolerance(acq.seq), 50e-6)
    se_valid = (
        all(n == 1 for n in shot_refs)
        and len(refs) == shots
        and bool(np.all(np.abs(acq.flips - 90) <= 2))
        and all(abs(p.flip - 180) <= 2 for p in refs)
        and max(midpoint_errors, default=1e6) <= tol_se
    )
    checks.append(
        check(
            "spin_echo",
            se_valid,
            {
                "refocusing_counts_per_shot": shot_refs,
                "max_spin_echo_alignment_error_s": float(max(midpoint_errors, default=1e6)),
                "tolerance_s": tol_se,
            },
        )
    )
    target_b = float(spec["application"]["b_value_s_per_mm2"])
    b_arr = np.asarray(b_values, dtype=float)
    res_limit = 0.02 / float(min(fov))
    b_conditions = {
        "one_refocusing_per_shot": all(n == 1 for n in shot_refs),
        "positive_diffusion_encoding": bool(len(peaks) and min(peaks) >= 100.0),
        "balanced_before_readout": bool(len(residuals) and max(residuals) <= res_limit),
        "b_value_within_tolerance": bool(
            len(b_arr) and np.all(np.abs(b_arr - target_b) <= 0.05 * target_b)
        ),
    }
    checks.append(
        check(
            "diffusion_b_value",
            all(b_conditions.values()),
            {
                "target_b_value_s_per_mm2": target_b,
                "measured_b_value_min_s_per_mm2": float(b_arr.min()) if len(b_arr) else None,
                "measured_b_value_max_s_per_mm2": float(b_arr.max()) if len(b_arr) else None,
                "tolerance_s_per_mm2": 0.05 * target_b,
                "max_residual_k_cycles_per_m": float(max(residuals, default=1e6)),
                "residual_k_limit_cycles_per_m": res_limit,
                "conditions": b_conditions,
            },
        )
    )
    all_ids = np.concatenate(acq.lines).astype(int)
    central = int(all_ids[np.argmin(np.linalg.norm(acq.k[all_ids], axis=1))])
    unique = len(np.unique(acq.indices[all_ids], axis=0))
    checks.append(
        check(
            "frame_0_full_cartesian_grid",
            unique == int(np.prod(spec["matrix"])) and len(all_ids) == int(np.prod(spec["matrix"])),
            {"unique_samples": unique, "samples": len(all_ids), "segments": shots},
        )
    )
    frame = {
        "adc_indices": all_ids.tolist(),
        "line_indices": list(range(len(acq.lines))),
        "central_adc_index": central,
        "preparation_pulse_indices": [],
        "conditioning_pulse_indices": [],
        "readout_pulse_indices": list(range(len(acq.pulses))),
        "train_start_s": float(exc[0].start),
        "trigger_time_s": 0.0,
        "segment_index": 0,
    }
    acq.checks = checks
    acq.measurements.update(
        frames=[frame],
        family="dwi_epi",
        reconstruction="bart_cartesian_fft@1",
        preparation_spoil_times_s=[float(p.start) for p in exc],
        centre_sample_indices=[central],
        b_values_s_per_mm2=b_values,
    )
    return acq


def _line_moments_at_te(seq, t_exc, t_echo):
    """Exact zeroth (cycles/m) and first (cycles*s/m) gradient moments on [t_exc, t_echo]."""
    t0 = np.asarray(t_exc, dtype=float)
    t1 = np.asarray(t_echo, dtype=float)
    m0 = np.zeros((len(t0), 3))
    m1 = np.zeros((len(t0), 3))
    for axis, grad in enumerate(seq.get_gradients()):
        if grad is not None:
            g1 = grad.antiderivative()
            g2 = g1.antiderivative()
            q0 = np.clip(t0, g1.x[0], g1.x[-1])
            q1 = np.clip(t1, g1.x[0], g1.x[-1])
            v0, v1 = g1(q0), g1(q1)
            w0, w1 = g2(q0), g2(q1)
            m0[:, axis] = v1 - v0
            m1[:, axis] = (q1 - q0) * v1 - (w1 - w0)
    return m0, m1


def _bloch_moving_spin_phase(seq, pulse, t_echo, velocity_m_per_s):
    """1D Bloch phase at t_echo for a spin moving at constant velocity r(t)=v*(t-t_exc)."""
    grads = seq.get_gradients()
    knots = [pulse.end, t_echo]
    for g in grads:
        if g is not None:
            bp = g.x[(g.x > pulse.end + 1e-12) & (g.x < t_echo - 1e-12)]
            knots.extend(bp.tolist())
    knots = np.unique(np.clip(knots, pulse.end, t_echo))
    gl_nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gl_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    a_seg, b_seg = knots[:-1], knots[1:]
    mid = 0.5 * (a_seg + b_seg)
    half = 0.5 * (b_seg - a_seg)
    quad_t = (mid[:, None] + half[:, None] * gl_nodes[None, :]).ravel()
    quad_w = (half[:, None] * gl_weights[None, :]).ravel()
    g_vals = np.zeros((len(quad_t), 3))
    for axis, g in enumerate(grads):
        if g is not None:
            g_vals[:, axis] = g(np.clip(quad_t, g.x[0], g.x[-1]))
    pos = (quad_t - pulse.center)[:, None] * np.asarray(velocity_m_per_s, dtype=float)[None, :]
    freq = np.sum(g_vals * pos, axis=1)
    return float(-2.0 * np.pi * np.sum(quad_w * freq))


def _inspect_venc_flow_2d(acq, case):
    spec = case["task"]["sequence"]
    matrix = np.asarray(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    ny = int(matrix[1])
    checks = [
        c
        for c in acq.checks
        if c["name"]
        in {
            "readout_samples",
            "rf_before_adc",
            "echo_in_adc",
            "cartesian",
            "te_ms",
            "tr_ms",
            "flip_angle_deg",
            "slice_thickness_mm",
            "no_refocusing",
            "gre",
        }
    ]
    n_lines = len(acq.lines)
    if n_lines != 2 * ny:
        checks.append(
            check(
                "cardiac_frame_count",
                False,
                {"acquired_lines": n_lines, "required_lines": 2 * ny},
            )
        )
        return _failed(acq, checks)

    exc = [p for p in acq.pulses if not p.refocusing]
    centres = np.asarray(acq.measurements["echo_times_s"], dtype=float)
    frames = []
    frame_m0, frame_m1, frame_central_line = [], [], []
    for frame_no in range(2):
        line_ids = list(range(frame_no * ny, (frame_no + 1) * ny))
        ids = np.concatenate([acq.lines[j] for j in line_ids]).astype(int)
        unique = len(np.unique(acq.indices[ids], axis=0))
        in_bounds = bool(((acq.indices[ids] >= 0) & (acq.indices[ids] < matrix)).all())
        checks.append(
            check(
                f"frame_{frame_no}_full_cartesian_grid",
                in_bounds
                and unique == int(np.prod(matrix))
                and len(ids) == int(np.prod(matrix)),
                {"unique_samples": unique, "samples": len(ids), "segments": 1},
            )
        )
        ky_vals = np.array([int(acq.indices[acq.lines[j][0], 1]) for j in line_ids])
        order = np.argsort(ky_vals)
        ordered_lines = [line_ids[i] for i in order]
        t_exc = np.array([exc[int(acq.excitation[j])].center for j in ordered_lines])
        t_echo = np.array([centres[j] for j in ordered_lines])
        m0, m1 = _line_moments_at_te(acq.seq, t_exc, t_echo)
        frame_m0.append(m0)
        frame_m1.append(m1)
        central = int(ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))])
        central_line = next(j for j in line_ids if central in acq.lines[j])
        frame_central_line.append(central_line)
        readout_ids = [int(acq.excitation[j]) for j in line_ids]
        frames.append(
            {
                "adc_indices": ids.tolist(),
                "line_indices": line_ids,
                "central_adc_index": central,
                "preparation_pulse_indices": [],
                "conditioning_pulse_indices": [],
                "readout_pulse_indices": readout_ids,
                "train_start_s": float(exc[readout_ids[0]].start),
                "trigger_time_s": 0.0,
                "segment_index": 0,
            }
        )

    dm0 = frame_m0[1] - frame_m0[0]
    dm1 = frame_m1[1] - frame_m1[0]
    mean_dm1 = np.mean(dm1, axis=0)
    norm_dm1 = float(np.linalg.norm(mean_dm1))
    max_dm0_err = float(np.max(np.linalg.norm(dm0, axis=1)))
    max_dm1_spread = float(
        np.max(np.linalg.norm(dm1 - mean_dm1[None, :], axis=1)) / max(norm_dm1, 1e-15)
    )
    measured_venc = 50.0 / max(norm_dm1, 1e-15)
    target_venc = float(spec["application"]["venc_cm_per_s"])
    direction = mean_dm1 / max(norm_dm1, 1e-15)
    v_test = 0.5 * (target_venc / 100.0) * direction
    phi0 = _bloch_moving_spin_phase(
        acq.seq,
        exc[int(acq.excitation[frame_central_line[0]])],
        centres[frame_central_line[0]],
        v_test,
    )
    phi1 = _bloch_moving_spin_phase(
        acq.seq,
        exc[int(acq.excitation[frame_central_line[1]])],
        centres[frame_central_line[1]],
        v_test,
    )
    dphi = float(abs(np.angle(np.exp(1j * (phi1 - phi0)))))
    m0_limit = 0.01 / float(min(fov))
    flow_conditions = {
        "two_velocity_frames": len(frames) == 2,
        "matched_zeroth_moment_at_te": max_dm0_err <= m0_limit,
        "uniform_first_moment_across_lines": max_dm1_spread <= 0.02,
        "venc_within_tolerance": abs(measured_venc - target_venc) <= 0.05 * target_venc,
        "bloch_moving_spin_phase_match": abs(dphi - np.pi / 2) <= 0.10,
    }
    checks.append(
        check(
            "flow_encoding_venc",
            all(flow_conditions.values()),
            {
                "target_venc_cm_per_s": target_venc,
                "measured_venc_cm_per_s": measured_venc,
                "venc_tolerance_cm_per_s": 0.05 * target_venc,
                "delta_m1_cycles_s_per_m": mean_dm1.tolist(),
                "max_delta_m0_cycles_per_m": max_dm0_err,
                "max_delta_m1_relative_spread": max_dm1_spread,
                "moving_spin_half_venc_phase_rad": dphi,
                "expected_half_venc_phase_rad": float(np.pi / 2),
                "conditions": flow_conditions,
            },
        )
    )
    checks.append(
        check(
            "cardiac_frame_count",
            len(frames) == 2,
            {"frames": len(frames), "required": 2},
        )
    )
    acq.checks = checks
    acq.measurements.update(
        frames=frames,
        family="venc_flow_2d",
        reconstruction="bart_cartesian_fft@1",
        preparation_spoil_times_s=[float(p.start) for p in exc],
        flow_encoding={
            "measured_venc_cm_per_s": measured_venc,
            "target_venc_cm_per_s": target_venc,
            "delta_m1_cycles_s_per_m": mean_dm1.tolist(),
        },
    )
    return acq


def _inversion_time_matches(measured_ms, target_ms, tolerance_ms):
    """Inclusive TI bound with a 1 ps guard for accumulated float arithmetic.

    This is numerical slack, far below the file rasters; it does not replace
    the physical tolerance or round a genuinely out-of-tolerance TI onto it.
    """
    return measured_ms is not None and abs(measured_ms - target_ms) <= (
        tolerance_ms + 1e-9
    )


def _phase(pulse):
    return pulse.phase + 2 * np.pi * pulse.frequency * (pulse.center - pulse.start)


def _phase_error(actual, expected):
    return float(abs(np.angle(np.exp(1j * (actual - expected)))))


def classify_preparation(pulses, window_ids, readout_flip, *, t2):
    """Assign pre-train RF a role from its waveform, never from labels.

    Preparation is an inversion-class pulse (nominal flip >=170 degrees) or,
    for T2 families, the first 90/>=178/>=178/90 consecutive block. Pulses no
    larger than the train's own flips (+2 degrees) are conditioning, e.g.
    alpha/2 catalyzation, ramp-up or flip-back. Other RF requires a measured
    RF-free crusher interval to establish an additional spoiled preparation.
    """
    flips = [pulses[j].flip for j in window_ids]
    prep = []
    if t2:
        for i in range(len(window_ids) - 3):
            down, ref1, ref2, up = flips[i : i + 4]
            if (
                abs(down - 90) <= 2
                and ref1 >= 178
                and ref2 >= 178
                and abs(up - 90) <= 2
            ):
                prep = window_ids[i : i + 4]
                break
    else:
        prep = [j for j, flip in zip(window_ids, flips) if flip >= 170]
    rest = [j for j in window_ids if j not in prep]
    conditioning = [
        j for j in rest if pulses[j].flip <= readout_flip + 2 and pulses[j].flip < 170
    ]
    unclassified = [j for j in rest if j not in conditioning]
    return prep, conditioning, unclassified


# A crusher must wrap transverse magnetization by >=2 full cycles across one
# in-plane image voxel (|m| * FOV / N >= 2 on x or y). The crushed coherence's
# echo is then displaced by twice the full sampled k-space width, i.e. >=1.5
# widths beyond its edge, so the ideal-dephasing oracle is a sound model.
# Through-plane (z) moments do not dephase a single-slice 2D object.
CRUSHER_MIN_CYCLES_PER_VOXEL = 2.0
# Pulseq trapezoid amplitudes serialize to six significant digits.
CRUSHER_RELATIVE_SLACK = 1e-5


def crusher_cycles_per_voxel(area, spec):
    """Cycles of phase across one image voxel along each in-plane axis."""
    voxel = np.asarray(spec["fov_mm"], float) / 1000 / np.asarray(spec["matrix"])
    return np.abs(np.asarray(area)[: len(voxel)]) * voxel


def spoiled_preparations(acq, candidates, spec):
    """Additional prep RF may dephase at its following RF-free crusher only.

    The crusher must reach CRUSHER_MIN_CYCLES_PER_VOXEL along an in-plane axis;
    weaker RF-free moments do not establish a spoiled preparation.
    """
    ids, times = [], []
    for j in candidates:
        if j + 1 >= len(acq.pulses):
            continue
        pulse, following = acq.pulses[j : j + 2]
        area = np.diff(
            gradient_integrals(acq.seq, [pulse.end, following.start]), axis=0
        )[0]
        cycles = crusher_cycles_per_voxel(area, spec)
        if np.max(cycles) >= CRUSHER_MIN_CYCLES_PER_VOXEL * (
            1 - CRUSHER_RELATIVE_SLACK
        ):
            ids.append(j)
            times.append(float(following.start))
    return ids, times


def _hs_shape(pulse):
    """Sech amplitude and tanh frequency sweep; RF use/name labels are ignored."""
    mids = (pulse.edges[:-1] + pulse.edges[1:]) / 2
    amplitude = np.abs(pulse.signal)
    if len(amplitude) < 3 or amplitude.max() <= 0:
        return {"valid": False, "reason": "insufficient shaped RF samples"}
    x = 2 * (mids - pulse.start) / (pulse.end - pulse.start) - 1
    normalized = amplitude / amplitude.max()
    # Infer a positive shape scale from the observed envelope, rather than
    # limiting valid HS pulses to a hand-picked beta range. Normalize the model
    # on the same sample lattice, whose centre need not contain x=0.
    observed = (normalized > 0) & (normalized < 1) & (np.abs(x) > 0)
    scales = np.arccosh(1 / normalized[observed]) / np.abs(x[observed])
    scales = scales[np.isfinite(scales) & (scales > 0)]
    initial = float(np.median(scales)) if len(scales) else np.sqrt(np.finfo(float).eps)

    def envelope(beta):
        decay = np.exp(-np.abs(beta * x))
        values = 2 * decay / (1 + decay**2)
        return values / values.max()

    fit = least_squares(
        lambda value: envelope(value[0]) - normalized,
        [initial],
        bounds=(np.finfo(float).tiny, np.inf),
    )
    beta = float(fit.x[0])
    amplitude_error = float(np.sqrt(np.mean(fit.fun**2)))
    frequency = np.gradient(np.unwrap(np.angle(pulse.signal)), mids) / (2 * np.pi)
    basis = np.column_stack((np.ones(len(x)), np.tanh(beta * x)))
    fit = basis @ np.linalg.lstsq(basis, frequency, rcond=None)[0]
    sweep = float(np.ptp(frequency))
    sweep_error = float(np.sqrt(np.mean((frequency - fit) ** 2)) / max(sweep, 1e-15))
    return {
        "valid": bool(amplitude_error <= 0.05 and sweep_error <= 0.05 and sweep > 0),
        "sech_amplitude_rms": amplitude_error,
        "tanh_sweep_relative_rms": sweep_error,
        "sweep_hz": sweep,
        "beta": beta,
    }


def _failed(acq, checks):
    acq.checks = checks
    acq.measurements["frames"] = []
    return acq


def inspect_sequence(path, case):
    seq = Sequence()
    seq.read(str(path))
    empty = [
        int(number)
        for number, events in seq.block_events.items()
        if events[5] and float(seq.adc_library.data[int(events[5])][0]) <= 0
    ]
    if empty:
        # Zero-sample ADC events make every Cartesian reduction undefined.
        detail = {"blocks_with_nonpositive_samples": empty[:20], "min": 1}
        return SimpleNamespace(
            checks=[check("adc_sample_count", False, detail)],
            measurements={"frames": []},
        )
    acq = inspect_cartesian(path, case)
    if family(case) == "water_selective_gre":
        return _inspect_water_selective_gre(acq, case)
    if family(case) == "dwi_epi":
        return _inspect_dwi_epi(acq, case)
    if family(case) == "venc_flow_2d":
        return _inspect_venc_flow_2d(acq, case)
    spec = case["task"]["sequence"]
    timing = spec["temporal"]
    segments_per_frame = timing.get("segments_per_frame", 1)
    rr = timing["rr_interval_ms"] / 1000
    delay = timing["trigger_delay_ms"] / 1000
    window = timing["acquisition_window_ms"] / 1000
    tolerance = max(acq.seq.block_duration_raster, acq.seq.rf_raster_time) * 2
    system = acq.seq.system
    clocks, triggers, adc_offsets, adc_frequencies = 0.0, [], [], []
    for number in acq.seq.block_events:
        block = acq.seq.get_block(number)
        for event in (getattr(block, "trigger", None) or {}).values():
            if event.type == "trigger" and event.channel == "physio1":
                triggers.append(clocks + float(event.delay))
        if (adc := block.adc) is not None:
            gamma, b0 = system.gamma, system.B0
            adc_offsets.append(
                float(adc.phase_offset + adc.phase_ppm * 1e-6 * gamma * b0)
            )
            adc_frequencies.append(
                float(adc.freq_offset + adc.freq_ppm * 1e-6 * gamma * b0)
            )
        clocks += acq.seq.block_durations[number]
    triggers = np.asarray(triggers)
    checks = [
        c
        for c in acq.checks
        if c["name"]
        in {
            "rf_before_adc",
            "echo_in_adc",
            "cartesian",
            "matrix",
            "fov_mm",
            "single_line_per_excitation",
            "slice_thickness_mm",
        }
    ]
    checks.append(
        check("imaging_adc_samples", len(acq.adc) > 0, {"samples": len(acq.adc)})
    )
    checks.append(
        check(
            "ecg_trigger_events",
            len(triggers) > 0,
            {"source": "physio1", "trigger_times_s": triggers.tolist()},
        )
    )
    if not all(item["passed"] for item in checks):
        # Established structural failures make dependent timing/role analysis
        # undefined; return them as a failed verdict rather than reducing over
        # possibly empty or wrapped indices.
        return _failed(acq, checks)
    cadence = np.diff(triggers) / rr
    checks.append(
        check(
            "nominal_rr_cadence",
            np.all(np.abs(cadence - np.rint(cadence)) <= (tolerance + 1e-9) / rr)
            and np.all(cadence >= 1 - (tolerance + 1e-9) / rr),
            {"rr_ms": rr * 1000, "trigger_intervals_rr": cadence.tolist()},
        )
    )
    assignments = np.searchsorted(triggers, acq.adc, side="right") - 1
    valid = assignments >= 0
    elapsed = acq.adc - triggers[np.maximum(assignments, 0)]
    checks.append(
        check(
            "cardiac_adc_window",
            valid.all()
            and np.all(elapsed >= delay - tolerance - 1e-9)
            and np.all(elapsed <= delay + window + tolerance + 1e-9),
            {
                "adc_trigger_delay_ms": [
                    float(elapsed.min() * 1000),
                    float(elapsed.max() * 1000),
                ],
                "required_ms": [delay * 1000, (delay + window) * 1000],
            },
        )
    )
    pulse_centers = np.array([p.center for p in acq.pulses])
    acquired_triggers = np.unique(assignments[valid])
    if len(acquired_triggers) != timing["frames"] * segments_per_frame:
        checks.append(
            check(
                "cardiac_frame_count",
                False,
                {
                    "acquired_windows": len(acquired_triggers),
                    "required_windows": timing["frames"] * segments_per_frame,
                },
            )
        )
        return _failed(acq, checks)
    frames = []
    # Lines per frame (all non-readout grid points) split evenly over segments.
    segment_lines = max(2, int(np.prod(spec["matrix"][1:])) // segments_per_frame)
    preparation_spoil_times = []
    prior_frame_end = 0.0
    prep_centers = []
    for segment_no, trigger_id in enumerate(acquired_triggers):
        frame_no = segment_no // segments_per_frame
        check_start = len(checks)
        line_ids = [
            j
            for j, line in enumerate(acq.lines)
            if np.all(assignments[line] == trigger_id)
        ]
        if len(line_ids) != segment_lines:
            # Each segment acquires exactly its declared share of lines (also
            # guarantees >=2 lines for TR, phase cycling and balance).
            checks.append(
                check(
                    f"frame_{frame_no}_segment_{segment_no % segments_per_frame}"
                    "_acquisition",
                    False,
                    {
                        "complete_lines_in_window": len(line_ids),
                        "required_lines": segment_lines,
                        "trigger_time_s": float(triggers[trigger_id]),
                    },
                )
            )
            return _failed(acq, checks)
        ids = np.concatenate([acq.lines[j] for j in line_ids])
        readout_ids = [
            int(
                np.searchsorted(pulse_centers, acq.adc[acq.lines[j][0]], side="right")
                - 1
            )
            for j in line_ids
        ]
        readout_times = pulse_centers[readout_ids]
        tr = float(np.median(np.diff(readout_times)))
        start = readout_ids[0]
        while (
            start > 0
            and abs(pulse_centers[start] - pulse_centers[start - 1] - tr) <= tolerance
        ):
            start -= 1
        train_start = acq.pulses[start].start
        window_ids = [
            j
            for j, pulse in enumerate(acq.pulses)
            if prior_frame_end <= pulse.start and pulse.end <= train_start - tolerance
        ]
        prep_ids, conditioning_ids, unclassified_ids = classify_preparation(
            acq.pulses,
            window_ids,
            max(acq.pulses[j].flip for j in readout_ids),
            t2=family(case) in {"t2_prepared", "t2_mapping"},
        )
        # Additional spoiled preparation is compatible with coherent bSSFP.
        # Infer its crusher from the RF-free gradient moment; RF labels, carrier
        # frequency and pulse shape do not establish this role.
        spoiled_ids, spoil_times = spoiled_preparations(acq, unclassified_ids, spec)
        preparation_spoil_times.extend(spoil_times)
        unclassified_ids = [j for j in unclassified_ids if j not in spoiled_ids]
        preparations = [acq.pulses[j] for j in prep_ids]
        roles = {
            "preparation_pulse_indices": prep_ids,
            "conditioning_pulse_indices": conditioning_ids,
            "unclassified_pulse_indices": unclassified_ids,
            "unclassified_flips_deg": [acq.pulses[j].flip for j in unclassified_ids],
            "spoiled_preparation_pulse_indices": spoiled_ids,
        }
        starts = np.array([acq.pulses[j].start for j in readout_ids])
        # Every TR, including the last, spans readout RF start to RF start + TR.
        ends = np.r_[starts[1:], starts[-1] + tr]
        areas = gradient_integrals(acq.seq, ends) - gradient_integrals(acq.seq, starts)
        fov = np.r_[np.asarray(spec["fov_mm"]) / 1000, 0.2][:3]
        balance = float(np.max(np.abs(areas) * fov))
        phases = np.array([_phase(acq.pulses[j]) for j in readout_ids])
        alternating = (
            np.max(np.abs(np.abs(np.angle(np.exp(1j * np.diff(phases)))) - np.pi))
            <= 0.01
        )
        # The ADC phase offset must equal the effective RF phase at the pulse
        # centre (event phase, envelope phase and the frequency offset's accrual
        # from pulse start to centre), the convention COMMON_EVALUATION.md states
        # and expected_receiver_phase demodulates with. The ADC's own frequency
        # accrual is not included: comparing the first sample's accumulated
        # demodulation phase would reject ordinary matched RF/ADC tuning.
        receiver_error = max(
            _phase_error(adc_offsets[j], _phase(acq.pulses[pulse]))
            for j, pulse in zip(line_ids, readout_ids, strict=True)
        )
        matched_carrier = all(
            abs(adc_frequencies[j] - acq.pulses[pulse].frequency) <= 1e-6
            for j, pulse in zip(line_ids, readout_ids, strict=True)
        )
        echo_te = np.asarray(acq.measurements["echo_times_s"])[line_ids] - readout_times
        checks.append(
            check(
                f"frame_{frame_no}_balanced_bssfp",
                tr > 0
                and balance <= 0.01
                and alternating
                and receiver_error <= 0.01
                and matched_carrier,
                {
                    "tr_ms": tr * 1000,
                    "max_gradient_area_fov": balance,
                    "alternating_rf": bool(alternating),
                    "receiver_phase_error_rad": receiver_error,
                    "matched_rf_adc_frequency": matched_carrier,
                    "echo_te_ms": (echo_te * 1000).tolist(),
                    "limits": {
                        "max_gradient_area_fov": 0.01,
                        "phase_error_rad": 0.01,
                        "carrier_error_hz": 1e-6,
                    },
                    "conditions": {
                        "positive_tr": bool(tr > 0),
                        "balanced_gradients": balance <= 0.01,
                        "alternating_rf": bool(alternating),
                        "matched_receiver_phase": receiver_error <= 0.01,
                        "matched_rf_adc_frequency": matched_carrier,
                    },
                },
            )
        )
        central = int(ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))])
        # TI uses the same interpolated k-space-centre echo time as TE. The
        # nearest ADC sample is up to half a dwell away on a half-sample lattice.
        central_echo = float(
            acq.measurements["echo_times_s"][
                next(j for j in line_ids if central in acq.lines[j])
            ]
        )
        application = spec.get("application") or {}
        if family(case) == "cardiac_respiratory_bssfp":
            resp_s = timing["respiratory_period_ms"] / 1000
            trig_s = float(triggers[trigger_id])
            phase_s = float(np.mod(trig_s, resp_s))
            phase_err_s = min(phase_s, abs(resp_s - phase_s))
            gating_conditions = {
                "end_expiratory_trigger_phase": phase_err_s <= tolerance + 1e-9,
                "no_magnetization_preparation": not preparations,
                "no_unclassified_rf": not unclassified_ids,
            }
            checks.append(
                check(
                    f"frame_{frame_no}_respiratory_gating",
                    all(gating_conditions.values()),
                    {
                        "trigger_time_s": trig_s,
                        "respiratory_period_s": resp_s,
                        "respiratory_phase_error_ms": phase_err_s * 1000,
                        "timing_tolerance_ms": tolerance * 1000,
                        **roles,
                        "conditions": gating_conditions,
                    },
                )
            )
        elif family(case) in {"adiabatic_inversion", "t1_mapping"}:
            target_ti = (
                application["inversion_time_ms"]
                if family(case) == "adiabatic_inversion"
                else application["inversion_times_ms"][frame_no]
            )
            inversion = preparations[-1] if preparations else None
            inversion_ok = (
                inversion is not None
                and len(preparations) == 1
                and not unclassified_ids
            )
            measured_ti = (
                (central_echo - inversion.center) * 1000
                if inversion is not None
                else None
            )
            ti_ok = _inversion_time_matches(measured_ti, target_ti, tolerance * 1000)
            checks.append(
                check(
                    f"frame_{frame_no}_inversion",
                    inversion_ok and ti_ok,
                    {
                        "ti_ms": measured_ti,
                        "required_ti_ms": target_ti,
                        "preparation_pulses": len(preparations),
                        **roles,
                        "nominal_flip_deg": inversion.flip
                        if inversion is not None
                        else None,
                        "min_nominal_flip_deg": 170,
                        "ti_error_ms": abs(measured_ti - target_ti)
                        if measured_ti is not None
                        else None,
                        "ti_tolerance_ms": tolerance * 1000,
                        "ti_numerical_slack_ms": 1e-9,
                        "conditions": {
                            "exactly_one_inversion": len(preparations) == 1,
                            "no_unclassified_rf": not unclassified_ids,
                            "inversion_time": ti_ok,
                        },
                    },
                )
            )
            if inversion is not None:
                prep_centers.append(inversion.center)
                if family(case) == "adiabatic_inversion":
                    hs = _hs_shape(inversion)
                    checks.append(
                        check(
                            "adiabatic_hyperbolic_secant",
                            hs["valid"]
                            and abs(
                                (inversion.end - inversion.start) * 1000
                                - application["preparation_duration_ms"]
                            )
                            <= tolerance * 1000 + 1e-6,
                            hs,
                        )
                    )
        else:
            target = (
                application["preparation_duration_ms"]
                if family(case) == "t2_prepared"
                else application["preparation_durations_ms"][frame_no]
            )
            valid_prep = not unclassified_ids and (
                not preparations if target == 0 else len(preparations) == 4
            )
            measured = None
            if target > 0 and len(preparations) == 4:
                down, ref1, ref2, up = preparations
                measured = (up.center - down.center) * 1000
                interval = up.center - down.center
                phase_errors = [
                    _phase_error(_phase(up), _phase(down) + np.pi),
                    _phase_error(_phase(ref1), _phase(down) + np.pi / 2),
                    _phase_error(_phase(ref2), _phase(ref1)),
                ]
                valid_prep = valid_prep and (
                    abs(down.flip - 90) <= 2
                    and abs(up.flip - 90) <= 2
                    and ref1.flip >= 178
                    and ref2.flip >= 178
                    and abs(measured - target) <= tolerance * 1000 + 1e-6
                    and abs(ref1.center - down.center - interval / 4)
                    <= tolerance + 1e-9
                    and abs(ref2.center - down.center - 3 * interval / 4)
                    <= tolerance + 1e-9
                    and all(error <= 0.01 for error in phase_errors)
                )
                prep_centers.append(down.center)
            detail = {
                "preparation_ms": measured if target else 0,
                "required_ms": target,
                "preparation_pulses": len(preparations),
                **roles,
                "nominal_flips_deg": [p.flip for p in preparations],
                "timing_tolerance_ms": tolerance * 1000,
                "tip_flip_tolerance_deg": 2,
                "min_refocusing_flip_deg": 178,
                "phase_tolerance_rad": 0.01,
                "conditions": {
                    "preparation_event_count": len(preparations)
                    == (4 if target > 0 else 0),
                    "no_unclassified_rf": not unclassified_ids,
                },
            }
            if target > 0 and len(preparations) == 4:
                quarter_errors = [
                    (ref1.center - down.center - interval / 4) * 1000,
                    (ref2.center - down.center - 3 * interval / 4) * 1000,
                ]
                detail.update(
                    refocuser_center_errors_ms=quarter_errors,
                    phase_errors_rad=phase_errors,
                    preparation_duration_error_ms=abs(measured - target),
                )
                detail["conditions"].update(
                    {
                        "tip_down_flip": abs(down.flip - 90) <= 2,
                        "tip_up_flip": abs(up.flip - 90) <= 2,
                        "first_refocusing_flip": ref1.flip >= 178,
                        "second_refocusing_flip": ref2.flip >= 178,
                        "duration": abs(measured - target) <= tolerance * 1000,
                        "first_refocuser_center": abs(quarter_errors[0])
                        <= tolerance * 1000,
                        "second_refocuser_center": abs(quarter_errors[1])
                        <= tolerance * 1000,
                        "tip_up_phase": phase_errors[0] <= 0.01,
                        "first_refocusing_phase": phase_errors[1] <= 0.01,
                        "second_refocusing_phase": phase_errors[2] <= 0.01,
                    }
                )
            checks.append(
                check(
                    f"frame_{frame_no}_t2_preparation",
                    valid_prep,
                    detail,
                )
            )
        frames.append(
            {
                "adc_indices": ids.tolist(),
                "line_indices": line_ids,
                "central_adc_index": central,
                "preparation_pulse_indices": prep_ids,
                "conditioning_pulse_indices": conditioning_ids,
                "readout_pulse_indices": readout_ids,
                "train_start_s": train_start,
                "trigger_time_s": float(triggers[trigger_id]),
                "segment_index": segment_no % segments_per_frame,
            }
        )
        prior_frame_end = float(acq.adc[ids[-1]])
        if segments_per_frame > 1:
            for item in checks[check_start:]:
                name = item["name"]
                item["name"] = name.replace(
                    f"frame_{frame_no}_",
                    f"frame_{frame_no}_segment_{segment_no % segments_per_frame}_",
                    1,
                )
    for item in checks:
        detail = item.get("detail")
        if isinstance(detail, dict) and "conditions" in detail:
            detail["conditions"] = {k: bool(v) for k, v in detail["conditions"].items()}
            detail["failed_conditions"] = [
                k for k, v in detail["conditions"].items() if not v
            ]
    frames = _assemble_frames(acq, frames, spec, checks, segments_per_frame)
    checks.append(
        check(
            "cardiac_frame_count",
            len(frames) == timing["frames"],
            {"frames": len(frames), "required": timing["frames"]},
        )
    )
    if family(case) in {"t1_mapping", "t2_mapping"}:
        separations = np.diff(prep_centers)
        checks.append(
            check(
                "preparation_recovery",
                np.all(separations >= 3 * rr - tolerance),
                {"preparation_intervals_s": separations.tolist(), "min_s": 3 * rr},
            )
        )
    if family(case) == "cardiac_respiratory_bssfp":
        resp_s = timing["respiratory_period_ms"] / 1000
        seg_triggers = triggers[acquired_triggers]
        seg_diffs = np.diff(seg_triggers)
        checks.append(
            check(
                "respiratory_cycle_separation",
                bool(np.all(seg_diffs >= resp_s - tolerance - 1e-9)),
                {
                    "segment_trigger_intervals_s": seg_diffs.tolist(),
                    "min_s": resp_s,
                },
            )
        )
    acq.checks = checks
    acq.measurements.update(
        frames=frames,
        family=family(case),
        reconstruction="bart_cartesian_fft@1",
        preparation_spoil_times_s=preparation_spoil_times,
    )
    return acq


def _assemble_frames(acq, segments, spec, checks, segments_per_frame):
    """Assemble measured trigger windows; require each imaging sample exactly once."""
    frames = []
    for number in range(spec["temporal"]["frames"]):
        parts = segments[
            number * segments_per_frame : (number + 1) * segments_per_frame
        ]
        ids = np.concatenate([part["adc_indices"] for part in parts]).astype(int)
        unique = len(np.unique(acq.indices[ids], axis=0))
        checks.append(
            check(
                f"frame_{number}_full_cartesian_grid",
                unique == np.prod(spec["matrix"])
                and len(ids) == np.prod(spec["matrix"]),
                {"unique_samples": unique, "samples": len(ids), "segments": len(parts)},
            )
        )
        if segments_per_frame > 1:
            for suffix in (
                "balanced_bssfp",
                "inversion",
                "t2_preparation",
                "respiratory_gating",
            ):
                measured = [
                    item
                    for item in checks
                    if item["name"].startswith(f"frame_{number}_segment_")
                    and item["name"].endswith("_" + suffix)
                ]
                if measured:
                    checks[:] = [item for item in checks if item not in measured]
                    checks.append(
                        check(
                            f"frame_{number}_{suffix}",
                            len(measured) == segments_per_frame
                            and all(item["passed"] for item in measured),
                            {"segments": [item["detail"] for item in measured]},
                        )
                    )
        frame = dict(parts[0])
        for key in (
            "adc_indices",
            "line_indices",
            "preparation_pulse_indices",
            "conditioning_pulse_indices",
            "readout_pulse_indices",
        ):
            frame[key] = [value for part in parts for value in part[key]]
        frame["central_adc_index"] = int(
            ids[np.argmin(np.linalg.norm(acq.k[ids], axis=1))]
        )
        if segments_per_frame > 1:
            frame["segments"] = parts
        frames.append(frame)
    hs = [item for item in checks if item["name"] == "adiabatic_hyperbolic_secant"]
    if len(hs) > 1:
        checks[:] = [item for item in checks if item not in hs]
        checks.append(
            check(
                "adiabatic_hyperbolic_secant",
                all(item["passed"] for item in hs),
                {"segments": [item["detail"] for item in hs]},
            )
        )
    return frames


def frame_acquisition(acq, frame):
    """Subset imaging ADCs while retaining every prior RF preparation and transient."""
    result = copy(acq)
    ids = np.asarray(frame["adc_indices"])
    result.adc, result.adc_phase = acq.adc[ids], acq.adc_phase[ids]
    result.k, result.indices = acq.k[ids], acq.indices[ids]
    result.pulses = [p for p in acq.pulses if p.start < result.adc[-1]]
    inverse = {int(old): new for new, old in enumerate(ids)}
    result.lines = [
        np.array([inverse[int(i)] for i in acq.lines[j]]) for j in frame["line_indices"]
    ]
    result.excitation = acq.excitation[frame["line_indices"]]
    result.checks = []
    result.measurements = {**acq.measurements, "frame": frame, "frames": [frame]}
    return result
