"""S2 identity from RF history and the actual ADC trajectory, never labels."""

import numpy as np
from scipy.spatial import cKDTree

from .s1_sequence import check, family
from .s1_sequence import inspect_sequence as inspect_base

FAMILIES = {
    "epi_single_shot",
    "epi_multishot",
    "radial_readout",
    "spiral_gre",
    "stack_of_stars",
    "propeller_tse",
    "ute_radial",
    "tse_t2w",
    "grase",
}
CARTESIAN = {"epi_single_shot", "epi_multishot", "tse_t2w", "grase"}
REFOCUSED = {"tse_t2w", "propeller_tse", "grase"}


def epi_sampling_mask(matrix, sampling):
    """Uniform physical ky multiples, plus the prescribed central ACS lines."""
    from .s3_sequence import central_indices

    step = sampling["acceleration"]
    if step not in (2, 3):
        raise ValueError("The fixed receive model supports EPI acceleration 2 or 3")
    mask = np.zeros(tuple(matrix), bool)
    mask[:, (np.arange(matrix[1]) - matrix[1] // 2) % int(step) == 0] = True
    mask[:, central_indices(matrix[1], sampling.get("calibration_lines") or 0)] = True
    return mask


def angular_details(angles, period):
    """Circular coverage: no angular gap above 1.6x the uniform spacing."""
    angles = np.sort(np.remainder(angles, period))
    gaps = np.diff(np.r_[angles, angles[0] + period])
    return {
        "passed": bool(len(angles) >= 2 and gaps.max() <= 1.6 * period / len(angles)),
        "count": len(angles),
        "period_rad": float(period),
        "min_gap_rad": float(gaps.min()),
        "max_gap_rad": float(gaps.max()),
        "max_gap_inclusive_rad": float(1.6 * period / len(angles)),
    }


def angular_coverage(angles, period):
    return angular_details(angles, period)["passed"]


def flip_details(values, target):
    values = np.asarray(values)
    return {
        "passed": bool(np.all(np.abs(values - target) <= 2)),
        "target_deg": target,
        "tolerance_deg": 2,
        "min_deg": float(values.min()) if values.size else None,
        "max_deg": float(values.max()) if values.size else None,
        "failed_count": int(np.sum(np.abs(values - target) > 2)),
    }


def line_axis(points):
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axis = vh[0]
    projection = points @ axis
    error = np.max(np.linalg.norm(points - projection[:, None] * axis, axis=1))
    return axis, projection, float(error)


def inspect_sequence(path, case):
    a = inspect_base(path, case)
    spec, fam = case["task"]["sequence"], family(case)
    matrix, fov = np.asarray(spec["matrix"]), np.asarray(spec["fov_mm"]) / 1000
    q = a.k[:, : len(matrix)] * fov
    read = spec.get("readout") or {}
    sampling = spec.get("sampling") or {}
    accelerated = fam == "epi_single_shot" and (sampling.get("acceleration") or 1) > 1
    checks = [
        c
        for c in a.checks
        if c["name"]
        in {
            "readout_samples",
            "readout_bandwidth",
            "rf_before_adc",
            "flip_angle_deg",
            "tr_ms",
        }
        or fam in CARTESIAN
        and c["name"] in {"cartesian", "matrix", "fov_mm"}
    ]
    # Every S2 readout must travel through k-space (at least one sample
    # spacing); a stationary ADC has no readout axis, angle or echo crossing.
    travel = [
        float(np.max(np.ptp(q[ids], axis=0))) if len(ids) else 0.0 for ids in a.lines
    ]
    stationary = [i for i, value in enumerate(travel) if value < 1.0]
    checks.append(
        check(
            "readout_travel",
            not stationary,
            {
                "min_travel_samples": 1.0,
                "measured_min_travel_samples": min(travel, default=0.0),
                "stationary_readouts": stationary[:20],
            },
        )
    )
    if (
        stationary
        or any(len(ids) < 2 for ids in a.lines)
        or any(c["name"] == "readout_samples" and not c["passed"] for c in a.checks)
    ):
        # Trajectory geometry needs at least two samples per readout and real
        # k travel; fail the structure instead of reducing over degenerate data.
        a.checks = checks + [check("sequence_identity", False, fam)]
        a.measurements.update(family=fam, fov_mm=spec["fov_mm"])
        return a
    if accelerated:
        # An undersampled grid cannot infer its integer origin from its extrema.
        # Use the same fixed centred origin and half-sample branch as S3.
        a.offset = np.angle(np.mean(np.exp(2j * np.pi * q), axis=0)) / (2 * np.pi)
        a.offset[(a.offset < 0) & (abs(a.offset + 0.5) <= 0.01)] += 1
        integer = np.rint(q - a.offset).astype(int)
        a.indices = integer + matrix // 2
        residual = float(np.max(abs(q - a.offset - integer)))
        cartesian = residual <= 0.01 and np.max(abs(a.k[:, 2])) <= 0.01 / min(fov)
        cartesian &= all(
            len(ids) == matrix[0]
            and np.all(abs(abs(np.diff(q[ids, 0])) - 1) <= 0.01)
            and np.ptp(q[ids, 1]) <= 0.01
            for ids in a.lines
        )
        checks = [
            item
            for item in checks
            if item["name"] not in {"cartesian", "matrix", "fov_mm"}
        ]
        mask = np.zeros(tuple(matrix), bool)
        in_bounds = bool(((a.indices >= 0) & (a.indices < matrix)).all())
        if in_bounds:
            mask[tuple(a.indices.T)] = True
        unique = int(mask.sum()) == len(a.adc)
        prescribed = epi_sampling_mask(matrix, sampling)
        checks += [
            check(
                "cartesian",
                cartesian,
                {"max_lattice_error_samples": residual, "readout_axis": "x"},
            ),
            check(
                "matrix",
                in_bounds and unique,
                {"matrix": matrix.tolist(), "adc_samples": len(a.adc)},
            ),
            check("fov_mm", cartesian and in_bounds, {"fov_mm": spec["fov_mm"]}),
            check(
                "sampling_pattern",
                np.array_equal(mask, prescribed) and unique,
                {
                    "acceleration": sampling["acceleration"],
                    "calibration_lines": sampling.get("calibration_lines") or 0,
                    "retained_ky_indices": np.flatnonzero(mask.any(axis=0)).tolist(),
                    "required_ky_indices": np.flatnonzero(
                        prescribed.any(axis=0)
                    ).tolist(),
                },
            ),
        ]
        a.measurements.update(
            parallel_imaging=True,
            sampling_mask=mask.tolist(),
            lattice_offset=a.offset.tolist(),
            receive_encoding="s3_trigonometric_comoving@1",
            receive_channels=4,
        )
    exc = [p for p in a.pulses if not p.refocusing]
    groups = [np.flatnonzero(a.excitation == j) for j in np.unique(a.excitation)]
    # Each ADC event can contain a complete curved trajectory. Its effective TE
    # is the acquired sample nearest the origin, not an endpoint interpolation.
    centre_ids = np.array(
        [ids[np.argmin(np.linalg.norm(q[ids, :2], axis=1))] for ids in a.lines]
    )
    if fam in CARTESIAN:
        centres = np.asarray(a.measurements["echo_times_s"])
    else:
        centres = a.adc[centre_ids]
    if fam == "propeller_tse":
        centres = []
        for ids in a.lines:
            direction = q[ids[-1], :2] - q[ids[0], :2]
            projection = q[ids, :2] @ (direction / np.linalg.norm(direction))
            fraction = -projection[0] / (projection[-1] - projection[0])
            centres.append(a.adc[ids[0]] + fraction * (a.adc[ids[-1]] - a.adc[ids[0]]))
        centres = np.asarray(centres)
    a.te = centres - np.array([exc[max(j, 0)].center for j in a.excitation])
    effective_line = int(np.argmin(np.linalg.norm(q[centre_ids], axis=1)))
    te = float(a.te[effective_line] * 1000)
    if spec.get("te_ms"):
        limit = spec["te_ms"]
        # Echo-train tasks prescribe effective TE at the central phase encode.
        values = np.array([te]) if fam in CARTESIAN | {"propeller_tse"} else a.te * 1000
        ok = np.all(values > 0)
        if limit.get("target") is not None:
            ok &= np.all(np.abs(values - limit["target"]) <= limit["tolerance"] + 1e-3)
        if limit.get("max") is not None:
            ok &= np.all(values <= limit["max"] + 1e-3)
        checks.append(check("te_ms", ok, {"effective_te_ms": te, "contract": limit}))
    checks.append(
        check("positive_echo_times", np.all(a.te > 0), {"effective_te_ms": te})
    )
    shots = len(groups)
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
    refs_per_line, ref_residual = [], []
    for j, centre in zip(a.excitation, centres, strict=True):
        start = exc[max(j, 0)].center
        refs = [p for p in a.pulses if p.refocusing and start < p.center < centre]
        delta = -start
        for p in refs:
            delta = -2 * p.center - delta
        refs_per_line.append(len(refs))
        ref_residual.append(float(centre + delta))
    if fam in REFOCUSED:
        tolerance = max(
            20e-6,
            max(
                float(np.min(np.diff(a.adc[ids]))) / 2
                for ids in a.lines
                if len(ids) > 1
            ),
        )
        conditions = {
            "refocusing_before_each_echo": all(n >= 1 for n in refs_per_line),
            "multiple_readouts_per_excitation": all(len(g) >= 2 for g in groups),
        }
        valid = all(n >= 1 for n in refs_per_line)
        if fam != "grase":
            conditions["coherence_time_residual"] = bool(
                np.max(np.abs(ref_residual)) <= tolerance
            )
            valid &= np.max(np.abs(ref_residual)) <= tolerance
        else:
            # The declared GRASE train has two GEs per spin echo. Its total
            # echo count determines the required number of refocusing intervals.
            required_echoes = 2
            required_intervals = read["echo_train_length"] // required_echoes
            valid &= read["echo_train_length"] % required_echoes == 0
            paired_echoes = []
            for g in groups:
                buckets = {}
                for i in g:
                    buckets.setdefault(refs_per_line[i], []).append(i)
                paired_echoes.append(
                    {
                        "distinct_intervals": len(buckets),
                        "required_distinct_intervals": required_intervals,
                        "echoes_per_interval": [len(b) for b in buckets.values()],
                        "required_echoes_per_interval": required_echoes,
                        "straddles_spin_echo": [
                            bool(
                                min(ref_residual[i] for i in b) <= tolerance
                                and max(ref_residual[i] for i in b) >= -tolerance
                            )
                            for b in buckets.values()
                        ],
                    }
                )
                valid &= len(buckets) == required_intervals and all(
                    len(b) == required_echoes for b in buckets.values()
                )
                valid &= all(
                    min(ref_residual[i] for i in b) <= tolerance
                    and max(ref_residual[i] for i in b) >= -tolerance
                    for b in buckets.values()
                )
            # The per-shot measurements identify which part of the original
            # paired-GE condition failed; they do not alter its decision.
            conditions["grase_pairs"] = paired_echoes
        valid &= all(len(g) >= 2 for g in groups)
        valid &= np.all(np.abs(a.flips - 90) <= 2) and all(
            abs(p.flip - 180) <= 2 for p in a.pulses if p.refocusing
        )
        conditions["excitation_rotation"] = flip_details(a.flips, 90)
        conditions["refocusing_rotation"] = flip_details(
            [p.flip for p in a.pulses if p.refocusing], 180
        )
        checks.append(
            check(
                "refocused_echo_train",
                valid,
                {
                    "conditions": conditions,
                    "coherence_time_tolerance_s": tolerance,
                    "refocusing_counts": refs_per_line,
                    "coherence_time_residual_s": ref_residual,
                },
            )
        )
        if read.get("echo_spacing_ms"):
            spacings = np.concatenate([np.diff(centres[g]) for g in groups]) * 1000
            checks.append(
                check(
                    "echo_spacing_ms",
                    np.all(np.abs(spacings - read["echo_spacing_ms"]) <= 0.05),
                    spacings.tolist(),
                )
            )
    else:
        checks.append(
            check(
                "gradient_echo",
                not any(p.refocusing for p in a.pulses)
                and np.all((a.flips > 0.1) & (a.flips < 135)),
                {
                    "refocusing_events": sum(p.refocusing for p in a.pulses),
                    "required_refocusing_events": 0,
                    "excitation_min_deg": float(a.flips.min()),
                    "excitation_max_deg": float(a.flips.max()),
                    "excitation_exclusive_bounds_deg": [0.1, 135],
                },
            )
        )
    if fam.startswith("epi") or fam == "grase":
        # Disclosed contract: EPI and GRASE read along x with unit sample steps
        # and bounded transverse motion; polarity is the sign of x travel.
        x_steps = [np.abs(np.abs(np.diff(q[ids, 0])) - 1) for ids in a.lines]
        transverse = [np.ptp(q[ids, 1:], axis=0) for ids in a.lines]
        x_readout = all(
            len(ids) == matrix[0] and np.all(step <= 0.01) and np.all(motion <= 0.01)
            for ids, step, motion in zip(a.lines, x_steps, transverse, strict=True)
        )
        checks.append(
            check(
                "readout_axis",
                x_readout,
                {
                    "required_axis": "x",
                    "samples_per_readout": int(matrix[0]),
                    "max_x_step_error_samples": float(max(map(np.max, x_steps))),
                    "max_transverse_motion_samples": float(
                        max(map(np.max, transverse))
                    ),
                    "limit_samples": 0.01,
                },
            )
        )
        polarity = np.array([np.sign(q[ids[-1], 0] - q[ids[0], 0]) for ids in a.lines])
        alternating = all(
            np.all(polarity[g][1:] * polarity[g][:-1] == -1) for g in groups
        )
        valid = alternating and all(len(g) >= 2 for g in groups)
        epi_detail = {}
        if fam == "epi_single_shot":
            valid &= shots == 1
            if accelerated:
                ky = q[centre_ids, 1]
                # One acquiring excitation. Other RF (e.g. CHESS + spoiler) must
                # end before it starts: no RF may follow it inside the shot, and
                # no other excitation is acquired. Unspoiled preparation
                # coherence is judged physically by the spoiling probe and image.
                shot = exc[int(a.excitation[0])] if shots == 1 else None
                preparation = [p for p in a.pulses if p is not shot]
                prepared = shot is not None and all(
                    p.end <= shot.start + 1e-12 for p in preparation
                )
                valid &= prepared and bool(
                    np.all(np.diff(ky) > 0) or np.all(np.diff(ky) < 0)
                )
                epi_detail = {
                    "acquiring_excitations": shots,
                    "preparation_rf_pulses": len(preparation),
                    "preparation_rf_before_excitation": bool(prepared),
                }
        elif fam == "epi_multishot":
            valid &= shots >= 2
            for g in groups:
                ky = q[[centre_ids[i] for i in g], 1]
                steps = np.diff(ky)
                valid &= (
                    len(ky) > 1
                    and np.all(np.abs(np.abs(steps) - shots) <= 0.02)
                    and (np.all(steps > 0) or np.all(steps < 0))
                )
        checks.append(
            check(
                "epi",
                valid,
                {"shots": shots, "alternating_polarity": alternating, **epi_detail},
            )
        )
    if fam in {"radial_readout", "ute_radial", "stack_of_stars"}:
        angles, valid = [], True
        line_errors, extents, steps, starts, backsteps = [], [], [], [], []
        for ids in a.lines:
            axis, proj, error = line_axis(q[ids, :2])
            line_errors.append(error)
            extents.append([float(proj.min()), float(proj.max())])
            steps.append(float(np.max(np.abs(np.diff(proj)))))
            starts.append(float(np.linalg.norm(q[ids[0], :2])))
            backsteps.append(float(np.min(np.diff(np.abs(proj)))))
            valid &= error <= 0.05 and np.max(np.abs(proj)) >= matrix[0] / 2 - 0.75
            valid &= np.max(np.abs(np.diff(proj))) <= 1.05
            centre_out = fam == "ute_radial" or read.get("center_out")
            valid &= (
                (
                    np.linalg.norm(q[ids[0], :2]) <= 0.6
                    and np.all(np.diff(np.abs(proj)) >= -0.05)
                )
                if centre_out
                else (
                    proj.min() < -matrix[0] / 2 + 0.75
                    and proj.max() > matrix[0] / 2 - 0.75
                )
            )
            direction = q[ids[-1], :2] - q[ids[0], :2]
            angles.append(float(np.arctan2(direction[1], direction[0])))
        extent = matrix[0] / 2 - 0.75
        conditions = {
            "straightness": {
                "passed": max(line_errors) <= 0.05,
                "max_error_samples": max(line_errors),
                "max": 0.05,
            },
            "sample_step": {
                "passed": max(steps) <= 1.05,
                "max_step_samples": max(steps),
                "max": 1.05,
            },
            "radial_extent": {
                "passed": bool(
                    all(max(abs(lo), abs(hi)) >= extent for lo, hi in extents)
                    if centre_out
                    else all(lo < -extent and hi > extent for lo, hi in extents)
                ),
                "least_negative_endpoint_samples": max(x[0] for x in extents),
                "least_positive_endpoint_samples": min(x[1] for x in extents),
                "required_extent_samples": float(extent),
                "two_sided_strict": not bool(centre_out),
            },
        }
        if centre_out:
            conditions["center_out"] = {
                "passed": max(starts) <= 0.6 and min(backsteps) >= -0.05,
                "max_start_radius_samples": max(starts),
                "max_start": 0.6,
                "min_absolute_radial_step_samples": min(backsteps),
                "min_step": -0.05,
            }
        if fam == "stack_of_stars":
            z = np.array([np.mean(q[ids, 2]) for ids in a.lines])
            partitions = np.rint(z).astype(int)
            partitioned = bool(
                np.max(np.abs(z - partitions)) <= 0.01
                and set(partitions) == set(range(-matrix[2] // 2, matrix[2] // 2))
                and all(np.ptp(q[ids, 2]) <= 0.01 for ids in a.lines)
            )
            valid &= partitioned
            angle_groups = [
                np.flatnonzero(partitions == p) for p in np.unique(partitions)
            ]
            conditions["partitions"] = {
                "passed": partitioned,
                "measured": sorted(map(int, set(partitions))),
                "required": list(range(-int(matrix[2]) // 2, int(matrix[2]) // 2)),
                "max_integer_error_samples": float(np.max(np.abs(z - partitions))),
                "max_within_readout_variation_samples": max(
                    float(np.ptp(q[ids, 2])) for ids in a.lines
                ),
                "tolerance_samples": 0.01,
            }
        else:
            angle_groups = [np.arange(len(a.lines))]
        conditions["angular_coverage"] = [
            angular_details(
                np.asarray(angles)[g], 2 * np.pi if fam == "ute_radial" else np.pi
            )
            for g in angle_groups
        ]
        for g, coverage in zip(angle_groups, conditions["angular_coverage"]):
            valid &= coverage["passed"]
            if read.get("spokes"):
                valid &= len(g) == read["spokes"]
        conditions["spoke_count_per_partition"] = {
            "passed": not read.get("spokes")
            or all(len(g) == read["spokes"] for g in angle_groups),
            "measured": [len(g) for g in angle_groups],
            "required": read.get("spokes"),
        }
        conditions["one_readout_per_excitation"] = all(len(g) == 1 for g in groups)
        valid &= all(len(g) == 1 for g in groups)
        checks.append(
            check(
                "radial",
                valid,
                {
                    "conditions": conditions,
                    "spokes": len(a.lines),
                    "angles_rad": angles,
                },
            )
        )
        if fam == "ute_radial":
            times = np.concatenate([np.linspace(p.start, p.end, 65) for p in exc])
            nonselective = all(
                grad is None or np.max(np.abs(grad(times))) <= 1e-6
                for grad in a.seq.get_gradients()
            )
            checks.append(
                check(
                    "ute",
                    np.all(a.te <= 0.001) and nonselective,
                    "Centre-out TE <=1 ms; nonselective excitation",
                )
            )
    if fam == "spiral_gre":
        valid, turns, angles = True, [], []
        trajectory_details = []
        for ids in a.lines:
            xy = q[ids, :2]
            r = np.linalg.norm(xy, axis=1)
            theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
            keep = r > 0.5
            winding = (
                abs(theta[keep][-1] - theta[keep][0]) / (2 * np.pi) if keep.any() else 0
            )
            turns.append(float(winding))
            angles.append(float(theta[keep][0]) if keep.any() else 0.0)
            max_step = np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))
            valid &= len(ids) >= matrix[0] and max_step <= 1.1
            trajectory = {
                "samples": len(ids),
                "min_samples": int(matrix[0]),
                "max_step_samples": float(max_step),
                "max_step": 1.1,
            }
            # A radius-versus-angle fit needs angular travel; a straight or
            # stationary trajectory has none and fails without a fit.
            fitted = keep.sum() > 2 and np.ptp(theta[keep]) > 1e-6
            fit = np.polyfit(theta[keep], r[keep], 1) if fitted else None
            residual = (
                float(np.sqrt(np.mean((np.polyval(fit, theta[keep]) - r[keep]) ** 2)))
                if fitted
                else None
            )
            valid &= r[0] <= 0.6 and r[-1] >= matrix[0] / 2 - 0.75 and winding >= 1
            valid &= (
                np.mean(np.diff(r) >= -0.05) >= 0.95
                and residual is not None
                and residual <= 0.1 * r.max()
            )
            if read.get("turns"):
                valid &= abs(winding - read["turns"]) <= 0.5
            trajectory.update(
                start_radius_samples=float(r[0]),
                max_start_radius=0.6,
                end_radius_samples=float(r[-1]),
                min_end_radius=float(matrix[0] / 2 - 0.75),
                winding_turns=float(winding),
                min_winding_turns=1,
                required_turns=read.get("turns"),
                turn_tolerance=0.5,
                nondecreasing_fraction=float(np.mean(np.diff(r) >= -0.05)),
                min_nondecreasing_fraction=0.95,
                radial_step_tolerance=0.05,
                archimedean_fit_rms_samples=residual,
                max_fit_rms_samples=float(0.1 * r.max()),
            )
            trajectory_details.append(trajectory)
        coverage = angular_details(angles, 2 * np.pi)
        if read.get("interleaves"):
            valid &= len(a.lines) == read["interleaves"]
        valid &= coverage["passed"]
        valid &= all(len(g) == 1 for g in groups)
        checks.append(
            check(
                fam,
                valid,
                {
                    "trajectories": trajectory_details,
                    "interleaves": len(a.lines),
                    "required_interleaves": read.get("interleaves"),
                    "one_readout_per_excitation": all(len(g) == 1 for g in groups),
                    "angular_coverage": coverage,
                    "turns": turns,
                },
            )
        )
    if fam == "propeller_tse":
        valid, angles = True, []
        blade_details = []
        for g in groups:
            ids = a.lines[g[0]]
            axis, _, _ = line_axis(q[ids, :2] - q[ids, :2].mean(axis=0))
            normal = np.array([-axis[1], axis[0]])
            offsets, transverse_errors, step_errors, endpoints = [], [], [], []
            for i in g:
                xy = q[a.lines[i], :2]
                t, u = xy @ axis, xy @ normal
                transverse_errors.append(float(np.ptp(u)))
                step_errors.append(float(np.max(np.abs(np.abs(np.diff(t)) - 1))))
                endpoints.append([float(t.min()), float(t.max())])
                valid &= (
                    np.ptp(u) <= 0.05
                    and t.min() <= -matrix[0] / 2 + 0.75
                    and t.max() >= matrix[0] / 2 - 0.75
                )
                valid &= np.all(np.abs(np.abs(np.diff(t)) - 1) <= 0.02)
                offsets.append(float(u.mean()))
            offsets = np.sort(offsets)
            valid &= (
                len(offsets) >= 2
                and np.all(np.abs(np.diff(offsets) - 1) <= 0.05)
                and abs(offsets.mean()) <= 0.6
            )
            if read.get("blade_width_lines"):
                valid &= len(g) == read["blade_width_lines"]
            blade_details.append(
                {
                    "lines": len(g),
                    "required_lines": read.get("blade_width_lines"),
                    "max_transverse_variation_samples": max(transverse_errors),
                    "max_transverse_variation": 0.05,
                    "least_negative_endpoint_samples": max(x[0] for x in endpoints),
                    "least_positive_endpoint_samples": min(x[1] for x in endpoints),
                    "required_extent_samples": float(matrix[0] / 2 - 0.75),
                    "max_unit_step_error_samples": max(step_errors),
                    "max_unit_step_error": 0.02,
                    "line_offsets_samples": offsets.tolist(),
                    "required_offset_step": 1,
                    "offset_step_tolerance": 0.05,
                    "mean_offset_samples": float(offsets.mean()),
                    "max_abs_mean_offset": 0.6,
                }
            )
            angles.append(float(np.arctan2(axis[1], axis[0])))
        coverage = angular_details(angles, np.pi)
        valid &= coverage["passed"]
        checks.append(
            check(
                "propeller",
                valid,
                {
                    "angular_coverage": coverage,
                    "blade_geometry": blade_details,
                    "blades": shots,
                    "angles_rad": angles,
                },
            )
        )
    if fam not in CARTESIAN:
        # Require acquired resolution and angular/partition coverage in physical
        # units. Missing sectors and scaled-FOV trajectories cannot self-reference.
        grid = np.indices(tuple(matrix)).reshape(len(matrix), -1).T - matrix // 2
        disk = np.linalg.norm(grid[:, :2], axis=1) <= min(matrix[:2]) / 2 - 1
        distances = cKDTree(q).query(grid[disk])[0]
        covered = bool(
            distances.max() <= 1.5
            and (len(matrix) == 3 or np.max(np.abs(a.k[:, 2])) <= 0.05 / min(fov))
        )
        checks.extend(
            [
                check(
                    "matrix",
                    covered,
                    {"max_gap_samples": float(distances.max()), "limit": 1.5},
                ),
                check(
                    "fov_mm",
                    covered,
                    {
                        "fov_mm": spec["fov_mm"],
                        "max_gap_samples": float(distances.max()),
                    },
                ),
            ]
        )
        a.offset = np.zeros(len(matrix))
    checks.append(check("sequence_identity", all(c["passed"] for c in checks), fam))
    if fam not in CARTESIAN:
        a.measurements["reconstruction_band_samples"] = int(
            np.sum(np.all(np.abs(q / matrix) <= 0.5 + 1e-5, axis=1))
        )
        a.measurements["reconstruction_total_samples"] = len(a.adc)
    a.checks = checks
    a.measurements.update(
        family=fam,
        te_ms=(a.te * 1000).tolist(),
        echo_times_s=centres.tolist(),
        effective_te_ms=te,
        shots=shots,
        fov_mm=spec["fov_mm"],
        centre_sample_indices=centre_ids.tolist(),
        reconstruction="bart_sampling_sense@1"
        if accelerated
        else "bart_cartesian_fft@1"
        if fam in CARTESIAN
        else "bart_nufft@1",
    )
    return a
