"""S2 identity from RF history and the actual ADC trajectory, never labels."""

from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from .s1_sequence import check, family
from .s1_sequence import inspect_sequence as inspect_base

FAMILIES = {
    "epi_single_shot",
    "epi_multishot",
    "radial_readout",
    "spiral_gre",
    "rosette_gre",
    "concentric_rings",
    "stack_of_stars",
    "propeller_tse",
    "ute_radial",
    "tse_t2w",
    "grase",
}
CARTESIAN = {"epi_single_shot", "epi_multishot", "tse_t2w", "grase"}
REFOCUSED = {"tse_t2w", "propeller_tse", "grase"}


def angular_coverage(angles, period):
    angles = np.sort(np.remainder(angles, period))
    gaps = np.diff(np.r_[angles, angles[0] + period])
    return bool(
        len(angles) >= 2
        and gaps.min() > 0.1 * period / len(angles)
        and gaps.max() <= 1.6 * period / len(angles)
    )


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
    detailed = case["task"].get("prompt_mode") != "research"
    checks = [
        c
        for c in a.checks
        if c["name"] in {"rf_before_adc", "flip_angle_deg", "tr_ms"}
        or fam in CARTESIAN
        and c["name"] in {"cartesian", "matrix", "fov_mm"}
    ]
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
        valid = all(n >= 1 for n in refs_per_line)
        if fam != "grase":
            valid &= np.max(np.abs(ref_residual)) <= tolerance
        else:
            # A GRASE refocusing interval contains >=2 opposite-polarity GEs.
            for g in groups:
                buckets = {}
                for i in g:
                    buckets.setdefault(refs_per_line[i], []).append(i)
                valid &= len(buckets) >= 2 and all(
                    len(b) >= 2 for b in buckets.values()
                )
                valid &= all(
                    min(ref_residual[i] for i in b) <= tolerance
                    and max(ref_residual[i] for i in b) >= -tolerance
                    for b in buckets.values()
                )
        valid &= all(len(g) >= 2 for g in groups)
        if detailed:
            valid &= np.all(np.abs(a.flips - 90) <= 2) and all(
                abs(p.flip - 180) <= 2 for p in a.pulses if p.refocusing
            )
        checks.append(
            check(
                "refocused_echo_train",
                valid,
                {
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
                "Non-refocused excitation/readout",
            )
        )
    if fam.startswith("epi") or fam == "grase":
        polarity = np.array([np.sign(q[ids[-1], 0] - q[ids[0], 0]) for ids in a.lines])
        alternating = all(
            np.all(polarity[g][1:] * polarity[g][:-1] == -1) for g in groups
        )
        valid = alternating and all(len(g) >= 2 for g in groups)
        if fam == "epi_single_shot":
            valid &= shots == 1
        elif fam == "epi_multishot":
            valid &= shots >= 2
            if detailed:
                for g in groups:
                    ky = q[[centre_ids[i] for i in g], 1]
                    valid &= len(ky) > 1 and np.all(np.abs(np.diff(ky) - shots) <= 0.02)
        checks.append(
            check("epi", valid, {"shots": shots, "alternating_polarity": alternating})
        )
    if fam in {"radial_readout", "ute_radial", "stack_of_stars"}:
        angles, valid = [], True
        for ids in a.lines:
            axis, proj, error = line_axis(q[ids, :2])
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
        if fam == "stack_of_stars":
            z = np.array([np.mean(q[ids, 2]) for ids in a.lines])
            partitions = np.rint(z).astype(int)
            valid &= np.max(np.abs(z - partitions)) <= 0.01 and set(partitions) == set(
                range(-matrix[2] // 2, matrix[2] // 2)
            )
            valid &= all(np.ptp(q[ids, 2]) <= 0.01 for ids in a.lines)
            angle_groups = [
                np.flatnonzero(partitions == p) for p in np.unique(partitions)
            ]
        else:
            angle_groups = [np.arange(len(a.lines))]
        for g in angle_groups:
            valid &= angular_coverage(
                np.asarray(angles)[g], 2 * np.pi if fam == "ute_radial" else np.pi
            )
            if read.get("spokes"):
                valid &= len(g) == read["spokes"]
        valid &= all(len(g) == 1 for g in groups)
        checks.append(
            check("radial", valid, {"spokes": len(a.lines), "angles_rad": angles})
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
                    np.all(a.te <= 0.001)
                    and nonselective
                    and all(p.end - p.start <= 0.0005 for p in exc),
                    "Centre-out TE <=1 ms; nonselective RF <=0.5 ms",
                )
            )
    if fam in {"spiral_gre", "rosette_gre", "concentric_rings"}:
        valid, turns, lobes, radii, angles = True, [], [], [], []
        for ids in a.lines:
            xy = q[ids, :2]
            r = np.linalg.norm(xy, axis=1)
            theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
            keep = r > (0.05 if fam == "concentric_rings" else 0.5)
            winding = (
                abs(theta[keep][-1] - theta[keep][0]) / (2 * np.pi) if keep.any() else 0
            )
            turns.append(float(winding))
            radii.append(float(r.mean()))
            angles.append(float(theta[keep][0]) if keep.any() else 0.0)
            valid &= (
                len(ids) >= matrix[0]
                and np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)) <= 1.1
            )
            if fam == "spiral_gre":
                fit = (
                    np.polyfit(theta[keep], r[keep], 1) if keep.sum() > 2 else [0, 1e6]
                )
                residual = (
                    np.sqrt(np.mean((np.polyval(fit, theta[keep]) - r[keep]) ** 2))
                    if keep.any()
                    else 1e6
                )
                valid &= r[0] <= 0.6 and r[-1] >= matrix[0] / 2 - 0.75 and winding >= 1
                valid &= (
                    np.mean(np.diff(r) >= -0.05) >= 0.95 and residual <= 0.1 * r.max()
                )
                if read.get("turns"):
                    valid &= abs(winding - read["turns"]) <= 0.5
            elif fam == "rosette_gre":
                peaks = find_peaks(r, prominence=max(r.max() * 0.4, 0.1))[0]
                lobes.append(len(peaks))
                valid &= len(peaks) >= 2 and r.max() >= matrix[0] / 2 - 0.75
                valid &= (
                    all(r[left:right].min() <= 0.6 for left, right in pairwise(peaks))
                    and np.ptp(theta) >= np.pi
                )
                if read.get("rosette_lobes"):
                    valid &= len(peaks) == read["rosette_lobes"]
            else:
                valid &= np.ptp(r) <= 0.06 and (
                    r.mean() < 0.1 or winding >= 1 - 2 / len(ids)
                )
        if read.get("interleaves"):
            valid &= len(a.lines) == read["interleaves"]
        if fam in {"spiral_gre", "rosette_gre"}:
            valid &= angular_coverage(angles, 2 * np.pi)
        if fam == "concentric_rings":
            ordered = np.sort(radii)
            valid &= (
                len(ordered) >= 2
                and ordered[0] <= 0.6
                and ordered[-1] >= matrix[0] / 2 - 0.75
                and np.all((np.diff(ordered) > 0.1) & (np.diff(ordered) <= 1.1))
            )
            if read.get("rings"):
                valid &= len(a.lines) == read["rings"]
        valid &= all(len(g) == 1 for g in groups)
        checks.append(
            check(fam, valid, {"turns": turns, "lobes": lobes, "radii_samples": radii})
        )
    if fam == "propeller_tse":
        valid, angles = True, []
        for g in groups:
            ids = a.lines[g[0]]
            axis, _, _ = line_axis(q[ids, :2] - q[ids, :2].mean(axis=0))
            normal = np.array([-axis[1], axis[0]])
            offsets = []
            for i in g:
                xy = q[a.lines[i], :2]
                t, u = xy @ axis, xy @ normal
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
            angles.append(float(np.arctan2(axis[1], axis[0])))
        valid &= angular_coverage(angles, np.pi)
        checks.append(
            check("propeller", valid, {"blades": shots, "angles_rad": angles})
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
        reconstruction="bart_cartesian_fft@1"
        if fam in CARTESIAN
        else "bart_nufft@1",
    )
    return a
