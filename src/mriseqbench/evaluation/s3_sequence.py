"""S3 Cartesian sampling identity inferred from RF, gradients and ADC nodes."""

from itertools import product
from types import SimpleNamespace

import numpy as np

from .s1_sequence import (
    MissingExcitation,
    check,
    family,
    gradient_integrals,
    midpoint_tolerance,
)
from .s1_sequence import inspect_sequence as inspect_base

PARALLEL = {
    "sense_sampling",
    "grappa_sampling",
    "caipirinha",
    "compressed_sensing",
    "variable_density",
}
REDUCED_FOV = {"reduced_fov", "rfov_epi"}
FAMILIES = PARALLEL | REDUCED_FOV | {"partial_fourier", "elliptical_kspace"}


def receive_model(fam):
    if fam == "variable_density":
        return "s3_gaussian_y8_comoving@1"
    return "s3_trigonometric_comoving@1" if fam in PARALLEL else "uniform@1"


def rf_block_endpoints(seq):
    """Physical end of each RF-containing block, including authored rewinding."""
    ends, clock = {}, 0.0
    for number in seq.block_events:
        end = clock + seq.block_durations[number]
        if seq.get_block(number).rf is not None:
            ends[clock] = end
        clock = end
    return ends


def central_indices(n, width):
    return np.arange(n // 2 - width // 2, n // 2 - width // 2 + width)


def prescribed_mask(fam, matrix, sampling):
    """Version-1 detailed mask, inclusive of ACS in the nominal sample budget.

    Poisson-disc is a one-dimensional, radius-two hard-core draw outside ACS.
    Linear density and the RNG algorithm are frozen here, not delegated to an
    agent library. Returned coordinates are array indices, not Pulseq labels.
    """
    _, ny, *nz = matrix
    mask = np.zeros(matrix, bool)
    width = sampling.get("calibration_lines") or 0
    central = central_indices(ny, width)
    if fam in {"sense_sampling", "grappa_sampling"}:
        mask[:, :: int(sampling.get("acceleration") or 2)] = True
        mask[:, central] = True
    elif fam == "caipirinha":
        # R_y=R_z=2; shift ky on successive *retained* kz partitions.
        for j, z in enumerate(range(0, nz[0], 2)):
            mask[:, (j * sampling.get("caipi_shift", 1)) % 2 :: 2, z] = True
        for y in central:
            mask[:, y, central_indices(nz[0], width)] = True
    elif fam in {"compressed_sensing", "variable_density"}:
        count = max(width, int(np.ceil(ny / (sampling.get("acceleration") or 3))))
        if count >= ny or width < 1 or count <= width:
            raise ValueError("Incoherent mask needs ACS and peripheral undersampling")
        retained = list(central)
        candidates = np.setdiff1d(np.arange(ny), central)
        rng = np.random.Generator(np.random.PCG64(sampling.get("mask_seed") or 0))
        while len(retained) < count:
            if not len(candidates):
                raise ValueError("Poisson-disc radius is infeasible for this mask")
            probability = 1 - np.abs(candidates - ny // 2) / (ny / 2 + 1)
            y = int(rng.choice(candidates, p=probability / probability.sum()))
            retained.append(y)
            candidates = candidates[
                abs(candidates - y) >= (2 if fam == "compressed_sensing" else 1)
            ]
        mask[:, retained] = True
    elif fam == "partial_fourier":
        count = int(np.ceil(ny * sampling.get("partial_fourier_fraction", 0.75)))
        mask[:, ny - count :] = True
    elif fam in REDUCED_FOV:
        step = round(1 / sampling.get("reduced_fov_fraction", 0.5))
        if step < 2:
            raise ValueError("Reduced FOV must be smaller than the prescribed FOV")
        mask[:, np.arange(ny)[(np.arange(ny) - ny // 2) % step == 0]] = True
    elif fam == "elliptical_kspace":
        y, z = np.meshgrid(
            (np.arange(ny) - ny // 2) / (ny / 2),
            (np.arange(nz[0]) - nz[0] // 2) / (nz[0] / 2),
            indexing="ij",
        )
        mask[:] = y**2 + z**2 <= 1 + 1e-12
    else:
        raise ValueError(f"Unknown S3 family: {fam}")
    return mask


def calibration_width(mask):
    lines = mask.all(axis=0)
    if lines.ndim != 1:
        return 0
    c = len(lines) // 2
    if not lines[c]:
        return 0
    lo, hi = c, c + 1
    while lo > 0 and lines[lo - 1]:
        lo -= 1
    while hi < len(lines) and lines[hi]:
        hi += 1
    return hi - lo


def lattice_offset(q, matrix, expected):
    """Fractional lattice origin per axis, in (-0.5, 0.5] by default.

    The circular mean has a branch cut at half a sample: within 0.01 sample of
    it, the -0.5-side and +0.5-side representatives index the same nodes one
    grid point apart. Choose the representative that keeps every sample on
    the grid, then the one matching more prescribed nodes, then the circular
    mean itself (so a legitimate offset of e.g. -0.495 is never rebranched).
    A band reaching +N/2 may also shift by whole samples when the shifted
    samples equal the same samples wrapped modulo N (+N/2 aliases to -N/2);
    the offset phase then reconstructs exactly that wrap.
    """
    mean = np.angle(np.mean(np.exp(2j * np.pi * q), axis=0)) / (2 * np.pi)
    mean[np.isclose(mean, -0.5, atol=1e-5)] = 0.5
    options = [
        [value]
        + ([value + 1] if value < 0 and abs(value + 0.5) <= 0.01 else [])
        + ([value - 1] if value > 0 and abs(value - 0.5) <= 0.01 else [])
        for value in mean
    ]
    base = np.rint(q - mean).astype(int) + matrix // 2
    if base.min() >= 0 and np.all(base.max(axis=0) <= matrix):
        wrapped = np.zeros(tuple(matrix), bool)
        wrapped[tuple((base % matrix).T)] = True
        for axis, n in enumerate(matrix):
            if base[:, axis].max() != n:
                continue
            for shift in range(1, base[:, axis].min() + 1):
                step = np.eye(len(matrix), dtype=int)[axis] * shift
                shifted = np.zeros(tuple(matrix), bool)
                shifted[tuple(((base - step) % matrix).T)] = True
                if np.array_equal(shifted, wrapped):
                    options[axis].append(mean[axis] + shift)
    best, best_score = mean, None
    for choice in product(*options):
        offset = np.asarray(choice)
        indices = np.rint(q - offset).astype(int) + matrix // 2
        inside = ((indices >= 0) & (indices < matrix)).all(axis=1)
        matched = int(np.sum(expected[tuple(indices[inside].T)]))
        score = (bool(inside.all()), matched, int(np.sum(offset == mean)))
        if best_score is None or score > best_score:
            best, best_score = offset, score
    return best


def inspect_sequence(path, case):
    try:
        a = inspect_base(path, case)
    except MissingExcitation as exc:
        return SimpleNamespace(
            checks=[check("rf_before_adc", False, str(exc))],
            measurements={
                "family": family(case),
                "reconstruction": "bart_sampling_sense@1",
            },
        )
    spec, fam = case["task"]["sequence"], family(case)
    matrix = np.asarray(spec["matrix"])
    dims = len(matrix)
    fov = np.asarray(spec["fov_mm"]) / 1000
    if fam == "rfov_epi":
        # A shaped 2D excitation prepares a spatial state at its RF endpoint.
        # Encoding starts there; a hard-pulse RF-centre k reset is inapplicable.
        excitation_pulses = [p for p in a.pulses if not p.refocusing]
        block_ends = rf_block_endpoints(a.seq)
        origins = np.array(
            [
                block_ends[excitation_pulses[max(a.excitation[j], 0)].block_start]
                for j, ids in enumerate(a.lines)
                for _ in ids
            ]
        )
        a.k = gradient_integrals(a.seq, a.adc) - gradient_integrals(a.seq, origins)
    q = a.k[:, :dims] * fov
    # Fixed centred physical grid. S3's asymmetric support cannot estimate its
    # origin from its extrema: doing so silently recentres partial acquisitions.
    expected = prescribed_mask(fam, tuple(matrix), spec.get("sampling") or {})
    offset = lattice_offset(q, matrix, expected)
    integer = np.rint(q - offset).astype(int)
    indices = integer + matrix // 2
    in_bounds = bool(((indices >= 0) & (indices < matrix)).all())
    residual = float(np.max(abs(q - offset - integer)))
    lattice = residual <= 0.01 and (
        dims == 3 or np.max(abs(a.k[:, 2])) <= 0.01 / min(fov)
    )
    straight = all(
        len(ids) >= 2
        and len(ids) == matrix[0]
        and np.all(abs(abs(np.diff(q[ids, 0])) - 1) <= 0.01)
        and np.max(np.ptp(q[ids, 1:], axis=0)) <= 0.01
        for ids in a.lines
    )
    mask = np.zeros(tuple(matrix), bool)
    if in_bounds:
        mask[tuple(indices.T)] = True
    no_duplicates = int(mask.sum()) == len(indices)
    centres = np.array([ids[np.argmin(abs(q[ids, 0]))] for ids in a.lines])
    checks = [
        c
        for c in a.checks
        if c["name"]
        in {
            "readout_samples",
            "readout_bandwidth",
            "rf_before_adc",
            "echo_in_adc",
            "te_ms",
            "tr_ms",
            "flip_angle_deg",
        }
    ]
    checks += [
        check(
            "cartesian",
            lattice and straight,
            {"max_lattice_error_samples": residual, "readout_axis": "x"},
        ),
        check(
            "fov_mm",
            lattice and straight,
            {"fov_mm": spec["fov_mm"], "lattice_offset": offset.tolist()},
        ),
        check(
            "matrix",
            in_bounds and no_duplicates,
            {
                "in_bounds": in_bounds,
                "unique_samples": int(mask.sum()),
                "adc_samples": len(indices),
            },
        ),
    ]
    if fam == "rfov_epi":
        checks = [c for c in checks if c["name"] not in {"echo_in_adc", "te_ms"}]
        echo_times = []
        for ids in a.lines:
            span = q[ids[-1], 0] - q[ids[0], 0]
            fraction = -q[ids[0], 0] / span if abs(span) > 1e-12 else 0.5
            echo_times.append(
                a.adc[ids[0]] + fraction * (a.adc[ids[-1]] - a.adc[ids[0]])
            )
        a.te = np.asarray(echo_times) - np.array(
            [excitation_pulses[max(j, 0)].center for j in a.excitation]
        )
        central_line = int(np.argmin(np.linalg.norm(q[centres], axis=1)))
        effective_te_ms = float(a.te[central_line] * 1000)
        a.measurements["echo_times_s"] = echo_times
        a.measurements["te_ms"] = (a.te * 1000).tolist()
        a.measurements["effective_te_ms"] = effective_te_ms
        limit = spec.get("te_ms")
        if limit:
            valid_te = effective_te_ms > 0
            if limit.get("max") is not None:
                valid_te &= effective_te_ms <= limit["max"] + 1e-6
            if limit.get("target") is not None:
                valid_te &= (
                    abs(effective_te_ms - limit["target"]) <= limit["tolerance"] + 1e-6
                )
            checks.append(
                check(
                    "te_ms",
                    valid_te,
                    {"effective_te_ms": effective_te_ms, "contract": limit},
                )
            )
        checks.append(
            check(
                "echo_in_adc",
                all(np.min(q[ids, 0]) <= 0 <= np.max(q[ids, 0]) for ids in a.lines),
                "Every EPI line brackets kx=0 after the excitation endpoint",
            )
        )
        polarities = np.array(
            [np.sign(q[ids[-1], 0] - q[ids[0], 0]) for ids in a.lines]
        )
        checks.append(
            check(
                "single_shot_epi",
                len(np.unique(a.excitation)) == 1
                and len(a.lines) == matrix[1] // 2
                and np.all(polarities[1:] * polarities[:-1] == -1),
                {
                    "readout_lines": len(a.lines),
                    "shots": len(np.unique(a.excitation)),
                    "polarities": polarities.tolist(),
                },
            )
        )
        gradients = a.seq.get_gradients()
        encoded = np.zeros(2, bool)
        for pulse in a.pulses:
            times = (pulse.edges[1:] + pulse.edges[:-1]) / 2
            active = np.abs(pulse.signal) > np.max(np.abs(pulse.signal)) * 1e-5
            for axis in range(2):
                grad = gradients[axis]
                if grad is not None and active.any():
                    values = grad(np.clip(times[active], grad.x[0], grad.x[-1]))
                    encoded[axis] |= np.max(np.abs(values)) > 1
        checks.append(
            check(
                "two_dimensional_rf_encoding",
                encoded.all(),
                {
                    "rf_overlaps_x_gradient": bool(encoded[0]),
                    "rf_overlaps_y_gradient": bool(encoded[1]),
                },
            )
        )
    valid = np.array_equal(mask, expected)
    detail = {
        "missing_samples": int(np.sum(expected & ~mask)),
        "extra_samples": int(np.sum(mask & ~expected)),
    }
    checks.append(check("sampling_pattern", valid, {"family": fam, **detail}))
    if fam != "reduced_fov":
        checks.extend(c for c in a.checks if c["name"] == "no_refocusing")
    else:
        # Spatial selectivity is tested by encoded Bloch probes, not RF labels.
        refs = [p for p in a.pulses if p.refocusing]
        exc = [p for p in a.pulses if not p.refocusing]
        counts, midpoint_errors = [], []
        for j, te in zip(a.excitation, a.te, strict=True):
            start = exc[max(j, 0)].center
            before_echo = [p for p in refs if start < p.center < start + te]
            counts.append(len(before_echo))
            midpoint_errors.append(
                abs(before_echo[0].center - start - te / 2)
                if len(before_echo) == 1
                else 1e6
            )
        checks.append(
            check(
                "inner_volume_rf",
                all(count == 1 for count in counts)
                and max(midpoint_errors) <= midpoint_tolerance(a.seq),
                {
                    "refocusing_per_readout": counts,
                    "max_midpoint_error_s": float(max(midpoint_errors)),
                },
            )
        )
    checks.append(check("positive_echo_times", np.all(a.te > 0), {}))
    a.indices, a.offset, a.checks = indices, offset, checks
    a.measurements.update(
        family=fam,
        fov_mm=spec["fov_mm"],
        coverage=float(mask.mean()),
        sampling_mask=mask.tolist(),
        calibration_lines=calibration_width(mask),
        lattice_offset=offset.tolist(),
        centre_sample_indices=centres.tolist(),
        reconstruction="bart_sampling_sense@1",
        receive_model=receive_model(fam),
    )
    if fam in REDUCED_FOV:
        ys = np.flatnonzero(mask.all(axis=0))
        step = int(np.diff(ys)[0]) if len(ys) >= 2 else 1
        a.measurements["reduced_fov_fraction"] = 1 / step
    return a
