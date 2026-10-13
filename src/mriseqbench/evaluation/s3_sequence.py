"""S3 Cartesian sampling identity inferred from RF, gradients and ADC nodes."""

from types import SimpleNamespace

import numpy as np

from .s1_sequence import MissingExcitation, check, family
from .s1_sequence import inspect_sequence as inspect_base

FAMILIES = {
    "sense_sampling",
    "grappa_sampling",
    "caipirinha",
    "compressed_sensing",
    "variable_density",
    "partial_fourier",
    "asymmetric_echo",
    "reduced_fov",
    "elliptical_kspace",
}
PARALLEL = {
    "sense_sampling",
    "grappa_sampling",
    "caipirinha",
    "compressed_sensing",
    "variable_density",
}


def central_indices(n, width):
    return np.arange(n // 2 - width // 2, n // 2 - width // 2 + width)


def prescribed_mask(fam, matrix, sampling):
    """Version-1 detailed mask, inclusive of ACS in the nominal sample budget.

    Poisson-disc is a one-dimensional, radius-two hard-core draw outside ACS.
    Linear density and the RNG algorithm are frozen here, not delegated to an
    agent library. Returned coordinates are array indices, not Pulseq labels.
    """
    nx, ny, *nz = matrix
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
    elif fam == "asymmetric_echo":
        count = int(np.ceil(nx * sampling.get("asymmetric_echo_fraction", 0.75)))
        mask[nx - count :, :] = True
    elif fam == "reduced_fov":
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


def research_pattern(fam, mask):
    """Qualitative research identity, with no hidden detailed mask/seed targets."""
    n = np.asarray(mask.shape)
    if fam == "asymmetric_echo":
        occupied = np.flatnonzero(mask.any(axis=1))
        return (
            n[0] / 2 < len(occupied) < n[0]
            and mask[occupied].all()
            and np.all(np.diff(occupied) == 1)
            and occupied[0] <= n[0] // 2 <= occupied[-1]
            and abs((occupied[0] + occupied[-1]) / 2 - (n[0] - 1) / 2) >= 0.5
        )
    if fam == "elliptical_kspace":
        expected = prescribed_mask(fam, tuple(n), {})
        return np.array_equal(mask, expected)
    phase = mask.all(axis=0)
    if fam == "caipirinha":
        zs = np.flatnonzero(phase.any(axis=0))
        if len(zs) < 2 or not 0 < mask.mean() < 1:
            return False
        patterns = [phase[:, z] for z in zs if not phase[:, z].all()]
        # Exclude central ACS rows; infer a period and a changing residue.
        for ry in range(2, max(3, n[1] // 2 + 1)):
            residues = []
            for pattern in patterns:
                ys = np.flatnonzero(pattern)
                outer = ys[abs(ys - n[1] // 2) >= n[1] // 4]
                if len(outer) < 2 or len(set(outer % ry)) != 1:
                    break
                residue = int(outer[0] % ry)
                required = np.arange(n[1])[np.arange(n[1]) % ry == residue]
                if not pattern[required].all():
                    break
                residues.append(residue)
            if len(residues) >= 2 and len(set(residues)) > 1:
                differences = np.diff(residues) % ry
                if np.all(differences == differences[0]) and differences[0] != 0:
                    return True
        return False
    ys = np.flatnonzero(phase)
    if (
        len(ys) < 2
        or len(ys) >= n[1]
        or not np.array_equal(mask, np.broadcast_to(phase, mask.shape))
    ):
        return False
    if fam == "partial_fourier":
        return (
            len(ys) > n[1] / 2
            and np.all(np.diff(ys) == 1)
            and ys[0] <= n[1] // 2 <= ys[-1]
            and abs((ys[0] + ys[-1]) / 2 - (n[1] - 1) / 2) >= 0.5
        )
    if fam == "reduced_fov":
        step = int(np.diff(ys)[0])
        return (
            np.all(np.diff(ys) == step)
            and step >= 2
            and n[1] % step == 0
            and np.array_equal(
                ys, np.arange(n[1])[np.arange(n[1]) % step == ys[0] % step]
            )
            and n[1] // 2 in ys
        )
    if fam in {"sense_sampling", "grappa_sampling"}:
        # A uniform accelerated lattice plus optional contiguous ACS.
        for step in range(2, int(n[1]) // 2 + 1):
            for offset in range(step):
                lattice = np.arange(n[1]) % step == offset
                extra = np.flatnonzero(phase & ~lattice)
                if (
                    phase[lattice].all()
                    and (not len(extra) or np.all(phase[extra[0] : extra[-1] + 1]))
                    and (
                        fam == "sense_sampling"
                        or calibration_width(mask) >= max(4, 2 * step)
                    )
                ):
                    return True
        return False
    centre = abs(np.arange(n[1]) - n[1] // 2) < n[1] / 4
    denser = phase[centre].mean() > phase[~centre].mean()
    if fam == "compressed_sensing":
        return denser and calibration_width(mask) >= 3 and len(set(np.diff(ys))) >= 2
    return bool(denser)


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
    q = a.k[:, :dims] * fov
    # Fixed centred physical grid. S3's asymmetric support cannot estimate its
    # origin from its extrema: doing so silently recentres partial acquisitions.
    offset = np.angle(np.mean(np.exp(2j * np.pi * q), axis=0)) / (2 * np.pi)
    offset[np.isclose(offset, -0.5, atol=1e-5)] = 0.5
    # The circular mean has a branch cut at half a sample. Small exported
    # gradient round-off must not move an even readout to the adjacent integer
    # origin. Choose the positive half-sample representative within the same
    # published lattice tolerance, retaining the measured fractional offset.
    offset[(offset < 0) & (abs(offset + 0.5) <= 0.01)] += 1
    integer = np.rint(q - offset).astype(int)
    indices = integer + matrix // 2
    in_bounds = bool(((indices >= 0) & (indices < matrix)).all())
    residual = float(np.max(abs(q - offset - integer)))
    lattice = residual <= 0.01 and (
        dims == 3 or np.max(abs(a.k[:, 2])) <= 0.01 / min(fov)
    )
    straight = all(
        len(ids) >= 2
        and (fam == "asymmetric_echo" or len(ids) == matrix[0])
        and np.all(abs(abs(np.diff(q[ids, 0])) - 1) <= 0.01)
        and np.max(np.ptp(q[ids, 1:], axis=0)) <= 0.01
        for ids in a.lines
    )
    mask = np.zeros(tuple(matrix), bool)
    if in_bounds:
        mask[tuple(indices.T)] = True
    no_duplicates = int(mask.sum()) == len(indices)
    checks = [
        c
        for c in a.checks
        if c["name"]
        in {
            "rf_before_adc",
            "echo_in_adc",
            "single_line_per_excitation",
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
    detailed = case["task"].get("prompt_mode") != "research"
    if detailed:
        expected = prescribed_mask(fam, tuple(matrix), spec.get("sampling") or {})
        valid = np.array_equal(mask, expected)
        detail = {
            "missing_samples": int(np.sum(expected & ~mask)),
            "extra_samples": int(np.sum(mask & ~expected)),
        }
    else:
        valid = research_pattern(fam, mask)
        detail = {"qualitative_identity": case["task"]["identity_requirements"]}
    checks.append(check("sampling_pattern", valid, {"family": fam, **detail}))
    if fam != "reduced_fov":
        checks.extend(c for c in a.checks if c["name"] in {"gre", "no_refocusing"})
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
                and max(midpoint_errors) <= max(a.seq.rf_raster_time, 1e-6),
                {
                    "refocusing_per_readout": counts,
                    "max_midpoint_error_s": float(max(midpoint_errors)),
                },
            )
        )
    checks.append(check("positive_echo_times", np.all(a.te > 0), {}))
    a.indices, a.offset, a.checks = indices, offset, checks
    centres = np.array([ids[np.argmin(abs(q[ids, 0]))] for ids in a.lines])
    a.measurements.update(
        family=fam,
        fov_mm=spec["fov_mm"],
        coverage=float(mask.mean()),
        sampling_mask=mask.tolist(),
        calibration_lines=calibration_width(mask),
        lattice_offset=offset.tolist(),
        centre_sample_indices=centres.tolist(),
        reconstruction="bart_sampling_sense@1",
        receive_model="s3_trigonometric_comoving@1" if fam in PARALLEL else "uniform@1",
    )
    if fam == "reduced_fov":
        ys = np.flatnonzero(mask.all(axis=0))
        step = int(np.diff(ys)[0]) if len(ys) >= 2 else 1
        a.measurements["reduced_fov_fraction"] = 1 / step
    return a
