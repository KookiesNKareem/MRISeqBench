"""Fixed-coordinate reconstruction and independently constructed ideal encoding."""

from copy import copy

import finufft
import numpy as np

from .s1_koma import simulate
from .s1_phantom import Spins
from .s1_sequence import rotate


def expected_receiver_phase(acq):
    """Reference receiver phase follows RF coherence, never submitted ADC phase."""
    phase = np.zeros(len(acq.adc))
    current = 0.0
    for p in acq.pulses:
        rf_phase = p.phase + 2 * np.pi * p.frequency * (p.center - p.start)
        current = 2 * rf_phase - current if p.refocusing else rf_phase
        phase[acq.adc >= p.center] = current
    return phase


def canonical_response(acq, properties, fam, *, finite_rf=False):
    """Family oracle using measured RF/timing, with perfect spoiling when required.

    This deliberately does not copy encoding gradients or spoiler effectiveness.
    bSSFP uses finite-RF, zero-gradient Bloch dynamics, preserving transients.
    """
    receiver = expected_receiver_phase(acq)
    n = len(properties)
    if finite_rf:
        reference_acq = copy(acq)
        reference_acq.pulses = [p for p in acq.pulses if p.start < acq.adc[-1]]
        reference_acq.adc_phase = receiver
        spins = Spins(
            np.zeros((n, 3)),
            np.arange(n),
            list(range(n)),
            properties,
            1.0,
            (n,),
            "intrinsic-material-rf-response",
        )
        return simulate(
            reference_acq,
            spins,
            zero_gradients=True,
            individual=True,
            ideal_spoiling=fam != "bssfp",
        ).T
    if fam == "bssfp":
        out = []
        for i in range(n):
            spins = Spins(
                np.zeros((1, 3)),
                np.array([i]),
                list(range(n)),
                properties[:, :3],
                1.0,
                (1,),
                "diagnostic",
            )
            reference_acq = copy(acq)
            reference_acq.adc_phase = receiver
            out.append(simulate(reference_acq, spins, zero_gradients=True))
        return np.asarray(out)
    response = np.zeros((n, len(acq.adc)), dtype=complex)
    m = np.zeros((n, 3))
    m[:, 2] = properties[:, 0]
    last = 0.0
    for pulse in acq.pulses:
        dt = pulse.center - last
        m[:, :2] *= np.exp(-dt / properties[:, 2, None])
        m[:, 2] = properties[:, 0] + (m[:, 2] - properties[:, 0]) * np.exp(
            -dt / properties[:, 1]
        )
        if not pulse.refocusing:
            m[:, :2] = 0.0  # The independent ideal family destroys old coherence.
        phase = pulse.phase + 2 * np.pi * pulse.frequency * (pulse.center - pulse.start)
        axis = np.deg2rad(pulse.flip) * np.array([np.cos(phase), np.sin(phase), 0.0])
        m = np.array([rotate(v, axis) for v in m])
        last = pulse.center
        next_pulses = [p.center for p in acq.pulses if p.center > last + 1e-12]
        end = next_pulses[0] if next_pulses else np.inf
        ids = np.flatnonzero((acq.adc >= last) & (acq.adc < end))
        dt_adc = acq.adc[ids] - last
        response[:, ids] = (
            (m[:, 0] + 1j * m[:, 1])[:, None]
            * np.exp(-dt_adc[None, :] / properties[:, 2, None])
            * np.exp(-1j * receiver[ids])
        )
    return response


def reference_signal(acq, spins, spec, response):
    """Exact discrete Fourier sampling of the oversampled analytic object.

    No registration, contrast fitting, per-tissue scaling or submitted gradients
    enter this encoding oracle. Fractional lattice origins are retained.
    """
    dims = len(spins.grid)
    fov = np.asarray(spec["fov_mm"]) / 1000
    coords = spins.positions[:, :dims]
    offset_phase = np.exp(-2j * np.pi * np.sum(coords / fov * acq.offset, axis=1))
    raw = np.zeros(len(acq.adc), dtype=complex)
    weights = np.broadcast_to(spins.weight, (len(spins.positions),))
    for label in np.unique(spins.labels):
        keep = spins.labels == label
        points = [
            np.ascontiguousarray(2 * np.pi * coords[keep, j] / fov[j])
            for j in range(dims)
        ]
        transform = finufft.nufft2d1 if dims == 2 else finufft.nufft3d1
        fourier = transform(
            *points,
            offset_phase[keep] * weights[keep],
            tuple(spec["matrix"]),
            eps=1e-10,
            isign=-1,
            nthreads=1,
        )
        raw += fourier[tuple(acq.indices.T)] * response[label]
    return raw


from .s1_koma import reconstruct  # noqa: F401 -- shared public reconstruction adapter


def nrmse(actual, reference):
    return float(
        np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-15)
    )


def material_masks(spins, spec):
    matrix = np.asarray(spec["matrix"])
    dims = len(matrix)
    fov = np.asarray(spec["fov_mm"]) / 1000
    indices = np.floor((spins.positions[:, :dims] / fov + 0.5) * matrix).astype(int)
    masks = {}
    for label in np.unique(spins.labels):
        occupancy = np.zeros(tuple(matrix), dtype=float)
        keep = spins.labels == label
        np.add.at(
            occupancy,
            tuple(indices[keep].T),
            np.broadcast_to(spins.weight, (len(spins.positions),))[keep],
        )
        # Erode in all directions to avoid partial-volume and boundary ringing.
        mask = occupancy > 0.95
        for axis in range(dims):
            mask &= np.roll(occupancy, 1, axis) > 0.95
            mask &= np.roll(occupancy, -1, axis) > 0.95
        if not mask.any():
            mask = occupancy > 0.75
        masks[spins.materials[label]] = mask
    return masks


def image_metrics(image, reference, spins, spec):
    magnitude, expected = np.abs(image), np.abs(reference)
    masks = material_masks(spins, spec)
    dims = len(spec["matrix"])
    fov = np.asarray(spec["fov_mm"]) / 1000
    loc = np.floor(
        (spins.positions[:, :dims] / fov + 0.5) * np.asarray(spec["matrix"])
    ).astype(int)
    support = np.zeros(tuple(spec["matrix"]), dtype=bool)
    support[tuple(loc.T)] = True
    energy = float(np.sum(magnitude * magnitude))
    metrics = {
        "image_nrmse": nrmse(magnitude, expected),
        "complex_image_nrmse": nrmse(image, reference),
        "outside_object_energy_fraction": float(
            np.sum(magnitude[~support] ** 2) / max(energy, 1e-30)
        ),
        "signal_fraction": float(
            np.linalg.norm(magnitude) / max(np.linalg.norm(expected), 1e-15)
        ),
        "absolute_rms_signal": float(np.sqrt(np.mean(magnitude * magnitude))),
    }
    for name, mask in masks.items():
        if mask.any():
            metrics[f"{name}_mean_signal"] = float(magnitude[mask].mean())
    if {"wm_mean_signal", "gm_mean_signal"} <= metrics.keys():
        metrics["wm_gm_ratio"] = metrics["wm_mean_signal"] / max(
            metrics["gm_mean_signal"], 1e-15
        )
    return metrics
