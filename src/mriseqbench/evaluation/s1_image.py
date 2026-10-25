"""Fixed-coordinate reconstruction and independently constructed ideal encoding."""

from copy import copy

import finufft
import numpy as np

from .s1_koma import simulate
from .s1_phantom import Spins


def expected_receiver_phase(acq):
    """Reference receiver phase follows RF coherence, never submitted ADC phase."""
    phase = np.zeros(len(acq.adc))
    current = 0.0
    for p in acq.pulses:
        rf_phase = p.phase + 2 * np.pi * p.frequency * (p.center - p.start)
        current = np.remainder(
            2 * rf_phase - current if p.refocusing else rf_phase, 2 * np.pi
        )
        phase[acq.adc >= p.center] = current
    return phase


def refocusing_parity(acq):
    """Sign that restores the excitation's -i convention after each refocusing.

    A refocusing rotation about RF phase p maps m+ to exp(2ip)*conj(m+), so
    demodulating with expected_receiver_phase leaves +i after an odd number of
    refocusings. Every echo of a spin-echo train is scored in the FID convention:
    the matching ADC phase is expected_receiver_phase + pi per refocusing.
    """
    parity = np.zeros(len(acq.adc), dtype=int)
    for pulse in acq.pulses:
        after = acq.adc >= pulse.center
        if pulse.refocusing:
            parity[after] += 1
        else:
            parity[after] = 0
    return (-1.0) ** parity


def rf_only_response(
    acq, properties, *, ideal_spoiling=False, spoil_times_s=(), compress=True
):
    """Finite-RF Bloch; all preparations/transients and intrinsic fat retained."""
    a = copy(acq)
    receiver = expected_receiver_phase(acq)
    a.adc_phase = receiver
    anchors = inverse = None
    if compress and len(a.adc) > 1 and a.pulses:
        starts = np.asarray([pulse.start for pulse in a.pulses])
        ends = np.asarray([pulse.end for pulse in a.pulses])
        if (
            np.isfinite(np.r_[starts, ends, a.adc]).all()
            and np.all(ends >= starts)
            and np.all(starts[1:] >= ends[:-1])
            and np.all(np.diff(a.adc) > 0)
        ):
            groups = np.searchsorted(ends, a.adc, side="right") - 1
            if np.all(groups >= 0) and np.all(
                a.adc < np.r_[starts[1:], np.inf][groups]
            ):
                _, anchors, inverse = np.unique(
                    groups, return_index=True, return_inverse=True
                )
                # Forward propagation avoids dividing by a decayed echo. All RF
                # events remain in the native replay; only ADC outputs shrink.
                a.adc = a.adc[anchors]
                a.adc_phase = a.adc_phase[anchors]
                a.k = a.k[anchors]
    n = len(properties)
    spins = Spins(
        np.zeros((n, 3)),
        np.arange(n),
        [str(i) for i in range(n)],
        properties,
        np.ones(n),
        (n,),
        "intrinsic-material-rf-response",
    )
    response = simulate(
        a,
        spins,
        zero_gradients=True,
        individual=True,
        ideal_spoiling=ideal_spoiling,
        **({"spoil_times_s": spoil_times_s} if len(spoil_times_s) else {}),
    ).T
    if anchors is None:
        return response
    elapsed = acq.adc - a.adc[inverse]
    frequency = (
        properties[:, 3] * acq.seq.system.B0 / 3 if properties.shape[1] > 3 else 0
    )
    evolution = np.exp(
        -elapsed[None, :] / properties[:, 2, None]
        - 2j * np.pi * np.asarray(frequency)[..., None] * elapsed
        - 1j * (receiver - a.adc_phase[inverse])
    )
    return response[:, inverse] * evolution


def canonical_response(acq, properties, fam, *, finite_rf):
    """Family oracle: finite-RF, zero-gradient Bloch on the measured RF/timing.

    This deliberately does not copy encoding gradients or spoiler effectiveness:
    spoiled families use ideal spoiling, while bSSFP/rFOV EPI keep their full
    coherent history. Refocused echoes share the excitation phase convention.
    Callers state ``finite_rf=True``; no other family oracle exists.
    """
    if not finite_rf:
        raise ValueError("Only the finite-RF family oracle is supported")
    reference_acq = copy(acq)
    reference_acq.pulses = [p for p in acq.pulses if p.start < acq.adc[-1]]
    return rf_only_response(
        reference_acq, properties, ideal_spoiling=fam not in {"bssfp", "rfov_epi"}
    ) * refocusing_parity(acq)


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
    return metrics
