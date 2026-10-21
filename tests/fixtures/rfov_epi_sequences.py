"""Independent 2D RF / reduced-FOV EPI reference and controlled corruptions."""

import copy
from functools import lru_cache

import numpy as np
import pypulseq as pp
from scipy.linalg import solve
from scipy.special import erf


def system_for(case):
    hw = case["hardware"]
    return pp.Opts(
        B0=hw["field_strength_T"],
        max_grad=hw["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hw["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )


def sampled_block(system, duration, *gradients):
    seq = pp.Sequence(system)
    seq.add_block(*gradients, pp.make_delay(duration))
    times = (np.arange(round(duration / 1e-5)) + 0.5) * 1e-5
    return np.column_stack(
        [
            np.zeros(len(times))
            if grad is None
            else grad(np.clip(times, grad.x[0], grad.x[-1]))
            for grad in seq.get_gradients()[:2]
        ]
    )


@lru_cache(maxsize=6)
def excitation_design(field, fat_offset, fat_t2, shifted=False):
    system = pp.Opts(max_grad=26, grad_unit="mT/m", max_slew=45, slew_unit="T/m/s")
    read = pp.make_trapezoid(
        "x", flat_area=80, flat_time=1000e-6, rise_time=60e-6, system=system
    )
    pre = [
        pp.make_trapezoid(axis, area=area, duration=600e-6, system=system)
        for axis, area in zip("xy", (-read.area / 2, -80), strict=True)
    ]
    chunks = [sampled_block(system, 0.0006, *pre)]
    active = [np.zeros(len(chunks[0]), bool)]
    for line in range(33):
        grad = copy.deepcopy(read)
        grad.amplitude *= (-1) ** line
        chunks.append(sampled_block(system, pp.calc_duration(grad), grad))
        active.append(np.ones(len(chunks[-1]), bool))
        if line < 32:
            blip = pp.make_trapezoid("y", area=5, duration=140e-6, system=system)
            chunks.append(sampled_block(system, pp.calc_duration(blip), blip))
            active.append(np.zeros(len(chunks[-1]), bool))
    rewind = [
        pp.make_trapezoid(axis, area=area, duration=600e-6, system=system)
        for axis, area in zip("xy", (-read.area / 2, -80), strict=True)
    ]
    chunks.append(sampled_block(system, 0.0006, *rewind))
    active.append(np.zeros(len(chunks[-1]), bool))
    half = np.vstack(chunks)
    gradient = np.vstack((half, -half[::-1]))
    dt = 10e-6
    count = len(gradient)
    selected = np.flatnonzero(np.concatenate(active))
    mirrored = count - 1 - selected
    remaining = (count - np.arange(count) - 0.5) * dt + 31e-6
    area = np.cumsum(gradient[::-1], axis=0)[::-1] * dt - gradient * dt / 2
    axis = ((np.arange(32) + 0.5) / 32 - 0.5) * 0.2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    positions = np.column_stack((x.ravel(), y.ravel()))
    # Supplement the uniform design grid with the published pass/stopband
    # locations so interpolation between grid cells is also constrained.
    additional = (
        [(x, y) for x in (-0.056, 0.0, 0.056) for y in np.linspace(-0.035, 0.035, 9)]
        + [
            (x, y)
            for x in (-0.056, 0.0, 0.056)
            for y in np.r_[np.linspace(-0.096, -0.065, 9), np.linspace(0.065, 0.096, 9)]
        ]
        + [(x, y) for x in (-0.096, -0.092, 0.092, 0.096) for y in (-0.025, 0.0, 0.025)]
    )
    positions = np.vstack((positions, additional))
    target = (
        0.4
        * (
            (
                erf((positions[:, 0] + 0.08) / 0.002)
                - erf((positions[:, 0] - 0.08) / 0.002)
            )
            / 2
        )
        * (
            (
                erf((positions[:, 1] + 0.05 - (0.03 if shifted else 0)) / 0.002)
                - erf((positions[:, 1] - 0.05 - (0.03 if shifted else 0)) / 0.002)
            )
            / 2
        )
    )
    matrices, targets = [], []
    for chemical, t2 in ((0.0, 0.08), (fat_offset, fat_t2)):
        offsets = positions[:, 0] * 1000 * field / 3 + chemical
        encoded = []
        for ids in (selected, mirrored):
            values = np.exp(
                -2j
                * np.pi
                * (positions @ area[ids].T + offsets[:, None] * remaining[ids])
            )
            values *= np.exp(-remaining[ids] / t2)[None, :] * (2j * np.pi * dt)
            encoded.append(values)
        matrices.append(np.hstack(encoded))
        targets.append(
            (0 if chemical else 1)
            * 1j
            * np.arcsin(target)
            * np.exp(-2j * np.pi * offsets * (count * dt / 2 + 31e-6))
        )
    encoding = np.vstack(matrices)
    target = np.concatenate(targets)
    gram = encoding @ encoding.conj().T
    gram.flat[:: len(gram) + 1] += 1e-7
    controls = encoding.conj().T @ solve(gram, target, assume_a="pos")
    rf = np.zeros(count, complex)
    rf[selected] = controls[: len(selected)]
    rf[mirrored] = controls[len(selected) :]
    return gradient, rf, dt


def make_sequence(case, mutation=None):
    system = system_for(case)
    tissue = case["object"]["tissue_properties"]["fat"]
    gradient, signal, dt = excitation_design(
        system.B0,
        tissue["chemical_shift_hz"],
        tissue["T2_s"],
        mutation == "shift_selection",
    )
    gradient, signal = gradient.copy(), signal.copy()
    if mutation == "no_selection":
        gradient[:] = 0
        signal[:] = 0
        signal[-100:] = 0.5 / (2 * np.pi * 100 * dt)
    elif mutation == "one_dimensional_rf":
        gradient[:, 0] = 0
    rf = pp.make_arbitrary_rf(
        signal,
        flip_angle=1,
        no_signal_scaling=True,
        dwell=dt,
        delay=100e-6,
        system=system,
        use="excitation",
    )
    gradients = [
        pp.make_arbitrary_grad(
            axis, gradient[:, j], first=0, last=0, delay=100e-6, system=system
        )
        for j, axis in enumerate("xy")
    ]
    seq = pp.Sequence(system)
    seq.add_block(rf, *gradients, pp.make_delay(100e-6 + len(signal) * dt + 30e-6))
    read = pp.make_trapezoid(
        "x", flat_area=160, flat_time=320e-6, rise_time=300e-6, system=system
    )
    full = mutation == "full_fov"
    ys = list(range(-16, 16, 1 if full else 2))
    if mutation == "missing_line":
        ys.pop()
    pre = [
        pp.make_trapezoid(axis, area=area, duration=800e-6, system=system)
        for axis, area in zip("xy", (-read.area / 2, ys[0] / 0.2), strict=True)
    ]
    seq.add_block(*pre)
    for line, ky in enumerate(ys):
        gx = copy.deepcopy(read)
        gx.amplitude *= (-1) ** line if mutation != "wrong_polarity" else 1
        adc = pp.make_adc(
            32,
            duration=320e-6,
            delay=gx.rise_time,
            phase_offset=0.6 if mutation == "wrong_receiver_phase" else 0,
            system=system,
        )
        seq.add_block(gx, adc)
        if line < len(ys) - 1:
            gy = pp.make_trapezoid(
                "y", area=(ys[line + 1] - ky) / 0.2, duration=200e-6, system=system
            )
            if mutation == "wrong_polarity":
                rewind = pp.make_trapezoid(
                    "x", area=-read.area, duration=900e-6, system=system
                )
                gy = pp.make_trapezoid(
                    "y", area=(ys[line + 1] - ky) / 0.2, duration=800e-6, system=system
                )
                seq.add_block(gy, rewind)
            else:
                seq.add_block(gy)
    seq.add_block(pp.make_delay(0.10))
    return seq
