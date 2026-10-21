"""Independent small-tip spiral reference author; never imported by the judge."""

import argparse
from pathlib import Path

import numpy as np
from pypulseq import Opts, Sequence, make_arbitrary_grad, make_arbitrary_rf, make_delay
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import solve

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.excitation_target import target_arrays

ROOT = Path(__file__).resolve().parents[2]


def spiral():
    """Arc-length speed planning with tangential/normal slew reservations."""
    gamma = 42.576e6
    radius = np.linspace(0, 180, 40001)
    angle = radius * 2 * np.pi / 8
    k = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    ds = np.linalg.norm(np.diff(k, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(ds)]
    first = np.gradient(k, arc, axis=0)
    second = np.gradient(first, arc, axis=0)
    curvature = (
        np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        / np.linalg.norm(first, axis=1) ** 3
    )
    speed = np.minimum(
        gamma * 0.024, np.sqrt(gamma * 40 * 0.8 / np.maximum(curvature, 1e-12))
    )
    acceleration = gamma * 40 * 0.5
    speed[0] = speed[-1] = 0
    for index in range(1, len(speed)):
        speed[index] = min(
            speed[index],
            np.sqrt(speed[index - 1] ** 2 + 2 * acceleration * ds[index - 1]),
        )
    for index in range(len(speed) - 2, -1, -1):
        speed[index] = min(
            speed[index], np.sqrt(speed[index + 1] ** 2 + 2 * acceleration * ds[index])
        )
    times = np.r_[0, np.cumsum(2 * ds / np.maximum(speed[1:] + speed[:-1], 1e-9))]
    dt = 10e-6
    count = int(np.ceil(times[-1] / dt))
    uniform = np.linspace(0, times[-1], count + 1)
    path = CubicHermiteSpline(times, k, first * speed[:, None])(uniform)[::-1]
    # Gradients are at interval centres; both explicit endpoint values are zero.
    gradient = np.diff(path, axis=0) / dt
    return gradient, dt


def make_sequence(case, mutation=None):
    hw = case["hardware"]
    system = Opts(
        B0=hw["field_strength_T"],
        max_grad=hw["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hw["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )
    gradient, dt = spiral()
    count = len(gradient)
    arrays = target_arrays()
    x, y = np.meshgrid(arrays["x_mm"], arrays["y_mm"], indexing="ij")
    mask = arrays["object_mask"]
    points = np.column_stack((x[mask], y[mask])) / 1000
    target = arrays["target_mxy_magnitude"].copy()
    if mutation == "shift":
        target = np.roll(target, 7, axis=0)
    elif mutation == "missing_i":
        target[arrays["x_mm"] > 0, :] = 0
    target = target[mask]
    # Exact small-tip encoding at interval midpoints, including relaxation/B0.
    remaining = (np.arange(count)[::-1] + 0.5) * dt + 31e-6
    integrated = np.cumsum(gradient[::-1], axis=0)[::-1] * dt - gradient * dt / 2
    b0 = sum(
        effect.get("offset_hz", 0)
        + points @ np.asarray(effect.get("gradient_hz_per_m", [0, 0]))
        for effect in case["physics"]["effects"]
    )
    encoding = np.exp(-2j * np.pi * (points @ integrated.T + b0[:, None] * remaining))
    encoding *= np.exp(-remaining / 0.1)[None, :] * (2j * np.pi * dt)
    desired = 1j * np.arcsin(target)
    gram = encoding @ encoding.conj().T
    gram.flat[:: len(gram) + 1] += 2e-9
    coefficients = solve(gram, desired, assume_a="pos")
    signal = encoding.conj().T @ coefficients
    if mutation == "wrong_amplitude":
        signal *= 0.5
    if mutation == "uniform":
        signal[:] = 0
        signal[-100:] = 0.5 / (2 * np.pi * 100 * dt)
        gradient[:] = 0
    if mutation == "no_rf":
        signal[:] = 0
    rf = make_arbitrary_rf(
        signal,
        flip_angle=1,
        no_signal_scaling=True,
        dwell=dt,
        delay=100e-6,
        system=system,
        use="excitation",
    )
    gx, gy = [
        make_arbitrary_grad(
            axis, gradient[:, index], first=0, last=0, delay=100e-6, system=system
        )
        for index, axis in enumerate("xy")
    ]
    seq = Sequence(system)
    seq.add_block(rf, gx, gy, make_delay(100e-6 + count * dt + 30e-6))
    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", default="high_performance")
    parser.add_argument("--mutation")
    args = parser.parse_args()
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "rf_excitation_2d"), hardware_ref=args.hardware
    )
    make_sequence(case, args.mutation).write(str(args.output))
