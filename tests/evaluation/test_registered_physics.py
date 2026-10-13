"""Independent finite-RF controls that activate every registered physics profile."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from pypulseq import (
    Opts,
    Sequence,
    make_adc,
    make_block_pulse,
    make_delay,
    make_trapezoid,
)
from scipy.integrate import quad, solve_ivp
from scipy.spatial.transform import Rotation

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_koma import simulate
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s1_sequence import inspect_sequence
from tests.fixtures.legacy_catalog import resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]
PROFILES = sorted(p.stem for p in (ROOT / "benchmark/physics").glob("*.yaml"))


def pad(values):
    return np.pad(np.asarray(values, float), (0, 3 - len(values)))


def physical_state(t, point, material, effects):
    """Definitions evaluated independently of the judge's environment adapters."""
    moved = point.copy()
    for effect in effects:
        if effect["kind"] != "motion":
            continue
        model = effect["model"]
        if model == "rigid_keyframes":
            pose = next(p for p in reversed(effect["poses"]) if p["time_s"] <= t)
            moved = Rotation.from_euler(
                "xyz", pose["rotation_xyz_deg"], degrees=True
            ).apply(point)
            moved += pad(pose["translation_mm"]) / 1000
        else:
            sine = np.sin(
                2 * np.pi * t / effect["period_s"]
                + np.deg2rad(effect.get("phase_deg", 0))
            )
            if model == "affine_sinusoidal":
                moved = point * (1 + pad(effect["strain_amplitude"]) * sine)
            else:
                moved = Rotation.from_euler(
                    "z", effect["rotation_amplitude_deg"] * sine, degrees=True
                ).apply(point)
                moved += pad(effect["translation_amplitude_mm"]) * sine / 1000
    field, density = 0.0, 1.0
    for effect in effects:
        if effect["kind"] == "b0":
            field += effect.get("offset_hz", 0)
            if effect["model"] == "linear":
                field += pad(effect["gradient_hz_per_m"]) @ moved
            elif effect["model"] == "gaussian":
                sigma = np.r_[
                    np.asarray(effect["sigma_mm"]) / 1000,
                    np.ones(3 - len(effect["sigma_mm"])),
                ]
                field += effect["amplitude_hz"] * np.exp(
                    -0.5
                    * np.sum(((moved - pad(effect["center_mm"]) / 1000) / sigma) ** 2)
                )
        elif effect["kind"] == "time_variation":
            sine = np.sin(
                2 * np.pi * t / effect["period_s"]
                + np.deg2rad(effect.get("phase_deg", 0))
            )
            if effect["target"] == "b0_offset":
                field += effect["amplitude_hz"] * sine
            elif (
                effect["target"] == "proton_density" and material == effect["material"]
            ):
                density *= 1 + effect["amplitude_scale"] * sine
    return moved, field, density


def authored_probe(tmp_path, starts=(0.24, 1.24, 2.24), gradient_scale=1.0):
    system = Opts(rf_dead_time=0, rf_ringdown_time=0, adc_dead_time=0)
    seq = Sequence(system)
    clock = 0.0
    windows = []
    for start in starts:
        seq.add_block(make_delay(start - clock))
        seq.add_block(
            make_block_pulse(0.6, duration=500e-6, system=system, use="excitation")
        )
        gradient_start = start + 500e-6
        gradients = [
            make_trapezoid(
                axis,
                amplitude=amplitude * gradient_scale,
                rise_time=200e-6,
                flat_time=8e-3,
                fall_time=200e-6,
                system=system,
            )
            for axis, amplitude in zip("xyz", (120, -70, 90), strict=True)
        ]
        seq.add_block(*gradients, make_adc(8, dwell=1e-3, delay=200e-6, system=system))
        windows.append(
            (gradient_start, np.array([120.0, -70.0, 90.0]) * gradient_scale)
        )
        clock = gradient_start + 8.4e-3
    path = tmp_path / "physics-probe.seq"
    seq.write(str(path))
    case = resolve_legacy_case(Catalog(ROOT), "gre_t1w", physics="ideal")
    return inspect_sequence(path, case), windows


def independent_signal(acq, windows, spins, effects):
    edges = [0, *acq.adc, 1.0, 2.0]
    for pulse in acq.pulses:
        edges.extend((pulse.start, pulse.end))
    for start, _gradient in windows:
        edges.extend(start + np.array([0, 200e-6, 8.2e-3, 8.4e-3]))
    edges = sorted({t for t in edges if t <= acq.adc[-1]})
    expected = np.zeros(len(acq.adc), complex)
    for point, label in zip(spins.positions, spins.labels, strict=True):
        rho, t1, t2 = spins.properties[label]
        material = spins.materials[label]
        initial_density = physical_state(0, point, material, effects)[2]
        state = np.array([0.0, 0.0, rho * initial_density])

        def frequency(t, point=point, material=material):
            moved, field, _density = physical_state(t, point, material, effects)
            for start, gradient in windows:
                u = t - start
                scale = max(0, min(u / 200e-6, 1, (8.4e-3 - u) / 200e-6))
                field += scale * (gradient @ moved)
            return field

        for left, right in pairwise(edges):
            middle = (left + right) / 2
            pulse = next((p for p in acq.pulses if p.start < middle < p.end), None)
            if pulse:
                wx = 2 * np.pi * pulse.signal[0].real

                def ode(
                    t,
                    values,
                    wx=wx,
                    rho=rho,
                    t1=t1,
                    t2=t2,
                    point=point,
                    material=material,
                    frequency=frequency,
                ):
                    x, y, z = values
                    wz = 2 * np.pi * frequency(t)
                    equilibrium = rho * physical_state(t, point, material, effects)[2]
                    return [
                        -x / t2 + wz * y,
                        -wz * x - y / t2 + wx * z,
                        -wx * y + (equilibrium - z) / t1,
                    ]

                result = solve_ivp(
                    ode, (left, right), state, method="DOP853", rtol=1e-11, atol=1e-13
                )
                assert result.success
                state = result.y[:, -1]
            else:
                area = quad(frequency, left, right, epsabs=1e-10, epsrel=1e-10)[0]
                xy = (state[0] + 1j * state[1]) * np.exp(
                    -(right - left) / t2 - 2j * np.pi * area
                )
                recovery = quad(
                    lambda t, rho=rho, point=point, material=material, t1=t1, right=right: (
                        rho
                        * physical_state(t, point, material, effects)[2]
                        / t1
                        * np.exp(-(right - t) / t1)
                    ),
                    left,
                    right,
                    epsabs=1e-12,
                )[0]
                state = np.array(
                    [
                        xy.real,
                        xy.imag,
                        state[2] * np.exp(-(right - left) / t1) + recovery,
                    ]
                )
            index = np.flatnonzero(acq.adc == right)
            expected[index] += (
                (state[0] + 1j * state[1])
                * spins.weight
                * np.exp(-1j * acq.adc_phase[index])
            )
    return expected


@pytest.mark.parametrize("profile", PROFILES)
def test_registered_profile_matches_independent_active_physics(tmp_path, profile):
    acq, windows = authored_probe(tmp_path)
    catalog = Catalog(ROOT)
    effects = catalog.get("physics", profile).model_dump(mode="json")["effects"]
    positions = np.array([[0.021, -0.013, 0.017], [-0.028, 0.019, -0.011]])
    if not profile.endswith("_3d"):
        positions[:, 2] = 0
    spins = Spins(
        positions,
        np.array([0, 1]),
        ["wm", "gm"],
        np.array([[0.7, 0.8, 2.0], [0.9, 1.1, 1.6]]),
        0.5,
        (2,),
        "independent-active-physics-probe",
    )
    expected = independent_signal(acq, windows, spins, effects)
    actual = simulate(
        acq, spins, effects, step_s=250e-6, rf_step_s=2e-6, integration_method="bloch"
    )
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    print(f"{profile}: independent active-physics NRMSE={error:.8g}")
    assert error < 0.005
    if profile != "ideal":
        baseline = independent_signal(acq, windows, spins, [])
        # Acquiring beyond 2 s activates both registered keyframe transitions.
        assert np.linalg.norm(expected - baseline) / np.linalg.norm(baseline) > 1e-4


def test_keyframe_jumps_inside_gradient_encoding(tmp_path):
    # Each first/second readout spans a declared discontinuity at 1/2 seconds.
    # An engine that merely evaluates the later pose at the ADC is incorrect.
    acq, windows = authored_probe(
        tmp_path, starts=(0.9955, 1.9955, 2.24), gradient_scale=100
    )
    effects = (
        Catalog(ROOT).get("physics", "rigid_steps").model_dump(mode="json")["effects"]
    )
    spins = Spins(
        np.array([[0.021, -0.013, 0.0], [-0.028, 0.019, 0.0]]),
        np.array([0, 1]),
        ["wm", "gm"],
        np.array([[0.7, 0.8, 2.0], [0.9, 1.1, 1.6]]),
        0.5,
        (2,),
        "independent-gradient-keyframe-jumps",
    )
    expected = independent_signal(acq, windows, spins, effects)
    actual = simulate(acq, spins, effects, step_s=250e-6, rf_step_s=2e-6)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    print(f"keyframe_gradient_jumps: independent active-physics NRMSE={error:.8g}")
    assert error < 2e-4
