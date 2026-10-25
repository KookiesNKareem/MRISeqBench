"""Exact integration controls retain independent targets and physical errors."""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import pytest

from mriseqbench.evaluation.s1_control import (
    FIDControl,
    build_controls,
    cell_fourier,
    corrected_raw,
    fourier,
)
from mriseqbench.evaluation.s1_phantom import Spins
from mriseqbench.evaluation.s1_sequence import rotate


def control(dims=2, gaussian=False):
    system = pp.Opts(B0=1.5, max_grad=40, grad_unit="mT/m")
    seq, pulses, clock = pp.Sequence(system), [], 0.0
    for j in range(5):
        rf = pp.make_block_pulse(0.2, duration=0.0004, system=system)
        seq.add_block(rf)
        pulses.append(
            SimpleNamespace(
                start=clock,
                center=clock + 0.0002,
                flip=np.rad2deg(0.2),
                phase=j * 0.37,
                frequency=0.0,
            )
        )
        gx = pp.make_trapezoid("x", area=40, duration=0.001, system=system)
        gy = pp.make_trapezoid("y", area=20, duration=0.001, system=system)
        seq.add_block(gx, gy)
        seq.add_block(pp.make_delay(0.01))
        clock += 0.0114
    acq = SimpleNamespace(
        seq=seq,
        pulses=pulses,
        adc=np.array([clock - 0.004, clock - 0.002]),
        adc_phase=np.array([0.3, 0.9]),
    )
    anatomy = SimpleNamespace(
        labels=np.ones((2,) * dims, int),
        voxel_size_m=np.full(dims, 0.002),
        center_m=np.zeros(dims),
        materials=["water"],
        properties=np.array([[1.0, 0.8, 0.2, 0.0]]),
    )
    case = {"task": {"sequence": {"fov_mm": [20] * dims, "matrix": [8] * dims}}}
    effects = [
        {"kind": "b0", "model": "linear", "gradient_hz_per_m": [500, 120, 300][:dims]}
    ]
    from mriseqbench.evaluation.s3 import gaussian_receive_maps

    return FIDControl(
        acq,
        case,
        anatomy,
        effects,
        gaussian_receive_maps if gaussian else None,
        None if gaussian else [[(np.zeros(3), 1)]],
        history=2,
    )


def test_finite_history_recurrence_equals_explicit_fourier_paths():
    c = control()
    xyz = np.random.default_rng(17).uniform(-0.003, 0.003, (13, 2))
    labels = np.zeros(len(xyz), int)
    pulse, state = list(c.states(xyz, labels))[-1]
    dt = c.acq.adc - c.rf[pulse]
    q = c.adc_moments - c.rf_moments[pulse] + dt[:, None] * c.b0
    recurrence = (
        state[:, None]
        * np.exp(-dt / 0.2)
        * np.exp(-2j * np.pi * (xyz @ q.T) - 1j * c.acq.adc_phase)
    )
    explicit = np.exp(-2j * np.pi * (xyz @ c.q.T)) * c.coefficients[0]
    explicit = explicit.reshape(len(xyz), len(c.acq.adc), c.history).sum(axis=2)
    np.testing.assert_allclose(recurrence, explicit, atol=1e-12, rtol=1e-12)
    # This sign is fixed by the Bloch convention, never fitted to native output.
    rotated = rotate(np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.0, 0.0]))
    assert rotated[0] + 1j * rotated[1] == pytest.approx(1j * np.sin(0.2))


@pytest.mark.parametrize("dims,gaussian", [(2, False), (3, False), (2, True)])
def test_known_cell_integral_matches_independent_gauss_quadrature(dims, gaussian):
    c = control(dims, gaussian)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    unit = np.stack(np.meshgrid(*[nodes] * dims, indexing="ij"), axis=-1).reshape(
        -1, dims
    )
    quadrature = (
        np.prod(
            np.stack(np.meshgrid(*[weights] * dims, indexing="ij"), axis=-1), axis=-1
        ).ravel()
        / 2**dims
    )
    centres = (np.indices((2,) * dims).reshape(dims, -1).T - 0.5) * 0.002
    points = np.concatenate([centre + unit * 0.001 for centre in centres])
    xyz = np.zeros((len(points), 3))
    xyz[:, :dims] = points
    spins = Spins(
        xyz,
        np.zeros(len(xyz), int),
        ["water"],
        c.anatomy.properties,
        np.tile(quadrature * 0.002**dims / 0.0025**dims, 2**dims),
        (8,) * dims,
        "gauss-test",
    )
    np.testing.assert_allclose(c.sampled(spins), c.known, rtol=1e-9, atol=1e-11)


def test_widely_separated_queries_match_direct_complex_fourier_sums():
    rng = np.random.default_rng(73)
    points = rng.uniform(-0.1, 0.1, (19, 2))
    weights = rng.normal(size=(2, 19)) + 1j * rng.normal(size=(2, 19))
    frequencies = np.array(
        [[0, 0], [2100, 0], [-2300, 100], [2500, 3000], [-5200, -9000], [10000, 6500]]
    )
    direct = weights @ np.exp(-2j * np.pi * points @ frequencies.T)
    np.testing.assert_allclose(
        fourier(points, weights, frequencies), direct, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_regular_cells_at_large_frequencies_match_direct_integrals(dims):
    rng = np.random.default_rng(47)
    sizes = np.array([0.001, 0.002, 0.00625])[:dims]
    centres = np.indices((5,) * dims).reshape(dims, -1).T * sizes - 0.073
    weights = rng.normal(size=len(centres)) + 1j * rng.normal(size=len(centres))
    frequencies = rng.uniform(-50000, 50000, (31, dims))
    direct = (weights @ np.exp(-2j * np.pi * centres @ frequencies.T)) * np.prod(
        np.sinc(frequencies * sizes), axis=1
    )
    np.testing.assert_allclose(
        cell_fourier(centres, weights, sizes, frequencies), direct, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize(
    "unsupported", ["balanced", "volume", "refocusing", "shaped", "motion"]
)
def test_unsupported_acquisitions_keep_the_existing_fourier_control(
    tmp_path, unsupported
):
    pulse = SimpleNamespace(refocusing=False, signal=np.ones(2))
    case = {
        "task": {"id": "gre_t1w", "sequence": {"matrix": [128, 128]}},
        "physics": {"effects": []},
        "object": {"phantom": "cardiac_2d@1"},
    }
    if unsupported == "balanced":
        case["task"]["id"] = "bssfp"
    elif unsupported == "volume":
        case["task"]["sequence"]["matrix"] = [128, 128, 32]
    elif unsupported == "refocusing":
        pulse.refocusing = True
    elif unsupported == "shaped":
        pulse.signal = np.array([1, 2])
    elif unsupported == "motion":
        case["physics"]["effects"] = [{"kind": "motion", "model": "rigid_sinusoidal"}]
    assert build_controls(SimpleNamespace(pulses=[pulse]), case, tmp_path) is None


def test_control_removes_sampling_noise_without_repairing_physical_errors():
    reference = np.ones((8, 8), complex)
    reference[4:] = 2
    known_control = reference * (0.7 + 0.2j)
    noise = 0.1 * (-1.0) ** np.indices(reference.shape).sum(axis=0)
    sampled_control = known_control + noise
    # Correct, wrong receiver phase, absent signal, and doubled gain.
    physical = np.array([reference, -1j * reference, reference * 0, 2 * reference])
    native = physical + noise
    original = native.copy()
    corrected = corrected_raw(native, sampled_control, known_control)
    np.testing.assert_allclose(corrected, physical, atol=1e-15)
    np.testing.assert_array_equal(native, original)
    # Any omitted/reflected/stimulated/finite-RF residual survives algebraically.
    np.testing.assert_allclose(corrected - known_control, native - sampled_control)


@pytest.mark.parametrize(
    "family", ["caipirinha", "elliptical_kspace", "stack_of_stars"]
)
def test_volume_control_uses_the_native_receive_operator(monkeypatch, tmp_path, family):
    from mriseqbench.evaluation import s1_control, s3

    monkeypatch.setattr(s1_control, "load_anatomy", lambda *_: None)
    monkeypatch.setattr(
        s1_control,
        "FIDControl",
        lambda acq, case, anatomy, effects, maps, terms, history: SimpleNamespace(
            maps=maps, terms=terms
        ),
    )
    acq = SimpleNamespace(
        pulses=[SimpleNamespace(refocusing=False, signal=np.ones(2))],
        measurements={"family": family},
    )
    case = {
        "task": {"id": family, "sequence": {"matrix": [128, 128, 32]}},
        "object": {"phantom": "cardiac_3d@1"},
        "physics": {"effects": []},
    }
    c = build_controls(acq, case, tmp_path)[0]
    points = np.array([[0.02, -0.03, 0.01], [-0.04, 0.01, 0.05]])
    fov = np.full(3, 0.2)
    actual = c.maps(points, fov) if c.maps else np.ones((1, len(points)))
    expected = s3.receive_maps(points, fov, s3.parallel_imaging(acq))
    np.testing.assert_array_equal(actual, expected)


def test_control_cannot_silently_expand_native_receive_channels():
    native = np.ones((1, 7), complex)
    with pytest.raises(ValueError, match="native receive channels"):
        corrected_raw(native, np.ones((16, 7)), np.ones((16, 7)))
    # Single-coil S3 keeps a channel axis; S1/S2 use the same scalar control flat.
    np.testing.assert_array_equal(corrected_raw(native, np.ones(7), np.ones(7)), native)


@pytest.mark.parametrize("bend", [0, 1e-10, 1])
def test_3d_flat_roundoff_and_noncollinear_queries_match_direct_sums(bend):
    rng = np.random.default_rng(17)
    points = rng.uniform(-0.1, 0.1, (23, 3))
    from mriseqbench.evaluation.s3 import receive_maps

    weights = receive_maps(points, np.full(3, 0.2), True) * rng.uniform(
        0.1, 1, (1, len(points))
    )
    # Actual declared z-B0 makes the readout line oblique in three dimensions.
    frequencies = np.array([17.0, 25.0, 6.0]) + np.linspace(-0.00064, 0.00064, 128)[
        :, None
    ] * np.array([201000.0, 0.0, 300.0])
    frequencies[64, 1] += bend
    direct = weights @ np.exp(-2j * np.pi * (points @ frequencies.T))
    np.testing.assert_allclose(
        fourier(points, weights, frequencies), direct, rtol=1e-8, atol=1e-8
    )
