"""Independent complex-valued reconstruction gold, without fitted gain or phase."""

from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import bart


def asymmetric_image(matrix):
    coordinates = [(np.arange(n) + 0.5) / n - 0.5 for n in matrix]
    axes = np.meshgrid(*coordinates, indexing="ij")
    centre = [0.13, -0.19, 0.09][: len(matrix)]
    widths = [0.09, 0.14, 0.12][: len(matrix)]
    gaussian = np.exp(
        -sum(
            ((axis - c) / width) ** 2
            for axis, c, width in zip(axes, centre, widths, strict=True)
        )
    )
    secondary = np.exp(
        -sum(
            ((axis + c * 1.3) / (width * 0.7)) ** 2
            for axis, c, width in zip(axes, centre, widths, strict=True)
        )
    )
    phase = 0.47 + sum((index + 1) * axis * 1.1 for index, axis in enumerate(axes))
    return (gaussian + 0.31 * secondary) * np.exp(1j * phase)


def encoded_cartesian(image, offset):
    matrix = np.array(image.shape)
    indices = np.indices(tuple(matrix)).reshape(len(matrix), -1).T
    coordinates = ((indices + 0.5) / matrix - 0.5).reshape((*matrix, len(matrix)))
    spatial = np.exp(-2j * np.pi * np.sum(coordinates * offset, axis=-1))
    grid = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image * spatial)))
    half_voxel = np.exp(-1j * np.pi * np.sum((indices - matrix // 2) / matrix, axis=1))
    return grid.ravel() * half_voxel, indices


@pytest.mark.parametrize("matrix", [(14, 10), (8, 6, 10)])
@pytest.mark.parametrize("fractional_offset", [False, True])
def test_cartesian_bart_matches_independent_physical_voxel_gold(
    matrix, fractional_offset
):
    image = asymmetric_image(matrix)
    offset = (
        np.array([0.25, -0.375, 0.125][: len(matrix)])
        if fractional_offset
        else np.zeros(len(matrix))
    )
    raw, indices = encoded_cartesian(image, offset)
    fov = np.array([0.21, 0.165, 0.19][: len(matrix)])
    k = np.zeros((len(raw), 3))
    k[:, : len(matrix)] = (indices - np.array(matrix) // 2 + offset) / fov
    acq = SimpleNamespace(
        indices=indices,
        offset=offset,
        k=k,
        measurements={"family": "gre_t1w", "fov_mm": (fov * 1000).tolist()},
    )
    actual = bart.reconstruct_cartesian(raw, acq, matrix)
    np.testing.assert_allclose(actual, image, rtol=2e-6, atol=2e-7)
    rotated = bart.reconstruct_cartesian(1j * raw, acq, matrix)
    np.testing.assert_allclose(rotated, 1j * image, rtol=2e-6, atol=2e-7)
    assert np.linalg.norm(rotated - image) / np.linalg.norm(image) > 1.4
    np.testing.assert_array_equal(
        bart.reconstruct_cartesian(np.zeros_like(raw), acq, matrix), np.zeros(matrix)
    )


@pytest.mark.parametrize("axes", [(0,), (0, 1), (0, 1, 2)])
def test_bart_fft_has_fixed_complex_scale_and_selected_axes(axes):
    data = asymmetric_image((8, 6, 10))
    expected = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data, axes=axes), axes=axes), axes=axes
    )
    actual = bart.fft(data, inverse=False, axes=axes)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        bart.fft(actual, inverse=True, axes=axes), data, rtol=2e-6, atol=2e-7
    )


def noncartesian_nodes(dims):
    if dims == 2:
        angles = np.arange(13) * np.pi / 13
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        z = np.linspace(-0.95, 0.95, 21)
        angle = np.arange(21) * np.pi * (3 - np.sqrt(5))
        directions = np.column_stack(
            (np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z)
        )
    nodes = (
        directions[:, None, :] * np.linspace(-0.5, 0.5, 12)[None, :, None]
    ).reshape(-1, dims)
    # Nyquist boundary wrapping and explicit out-of-box sample rejection.
    return np.concatenate(
        (nodes, np.eye(dims) * 0.5, np.eye(dims) * -0.5, np.ones((1, dims)) * 0.7)
    )


@pytest.mark.parametrize("matrix", [(12, 10), (8, 6, 6)])
def test_noncartesian_bart_inverse_matches_independent_dense_normal_equations(matrix):
    rng = np.random.default_rng(20261018)
    nodes = np.concatenate(
        (
            rng.uniform(-0.5, 0.5, (4 * np.prod(matrix), len(matrix))),
            noncartesian_nodes(len(matrix)),
        )
    )
    fov_mm = np.array([210.0, 165.0, 190.0][: len(matrix)])
    k = np.zeros((len(nodes), 3))
    k[:, : len(matrix)] = nodes * matrix / (fov_mm / 1000)
    acq = SimpleNamespace(k=k)
    image = asymmetric_image(matrix)
    indices = np.indices(matrix).reshape(len(matrix), -1).T
    physical = ((indices + 0.5) / matrix - 0.5) * (fov_mm / 1000)
    encoder = np.exp(-2j * np.pi * (k[:, : len(matrix)] @ physical.T))
    raw = encoder @ image.ravel()
    raw[-1] = 1e7 + 1e7j  # The out-of-box sample must have no effect.
    keep = np.all(np.abs(nodes) <= 0.5 + 1e-5, axis=1)
    assert not keep[-1]
    operator = encoder[keep] / np.sqrt(np.prod(matrix))
    expected = np.linalg.solve(
        operator.conj().T @ operator + 0.001 * np.eye(np.prod(matrix)),
        operator.conj().T @ (raw[keep] / np.sqrt(np.prod(matrix))),
    )
    actual = bart.reconstruct_noncartesian(raw, acq, matrix, fov_mm)
    error = np.linalg.norm(actual.ravel() - expected) / np.linalg.norm(expected)
    print(f"non-Cartesian {matrix}: dense-normal-equation NRMSE={error:.6g}")
    assert error < 0.00005
    rotated = bart.reconstruct_noncartesian(1j * raw, acq, matrix, fov_mm)
    np.testing.assert_allclose(rotated, 1j * actual, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(
        bart.reconstruct_noncartesian(np.zeros_like(raw), acq, matrix, fov_mm),
        np.zeros(matrix),
    )


def sense_problem(matrix, family):
    from itertools import product

    from mriseqbench.evaluation import s3

    assert s3.REGULARIZATION == 0.001
    assert s3.MAX_ITERATIONS == 50
    dims = len(matrix)
    indices = np.indices(matrix).reshape(dims, -1).T
    positions = (indices + 0.5) / matrix - 0.5
    normalization = np.sqrt(4 * (1 + 0.8**2 / 2))
    maps = np.stack(
        [
            np.prod(
                [
                    (1 + 0.8 * np.cos(2 * np.pi * positions[:, axis] - angle))
                    / normalization
                    for axis, angle in enumerate(angles, start=1)
                ],
                axis=0,
            )
            for angles in product(np.arange(4) * np.pi / 2, repeat=dims - 1)
        ]
    )
    np.testing.assert_allclose(np.sum(maps**2, axis=0), 1, rtol=2e-14, atol=2e-14)
    mask = np.zeros(matrix, bool)
    if family == "compressed_sensing":
        mask[:, [1, 5, 9, *range(12, 20), 22, 26, 30]] = True
    elif family == "variable_density":
        mask[:, [0, 6, 10, *range(12, 20), 21, 25, 31]] = True
    elif dims == 2:
        mask[:, ::2] = True
    else:
        mask[:, ::2, ::2] = True
        mask[:, 1::2, 1::2] = True
    acquired = indices[mask.ravel()]
    offset = np.array([0.25, -0.375, 0.125][:dims])
    q = acquired - np.array(matrix) // 2 + offset
    fourier = np.exp(-2j * np.pi * (q @ positions.T))
    operator = (
        maps[:, None, :] * fourier[None, :, :] / np.sqrt(np.prod(matrix))
    ).reshape(-1, np.prod(matrix))
    image = asymmetric_image(matrix)
    raw = (operator @ image.ravel() * np.sqrt(np.prod(matrix))).reshape(len(maps), -1)
    expected = np.linalg.solve(
        operator.conj().T @ operator + 0.001 * np.eye(np.prod(matrix)),
        operator.conj().T @ (raw.ravel() / np.sqrt(np.prod(matrix))),
    )
    acq = SimpleNamespace(
        indices=acquired,
        offset=offset,
        measurements={
            "family": family,
            "sampling_mask": mask.tolist(),
            "fov_mm": [210, 165, 190][:dims],
        },
    )
    return acq, raw, operator, expected


@pytest.mark.parametrize(
    "matrix,family",
    [
        ((8, 8), "sense_sampling"),
        ((8, 6, 4), "caipirinha"),
        ((8, 32), "compressed_sensing"),
        ((8, 32), "variable_density"),
    ],
)
def test_bart_sense_matches_independent_complex_dense_normal_equations(matrix, family):
    from mriseqbench.evaluation import s3

    acq, raw, operator, expected = sense_problem(matrix, family)
    actual = s3.reconstruct(raw, acq, matrix)
    error = np.linalg.norm(actual.ravel() - expected) / np.linalg.norm(expected)
    print(f"{family} {matrix}: dense-normal-equation NRMSE={error:.6g}")
    assert error < 0.00005
    rhs = operator.conj().T @ (raw.ravel() / np.sqrt(np.prod(matrix)))
    normal = operator.conj().T @ operator + 0.001 * np.eye(np.prod(matrix))
    assert np.linalg.norm(normal @ actual.ravel() - rhs) / np.linalg.norm(rhs) < 1e-6
    rotated = s3.reconstruct(1j * raw, acq, matrix)
    # Measure the global complex residual; pointwise relative error is undefined
    # at the near-zero pixels produced by an ill-conditioned sampling mask.
    assert np.linalg.norm(rotated - 1j * actual) / np.linalg.norm(actual) < 5e-5
    np.testing.assert_array_equal(
        s3.reconstruct(np.zeros_like(raw), acq, matrix), np.zeros(matrix)
    )
