"""Fourier operators for independent reference images."""

import finufft
import numpy as np


def fourier(points, weights, frequencies):
    """Type-3 Fourier sums; coil batches share a plan, each plan uses one thread."""
    dims = points.shape[1]
    if dims in {2, 3}:
        if len(frequencies) > 1:
            # Uniform readouts are a type-1 transform after a phase shift.
            # Check the complete query grid before taking this faster path.
            count = len(frequencies)
            step = (frequencies[-1] - frequencies[0]) / (count - 1)
            modes = np.arange(count) - count // 2
            centre = frequencies[0] + (count // 2) * step
            residual = frequencies - (centre + modes[:, None] * step)
            bound = (
                2 * np.pi * np.max(abs(points), axis=0)
                @ np.max(abs(residual), axis=0)
            )
            if bound <= 1e-9:
                return finufft.nufft1d1(
                    np.ascontiguousarray(2 * np.pi * (points @ step)),
                    np.ascontiguousarray(
                        weights * np.exp(-2j * np.pi * (points @ centre)), dtype=complex
                    ),
                    n_modes=count,
                    isign=-1,
                    eps=1e-10,
                    nthreads=1,
                )
        offset = frequencies[0]
        delta = frequencies - offset
        direction = delta[np.argmax(np.linalg.norm(delta, axis=1))]
        length = np.linalg.norm(direction)
        if not length:
            value = weights @ np.exp(-2j * np.pi * (points @ offset))
            return np.broadcast_to(
                np.asarray(value)[..., None],
                np.asarray(weights).shape[:-1] + (len(frequencies),),
            ).copy()
        direction = direction / length
        coordinate = delta @ direction
        residual = delta - coordinate[:, None] * direction
        phase_bound = (
            2 * np.pi * np.max(abs(points), axis=0) @ np.max(abs(residual), axis=0)
        )
        # Flat Cartesian/radial readouts lie on a line even with affine B0.
        # Gradient-moment subtraction accumulates tiny roundoff in long trains.
        # Dropping it changes the complex sum by at most phase_bound * sum(abs(weights)).
        if phase_bound <= 1e-9:
            return finufft.nufft1d3(
                np.ascontiguousarray(points @ direction),
                np.ascontiguousarray(
                    weights * np.exp(-2j * np.pi * (points @ offset)), dtype=complex
                ),
                np.ascontiguousarray(2 * np.pi * coordinate),
                isign=-1,
                eps=1e-10,
                nthreads=1,
            )
    # Long RF histories can span enormous frequency grids. Partition queries
    # without changing their physical coordinates or the Fourier operator.
    width = 200 if dims == 3 else 2000
    if len(frequencies) > 1 and np.any(np.ptp(frequencies, axis=0) > width):
        _, groups = np.unique(
            np.floor(frequencies / width).astype(np.int64), axis=0, return_inverse=True
        )
        result = np.empty(np.asarray(weights).shape[:-1] + (len(frequencies),), complex)
        for group in range(int(groups.max()) + 1):
            ids = np.flatnonzero(groups == group)
            result[..., ids] = fourier(points, weights, frequencies[ids])
        return result
    return getattr(finufft, f"nufft{dims}d3")(
        *[np.ascontiguousarray(points[:, axis]) for axis in range(dims)],
        np.ascontiguousarray(weights, dtype=complex),
        *[
            np.ascontiguousarray(2 * np.pi * frequencies[:, axis])
            for axis in range(dims)
        ],
        isign=-1,
        eps=1e-10,
        nthreads=1,
    )


def cell_fourier(centres, weights, sizes, frequencies):
    dims = centres.shape[1]
    if dims in {2, 3}:
        lower = centres.min(axis=0)
        indices = np.rint((centres - lower) / sizes).astype(int)
        if np.allclose(centres, lower + indices * sizes, rtol=0, atol=2e-15):
            shape = tuple(indices.max(axis=0) + 1)
            grid = np.zeros(shape, complex)
            np.add.at(grid, tuple(indices.T), weights)
            origin = lower + (np.asarray(shape) // 2) * sizes
            angles = 2 * np.pi * frequencies * sizes
            values = getattr(finufft, f"nufft{dims}d2")(
                *[np.ascontiguousarray(angles[:, axis]) for axis in range(dims)],
                grid,
                isign=-1,
                eps=1e-10,
                nthreads=6 if dims == 3 else 1,
            )
            del angles
            for first in range(0, len(frequencies), 262144):
                part = frequencies[first : first + 262144]
                factor = np.exp(-2j * np.pi * (part @ origin))
                for axis in range(dims):
                    factor *= np.sinc(part[:, axis] * sizes[axis])
                values[first : first + 262144] *= factor
            return values
    return fourier(centres, weights, frequencies) * np.prod(
        np.sinc(frequencies * sizes), axis=1
    )
