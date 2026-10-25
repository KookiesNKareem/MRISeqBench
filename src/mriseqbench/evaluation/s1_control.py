"""Known-integral FID histories reduce quadrature noise, retaining native residuals.

This is an integration control, not an image or dynamics model. Its cell integral
is separate from the independent intended-contrast target. Reflected, stimulated,
finite-RF and all other omitted dynamics remain in native minus sampled control.
"""

from concurrent.futures import ThreadPoolExecutor

import finufft
import numpy as np

from .s1_anatomy import _geometry, is_anatomical, load_anatomy
from .s1_sequence import gradient_integrals
from .s3_sequence import PARALLEL

FAMILIES = {
    "gre_t1w",
    "gre_t2star",
    "compressed_sensing",
    "variable_density",
    "partial_fourier",
    "sense_sampling",
    "grappa_sampling",
    "radial_readout",
    "spiral_gre",
    "ute_radial",
    "caipirinha",
    "elliptical_kspace",
    "stack_of_stars",
}
HISTORY = 64
VOLUMES = {"caipirinha", "elliptical_kspace", "stack_of_stars"}


def fourier(points, weights, frequencies):
    """Type-3 Fourier sums; coil batches share a plan, each plan uses one thread."""
    dims = points.shape[1]
    if dims == 3:
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


def corrected_raw(native, sampled_control, known_control):
    if np.broadcast_shapes(
        np.shape(native), np.shape(sampled_control), np.shape(known_control)
    ) != np.shape(native):
        raise ValueError("Integration controls must preserve native receive channels")
    return known_control + native - sampled_control


class FIDControl:
    def __init__(self, acq, case, anatomy, effects, maps, terms, history=HISTORY):
        self.acq, self.anatomy = acq, anatomy
        self.maps, self.terms, self.history = maps, terms, history
        self.single = maps is None
        self.fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
        self.matrix = np.asarray(case["task"]["sequence"]["matrix"])
        self.dims = len(self.matrix)
        self.b0, self.offset = np.zeros(self.dims), 0.0
        for effect in effects:
            self.b0 += np.asarray(effect.get("gradient_hz_per_m", np.zeros(self.dims)))[
                : self.dims
            ]
            self.offset += effect.get("offset_hz", 0)
        self.rf = np.array([p.center for p in acq.pulses])
        self.flip = np.deg2rad([p.flip for p in acq.pulses])
        self.current = np.searchsorted(self.rf, acq.adc, side="right") - 1
        moments = gradient_integrals(acq.seq, np.r_[self.rf, acq.adc])[:, : self.dims]
        self.rf_moments, self.adc_moments = np.split(moments, [len(self.rf)])
        self.logc = np.r_[0, np.cumsum(np.log(np.cos(self.flip / 2) ** 2))]
        props = anatomy.properties
        self.chemical = props[:, 3] * acq.seq.system.B0 / 3
        mz, last, source = props[:, 0].copy(), 0.0, []
        for pulse, angle in zip(acq.pulses, self.flip, strict=True):
            mz = props[:, 0] + (mz - props[:, 0]) * np.exp(
                -(pulse.center - last) / props[:, 1]
            )
            phase = pulse.phase + 2 * np.pi * pulse.frequency * (
                pulse.center - pulse.start
            )
            source.append(1j * mz * np.sin(angle) * np.exp(1j * phase))
            mz *= np.cos(angle)
            last = pulse.center
        self.source = np.asarray(source).T
        lag = self.current[:, None] - np.arange(history)
        valid, lag = lag >= 0, np.maximum(lag, 0)
        dt = acq.adc[:, None] - self.rf[lag]
        self.q = (
            self.adc_moments[:, None] - self.rf_moments[lag] + dt[:, :, None] * self.b0
        ).reshape(-1, self.dims)
        attenuation = np.exp(self.logc[self.current[:, None] + 1] - self.logc[lag + 1])
        self.coefficients = (
            self.source[:, lag]
            * attenuation[None]
            * np.exp(-dt[None] / props[:, 2, None, None])
            * np.exp(-2j * np.pi * (self.chemical[:, None, None] + self.offset) * dt)
            * np.exp(-1j * acq.adc_phase[None, :, None])
            * valid
        ).reshape(len(props), -1)
        # Share geometry transforms across receive coils and independent scrambles.
        with ThreadPoolExecutor(max_workers=3) as pool:
            self.known = sum(pool.map(self._material_integral, range(len(props))))
        if self.single:
            self.known = self.known[0]

    def _material_integral(self, label):
        anatomy = self.anatomy
        _shape, lower = _geometry(anatomy)
        index = np.column_stack(np.nonzero(anatomy.labels == label + 1))
        centres = lower + (index + 0.5) * anatomy.voxel_size_m
        voxel = np.prod(self.fov / self.matrix)
        weights = np.full(len(index), np.prod(anatomy.voxel_size_m) / voxel)
        if self.terms is None:
            maps = self.maps(centres, self.fov, True)
            if not len(index):
                return np.zeros((len(maps), len(self.acq.adc)), complex)
            return np.array(
                [
                    (
                        cell_fourier(
                            centres, weights * receive, anatomy.voxel_size_m, self.q
                        )
                        * self.coefficients[label]
                    )
                    .reshape(len(self.acq.adc), self.history)
                    .sum(axis=1)
                    for receive in maps
                ]
            )
        result = np.zeros((len(self.terms), len(self.acq.adc)), complex)
        if not len(index):
            return result
        transforms = {}
        if self.dims == 3:
            shifts = list(
                dict.fromkeys(
                    tuple(shift) for terms in self.terms for shift, _ in terms
                )
            )
            queries = np.concatenate(
                [self.q - np.asarray(shift) / self.fov for shift in shifts]
            )
            values = cell_fourier(centres, weights, anatomy.voxel_size_m, queries)
            transforms = dict(
                zip(shifts, values.reshape(len(shifts), len(self.q)), strict=True)
            )
        for channel, terms in enumerate(self.terms):
            value = np.zeros(len(self.q), complex)
            for shift, coefficient in terms:
                key = tuple(shift)
                if key not in transforms:
                    transforms[key] = cell_fourier(
                        centres,
                        weights,
                        anatomy.voxel_size_m,
                        self.q - np.asarray(shift[: self.dims]) / self.fov,
                    )
                value += coefficient * transforms[key]
            result[channel] = (
                (value * self.coefficients[label])
                .reshape(len(self.acq.adc), self.history)
                .sum(axis=1)
            )
        return result

    def states(self, positions, labels):
        """Finite-history recurrence, algebraically equal to an explicit path sum."""
        props = self.anatomy.properties
        state = np.zeros(len(positions), complex)
        saved = [None] * self.history
        last_t, last_m = 0.0, np.zeros(self.dims)

        def propagate(value, start, moment, end_t, end_moment, attenuation=1.0):
            dt = end_t - start
            return (
                value
                * attenuation
                * np.exp(-dt / props[labels, 2])
                * np.exp(
                    -2j
                    * np.pi
                    * (
                        positions @ (end_moment - moment + self.b0 * dt)
                        + (self.chemical[labels] + self.offset) * dt
                    )
                )
            )

        for pulse, t in enumerate(self.rf):
            state = (
                propagate(state, last_t, last_m, t, self.rf_moments[pulse])
                * np.cos(self.flip[pulse] / 2) ** 2
            )
            state += self.source[labels, pulse]
            part = state
            if pulse >= self.history:
                old = pulse - self.history
                part = state - propagate(
                    saved[pulse % self.history],
                    self.rf[old],
                    self.rf_moments[old],
                    t,
                    self.rf_moments[pulse],
                    np.exp(self.logc[pulse + 1] - self.logc[old + 1]),
                )
            saved[pulse % self.history] = state.copy()
            yield pulse, part
            last_t, last_m = t, self.rf_moments[pulse]

    def sampled(self, spins, *, check=None):
        xyz, labels = spins.positions[:, : self.dims], spins.labels
        receive_positions = spins.positions
        if self.terms is None:
            _, lower = _geometry(self.anatomy)
            size = self.anatomy.voxel_size_m
            receive_positions = lower + (np.floor((xyz - lower) / size) + 0.5) * size
        maps = (
            np.ones((1, len(xyz)))
            if self.single
            else self.maps(receive_positions, self.fov, True)
        )
        weights = np.broadcast_to(spins.weight, (len(xyz),))
        result = np.zeros((len(maps), len(self.acq.adc)), complex)
        for pulse, state in self.states(xyz, labels):
            if check is not None:
                check()
            ids = np.flatnonzero(self.current == pulse)
            if not len(ids):
                continue
            dt = self.acq.adc[ids] - self.rf[pulse]
            q = self.adc_moments[ids] - self.rf_moments[pulse] + dt[:, None] * self.b0
            for label in np.unique(labels):
                if check is not None:
                    check()
                keep = labels == label
                value = fourier(
                    xyz[keep], maps[:, keep] * (weights[keep] * state[keep]), q
                )
                result[:, ids] += (
                    value
                    * np.exp(-dt / self.anatomy.properties[label, 2])
                    * np.exp(
                        -2j * np.pi * (self.chemical[label] + self.offset) * dt
                        - 1j * self.acq.adc_phase[ids]
                    )
                )
        result = result[0] if self.single else result
        if result.shape != self.known.shape:
            raise ValueError(
                "Sampled and known controls must preserve receive channels"
            )
        return result


def build_controls(acq, case, root):
    """Optional spoiled-GRE control with static affine B0 and known RF axes."""
    effects = case["physics"]["effects"]
    dims = len(case["task"]["sequence"]["matrix"])
    if (
        case["task"]["id"] not in FAMILIES
        or dims not in {2, 3}
        or (dims == 3) != (case["task"]["id"] in VOLUMES)
        or not is_anatomical(case)
        or any(p.refocusing for p in acq.pulses)
        or any(
            e["kind"] != "b0" or e["model"] not in {"linear", "constant"}
            for e in effects
        )
    ):
        return None
    shaped = [p for p in acq.pulses if np.any(p.signal != p.signal[0])]
    if shaped:
        if case["task"]["id"] != "spiral_gre":
            return None
        for pulse in shaped:
            axis = pulse.signal[np.argmax(abs(pulse.signal))]
            if not np.all(
                pulse.signal.real * axis.imag == pulse.signal.imag * axis.real
            ):
                return None
            for gradient in acq.seq.get_gradients():
                if gradient is not None:
                    overlap = (gradient.x[:-1] < pulse.end) & (
                        gradient.x[1:] > pulse.start
                    )
                    if np.any(gradient.c[:, overlap] != 0):
                        return None
    maps, terms = None, [[(np.zeros(3), 1.0)]]
    if case["task"]["id"] in PARALLEL:
        from .s3 import gaussian_receive_maps, receive_maps, receive_terms

        if case["task"]["id"] == "variable_density":
            # Only the control uses cell-constant maps; native receive stays continuous.
            maps, terms = gaussian_receive_maps, None
        else:
            maps, terms = receive_maps, receive_terms(dims, True)
    anatomy = load_anatomy(root, case["object"]["phantom"])
    history = 128 if case["task"]["id"] == "ute_radial" else HISTORY
    ideal = FIDControl(acq, case, anatomy, (), maps, terms, history=history)
    physical = (
        FIDControl(acq, case, anatomy, effects, maps, terms, history=history)
        if effects
        else ideal
    )
    return ideal, physical
