"""Independent midpoint/Rodrigues Bloch oracle for selected qualification spins."""

from itertools import pairwise

import numpy as np
from pypulseq import Opts, Sequence

from mriseqbench.evaluation.rf_waveform import rf_waveform


def independent_response(path, case, positions, *, max_step=1e-6):
    """Integrate authored waveforms directly, without any KomaMRI dynamics."""
    seq = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    seq.read(str(path))
    end = seq.duration()[0] + 1e-6
    gradients = seq.get_gradients()
    boundaries = [0.0, end]
    for gradient in gradients:
        if gradient is not None:
            boundaries.extend(gradient.x)
    clock = 0.0
    for number in seq.block_events:
        rf = seq.get_block(number).rf
        if rf is not None:
            boundaries.extend(clock + rf.delay + rf_waveform(rf, seq.rf_raster_time)[0])
        clock += seq.block_durations[number]
    boundaries = np.unique(np.clip(boundaries, 0, end))
    nodes = np.concatenate(
        [
            np.linspace(
                left, right, max(1, int(np.ceil((right - left) / max_step - 1e-9))) + 1
            )[:-1]
            for left, right in pairwise(boundaries)
            if right - left > 1e-14
        ]
        + [np.array([end])]
    )
    widths = np.diff(nodes)
    times = (nodes[1:] + nodes[:-1]) / 2
    count = len(times)
    fields = np.zeros((count, 3))
    for axis, grad in enumerate(gradients):
        if grad is not None:
            keep = (times >= grad.x[0]) & (times <= grad.x[-1])
            fields[keep, axis] = grad(times[keep])
    rf_samples = np.zeros(count, dtype=complex)
    clock = 0.0
    for number in seq.block_events:
        block = seq.get_block(number)
        rf = block.rf
        if rf is not None:
            edges, signal, linear = rf_waveform(rf, seq.rf_raster_time)
            local = times - clock - rf.delay
            keep = (local >= edges[0]) & (local < edges[-1])
            if linear:
                values = np.interp(local[keep], edges, signal)
            else:
                values = signal[np.searchsorted(edges, local[keep], side="right") - 1]
            frequency = (
                rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            phase = (
                rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.gamma * seq.system.B0
            )
            rf_samples[keep] = values * np.exp(
                1j * (phase + 2 * np.pi * frequency * local[keep])
            )
        clock += seq.block_durations[number]
    offset = np.zeros(len(positions))
    for effect in case["physics"]["effects"]:
        if effect["kind"] != "b0" or effect["model"] != "linear":
            raise ValueError("Oracle supports only registered linear static B0")
        gradient = np.pad(effect["gradient_hz_per_m"], (0, 1))
        offset += effect.get("offset_hz", 0) + positions @ gradient
    longitudinal = fields @ positions.T + offset
    state = np.tile([0.0, 0.0, 1.0], (len(positions), 1))
    decays = np.exp(-widths[:, None] / np.array([0.1, 0.1, 1.0]) / 2)
    for index in range(count):
        decay = decays[index]
        recovery = np.array([0.0, 0.0, 1 - decay[2]])
        state = state * decay + recovery
        axis = np.column_stack(
            (
                np.full(len(state), rf_samples[index].real),
                np.full(len(state), rf_samples[index].imag),
                longitudinal[index],
            )
        ) * (2 * np.pi * widths[index])
        angle = np.linalg.norm(axis, axis=1)
        unit = axis / np.maximum(angle[:, None], 1e-300)
        state = (
            state * np.cos(angle)[:, None]
            - np.cross(unit, state) * np.sin(angle)[:, None]
            + unit
            * np.sum(unit * state, axis=1)[:, None]
            * (1 - np.cos(angle))[:, None]
        )
        state = state * decay + recovery
    return state[:, 0] + 1j * state[:, 1]
