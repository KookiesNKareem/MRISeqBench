"""Restartable, bounded-memory replays for adaptive numerical integration."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


class ReplayCache:
    def __init__(
        self, directory, identity, simulator, chunk_size=32768, signal_shape=None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity.encode()
        self.simulator = simulator
        self.chunk_size = chunk_size
        self.hits = 0
        self.signal_shape = signal_shape

    def __call__(self, acq, spins, effects=(), **steps):
        """Bloch sums are linear in spin density, including signed inclusions."""
        weights = np.broadcast_to(spins.weight, (len(spins.positions),))
        signal = None
        # Smaller batches at fine time steps bound work lost on interruption.
        # Native CPU spin workspaces remain bounded by the batch size.
        chunk_size = max(
            64, int(self.chunk_size * min(1, steps.get("rf_step_s", 20e-6) / 20e-6))
        )
        for start in range(0, len(weights), chunk_size):
            stop = start + chunk_size
            part = replace(
                spins,
                positions=spins.positions[start:stop],
                labels=spins.labels[start:stop],
                weight=weights[start:stop],
            )
            digest = hashlib.sha256(self.identity)
            digest.update(
                json.dumps(
                    [effects, steps, part.materials, part.source_sha256], sort_keys=True
                ).encode()
            )
            for values in (part.positions, part.labels, part.weight, part.properties):
                digest.update(np.ascontiguousarray(values).tobytes())
            path = self.directory / (digest.hexdigest() + ".npy")
            if path.exists():
                try:
                    values = np.load(path, allow_pickle=False)
                    if (
                        values.shape != (self.signal_shape or acq.adc.shape)
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError("invalid cached replay")
                    self.hits += 1
                except (OSError, ValueError, EOFError):
                    path.unlink(missing_ok=True)
                    values = None
            else:
                values = None
            if values is None:
                values = self.simulator(acq, part, effects, **steps)
                if (
                    values.shape != (self.signal_shape or acq.adc.shape)
                    or not np.isfinite(values).all()
                ):
                    raise RuntimeError("KomaMRI returned a nonfinite or invalid signal")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(path)
            signal = values.copy() if signal is None else signal + values
        if signal is None:
            raise ValueError("Registered phantom has no integration nodes")
        return signal
