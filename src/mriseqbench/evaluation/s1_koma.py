"""Headless adapter to pinned KomaMRICore simulation and BART reconstruction.

There is no Bloch or FFT implementation in this module. One Julia worker is shared
within a judge process to amortize package loading; its kernels own all dynamics.
"""

import atexit
import copy
import hashlib
import json
import selectors
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import h5py
import numpy as np
from pypulseq import Sequence, make_adc, make_delay


class KomaWorker:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[3]
        self.directory = tempfile.TemporaryDirectory(prefix="mriseqbench-koma-")
        self.path = Path(self.directory.name)
        self.lock = threading.Lock()
        self.sequence_cache = {}
        self.log = (self.path / "worker.log").open("w+")
        julia = shutil.which("julia")
        if julia is None:
            raise RuntimeError(
                "Julia is required for KomaMRI evaluation; run julia --project=. -e 'using Pkg; Pkg.instantiate()'"
            )
        self.process = subprocess.Popen(
            [
                julia,
                "--startup-file=no",
                "--threads=4",
                f"--project={self.root}",
                str(Path(__file__).with_suffix(".jl")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.directory.cleanup()

    def call(self, request, write=None, read=None):
        with self.lock:
            input_path, output_path = self.path / "input.h5", self.path / "output.h5"
            if write:
                with h5py.File(input_path, "w") as handle:
                    write(handle)
            request.update(input=str(input_path), output=str(output_path))
            if self.process.poll() is not None:
                raise RuntimeError(
                    "KomaMRI worker exited: "
                    + (self.path / "worker.log").read_text()[-4000:]
                )
            self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                selector.select()
                line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "KomaMRI worker failed: "
                    + (self.path / "worker.log").read_text()[-4000:]
                )
            response = json.loads(line)
            if not response["ok"]:
                raise RuntimeError("KomaMRI: " + response["error"])
            if read:
                with h5py.File(output_path) as handle:
                    return read(handle)
            return response

    def sequence(self, acq, zero_gradients):
        key = (
            hashlib.sha256(Path(acq.source_path).read_bytes()).hexdigest()
            if getattr(acq, "source_path", None)
            else id(acq.seq),
            tuple((p.start, p.end) for p in acq.pulses),
            acq.adc.tobytes(),
            zero_gradients,
        )
        if key in self.sequence_cache:
            return self.sequence_cache[key]
        # The submitted file itself is read by KomaMRIFiles for normal replays.
        if (
            not zero_gradients
            and getattr(acq, "source_path", None)
            and len(acq.adc) == acq.seq.adc_times()[0].size
        ):
            path = str(acq.source_path)
        else:
            seq = Sequence(acq.seq.system)
            clock = 0.0
            for number in acq.seq.block_events:
                block = acq.seq.get_block(number)
                duration = acq.seq.block_durations[number]
                events = [make_delay(duration)]
                if block.rf is not None and any(
                    abs(p.start - clock - block.rf.delay) < 1e-9 for p in acq.pulses
                ):
                    events.append(copy.deepcopy(block.rf))
                if not zero_gradients:
                    events.extend(
                        copy.deepcopy(g)
                        for g in (block.gx, block.gy, block.gz)
                        if g is not None
                    )
                times = acq.adc[(acq.adc >= clock) & (acq.adc < clock + duration)]
                if len(times):
                    if len(times) > 1:
                        dwell = float(np.min(np.diff(times)))
                    else:
                        # ADC delays serialize in integer microseconds. Preserve
                        # an arbitrary selected sample centre with its dwell.
                        fraction = float(np.remainder((times[0] - clock) * 1e6, 1))
                        fraction = (
                            0.0 if min(fraction, 1 - fraction) < 1e-6 else fraction
                        )
                        dwell = 2 * fraction * 1e-6 if fraction else 2e-6
                    events.append(
                        make_adc(
                            len(times),
                            dwell=dwell,
                            delay=float(times[0] - clock - dwell / 2),
                            system=seq.system,
                        )
                    )
                seq.add_block(*events)
                clock += duration
                if clock > acq.adc[-1]:
                    break
            path = str(self.path / f"diagnostic-{len(self.sequence_cache)}.seq")
            seq.write(path, check_timing=False)
        self.sequence_cache[key] = path
        return path


_WORKER = None


def worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = KomaWorker()
    return _WORKER


def versions():
    return worker().call({"operation": "versions"})["versions"]


def simulate(
    acq,
    spins,
    effects=(),
    *,
    step_s=0.001,
    rf_step_s=20e-6,
    frequency_offsets=None,
    zero_gradients=False,
    demodulate_encoding=False,
    individual=False,
    integration_method="bloch",
    phase_windows=None,
    ideal_spoiling=False,
    receive_weights=None,
):
    if integration_method not in {"bloch", "magnus4"}:
        raise ValueError("Unsupported integration method")
    if integration_method != "bloch" and (individual or demodulate_encoding):
        raise ValueError("Higher-order integration requires summed encoded replay")
    if individual and demodulate_encoding:
        raise ValueError(
            "Individual responses and encoding-demodulated sums are separate modes"
        )
    if phase_windows is not None and not (individual and zero_gradients):
        raise ValueError("Echo-local phase windows require individual RF-only replay")
    if ideal_spoiling and not (individual and zero_gradients):
        raise ValueError("Ideal spoiling requires individual RF-only replay")
    if ideal_spoiling and phase_windows is not None:
        raise ValueError("Ideal spoiling and equilibrium-reset phase windows differ")
    if receive_weights is not None:
        receive_weights = np.asarray(receive_weights, dtype=float)
        if individual or demodulate_encoding or integration_method != "bloch":
            raise ValueError("Receive encoding requires summed native Bloch replay")
        if (
            receive_weights.ndim != 2
            or receive_weights.shape[1] != len(spins.positions)
            or not np.isfinite(receive_weights).all()
        ):
            raise ValueError(
                "Receive weights must be a finite channels-by-spins matrix"
            )
    engine = worker()
    path = engine.sequence(acq, zero_gradients)
    weights = np.broadcast_to(spins.weight, (len(spins.positions),))

    offsets = (
        np.zeros(len(weights))
        if frequency_offsets is None
        else np.broadcast_to(
            np.asarray(frequency_offsets, dtype=float), (len(weights),)
        ).copy()
    )
    if spins.properties.shape[1] > 3:
        # The cardiac asset records intrinsic chemical shift in Hz at 3 T.
        offsets += spins.properties[spins.labels, 3] * acq.seq.system.B0 / 3.0

    def write(handle):
        for axis, name in enumerate(("x", "y", "z")):
            handle[name] = spins.positions[:, axis]
        handle["rho"] = spins.properties[spins.labels, 0] * weights
        handle["t1"] = spins.properties[spins.labels, 1]
        handle["t2"] = spins.properties[spins.labels, 2]
        handle["labels"] = spins.labels
        handle["offset_hz"] = offsets
        handle["adc_times"] = acq.adc
        if receive_weights is not None:
            handle["receive_weights"] = receive_weights
        if demodulate_encoding:
            for axis in range(3):
                handle[f"k{axis}"] = acq.k[:, axis]

    def read(handle):
        result = handle["signal_real"][...] + 1j * handle["signal_imag"][...]
        return result.transpose() if individual else result

    result = engine.call(
        {
            "operation": "simulate",
            "sequence": path,
            "sequence_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "effects": list(effects),
            "materials": list(spins.materials),
            "step_s": step_s,
            "rf_step_s": rf_step_s,
            "rf_frequency_hz": [p.frequency for p in acq.pulses],
            "rf_phase_correction_rad": [p.phase_correction for p in acq.pulses],
            "field_strength_T": acq.seq.system.B0,
            "demodulate_encoding": demodulate_encoding,
            "individual": individual,
            "integration_method": integration_method,
            "phase_windows": phase_windows,
            "spoil_times_s": [float(p.start) for p in acq.pulses if not p.refocusing]
            if ideal_spoiling
            else None,
        },
        write,
        read,
    )
    receiver = np.exp(-1j * acq.adc_phase)
    return result * (
        receiver[:, None]
        if individual
        else receiver[None, :]
        if receive_weights is not None
        else receiver
    )


def reconstruct(raw, acq, matrix):
    from .bart import reconstruct_cartesian

    return reconstruct_cartesian(raw, acq, matrix)
