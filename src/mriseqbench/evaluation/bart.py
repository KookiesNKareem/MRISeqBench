"""Strict BART CLI reconstruction with fixed physical coordinate conventions.

CFL dimensions are column major: spatial axes 0/1/2, receive coils on axis 3.
BART's centered FFT is unnormalized in both directions; ``fft`` applies the
fixed inverse 1/N scale. NUFFT uses BART's unitary operator and trajectory in
cycles across the FOV (k [cycles/m] times FOV [m]). No local transform fallback.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

QUALIFIED_VERSION = "v1.0.00"
QUALIFIED_SOURCE_COMMIT = "37f3c5f0feaf0a19a7fca85c926ad74101f4a034"
NONCART_REGULARIZATION = 0.001
NONCART_MAX_ITERATIONS = 50
LOCAL_INSTALL = Path.home() / ".local/share/mriseqbench-bart-v1.0.00"
_commands = Counter()
_lock = threading.Lock()


def executable():
    """Find BART explicitly, on PATH, or in the reproducible local installation."""
    configured = os.environ.get("MRISEQBENCH_BART")
    toolbox = os.environ.get("BART_TOOLBOX_PATH")
    candidates = (
        [configured]
        if configured
        else [
            shutil.which("bart"),
            str(Path(toolbox) / "bart") if toolbox else None,
            str(LOCAL_INSTALL / "bart"),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            binary = Path(candidate).resolve()
            stat = binary.stat()
            _binary_identity(str(binary), stat.st_size, stat.st_mtime_ns)
            return str(binary)
    raise RuntimeError(
        "BART is required for all reconstruction; install BART v1.0.00 or set MRISEQBENCH_BART"
    )


def write_cfl(path, array):
    """Write official BART complex64 CFL/HDR with no axis permutation."""
    path = Path(path)
    value = np.asarray(array, dtype=np.complex64)
    if value.ndim > 16 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(
            "BART arrays must be nonempty finite arrays with at most 16 dimensions"
        )
    path.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(map(str, value.shape or (1,))) + "\n"
    )
    value.ravel(order="F").tofile(path.with_suffix(".cfl"))


def read_cfl(path):
    path = Path(path)
    # New BART headers append command/provenance sections after Dimensions.
    lines = path.with_suffix(".hdr").read_text().splitlines()
    try:
        dimensions = tuple(map(int, lines[lines.index("# Dimensions") + 1].split()))
    except (ValueError, IndexError) as error:
        raise RuntimeError("BART output has no valid Dimensions header") from error
    value = np.fromfile(path.with_suffix(".cfl"), dtype=np.complex64)
    if value.size != np.prod(dimensions):
        raise RuntimeError("BART output CFL has inconsistent dimensions")
    last = max((i + 1 for i, n in enumerate(dimensions) if n != 1), default=1)
    result = value.reshape(dimensions, order="F").reshape(dimensions[:last], order="F")
    if not np.isfinite(result).all():
        raise RuntimeError("BART reconstruction produced nonfinite values")
    return result


def run(command, arrays, *, output="out", timeout=300):
    """Run BART using named CFL placeholders, e.g. ['fft','-i','3','{data}','{out}'].

    ``arrays`` maps input names to NumPy arrays; ``output`` names the output
    placeholder. Returned arrays omit trailing singleton dimensions only.
    """
    command = [str(token) for token in command]
    if not command or "{" + output + "}" not in command:
        raise ValueError("BART command must contain its named output placeholder")
    with tempfile.TemporaryDirectory(prefix="mriseqbench-bart-") as directory:
        paths = {name: str(Path(directory) / name) for name in (*arrays, output)}
        for name, value in arrays.items():
            if not name.isidentifier() or name == output:
                raise ValueError(
                    "BART input names must be identifiers distinct from output"
                )
            write_cfl(paths[name], value)
        argv = [
            paths.get(token[1:-1], token)
            if token.startswith("{") and token.endswith("}")
            else token
            for token in command
        ]
        if any("{" in token or "}" in token for token in argv):
            raise ValueError("Unknown BART array placeholder")
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(
            [executable(), *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
        if completed.returncode:
            raise RuntimeError(
                f"BART {' '.join(command)} failed: {(completed.stderr + completed.stdout)[-4000:]}"
            )
        result = read_cfl(paths[output])
        with _lock:
            _commands[tuple(command)] += 1
        return result


@lru_cache(maxsize=4)
def _binary_identity(path, size, mtime_ns):
    binary = Path(path)
    version = subprocess.run(
        [path, "version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    if version != QUALIFIED_VERSION:
        raise RuntimeError(
            f"BART {QUALIFIED_VERSION} is required by the calibrated evaluator; found {version!r}"
        )
    commit = None
    if (binary.parent / ".git").is_dir():
        commit = subprocess.run(
            ["git", "-C", str(binary.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if commit != QUALIFIED_SOURCE_COMMIT:
            raise RuntimeError(
                "BART source checkout does not match the qualified release"
            )
    return {
        "engine": "BART",
        "version": version,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_commit": commit,
    }


def provenance():
    binary = Path(executable())
    stat = binary.stat()
    identity = dict(_binary_identity(str(binary), stat.st_size, stat.st_mtime_ns))
    with _lock:
        identity["commands"] = [
            {"argv": list(argv), "calls": count}
            for argv, count in sorted(_commands.items())
        ]
    return identity


def fft(array, *, inverse, axes):
    array = np.asarray(array)
    axes = tuple(axes)
    if (
        not axes
        or len(set(axes)) != len(axes)
        or any(a < 0 or a >= array.ndim for a in axes)
    ):
        raise ValueError("FFT axes must be distinct valid array dimensions")
    command = [
        "fft",
        *(["-i"] if inverse else []),
        str(sum(1 << a for a in axes)),
        "{data}",
        "{out}",
    ]
    result = run(command, {"data": array}).reshape(array.shape)
    return result / np.prod([array.shape[a] for a in axes]) if inverse else result


def reconstruct_cartesian(raw, acq, matrix):
    matrix = tuple(map(int, matrix))
    raw = np.asarray(raw)
    indices = np.asarray(acq.indices, dtype=int)
    if raw.shape != (len(indices),) or indices.shape[1] != len(matrix):
        raise ValueError("Cartesian samples and encoding indices disagree")
    if np.any(indices < 0) or np.any(indices >= np.asarray(matrix)):
        raise ValueError("Cartesian encoding index outside matrix")
    grid, counts = np.zeros(matrix, complex), np.zeros(matrix, int)
    np.add.at(grid, tuple(indices.T), raw)
    np.add.at(counts, tuple(indices.T), 1)
    grid /= np.maximum(counts, 1)
    q = np.indices(matrix) - (np.asarray(matrix) // 2).reshape(
        (-1,) + (1,) * len(matrix)
    )
    phase = sum(q[a] / matrix[a] for a in range(len(matrix)))
    image = fft(
        grid * np.exp(1j * np.pi * phase), inverse=True, axes=tuple(range(len(matrix)))
    )
    offset = np.asarray(acq.offset)
    return image * np.exp(
        2j
        * np.pi
        * sum((q[a] + 0.5) / matrix[a] * offset[a] for a in range(len(matrix)))
    )


def _trajectory(nodes, matrix):
    trajectory = np.zeros((3, len(nodes), 1), complex)
    trajectory[: len(matrix), :, 0] = (nodes * np.asarray(matrix)).T
    return trajectory


def _inverse_nufft(data, trajectory, shape):
    command = [
        "nufft",
        "-i",
        "-P",
        "-r",
        "-x",
        ":".join(map(str, shape)),
        "-m",
        str(NONCART_MAX_ITERATIONS),
        "-l",
        str(NONCART_REGULARIZATION),
        "{trajectory}",
        "{data}",
        "{out}",
    ]
    return run(command, {"trajectory": trajectory, "data": data})


def _stack_inverse(data, nodes, matrix):
    """Factor a complete Cartesian partition axis into BART FFT and 2D NUFFT.

    Return None unless every partition has identical transverse sample nodes.
    The z transform is unitary, so the same Tikhonov lambda applies to every
    slice. Source sample values never enter geometry detection or scaling.
    """
    if len(matrix) != 3:
        return None
    nz = matrix[2]
    kz = nodes[:, 2] * nz
    if np.max(np.abs(kz - np.rint(kz))) > 1e-7:
        return None
    groups = (np.rint(kz).astype(int) + nz // 2) % nz
    if len(np.unique(groups)) != nz:
        return None
    partitions = [np.flatnonzero(groups == z) for z in range(nz)]
    count = len(partitions[0])
    if any(len(indices) != count for indices in partitions):
        return None
    # Sorting permits interleaved partitions and independent spoke order while
    # keeping repeated centre samples. Exact geometry is checked after sorting.
    for z, indices in enumerate(partitions):
        key = np.rint(nodes[indices, :2] * 1e8).astype(np.int64)
        partitions[z] = indices[np.lexsort((key[:, 1], key[:, 0]))]
    xy = nodes[partitions[0], :2]
    if any(
        not np.allclose(nodes[indices, :2], xy, rtol=0, atol=1e-8)
        for indices in partitions[1:]
    ):
        return None
    ordered = np.column_stack([data[indices] for indices in partitions])
    slices = fft(ordered, inverse=True, axes=(1,)) * np.sqrt(nz)
    result = _inverse_nufft(
        slices.reshape(1, count, 1, nz), _trajectory(xy, matrix[:2]), (*matrix[:2], 1)
    )
    return result.reshape(matrix)


def reconstruct_noncartesian(raw, acq, matrix, fov_mm):
    """Standard BART Tikhonov inverse NUFFT, fixed lambda and iteration bound.

    Nyquist overscan is excluded; the retained band wraps periodically only
    after the fixed physical half-voxel phase correction. BART receives its
    unitary-operator signal scale, with no fitted gain or field-map correction.
    """
    matrix = tuple(map(int, matrix))
    original_nodes = np.asarray(
        acq.k[:, : len(matrix)] * (np.asarray(fov_mm) / 1000) / np.asarray(matrix),
        dtype=np.float64,
    )
    keep = np.all(np.abs(original_nodes) <= 0.5 + 1e-5, axis=1)
    nodes = np.mod(original_nodes[keep] + 0.5, 1.0) - 0.5
    if not len(nodes):
        raise ValueError("No samples inside reconstruction Nyquist box")
    unwrapped = (
        acq.k[keep, : len(matrix)] * (np.asarray(fov_mm) / 1000) / np.asarray(matrix)
    )
    corrected = np.asarray(raw)[keep] * np.exp(1j * np.pi * np.sum(unwrapped, axis=1))
    data = corrected / np.sqrt(np.prod(matrix))
    stack = _stack_inverse(data, nodes, matrix)
    if stack is not None:
        return stack
    shape = matrix + (1,) * (3 - len(matrix))
    return _inverse_nufft(
        data.reshape(1, -1, 1), _trajectory(nodes, matrix), shape
    ).reshape(matrix)
