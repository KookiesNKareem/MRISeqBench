"""Independent, font-free AI target, shared with participant input generation."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
from scipy.special import erf

# Rows run from positive to negative y. Two blank columns separate the letters.
BITMAP = (
    "011100011111",
    "110110000100",
    "110110000100",
    "111110000100",
    "110110000100",
    "110110000100",
    "110110011111",
)
PITCH_MM = 7.5
SIGMA_MM = 1.25
AMPLITUDE = 0.5
TARGET_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmark/references/rf_excitation_2d/target.npz"
)
TARGET_SHA256 = "4c58e1554762f8dee59f710ebb42602e7a0c89dda5d7116ef1af1dada2e7aa90"


def target_at(positions_mm):
    """Gaussian convolution of the union of lit bitmap cells, in physical mm."""
    points = np.asarray(positions_mm, dtype=float)
    x, y = points[..., 0], points[..., 1]
    result = np.zeros(x.shape)
    scale = np.sqrt(2) * SIGMA_MM
    for row, line in enumerate(BITMAP):
        for col, bit in enumerate(line):
            if bit != "1":
                continue
            lo_x = (col - len(line) / 2) * PITCH_MM
            hi_y = (len(BITMAP) / 2 - row) * PITCH_MM
            wx = (erf((x - lo_x) / scale) - erf((x - lo_x - PITCH_MM) / scale)) / 2
            wy = (erf((y - hi_y + PITCH_MM) / scale) - erf((y - hi_y) / scale)) / 2
            result += wx * wy
    return AMPLITUDE * np.clip(result, 0, 1)


def target_sha256():
    digest = hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest()
    if digest != TARGET_SHA256:
        raise ValueError("Canonical AI target asset SHA-256 mismatch")
    return digest


def target_arrays():
    target_sha256()
    with np.load(TARGET_PATH, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_target(path):
    target_sha256()
    shutil.copyfile(TARGET_PATH, path)


def excitation_metrics(response, target):
    response, target = np.asarray(response), np.asarray(target, dtype=float)
    if response.shape != target.shape or not np.isfinite(response).all():
        raise ValueError("Excitation response must be finite and match the target")
    inside, outside = target >= 0.9 * AMPLITUDE, target <= 0.01 * AMPLITUDE
    if not np.any(inside) or not np.any(outside) or not np.any(target):
        raise ValueError("Excitation grid must cover the letters and their background")
    magnitude = np.abs(response)
    return {
        "excitation_magnitude_nrmse": float(
            np.linalg.norm(magnitude - target) / np.linalg.norm(target)
        ),
        "inside_signal_fraction": float(
            np.mean(magnitude[inside]) / np.mean(target[inside])
        ),
        "outside_rms_fraction": float(
            np.sqrt(np.mean(magnitude[outside] ** 2)) / AMPLITUDE
        ),
    }
