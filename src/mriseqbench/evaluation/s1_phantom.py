"""Judge spin sets; catalog S1-S4 objects are all pinned cardiac anatomy."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Spins:
    positions: np.ndarray
    labels: np.ndarray
    materials: list
    properties: np.ndarray
    weight: float | np.ndarray
    grid: tuple
    source_sha256: str


def phantom(case, root):
    """Exact anatomical cell/image-voxel fractions positioned at voxel centres."""
    from .s1_anatomy import fraction_phantom

    return fraction_phantom(case, root)
