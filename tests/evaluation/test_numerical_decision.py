"""Nonnegative error intervals and independent receiver/dropout stress controls."""

import numpy as np
import pytest

from mriseqbench.evaluation.s1 import resolve_decision, score_precision
from mriseqbench.evaluation.s1_case import case_bounds, case_image_scores
from mriseqbench.models import ImagePrecision


@pytest.mark.parametrize(
    "name,squared",
    [
        ("phase_error_rad", "normalized_squared_phase_error"),
        ("magnitude_image_nrmse", "normalized_squared_magnitude_error"),
    ],
)
def test_small_error_interval_crossing_zero_can_certify_pass(name, squared):
    metrics = {name: np.sqrt(0.0005), squared: 0.0005}
    intervals = {name: (0, np.sqrt(0.00278)), squared: (-0.00177, 0.00278)}
    outcome, basis = resolve_decision(
        [{"name": name, "min": None, "max": 0.35}],
        metrics,
        intervals,
        precise=False,
    )
    assert outcome is True and basis == "interval_certified"


@pytest.mark.parametrize("precise", [False, True])
def test_negative_corrected_point_needs_a_decisive_interval(precise):
    metrics = {"phase_error_rad": 0.0, "normalized_squared_phase_error": -0.0005}
    intervals = {
        "phase_error_rad": (0, np.sqrt(0.003)),
        "normalized_squared_phase_error": (-0.002, 0.003),
    }

    def resolve(limit):
        return resolve_decision(
            [{"name": "phase_error_rad", "min": None, "max": limit}],
            metrics,
            intervals,
            precise,
        )

    assert resolve(0.35) == (True, "interval_certified")
    assert resolve(0.01) == (None, "unresolved")


@pytest.mark.parametrize(
    "squared_interval", [(0.2, 0.3), (float("nan"), 0.01), (0.1, 0.01)]
)
def test_large_or_invalid_error_interval_does_not_pass(squared_interval):
    metrics = {"phase_error_rad": 0.5, "normalized_squared_phase_error": 0.25}
    intervals = {
        "phase_error_rad": (0, 0),
        "normalized_squared_phase_error": squared_interval,
    }
    outcome, _ = resolve_decision(
        [{"name": "phase_error_rad", "min": None, "max": 0.35}],
        metrics,
        intervals,
        False,
    )
    assert outcome is not True


@pytest.mark.parametrize("corruption", ["none", "receiver", "blank", "gain", "ghost"])
def test_physical_image_requirements_still_reject_corruption(corruption):
    reference = np.ones((8, 8), complex)
    reference[4:] = 2
    images = np.repeat(reference[None], 8, axis=0)
    if corruption == "receiver":
        images *= -1j
    elif corruption == "blank":
        images[:] = 0
    elif corruption == "gain":
        images *= 2
    elif corruption == "ghost":
        images = np.roll(images, 4, axis=1)
    support = np.ones((8, 8), bool)
    masks = {
        "myocardium": np.indices((8, 8))[0] < 4,
        "blood": np.indices((8, 8))[0] >= 4,
    }
    metrics, intervals, _ = case_image_scores(
        images, reference, support, np.ones_like(reference), support, masks
    )
    tolerances = ImagePrecision().model_dump()
    precise, _ = score_precision(metrics, intervals, tolerances)
    criteria = {
        "complex_nrmse_max": 0.45,
        "magnitude_nrmse_max": 0.35,
        "phase_rms_rad_max": 0.35,
        "tissue_signal_fraction_min": 0.5,
        "contrast_retention_min": 0.5,
        "outside_energy_fraction_max": 0.25,
    }
    outcome, _ = resolve_decision(
        case_bounds(criteria),
        metrics,
        intervals,
        precise,
    )
    assert outcome is (corruption == "none")
