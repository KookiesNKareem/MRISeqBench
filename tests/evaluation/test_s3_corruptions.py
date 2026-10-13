"""Independent checks of S3 derived-control estimands and clipped scoring regions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s3_stresses", ROOT / "tests/fixtures/score_s3_calibration.py"
)
stresses = importlib.util.module_from_spec(loader)
loader.loader.exec_module(stresses)


def context():
    reference = np.ones((4, 4, 4), complex)
    reference[2:] = 3
    support = np.ones(reference.shape, bool)
    masks = {"a": np.indices(reference.shape)[0] < 2}
    masks["b"] = ~masks["a"]
    stored = {
        "reference": reference,
        "object_support": support,
        "case_reference_phasor": np.ones_like(reference),
        "case_reference_phase_valid": support,
    }
    policy = Catalog(ROOT).get("evaluators", "s3").model_dump(mode="json")
    case = {"evaluation": policy, "image_policy": "static"}
    return reference, masks, stored, case, policy


def test_contrast_erasure_is_of_scored_mean_not_mean_of_erased_replicates():
    reference, masks, stored, case, policy = context()
    images = np.repeat(reference[None], 8, axis=0)
    for i, image in enumerate(images):
        image[masks["b"]] *= np.exp(1j * (0.9 if i % 2 else -0.9))
    legacy = np.array([stresses.erase_contrast(image, masks) for image in images])
    means = [np.abs(legacy.mean(axis=0))[m].mean() for m in masks.values()]
    assert abs(means[0] - means[1]) > 0.5
    result = stresses.score_image_transform(
        images,
        stored,
        case,
        policy,
        lambda image: stresses.erase_contrast(image, masks),
        masks=masks,
    )
    assert result["metrics"]["contrast_retention"] == pytest.approx(0, abs=1e-12)
    assert max(abs(x) for x in result["intervals"]["contrast_retention"]) < 1e-12
    assert result["passed"] is False
    assert result["basis"] != "unresolved"


def test_transform_jackknife_recovers_unbiased_quadratic_identity_score():
    reference, masks, stored, case, policy = context()
    rng = np.random.default_rng(7631)
    images = reference + 0.1 * (
        rng.normal(size=(8, *reference.shape))
        + 1j * rng.normal(size=(8, *reference.shape))
    )
    native = stresses.score(images, stored, case, policy, masks=masks)
    transformed = stresses.score_image_transform(
        images, stored, case, policy, lambda image: image, masks=masks
    )
    assert transformed["metrics"]["normalized_squared_error"] == pytest.approx(
        native["metrics"]["normalized_squared_error"], abs=1e-13
    )


def test_exterior_material_is_excluded_from_selected_strip_scores():
    reference, masks, stored, case, policy = context()
    support = stored["object_support"].copy()
    support[:, 0] = False
    stored["object_support"] = support
    stored["reference"] = np.where(support, reference, 0)
    images = np.repeat(stored["reference"][None], 8, axis=0)
    clipped = {name: mask & support for name, mask in masks.items()}
    full = stresses.score(images, stored, case, policy, masks=masks)
    inner = stresses.score(images, stored, case, policy, masks=clipped)
    assert full == inner
    assert full["passed"] is True
