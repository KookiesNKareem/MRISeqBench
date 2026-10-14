"""Independent clean targets and receiver-invariant mechanism checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.s1_phantom import phantom
from mriseqbench.evaluation.s1_sampling import analytic_reference
from mriseqbench.evaluation.s2 import canonical_response, reference_image
from mriseqbench.evaluation.s2_sequence import inspect_sequence
from mriseqbench.models import Evaluation
from tests.fixtures.legacy_catalog import resolve_legacy_case

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "expert_controls", ROOT / "tests/fixtures/s2_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)


def case_for(profile="ideal", obj="uniform_disc"):
    catalog = Catalog(ROOT)
    case = resolve_legacy_case(catalog, "epi_single_shot", profile, obj)
    # Isolated analytic regression: qualify the fixture policy, not active cardiac cases.
    case["evaluation"] = Evaluation.model_validate(
        load_yaml(ROOT / "tests/fixtures/legacy_evaluators/s2.yaml")
    ).model_dump(mode="json")
    case["image_policy"] = case["object_level"]
    case["task"]["sequence"]["matrix"] = [8, 8]
    case["evaluation"].update(version=3, spatial_samples=4096, spatial_replicates=4)
    return case


def test_clean_target_does_not_absorb_echo_train_modulation(tmp_path):
    case = case_for()
    properties = phantom(case, ROOT, 4).properties
    clean = []
    for spacing in (0.00074, 0.0016):
        path = tmp_path / f"epi-{spacing}.seq"
        controls.make_sequence(case, epi_flat=0.00032, epi_spacing=spacing).write(
            str(path)
        )
        acq = inspect_sequence(path, case)
        response = canonical_response(acq, properties, "epi_single_shot")
        raw = analytic_reference(acq, case, ROOT, response)
        clean.append(reference_image(acq, case, ROOT, response, raw))
    assert np.linalg.norm(clean[0] - clean[1]) / np.linalg.norm(clean[0]) < 1e-9


@pytest.mark.parametrize("rotation", [False, True])
def test_receiver_tuning_preserves_spoiling_but_not_image_rotation(tmp_path, rotation):
    case = case_for("b0_offset", "tissue_discs")
    path = tmp_path / "sequence.seq"
    controls.make_sequence(
        case,
        "receiver_phase" if rotation else None,
        epi_flat=0.00032,
        epi_spacing=0.00074,
        receiver_frequency=100,
    ).write(str(path))
    result = evaluate(path, case, ROOT, tmp_path / "evaluation")
    assert result["sequence_valid"] is True
    assert next(c for c in result["checks"] if c["name"] == "spoiled_gre")["passed"]
    assert result["status"] == ("failed" if rotation else "passed")
    if rotation:
        assert any(not c["passed"] for c in result["case_image_checks"])
