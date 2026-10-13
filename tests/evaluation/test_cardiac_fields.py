"""Field selection and intrinsic/declared frequency composition at engine boundaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_koma_engine import probe

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_koma import simulate

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "hardware,field,tag,myocardium",
    [
        ("low_field", 0.55, "055t", (0.701, 0.058)),
        ("standard_1_5t", 1.5, "15t", (0.955, 0.049)),
        ("high_performance", 3.0, "3t", (1.122, 0.052)),
    ],
)
def test_scanner_selects_anatomy_relaxation_and_field(hardware, field, tag, myocardium):
    catalog = Catalog(ROOT)
    for task in ("gre_t1w", "stack_of_stars"):
        case = catalog.resolve(catalog.get("tasks", task), hardware_ref=hardware)
        dims = len(case["task"]["sequence"]["matrix"])
        assert case["object"]["phantom"] == f"cardiac_{dims}d_{tag}@1"
        tissue = case["object"]["tissue_properties"]
        assert (
            tissue["myocardium"]["T1_s"],
            tissue["myocardium"]["T2_s"],
        ) == pytest.approx(myocardium)
        assert tissue["fat"]["chemical_shift_hz"] == pytest.approx(
            -434.2902819118434 * field / 3
        )
        effect = case["physics"]["effects"][0]
        assert effect["gradient_hz_per_m"][0] == pytest.approx(1000 * field / 3)
        assert all(e["kind"] == "b0" for e in case["physics"]["effects"])
        assert "object_level" not in case


def test_explicit_mismatched_tissue_profile_is_rejected():
    catalog = Catalog(ROOT)
    with pytest.raises(ValueError, match="profile and scanner field"):
        catalog.resolve(
            catalog.get("tasks", "gre_t1w"),
            object_ref="cardiac_2d_3t@1",
            hardware_ref="low_field",
        )


@pytest.mark.parametrize("field", [0.55, 1.5, 3.0])
@pytest.mark.parametrize("method", ["bloch", "magnus4"])
def test_intrinsic_fat_shift_matches_explicit_hz_and_adds_declared_b0(
    tmp_path, field, method
):
    acq, water = probe(tmp_path, field=field)
    properties = np.column_stack((water.properties, [-434.2902819118434]))
    fat = replace(water, properties=properties)
    effects = [{"kind": "b0", "model": "constant", "offset_hz": 73.0}]
    expected = simulate(
        acq,
        water,
        effects,
        frequency_offsets=np.full(2, -434.2902819118434 * field / 3),
        integration_method=method,
    )
    actual = simulate(acq, fat, effects, integration_method=method)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
