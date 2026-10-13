"""Independent invariants for the supplied static cardiac anatomy."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mriseqbench.evaluation.s1_anatomy import (
    anatomical_reference,
    fraction_phantom,
    load_anatomy,
    sampled_anatomy,
    validate_assets,
)

ROOT = Path(__file__).resolve().parents[2]


def case(dims):
    return {
        "object": {"phantom": f"cardiac_{dims}d@1"},
        "evaluation": {"spatial_seed": 1729},
        "task": {"sequence": {"matrix": [32] * dims, "fov_mm": [200] * dims}},
    }


@pytest.mark.parametrize("dims", [2, 3])
def test_pinned_tissue_volumes_match_source_and_masks(dims):
    anatomy = load_anatomy(str(ROOT), f"cardiac_{dims}d@1")
    spins = sampled_anatomy(case(dims), ROOT, 4096, 0)
    fine = fraction_phantom(case(dims), ROOT)
    voxel = (0.2 / 32) ** dims
    for label in range(3):
        volume = np.count_nonzero(anatomy.labels == label + 1) * np.prod(
            anatomy.voxel_size_m
        )
        assert spins.weight[spins.labels == label].sum() * voxel == pytest.approx(
            volume, rel=1e-13
        )
        assert fine.weight[fine.labels == label].sum() * voxel == pytest.approx(
            volume, rel=1e-13
        )
    assert np.all(np.abs(spins.positions) < 0.1)
    lower = -np.asarray(anatomy.labels.shape) * anatomy.voxel_size_m / 2
    cells = np.floor((spins.positions[:, :dims] - lower) / anatomy.voxel_size_m).astype(
        int
    )
    np.testing.assert_array_equal(anatomy.labels[tuple(cells.T)] - 1, spins.labels)
    assert np.array_equal(anatomy.properties[:, 0], [0.10000000149011612] * 3)
    assert anatomy.properties[0, 3] == pytest.approx(-2728.726318359375 / (2 * np.pi))
    assert np.array_equal(anatomy.properties[1:, 3], [0, 0])
    assert validate_assets(ROOT)


def test_2d_exact_fourier_matches_direct_cell_integral():
    anatomy = load_anatomy(str(ROOT), "cardiac_2d@1")
    k = np.array([[0, 0, 0], [11.7, -27.3, 0], [-19.1, 32.8, 0]])
    response = np.array([[1, 2, 3], [0.4j, 0.7j, 0.2j], [0.3, 0.9, 0.8]])
    result = anatomical_reference(SimpleNamespace(k=k), case(2), ROOT, response)
    direct = np.zeros(3, complex)
    shape = np.asarray(anatomy.labels.shape)
    for label in range(3):
        indices = np.column_stack(np.nonzero(anatomy.labels == label + 1))
        points = (indices + 0.5 - shape / 2) * anatomy.voxel_size_m
        exact = np.exp(-2j * np.pi * (k[:, :2] @ points.T)).sum(axis=1)
        direct += (
            exact
            * np.prod(
                anatomy.voxel_size_m * np.sinc(k[:, :2] * anatomy.voxel_size_m), axis=1
            )
            / (0.2 / 32) ** 2
            * response[label]
        )
    np.testing.assert_allclose(result, direct, rtol=2e-9, atol=1e-8)


def test_3d_dc_matches_original_source_cell_mass():
    anatomy = load_anatomy(str(ROOT), "cardiac_3d@1")
    result = anatomical_reference(
        SimpleNamespace(k=np.zeros((1, 3))), case(3), ROOT, np.ones((3, 1))
    )
    with h5py.File(ROOT / "benchmark/objects/assets/cardiac_source.phantom") as handle:
        count = len(handle["contrast/T1"])
    assert result[0] == pytest.approx(
        count * np.prod(anatomy.voxel_size_m) / (0.2 / 32) ** 3
    )


def test_slab_clipping_dc_matches_exact_cell_intersections():
    anatomy = load_anatomy(str(ROOT), "cardiac_2d@1")
    lo, hi = -0.0173, 0.0291
    result = anatomical_reference(
        SimpleNamespace(k=np.zeros((1, 3))),
        case(2),
        ROOT,
        np.ones((3, 1)),
        clip_axis=1,
        clip_bounds_m=(lo, hi),
    )
    y = (
        np.arange(anatomy.labels.shape[1]) + 0.5 - anatomy.labels.shape[1] / 2
    ) * anatomy.voxel_size_m[1]
    width = np.maximum(0, np.minimum(y + 0.00075, hi) - np.maximum(y - 0.00075, lo))
    expected = (
        (np.count_nonzero(anatomy.labels, axis=0) * width).sum()
        * 0.0015
        / (0.2 / 32) ** 2
    )
    assert result[0] == pytest.approx(expected, rel=1e-12)


def test_scrambles_differ_and_are_reproducible():
    first = sampled_anatomy(case(2), ROOT, 4096, 0)
    again = sampled_anatomy(case(2), ROOT, 4096, 0)
    other = sampled_anatomy(case(2), ROOT, 4096, 1)
    assert np.array_equal(first.positions, again.positions)
    assert not np.array_equal(first.positions, other.positions)
    assert np.array_equal(first.weight, other.weight)


def test_asset_escape_and_cached_mutation_are_rejected(tmp_path):
    directory = tmp_path / "benchmark/objects"
    shutil.copytree(ROOT / "benchmark/objects", directory)
    metadata_path = directory / "anatomy.json"
    meta = json.loads(metadata_path.read_text())
    first = load_anatomy(tmp_path, "cardiac_2d@1")
    meta["field_profiles"]["3t"]["properties"][0][1] *= 1.1
    metadata_path.write_text(json.dumps(meta))
    changed = load_anatomy(tmp_path, "cardiac_2d@1")
    assert changed.properties[0, 1] != first.properties[0, 1]
    assert changed.source_sha256 != first.source_sha256
    asset = directory / meta["objects"]["cardiac_2d@1"]["path"]
    asset.write_bytes(asset.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_anatomy(tmp_path, "cardiac_2d@1")
    meta["objects"]["cardiac_2d@1"]["path"] = "../../../outside.h5"
    metadata_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="inside benchmark/objects"):
        load_anatomy(tmp_path, "cardiac_2d@1")
    with pytest.raises(ValueError, match="inside benchmark/objects"):
        validate_assets(tmp_path)


def test_native_materialized_cardiac_matches_python_with_fat_and_static_b0(tmp_path):
    field = 0.55
    anatomy = load_anatomy(ROOT, "cardiac_2d_055t@1")
    source = {
        "object": {"phantom": "cardiac_2d_055t@1"},
        "hardware": {"field_strength_T": field},
        "physics": {
            "effects": [
                {
                    "kind": "b0",
                    "model": "linear",
                    "offset_hz": 7.0,
                    "gradient_hz_per_m": [110.0, -90.0],
                }
            ]
        },
        "seed": 42,
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(source))
    subprocess.run(
        [
            "julia",
            "--startup-file=no",
            f"--project={ROOT}",
            str(ROOT / "src/mriseqbench/phantoms/loader.jl"),
            str(case_path),
            str(tmp_path / "materialized"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    indices = np.column_stack(np.nonzero(anatomy.labels))
    expected_xy = (
        indices + 0.5 - np.asarray(anatomy.labels.shape) / 2
    ) * anatomy.voxel_size_m
    tissue = anatomy.labels[tuple(indices.T)] - 1
    order = np.lexsort((expected_xy[:, 1], expected_xy[:, 0]))
    properties = anatomy.properties[tissue][order]
    expected_xy = expected_xy[order]
    with h5py.File(tmp_path / "materialized/phantom.phantom") as handle:
        xy = np.column_stack([handle[f"position/{axis}"][:] for axis in ("x", "y")])
        native_order = np.lexsort((xy[:, 1], xy[:, 0]))
        np.testing.assert_allclose(xy[native_order], expected_xy, atol=1e-15, rtol=0)
        for column, key in enumerate(("ρ", "T1", "T2")):
            np.testing.assert_allclose(
                handle[f"contrast/{key}"][:][native_order],
                properties[:, column],
                rtol=1e-7,
            )
        expected_hz = (
            properties[:, 3] * field / 3 + 7 + expected_xy @ np.array([110.0, -90.0])
        )
        np.testing.assert_allclose(
            handle["contrast/Δw"][:][native_order] / (2 * np.pi),
            expected_hz,
            rtol=1e-7,
            atol=1e-5,
        )


def test_native_constructor_selects_field_profiles_and_rejects_mismatch():
    script = """
    using Test
    include(ARGS[1])
    using .MRISeqBenchPhantoms
    low, labels = build_phantom("cardiac_2d@1"; field_strength_T=0.55)
    @test low.name == "cardiac_2d_055t@1"
    for (name, t1, t2) in (("fat", .187, .093), ("myocardium", .701, .058), ("blood", 1.122, .263))
        indices = findall(==(name), labels)
        @test !isempty(indices)
        @test all(low.T1[indices] .≈ t1)
        @test all(low.T2[indices] .≈ t2)
    end
    fat = labels .== "fat"
    @test all(low.Δw[fat] .≈ (-2728.726318359375 * .55 / 3))
    medium, labels = build_phantom("cardiac_2d_15t@1")
    @test medium.name == "cardiac_2d_15t@1"
    for (name, t1, t2) in (("fat", .343, .058), ("myocardium", .955, .049), ("blood", 1.441, .290))
        indices = findall(==(name), labels)
        @test !isempty(indices)
        @test all(medium.T1[indices] .≈ t1)
        @test all(medium.T2[indices] .≈ t2)
    end
    @test all(medium.Δw[labels .== "fat"] .≈ (-2728.726318359375 / 2))
    @test_throws ErrorException build_phantom("cardiac_2d_3t@1"; field_strength_T=.55)
    """
    subprocess.run(
        [
            "julia",
            "--startup-file=no",
            f"--project={ROOT}",
            "-e",
            script,
            str(ROOT / "src/mriseqbench/phantoms/loader.jl"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_phase_oracle_treats_intrinsic_fat_as_nominal_contrast(tmp_path):
    from test_koma_engine import probe

    from mriseqbench.catalog import Catalog
    from mriseqbench.evaluation.s1_case import phase_reference
    from mriseqbench.evaluation.s1_image import material_masks

    acq, _ = probe(tmp_path, field=0.55)
    catalog = Catalog(ROOT)
    resolved = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), hardware_ref="low_field"
    )
    resolved["task"]["sequence"]["matrix"] = [32, 32]
    resolved["physics"]["effects"] = []
    geometry = fraction_phantom(resolved, ROOT)
    masks = material_masks(geometry, resolved["task"]["sequence"])
    phasor, valid, diagnostic = phase_reference(acq, resolved, ROOT, np.ones((32, 32)))
    fat = masks["fat"] & valid
    water = (masks["blood"] | masks["myocardium"]) & valid
    assert fat.any() and water.any()
    np.testing.assert_allclose(phasor[fat], 1, atol=1e-12)
    np.testing.assert_allclose(phasor[water], 1, atol=1e-12)
    assert diagnostic["converged"]
    # Frozen pre-cardiac contracts keep their original zero-chemistry baseline.
    resolved["evaluation"]["intrinsic_tissue_response"] = False
    legacy, _, _ = phase_reference(acq, resolved, ROOT, np.ones((32, 32)))
    assert np.max(np.abs(legacy[fat] - 1)) > 0.1
