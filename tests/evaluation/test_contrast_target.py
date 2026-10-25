"""Which acquired samples set the S1/S2 image target contrast; no simulation."""

from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_probes import reference_index
from mriseqbench.evaluation.s1_sequence import inspect_sequence as inspect_s1
from mriseqbench.evaluation.s2 import contrast_indices, reference_response
from mriseqbench.evaluation.s2_sequence import inspect_sequence
from tests.fixtures.reference128_readouts import build

ROOT = Path(__file__).resolve().parents[2]


def acquisition(tmp_path, task, inspector=inspect_sequence, **options):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", task), hardware_ref="standard_1_5t")
    path = tmp_path / f"{task}.seq"
    build(case).write(str(path))
    return inspector(path, case), case["task"]["sequence"]


@pytest.mark.parametrize(
    "task,readouts",
    [("radial_readout", 202), ("ute_radial", 404), ("spiral_gre", 16)],
)
def test_noncartesian_target_uses_every_centre_readout(tmp_path, task, readouts):
    acq, spec = acquisition(tmp_path, task)
    ids = contrast_indices(acq, spec)
    assert len(ids) == len(acq.lines) == readouts
    q = acq.k[ids, :2] * np.asarray(spec["fov_mm"])[:2] / 1000
    assert np.linalg.norm(q, axis=1).max() <= 0.75


def test_propeller_target_uses_the_central_line_of_each_blade(tmp_path):
    acq, spec = acquisition(tmp_path, "propeller_tse")
    ids = contrast_indices(acq, spec)
    blades = len(np.unique(acq.excitation))
    assert len(ids) == blades == len(acq.lines) // 8


@pytest.mark.parametrize(
    "task,inspector",
    [("epi_single_shot", inspect_sequence), ("gre_t1w", inspect_s1)],
)
def test_cartesian_target_is_the_ky0_echo(tmp_path, task, inspector):
    acq, spec = acquisition(tmp_path, task, inspector)
    ids = contrast_indices(acq, spec)
    np.testing.assert_array_equal(ids, [reference_index(acq, spec)])
    q = acq.k[ids[0], : len(spec["matrix"])] * np.asarray(spec["fov_mm"]) / 1000
    assert np.linalg.norm(q) <= 0.51


def test_target_response_is_the_mean_over_contrast_samples(tmp_path):
    acq, spec = acquisition(tmp_path, "radial_readout")
    response = np.vstack([np.arange(len(acq.adc)), np.ones(len(acq.adc))]) + 0j
    ids = contrast_indices(acq, spec)
    np.testing.assert_allclose(
        reference_response(acq, spec, response), [ids.mean(), 1.0]
    )


def test_s1_target_and_phase_oracle_share_one_central_echo(tmp_path, monkeypatch):
    from mriseqbench.evaluation import s2 as s2_module

    acq, spec = acquisition(tmp_path, "gre_t1w", inspect_s1)
    fov = np.asarray(spec["fov_mm"]) / 1000
    acq.k = acq.k.copy()
    acq.k[:, 1] += 0.5 / fov[1]  # half-sample y lattice: ky = +/-0.5 lines tie
    q = acq.k[:, :2] * fov
    radius = np.linalg.norm(q, axis=1)
    tied = np.flatnonzero(radius <= radius.min() + 1e-3)
    assert (
        len({next(j for j, ids in enumerate(acq.lines) if i in ids) for i in tied}) > 1
    )
    target = contrast_indices(acq, spec)
    np.testing.assert_array_equal(target, [tied.min()])  # earliest echo only
    seen = {}
    monkeypatch.setattr(
        s2_module,
        "shared_phase_reference",
        lambda *args, sample_indices, **kwargs: seen.setdefault("ids", sample_indices),
    )
    case = Catalog(ROOT).resolve(
        Catalog(ROOT).get("tasks", "gre_t1w"), hardware_ref="standard_1_5t"
    )
    s2_module.phase_reference(acq, case, ROOT, None)
    np.testing.assert_array_equal(seen["ids"], target)
