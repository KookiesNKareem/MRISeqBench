"""Target construction regression; unit fixtures are not physical qualification."""

from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1 import evaluate_s1

ROOT = Path(__file__).resolve().parents[2]


def test_current_s1_target_excludes_missing_lines_and_distorted_encoding(
    tmp_path, monkeypatch
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "gre_t1w"))
    spec = case["task"]["sequence"]
    spec["matrix"] = [8, 8]
    matrix = np.asarray(spec["matrix"])
    indices = np.indices(tuple(matrix)).reshape(2, -1).T
    k = np.zeros((len(indices), 3))
    k[:, :2] = (indices - matrix // 2) / (np.asarray(spec["fov_mm"]) / 1000)
    acquisition = SimpleNamespace(
        adc=np.arange(len(indices), dtype=float),
        k=k,
        indices=indices,
        offset=np.zeros(2),
        checks=[],
        measurements={},
    )
    response = np.array([0.1 - 0.05j, 0.2 + 0.07j, 0.3 - 0.1j])[:, None] * np.linspace(
        0.8, 1, len(indices)
    )
    submitted = tmp_path / "unit-fixture.seq"
    submitted.write_text("Target-only fixture; no physical sequence claim")
    captured = []

    class TargetCaptured(Exception):
        pass

    def capture_phase(_acq, _case, _root, reference, **_kwargs):
        captured.append(reference.copy())
        raise TargetCaptured

    def fixed_fft(raw, grid, shape):
        np.testing.assert_array_equal(grid.indices, indices)
        np.testing.assert_allclose(grid.k, k)
        return np.fft.ifftn(np.fft.ifftshift(raw.reshape(shape)))

    monkeypatch.setattr("mriseqbench.evaluation.s2.cartesian_reconstruct", fixed_fft)
    for variant in ("complete", "missing_outer_line", "distorted_encoding"):
        acq = copy(acquisition)
        coefficients = response.copy()
        if variant == "missing_outer_line":
            acq.adc, acq.k, acq.indices = acquisition.adc[8:], k[8:], indices[8:]
            coefficients = coefficients[:, 8:]
        elif variant == "distorted_encoding":
            acq.k, acq.indices = k.copy(), indices.copy()
            outer = np.linalg.norm(k, axis=1) > 0
            acq.k[outer] *= 1.37
            acq.indices[outer] = 0
        try:
            evaluate_s1(
                submitted,
                case,
                ROOT,
                tmp_path / variant,
                inspector=lambda _path, _case, acq=acq: acq,
                probes=lambda *_args, coefficients=coefficients: ([], {}, coefficients),
                reconstructor=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("Candidate encoding entered the target")
                ),
                phase_builder=capture_phase,
            )
        except TargetCaptured:
            pass
        else:
            raise AssertionError("Target was not constructed")
    assert len(captured) == 3
    np.testing.assert_allclose(captured[1], captured[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(captured[2], captured[0], rtol=0, atol=1e-12)
