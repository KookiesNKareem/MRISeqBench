"""S3 task identity and per-family image limits."""

from pathlib import Path

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s3
from mriseqbench.evaluation.s3_sequence import calibration_width, prescribed_mask

ROOT = Path(__file__).resolve().parents[2]


def test_grappa_mask_is_distinct_autocalibrating_r3():
    catalog = Catalog(ROOT)
    grappa = catalog.get("tasks", "grappa_sampling").sequence.sampling
    sense = catalog.get("tasks", "sense_sampling").sequence.sampling
    g = prescribed_mask("grappa_sampling", (128, 128), grappa.model_dump())
    s = prescribed_mask("sense_sampling", (128, 128), sense.model_dump())
    lines = np.flatnonzero(g.all(axis=0))
    assert len(lines) == 59 and not np.array_equal(g, s)
    assert set(range(52, 76)) <= set(lines)
    assert set(lines) - set(range(52, 76)) == {y for y in range(0, 128, 3)} - set(
        range(52, 76)
    )
    # iy 51 (a multiple of 3) abuts the ACS block: 51..75 is fully sampled.
    assert calibration_width(g) == 25


def test_task_image_limits_replace_static_limits():
    catalog = Catalog(ROOT)
    evaluator = catalog.get("evaluators", "s3")
    static = evaluator.case_image_criteria["static"].model_dump(mode="json")
    for task, override in evaluator.family_image_criteria.items():
        case = catalog.resolve(catalog.get("tasks", task))
        limits = case["evaluation"]["case_image_criteria"]["static"]
        assert limits == override.model_dump(mode="json")
        assert "family_image_criteria" not in case["evaluation"]
        # Calibrated limits still reject a +0.6 rad receiver phase error
        # (measured complex NRMSE >= 0.57 and phase RMS >= 0.57 rad on GPU).
        assert limits["complex_nrmse_max"] < 0.5
        assert limits["phase_rms_rad_max"] < 0.5
        # Never looser than the static limits.
        for name, value in limits.items():
            assert (
                (value <= static[name])
                if name.endswith("_max")
                else (value >= static[name])
            )


def test_autocalibration_applies_only_to_grappa():
    for fam in ("grappa_sampling", "sense_sampling", "caipirinha"):
        acq = type("A", (), {"measurements": {"family": fam}})()
        assert s3.autocalibrated(acq) == (fam == "grappa_sampling")


def test_zero_sample_adc_fails_cleanly(tmp_path):
    import pypulseq as pp

    from mriseqbench.evaluation.s3_sequence import inspect_sequence

    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "grappa_sampling"))
    system = pp.Opts()
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(0.1, duration=4e-4, system=system))
    seq.add_block(pp.make_adc(128, dwell=1e-5, system=system))
    empty = pp.make_adc(1, dwell=1e-5, system=system)
    empty.num_samples = 0
    seq.add_block(empty, pp.make_delay(1e-3))
    path = tmp_path / "empty.seq"
    seq.write(str(path))
    acq = inspect_sequence(path, case)
    failed = [c["name"] for c in acq.checks if not c["passed"]]
    assert "readout_samples" in failed


def test_lattice_offset_preserves_valid_negative_half_sample_offsets():
    from mriseqbench.evaluation.s3_sequence import lattice_offset

    n = 128
    full = np.ones(n, bool)
    grid = np.arange(n) - n // 2
    # A legitimate -0.495 sample offset keeps every node on the grid.
    q = (grid - 0.495)[:, None]
    np.testing.assert_allclose(lattice_offset(q, np.array([n]), full), [-0.495])
    # Both sides of the branch cut: -0.5 + eps and +0.5 - eps.
    for value in (-0.499, -0.4999, 0.4999, 0.499, 0.5001, -0.5001):
        q = (grid + value)[:, None]
        offset = lattice_offset(q, np.array([n]), full)
        indices = np.rint(q[:, 0] - offset[0]).astype(int) + n // 2
        assert indices.min() == 0 and indices.max() == n - 1, value
        # The representative may differ by one sample; the nodes may not.
        assert (
            abs((offset[0] - value) % 1) < 1e-9
            or abs((offset[0] - value) % 1 - 1) < 1e-9
        )


def test_spoiling_probe_spans_the_anisotropic_voxel(monkeypatch):
    seen = []

    def fake(a, spins, *args, **kwargs):
        seen.append(np.ptp(spins.positions, axis=0))
        return np.zeros(len(a.adc), complex)

    monkeypatch.setattr(s3, "simulate", fake)
    monkeypatch.setattr(
        s3, "canonical_response", lambda *_a, **_k: np.ones((1, 4), complex)
    )
    monkeypatch.setattr(s3, "expected_receiver_phase", lambda acq: np.zeros(4))
    for matrix, fov, z in (
        ([128, 128, 32], [200, 200, 200], 6.25),
        ([128, 128], [200, 200], 0),
    ):
        seen.clear()
        acq = type(
            "A",
            (),
            {
                "adc": np.arange(4.0),
                "adc_phase": np.zeros(4),
                "k": np.zeros((4, 3)),
                "measurements": {"centre_sample_indices": [0, 1, 2, 3]},
            },
        )()
        case = {
            "task": {
                "id": "caipirinha",
                "sequence": {"matrix": matrix, "fov_mm": fov},
            }
        }
        s3.spoiling_probes(acq, case, [], np.empty((0, 3)))
        np.testing.assert_allclose(seen[-1] * 1000, [1.5625, 1.5625, z], rtol=0.01)


def test_grappa_temporal_error_compares_calibrated_reconstructions(monkeypatch):
    acq = type("A", (), {"measurements": {"family": "grappa_sampling"}})()
    reference = np.ones((4, 4))

    def fake_reconstruct(raw, acq, matrix, *, calibrate=True):
        raw = np.asarray(raw, float)
        # Calibrated reconstruction is nonlinear in the data.
        return np.full(matrix, raw.sum() ** 2 if calibrate else raw.sum())

    monkeypatch.setattr(s3, "reconstruct", fake_reconstruct)
    delta, coarse = np.full((1, 2), 0.5), np.full((1, 2), 1.0)
    error = s3.temporal_image_error(delta, acq, (4, 4), reference, coarse=coarse)
    terms = acq.measurements["temporal_image_error_terms"]
    np.testing.assert_allclose(terms["calibrated"], 4 * (3**2 - 2**2) / 4)
    np.testing.assert_allclose(terms["known_map_projection"], 4 * 1.0 / 4)
    np.testing.assert_allclose(error, terms["calibrated"])


def test_target_and_phase_oracle_share_one_central_echo(monkeypatch):
    """Half-sample y lattice: two samples tie for the k-space origin 12 ms apart."""
    from types import SimpleNamespace

    from mriseqbench.evaluation import s2

    fov, n = 0.2, 128
    lines, k, adc = [], [], []
    for j, ky in enumerate((-0.5, 0.5)):  # cycles/FOV, symmetric about origin
        ids = []
        for i in range(n):
            ids.append(len(k))
            k.append(((i - n // 2) / fov, ky / fov, 0.0))
            adc.append(0.012 * j + 0.003 + i * 1e-5)
        lines.append(ids)
    acq = SimpleNamespace(
        k=np.asarray(k), adc=np.asarray(adc), lines=lines, measurements={}
    )
    spec = {"matrix": [n, n], "fov_mm": [200, 200]}
    case = {"task": {"sequence": spec}}
    chosen = s3.reference_context(acq, spec)
    assert chosen == lines[0][n // 2]
    # The shared S1/S2 Cartesian selector applies the same earliest-echo rule.
    assert list(s2.reference_indices(acq, spec)) == [chosen]
    seen = {}

    def capture(*args, sample_indices=None, **kwargs):
        seen["indices"] = list(sample_indices)
        return "phase"

    monkeypatch.setattr(s3, "shared_phase_reference", capture)
    assert s3.phase_reference(acq, case, ROOT, None) == "phase"
    assert seen["indices"] == [chosen]
