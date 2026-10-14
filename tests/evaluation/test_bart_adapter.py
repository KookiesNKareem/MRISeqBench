"""BART process boundary: CFL fidelity, strict availability and provenance."""

from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.evaluation import bart


def test_cfl_preserves_bart_spatial_and_coil_axes(tmp_path):
    value = np.arange(3 * 4 * 2 * 5).reshape(3, 4, 2, 5) * (1 + 2j)
    path = tmp_path / "array"
    bart.write_cfl(path, value)
    # Official CFL layout is first dimension fastest, independent of C strides.
    assert np.array_equal(
        np.fromfile(path.with_suffix(".cfl"), np.complex64), value.ravel(order="F")
    )
    path.with_suffix(".hdr").write_text(
        path.with_suffix(".hdr").read_text() + "# Command\nfft -i 7 input output\n"
    )
    assert np.array_equal(bart.read_cfl(path), value)


def test_explicit_missing_bart_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("MRISEQBENCH_BART", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="BART is required"):
        bart.fft(np.ones((4, 4)), inverse=True, axes=(0, 1))


def test_unqualified_bart_version_is_rejected_before_reconstruction(
    monkeypatch, tmp_path
):
    binary = tmp_path / "bart"
    binary.write_text("#!/bin/sh\nprintf 'v0.9.00\\n'\n")
    binary.chmod(0o755)
    monkeypatch.setenv("MRISEQBENCH_BART", str(binary))
    with pytest.raises(RuntimeError, match="v1.0.00 is required.*v0.9.00"):
        bart.fft(np.ones((4, 4)), inverse=True, axes=(0, 1))


def test_named_bart_command_runs_actual_binary_and_records_identity():
    value = np.arange(24).reshape(3, 4, 1, 2) * (1 - 1j)
    result = bart.run(["scale", "2", "{data}", "{out}"], {"data": value})
    assert np.array_equal(result, 2 * value)
    provenance = bart.provenance()
    assert provenance["engine"] == "BART"
    assert provenance["version"] == bart.QUALIFIED_VERSION
    assert len(provenance["binary_sha256"]) == 64
    assert any(
        item["argv"] == ["scale", "2", "{data}", "{out}"] and item["calls"] >= 1
        for item in provenance["commands"]
    )


def test_bart_error_does_not_fall_back_to_local_transform():
    with pytest.raises(RuntimeError, match="failed"):
        bart.run(["not_a_bart_command", "{data}", "{out}"], {"data": np.ones(4)})


def test_hybrid_stack_matches_independent_complex_dense_normal_equations():
    matrix = (8, 6, 4)
    rng = np.random.default_rng(20261020)
    xy = np.concatenate((rng.uniform(-0.49, 0.49, (180, 2)), [[0.5, 0], [-0.5, 0]]))
    nodes = np.concatenate(
        [np.column_stack((xy, np.full(len(xy), z / 4))) for z in range(-2, 2)]
    )
    nodes = nodes[rng.permutation(len(nodes))]
    fov = np.array([0.21, 0.165, 0.19])
    k = nodes * matrix / fov
    indices = np.indices(matrix).reshape(3, -1).T
    coordinates = ((indices + 0.5) / matrix - 0.5) * fov
    image = rng.normal(size=matrix) + 1j * rng.normal(size=matrix)
    encoder = np.exp(-2j * np.pi * k @ coordinates.T)
    raw = encoder @ image.ravel()
    operator = encoder / np.sqrt(np.prod(matrix))
    expected = np.linalg.solve(
        operator.conj().T @ operator + 0.001 * np.eye(np.prod(matrix)),
        operator.conj().T @ (raw / np.sqrt(np.prod(matrix))),
    )
    acq = SimpleNamespace(k=k)
    actual = bart.reconstruct_noncartesian(raw, acq, matrix, fov * 1000)
    assert np.linalg.norm(actual.ravel() - expected) / np.linalg.norm(expected) < 5e-5
    rotated = bart.reconstruct_noncartesian(1j * raw, acq, matrix, fov * 1000)
    np.testing.assert_allclose(rotated, 1j * actual, rtol=2e-5, atol=2e-6)
    assert np.linalg.norm(rotated - actual) / np.linalg.norm(actual) > 1.4
    np.testing.assert_array_equal(
        bart.reconstruct_noncartesian(np.zeros_like(raw), acq, matrix, fov * 1000),
        np.zeros(matrix),
    )
    assert any(
        item["argv"] == ["fft", "-i", "2", "{data}", "{out}"]
        for item in bart.provenance()["commands"]
    )


@pytest.mark.parametrize(
    "corruption",
    ["missing_partition", "noninteger_kz", "different_xy", "unequal_counts"],
)
def test_hybrid_stack_rejects_nonseparable_geometry(corruption):
    xy = np.array([[-0.25, 0], [0, 0], [0.25, 0]])
    nodes = np.concatenate(
        [np.column_stack((xy, np.full(len(xy), z / 4))) for z in range(-2, 2)]
    )
    if corruption == "missing_partition":
        nodes = nodes[:-3]
    elif corruption == "noninteger_kz":
        nodes[0, 2] += 0.01
    elif corruption == "different_xy":
        nodes[0, 0] += 0.01
    else:
        nodes = nodes[:-1]
    assert bart._stack_inverse(np.ones(len(nodes)), nodes, (8, 6, 4)) is None
