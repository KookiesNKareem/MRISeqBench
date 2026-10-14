import json
import os
from pathlib import Path

import numpy as np
import pytest

from mriseqbench.storage import (
    ArtifactStore,
    atomic_savez_compressed,
    finalize_evaluation,
    migrate_runs,
    restore_checkpoints,
    sha256,
    verify_migration,
)


def evaluation(directory, status="passed"):
    physical = directory / "physical/s2"
    physical.mkdir(parents=True)
    images = physical / "images.npz"
    atomic_savez_compressed(images, physical=np.arange(20), raw=np.arange(20) + 1j)
    replay = physical / "replays/abc.npy"
    replay.parent.mkdir()
    np.save(replay, np.arange(20) + 1j)
    report = {"status": status, "artifacts": {"images": str(images)}}
    (directory / "result.json").write_text(json.dumps(report))
    return report, images, replay


def test_duplicate_evidence_and_atomic_rewrite_are_independent(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    atomic_savez_compressed(first, a=np.arange(10))
    second.write_bytes(first.read_bytes())
    digest = sha256(first)
    store = ArtifactStore(tmp_path / "store")
    with store.locked():
        a = store.deduplicate(first)
        b = store.deduplicate(second)
    assert a["sha256"] == b["sha256"] == digest
    assert os.path.samefile(first, second)
    assert first.stat().st_mode & 0o222 == 0
    atomic_savez_compressed(first, a=np.arange(50))
    assert sha256(second) == sha256(a["blob"]) == digest
    assert not os.path.samefile(first, second)


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_seal_archive_restore_and_repeat_preserve_all_bytes(tmp_path, status):
    report, images, replay = evaluation(tmp_path / "runs/case", status)
    original = replay.read_bytes()
    evidence = images.read_bytes()
    receipt = finalize_evaluation(report, images.parents[2])
    assert receipt["sealed_result"] == report
    assert images.read_bytes() == evidence
    assert not replay.exists()
    seal = images.parents[2] / "storage-finalization.json"
    assert finalize_evaluation(report, images.parents[2]) == receipt
    with pytest.raises(ValueError, match="Sealed result changed"):
        finalize_evaluation({**report, "metrics": {"changed": True}}, images.parents[2])
    assert restore_checkpoints(seal) == 1
    assert replay.read_bytes() == original
    assert restore_checkpoints(seal) == 0
    finalize_evaluation(report, images.parents[2])
    assert not replay.exists()


@pytest.mark.parametrize(
    "status",
    ["evaluator_error", "numerically_unresolved", "incomplete", "uncalibrated"],
)
def test_unfinished_jobs_keep_restart_files(tmp_path, status):
    report, images, replay = evaluation(tmp_path / "runs/case", status)
    assert finalize_evaluation(report, images.parents[2]) is None
    assert replay.exists()
    assert not (tmp_path / "runs/.artifacts").exists()


def test_converged_uncalibrated_evidence_is_archived_without_scoring(tmp_path):
    report, images, replay = evaluation(tmp_path / "runs/case", "uncalibrated")
    report["numerically_converged"] = True
    receipt = finalize_evaluation(report, images.parents[2])
    assert receipt["sealed_result"]["status"] == "uncalibrated"
    assert report["status"] == "uncalibrated"
    assert not replay.exists()


def test_corruption_blocks_cache_removal_and_evidence_replacement(tmp_path):
    report, images, replay = evaluation(tmp_path / "runs/case")
    store = ArtifactStore(tmp_path / "runs/.artifacts")
    with store.locked():
        record = store.archive_checkpoint(replay)
    blob = Path(record["blob"])
    blob.chmod(0o644)
    blob.write_bytes(b"corrupt")
    with pytest.raises((OSError, ValueError)):
        finalize_evaluation(report, images.parents[2])
    assert replay.exists()
    assert not (images.parents[2] / "storage-finalization.json").exists()


def test_lossless_migration_verifies_receipts_sources_and_archived_files(tmp_path):
    root = tmp_path / "runs"
    _, images, replay = evaluation(root / "first")
    _, duplicate, _ = evaluation(root / "second")
    duplicate.write_bytes(images.read_bytes())
    source = root / "first/source/src/evaluator.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen source")
    receipt = root / "first/record.json"
    receipt.write_text(json.dumps({"artifact_sha256": {"images": sha256(images)}}))
    original = {path: path.read_bytes() for path in (source, receipt, replay)}
    summary = migrate_runs(root)
    assert summary["image_paths"] == 2
    assert summary["unique_image_blobs"] == 1
    assert summary["duplicate_image_bytes"] == images.stat().st_size
    assert summary["archived_files"] == 2
    assert os.path.samefile(images, duplicate)
    assert source.read_bytes() == original[source]
    assert receipt.read_bytes() == original[receipt]
    verify_migration(summary["migration"])
    assert restore_checkpoints(root / "first/storage-finalization.json") == 1
    assert replay.read_bytes() == original[replay]
    receipt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed evidence"):
        verify_migration(summary["migration"])


def test_migration_excludes_active_campaign(tmp_path):
    root = tmp_path / "runs"
    _, _, replay = evaluation(root / "active")
    summary = migrate_runs(root, exclude=[root / "active"])
    assert summary["verified_original_files"] == 0
    assert replay.exists()


def test_migration_leaves_unresolved_images_writable(tmp_path):
    root = tmp_path / "runs"
    _, images, replay = evaluation(root / "unresolved", "numerically_unresolved")
    summary = migrate_runs(root)
    assert summary["image_paths"] == summary["archived_files"] == 0
    assert images.stat().st_mode & 0o200
    assert replay.exists()


def test_migration_verification_recovers_index_after_interruption(tmp_path):
    root = tmp_path / "runs"
    evaluation(root / "case")
    summary = migrate_runs(root)
    (Path(summary["migration"]) / "archives.json").unlink()
    assert verify_migration(summary["migration"])["archived_files"] == 1


def test_corrupt_image_blob_never_replaces_original(tmp_path):
    image = tmp_path / "images.npz"
    image.write_bytes(b"original")
    store = ArtifactStore(tmp_path / "store")
    blob = store.directory / "sha256" / sha256(image)[:2] / sha256(image)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"corrupt")
    with store.locked(), pytest.raises(ValueError, match="Corrupt artifact"):
        store.deduplicate(image)
    assert image.read_bytes() == b"original"


def test_cache_finalization_recovers_after_receipt_published(monkeypatch, tmp_path):
    report, images, replay = evaluation(tmp_path / "runs/case")
    original_unlink = Path.unlink

    def interrupted_unlink(path, *args, **kwargs):
        if path == replay:
            raise OSError("interrupted after sealing")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted_unlink)
        with pytest.raises(OSError, match="interrupted after sealing"):
            finalize_evaluation(report, images.parents[2])
    assert replay.exists()
    assert (images.parents[2] / "storage-finalization.json").is_file()
    finalize_evaluation(report, images.parents[2])
    assert not replay.exists()


def test_failed_atomic_image_write_preserves_previous_bundle(monkeypatch, tmp_path):
    path = tmp_path / "images.npz"
    atomic_savez_compressed(path, a=np.arange(10))
    digest = sha256(path)

    def failed_write(stream, **arrays):
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", failed_write)
    with pytest.raises(OSError, match="disk full"):
        atomic_savez_compressed(path, a=np.arange(20))
    assert sha256(path) == digest
    assert list(tmp_path.iterdir()) == [path]


def test_evaluate_uses_common_finalization(monkeypatch, tmp_path):
    import importlib

    module = importlib.import_module("mriseqbench.evaluation.evaluate")
    report, images, replay = evaluation(tmp_path / "runs/case")
    monkeypatch.setattr(module, "_evaluate", lambda *args, **kwargs: report)
    assert module.evaluate(None, None, None, images.parents[2]) == report
    assert not replay.exists()


def test_migration_rejects_checkpoint_replaced_by_symlink(tmp_path):
    root = tmp_path / "runs"
    _, _, replay = evaluation(root / "case")
    summary = migrate_runs(root)
    replay.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="Changed evidence type"):
        verify_migration(summary["migration"])


def test_migration_verifies_each_declared_archive_identity(tmp_path):
    root = tmp_path / "runs"
    evaluation(root / "first")
    _, _, second = evaluation(root / "second")
    np.save(second, np.arange(20) * 2j)
    summary = migrate_runs(root)
    index = Path(summary["migration"]) / "archives.json"
    archives = json.loads(index.read_text())
    first_record, second_record = archives.values()
    second_record["blob"] = first_record["blob"]
    index.write_text(json.dumps(archives))
    with pytest.raises(ValueError, match="Corrupt checkpoint archive"):
        verify_migration(summary["migration"])


@pytest.mark.parametrize("converged", [False, True])
def test_cardiac_frames_and_shared_cache_finalize_only_after_convergence(
    tmp_path, converged
):
    workspace = tmp_path / "runs/case"
    frames, images, checkpoints = [], [], []
    for number in range(2):
        report, image, replay = evaluation(workspace / f"s4/frame-{number}")
        frames.append(report)
        images.append(image)
        checkpoints.append(replay)
    images[1].write_bytes(images[0].read_bytes())
    shared = workspace / "s4/full-encoded-replays/abc.npy"
    shared.parent.mkdir()
    np.save(shared, np.arange(20))
    checkpoints.append(shared)
    original = {path: path.read_bytes() for path in [*images, *checkpoints]}
    report = {
        "status": "passed" if converged else "numerically_unresolved",
        "numerically_converged": converged,
        "frames": frames,
        "encoded_replay_cache": {"directory": str(shared.parent)},
    }
    snapshot = json.dumps(report)
    receipt = finalize_evaluation(report, workspace)
    assert json.dumps(report) == snapshot
    if not converged:
        assert receipt is None
        assert all(path.read_bytes() == content for path, content in original.items())
        return
    assert receipt["sealed_result"] == report
    assert len(receipt["image_bundles"]) == 2
    assert all(image.read_bytes() == original[image] for image in images)
    assert os.path.samefile(*images)
    assert all(not path.exists() for path in checkpoints)
    assert restore_checkpoints(workspace / "storage-finalization.json") == 3
    assert all(path.read_bytes() == content for path, content in original.items())
