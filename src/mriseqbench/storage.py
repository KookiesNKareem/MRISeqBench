"""Lossless run storage: immutable image blobs and recoverable replay archives.

Original evidence paths remain ordinary files. Only trusted image bundles are
hard-linked; mutable agent files, reports and frozen source trees are untouched.
"""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary(path.parent) as tmp:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        publish(tmp, path)


def publish(temporary_path, destination):
    """Persist bytes and their directory entry before removing original data."""
    with Path(temporary_path).open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_path, destination)
    directory_fd = os.open(Path(destination).parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def temporary(directory):
    fd, name = tempfile.mkstemp(prefix=".storage-", dir=directory)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def atomic_savez_compressed(path, **arrays):
    """Replacing a directory entry never modifies an older shared inode."""
    import numpy as np

    path = Path(path)
    with temporary(path.parent) as tmp:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        publish(tmp, path)


def store_directory(workspace):
    workspace = Path(workspace).resolve()
    for parent in (workspace, *workspace.parents):
        if parent.name == "runs":
            return parent / ".artifacts"
    # Standalone evaluations remain self-contained outside a runs tree.
    return workspace / ".artifacts"


def _identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class ArtifactStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def deduplicate(self, path):
        """Called under locked(); reject concurrent content changes and corruption."""
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular artifact: {path}")
        identity = _identity(path)
        digest = sha256(path)
        blob = self.directory / "sha256" / digest[:2] / digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            if blob.is_symlink() or sha256(blob) != digest:
                raise ValueError(f"Corrupt artifact blob: {blob}")
        else:
            # Copy the first instance so pre-existing third-party hard links do
            # not gain write access to the shared store.
            with temporary(blob.parent) as tmp:
                shutil.copyfile(path, tmp)
                if sha256(tmp) != digest:
                    raise ValueError(f"Artifact changed while copying: {path}")
                tmp.chmod(0o444)
                publish(tmp, blob)
        if _identity(path) != identity or sha256(path) != digest:
            raise ValueError(f"Artifact changed during deduplication: {path}")
        if not os.path.samefile(path, blob):
            with temporary(path.parent) as tmp:
                tmp.unlink()
                os.link(blob, tmp)
                os.replace(tmp, path)
        return {"sha256": digest, "bytes": identity[2], "blob": str(blob)}


def verify_checkpoint(record):
    blob = Path(record["blob"])
    if blob.is_symlink():
        raise ValueError(f"Symlink checkpoint blob: {blob}")
    digest = hashlib.sha256()
    size = 0
    with gzip.open(blob, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != record["sha256"] or size != record["bytes"]:
        raise ValueError(f"Corrupt checkpoint archive: {blob}")


def completed_evidence(report):
    """Numerically complete uncalibrated evidence is safe to archive, not score."""
    return (
        report.get("status") in {"passed", "failed"}
        and report.get("numerically_converged") is not False
    ) or (
        report.get("status") == "uncalibrated"
        and report.get("numerically_converged") is True
    )


def result_artifacts(report):
    """S4 keeps each cardiac image's artifacts in its frame result."""
    artifacts = dict(report.get("artifacts", {}))
    for number, frame in enumerate(report.get("frames", [])):
        artifacts.update(
            (f"frame_{number}/{name}", path)
            for name, path in frame.get("artifacts", {}).items()
        )
    return artifacts


def finalize_evaluation(report, workspace):
    """Seal complete evidence and deduplicate images; restart caches stay raw.

    Errors/unresolved jobs retain their original restart files. No report is edited.
    """
    workspace = Path(workspace).resolve()
    if not completed_evidence(report):
        return None
    artifacts = result_artifacts(report)
    image_names = [name for name in artifacts if name.rsplit("/", 1)[-1] == "images"]
    if not image_names:
        return None
    if any(Path(path).is_symlink() for path in artifacts.values()):
        raise ValueError("Cannot seal symlink artifacts")
    paths = {name: Path(path).resolve() for name, path in artifacts.items()}
    if any(not path.is_relative_to(workspace) for path in paths.values()):
        return None  # Reused attempt results belong to their original workspace.
    store = ArtifactStore(store_directory(workspace))
    receipt_path = workspace / "storage-finalization.json"
    with store.locked():
        hashes = {name: sha256(path) for name, path in paths.items()}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["sealed_result"] != report:
                raise ValueError(f"Sealed result changed: {receipt_path}")
            if receipt["artifact_sha256"] != hashes:
                raise ValueError(f"Sealed evidence changed: {receipt_path}")
        else:
            receipt = {
                "schema_version": 1,
                "sealed_result": report,
                "artifact_sha256": hashes,
            }
        receipt["image_bundles"] = {
            name: store.deduplicate(paths[name]) for name in image_names
        }
        atomic_json(receipt_path, receipt)
        return receipt


def restore_checkpoints(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    restored = 0
    # Older campaigns archived replay checkpoints; restore their exact bytes.
    for record in receipt.get("checkpoints", []):
        verify_checkpoint(record)
        path = Path(record["path"])
        if path.exists():
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Refusing to overwrite changed checkpoint: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary(path.parent) as tmp:
            with gzip.open(record["blob"], "rb") as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target)
            tmp.chmod(record["mode"])
            publish(tmp, path)
        restored += 1
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description="Restore exact checkpoint bytes")
    parser.add_argument("restore", choices=["restore"])
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps({"restored_checkpoints": restore_checkpoints(args.receipt)}))


if __name__ == "__main__":
    main()
