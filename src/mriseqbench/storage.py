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
import stat
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
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

    def archive_checkpoint(self, path):
        """Keep the exact NPY bytes in a deterministic, compressed shared blob."""
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular checkpoint: {path}")
        identity = _identity(path)
        digest = sha256(path)
        blob = self.directory / "replays" / digest[:2] / (digest + ".gz")
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            with temporary(blob.parent) as tmp:
                with (
                    path.open("rb") as source,
                    tmp.open("wb") as destination,
                    gzip.GzipFile(
                        filename="", mode="wb", fileobj=destination, mtime=0
                    ) as compressed,
                ):
                    shutil.copyfileobj(source, compressed)
                tmp.chmod(0o444)
                publish(tmp, blob)
        record = {
            "sha256": digest,
            "bytes": identity[2],
            "blob": str(blob),
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        verify_checkpoint(record)
        if _identity(path) != identity or sha256(path) != digest:
            raise ValueError(f"Checkpoint changed during archival: {path}")
        return record


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
        report.get("status") in {"passed", "failed", "smoke_passed"}
        and report.get("numerically_converged") is not False
    ) or (
        report.get("status") == "uncalibrated"
        and report.get("numerically_converged") is True
    )


def finalize_evaluation(report, workspace, store=None):
    """Seal complete evidence, deduplicate images, then archive restart caches.

    Archive bytes and their receipt are durable before any checkpoint is removed.
    Errors/unresolved jobs retain their original restart files. No report is edited.
    """
    workspace = Path(workspace).resolve()
    if not completed_evidence(report):
        return None
    artifacts = report.get("artifacts", {})
    if "images" not in artifacts:
        return None
    if any(Path(path).is_symlink() for path in artifacts.values()):
        raise ValueError("Cannot seal symlink artifacts")
    paths = {name: Path(path).resolve() for name, path in artifacts.items()}
    if any(not path.is_relative_to(workspace) for path in paths.values()):
        return None  # Reused attempt results belong to their original workspace.
    store = ArtifactStore(store or store_directory(workspace))
    receipt_path = workspace / "storage-finalization.json"
    with store.locked():
        hashes = {name: sha256(path) for name, path in paths.items()}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["artifact_sha256"] != hashes:
                raise ValueError(f"Sealed evidence changed: {receipt_path}")
            for record in receipt["checkpoints"]:
                verify_checkpoint(record)
        else:
            receipt = {
                "schema_version": 1,
                "sealed_result": report,
                "artifact_sha256": hashes,
                "checkpoints": [],
            }
        receipt["images"] = store.deduplicate(paths["images"])
        cache = paths["images"].parent / "replays"
        known = {row["path"]: row for row in receipt["checkpoints"]}
        pending = []
        if cache.is_dir() and not cache.is_symlink():
            for path in sorted(cache.glob("*.npy")):
                if path.is_symlink():
                    raise ValueError(f"Symlink checkpoint: {path}")
                record = {"path": str(path), **store.archive_checkpoint(path)}
                if (
                    str(path) in known
                    and known[str(path)]["sha256"] != record["sha256"]
                ):
                    raise ValueError(f"Sealed checkpoint changed: {path}")
                known[str(path)] = record
                pending.append(record)
        receipt["checkpoints"] = list(known.values())
        atomic_json(receipt_path, receipt)
        for record in pending:
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Checkpoint changed before removal: {path}")
            path.unlink()
        return receipt


def restore_checkpoints(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    restored = 0
    for record in receipt["checkpoints"]:
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


def inventory(root, excluded=()):
    root = Path(root).resolve()
    entries = {}
    for directory, children, files in os.walk(root):
        for name in children:
            path = Path(directory) / name
            if path.is_symlink() and not any(
                path.is_relative_to(item) for item in excluded
            ):
                entries[str(path.relative_to(root))] = {"symlink": os.readlink(path)}
        children[:] = sorted(
            name
            for name in children
            if name not in {".artifacts", ".storage-migrations"}
            and not (Path(directory) / name).is_symlink()
            and not any(
                (Path(directory) / name).is_relative_to(item) for item in excluded
            )
        )
        for name in sorted(files):
            path = Path(directory) / name
            rel = str(path.relative_to(root))
            if any(path.is_relative_to(item) for item in excluded):
                continue
            if path.is_symlink():
                entries[rel] = {"symlink": os.readlink(path)}
            elif path.is_file():
                before = _identity(path)
                entries[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                if _identity(path) != before:
                    raise ValueError(f"File changed during inventory: {path}")
    return entries


def verify_migration(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "before.json").read_text())
    root = Path(manifest["root"])
    archives_path = directory / "archives.json"
    archives = json.loads(archives_path.read_text()) if archives_path.exists() else {}
    # A crash between cache finalization and the migration index is recoverable
    # from the receipt that was persisted before removing any checkpoint.
    for receipt_path in root.rglob("storage-finalization.json"):
        if {"source", "agent", ".artifacts", ".storage-migrations"}.intersection(
            receipt_path.relative_to(root).parts
        ):
            continue
        for row in json.loads(receipt_path.read_text())["checkpoints"]:
            path = Path(row["path"])
            if path.is_relative_to(root):
                rel = str(path.relative_to(root))
                if rel in manifest["files"]:
                    archives.setdefault(rel, row)
    verified_archives = set()
    for archive in archives.values():
        if archive["blob"] not in verified_archives:
            verify_checkpoint(archive)
            verified_archives.add(archive["blob"])
    for rel, record in manifest["files"].items():
        path = root / rel
        if "symlink" in record:
            if not path.is_symlink() or os.readlink(path) != record["symlink"]:
                raise ValueError(f"Changed symlink: {path}")
        elif path.is_file() and not path.is_symlink():
            if (
                path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(f"Changed evidence: {path}")
        elif rel in archives:
            archive = archives[rel]
            if any(archive[key] != record[key] for key in ("sha256", "bytes")):
                raise ValueError(f"Archive identity mismatch: {path}")
        else:
            raise ValueError(f"Missing evidence: {path}")
    return {
        "verified_original_files": len(manifest["files"]),
        "archived_files": len(archives),
    }


def migrate_runs(root, *, exclude=()):
    """Inventory first; add lossless storage without modifying existing receipts."""
    root = Path(root).resolve()
    excluded = tuple(Path(path).resolve() for path in exclude)
    if any(not path.is_relative_to(root) or path == root for path in excluded):
        raise ValueError("Excluded campaigns must be inside the runs directory")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = root / ".storage-migrations" / stamp
    directory.mkdir(parents=True)
    before = {
        "schema_version": 1,
        "root": str(root),
        "excluded": list(map(str, excluded)),
        "files": inventory(root, excluded),
    }
    atomic_json(directory / "before.json", before)
    store = ArtifactStore(root / ".artifacts")
    completed = {}
    for rel in before["files"]:
        path = root / rel
        if path.name != "result.json" or {"agent", "source"}.intersection(
            path.relative_to(root).parts
        ):
            continue
        try:
            report = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(report, dict) and completed_evidence(report):
            completed[rel] = report
    eligible_images = {
        Path(report["artifacts"]["images"]).resolve()
        for report in completed.values()
        if "images" in report.get("artifacts", {})
    }
    images = []
    with store.locked():
        for rel, record in before["files"].items():
            path = root / rel
            if (
                path.name == "images.npz"
                and path.resolve() in eligible_images
                and not {"agent", "source"}.intersection(path.relative_to(root).parts)
                and "sha256" in record
            ):
                if sha256(path) != record["sha256"]:
                    raise ValueError(f"Artifact changed since inventory: {path}")
                images.append({"path": rel, **store.deduplicate(path)})
    atomic_json(directory / "images.json", images)
    archives = {}
    for rel, report in completed.items():
        path = root / rel
        receipt = finalize_evaluation(report, path.parent, store.directory)
        if receipt:
            for record in receipt["checkpoints"]:
                cache = Path(record["path"])
                if cache.is_relative_to(root):
                    archives[str(cache.relative_to(root))] = record
    # Individual durable receipts already journal every archival. Avoid rewriting
    # the growing campaign-wide index once per result (quadratic I/O).
    atomic_json(directory / "archives.json", archives)
    verification = verify_migration(directory)
    summary = {
        **verification,
        "image_paths": len(images),
        "unique_image_blobs": len({row["sha256"] for row in images}),
        "duplicate_image_bytes": sum(row["bytes"] for row in images)
        - sum({row["sha256"]: row["bytes"] for row in images}.values()),
        "archived_checkpoint_bytes": sum(row["bytes"] for row in archives.values()),
        "migration": str(directory),
    }
    atomic_json(directory / "verification.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("migrate", help="losslessly migrate existing runs")
    migrate.add_argument("runs", type=Path)
    migrate.add_argument("--exclude", type=Path, action="append", default=[])
    verify = commands.add_parser("verify", help="verify every original file/hash")
    verify.add_argument("migration", type=Path)
    restore = commands.add_parser("restore", help="restore exact checkpoint bytes")
    restore.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "migrate":
        result = migrate_runs(args.runs, exclude=args.exclude)
    elif args.command == "verify":
        result = verify_migration(args.migration)
    else:
        result = {"restored_checkpoints": restore_checkpoints(args.receipt)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
