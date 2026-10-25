#!/usr/bin/env python3
"""Archive local generated data on Ultron, verify it, then retain only figures.

Dry run by default. Run only while local result writers are idle. Full inventories
and cleanup journals stay on Ultron; only a compact receipt stays in docs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ("runs", "output", "tmp", "paper")
VISUAL = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp", ".gif", ".mp4"}
REMOTE_ROOT = "/ssd-pool/kfareed/MRISeqBench-results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retained(path: Path) -> bool:
    if path.name == "REMOTE_RESULTS.md" and path.parent == Path("runs"):
        return True
    if "__pycache__" in path.parts:
        return False
    figure = (
        path.suffix.lower() in VISUAL
        or path.suffix in {".py", ".tex", ".md"}
        or path.name.endswith(".caption.txt")
    )
    # Frozen source snapshots in runs belong with the remote evidence, including
    # their bundled assets. Actual campaign figure directories remain local.
    if path.parts[0] == "runs":
        return "figures" in path.parts and figure
    return figure or path.name == "README.txt"


def ssh(code: str, *args: str) -> str:
    command = "python3 - " + " ".join(shlex.quote(arg) for arg in args)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "Ultron", command],
        input=code,
        text=True,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"Ultron command failed: {result.stderr.strip()}")
    return result.stdout


def upload(source: Path, destination: str) -> None:
    subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes", str(source), destination],
        check=True,
    )


VERIFY = """import hashlib, json, os, pathlib, sys
base = pathlib.Path(sys.argv[1])
m = json.loads((base / "manifest.json").read_text())
for e in m["entries"]:
    p = base / e["path"]
    if e["type"] == "symlink":
        assert p.is_symlink() and os.readlink(p) == e["target"], str(p)
    else:
        assert not p.is_symlink() and p.stat().st_size == e["size"], str(p)
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        assert h.hexdigest() == e["sha256"], str(p)
result = {"status": "passed", "records": len(m["entries"]),
          "manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest()}
(base / "verification.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="archive, verify and remove raw local files",
    )
    args = parser.parse_args()
    rp = ROOT / "docs/local-storage-receipt.json"
    previous = json.loads(rp.read_text()) if rp.exists() else None
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    entries = []
    kept = []
    for tree in TREES:
        for p in sorted((ROOT / tree).rglob("*")):
            relative = p.relative_to(ROOT)
            if (p.is_file() or p.is_symlink()) and (
                retained(relative) or str(relative) in tracked
            ):
                kept.append(str(relative))
                continue
            if p.is_symlink():
                e = {"path": str(relative), "type": "symlink", "target": os.readlink(p)}
            elif p.is_file():
                e = {
                    "path": str(relative),
                    "type": "file",
                    "size": p.stat().st_size,
                    "sha256": digest(p),
                }
            else:
                continue
            entries.append(e)
    summary = {
        "records": len(entries) + len(kept),
        "remove_records": len(entries),
        "remove_logical_bytes": sum(e.get("size", 0) for e in entries),
        "keep_records": len(kept),
    }
    print(json.dumps(summary), flush=True)
    if not args.apply or not entries:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    remote = f"{REMOTE_ROOT}/archives/laptop-generated-{stamp}"
    manifest = {
        "source": str(ROOT),
        "remote": f"Ultron:{remote}",
        "created_utc": stamp,
        "entries": entries,
        "retained_local_paths": kept,
    }
    before = shutil.disk_usage(ROOT).free
    with tempfile.TemporaryDirectory(prefix="mriseqbench-offload-") as temp:
        temp = Path(temp)
        mp = temp / "manifest.json"
        mp.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        filelist = temp / "files.list"
        filelist.write_bytes(b"".join(os.fsencode(e["path"]) + b"\0" for e in entries))
        ssh(
            "import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=False)",
            remote,
        )
        print(f"Copying raw-data snapshot to Ultron:{remote}", flush=True)
        subprocess.run(
            [
                "rsync",
                "-azHR",
                "--from0",
                f"--files-from={filelist}",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=15",
                "./",
                f"Ultron:{remote}/",
            ],
            cwd=ROOT,
            check=True,
        )
        upload(mp, f"Ultron:{remote}/manifest.json")
        verification = json.loads(ssh(VERIFY, remote))
        assert verification["status"] == "passed"
        assert verification["records"] == len(entries)
        assert verification["manifest_sha256"] == digest(mp)
        print(
            f"All {len(entries)} remote file/link records verified; checking local stability.",
            flush=True,
        )

        def unchanged(e: dict) -> bool:
            p = ROOT / e["path"]
            if e["type"] == "symlink":
                return p.is_symlink() and os.readlink(p) == e["target"]
            return (
                not p.is_symlink()
                and p.is_file()
                and p.stat().st_size == e["size"]
                and digest(p) == e["sha256"]
            )

        # Abort before deleting anything if a writer changed the snapshot.
        for e in entries:
            if not unchanged(e):
                raise RuntimeError(
                    f"Local file changed; no cleanup started: {e['path']}"
                )
        receipt = {
            "status": "verified_cleanup_pending",
            "remote": f"Ultron:{remote}",
            "verification": verification,
            **summary,
            "removed_records": 0,
            "removed_logical_bytes": 0,
            "free_before": before,
            "note": "Full inventory and original absolute symlink targets are in the remote manifest. Restore the original local layout for absolute-path replay receipts.",
        }
        if previous:
            receipt["previous_cleanup"] = {
                key: previous[key]
                for key in (
                    "remote",
                    "status",
                    "removed_records",
                    "removed_logical_bytes",
                )
            }

        def save() -> None:
            target = rp.with_suffix(".json.tmp")
            target.write_text(json.dumps(receipt, indent=2) + "\n")
            target.replace(rp)

        save()
        try:
            skipped = []
            for e in entries:
                if not unchanged(e):
                    skipped.append(e["path"])
                    continue
                (ROOT / e["path"]).unlink()
                receipt["removed_records"] += 1
                receipt["removed_logical_bytes"] += e.get("size", 0)
            for tree in TREES:
                for p in sorted(
                    (ROOT / tree).rglob("*"), key=lambda p: len(p.parts), reverse=True
                ):
                    if p.is_dir() and not p.is_symlink():
                        try:
                            p.rmdir()
                        except OSError:
                            pass
            receipt["skipped_changed_paths"] = skipped
            receipt["status"] = "partial_cleanup" if skipped else "complete"
        except Exception:
            receipt["status"] = "partial_cleanup"
            raise
        finally:
            receipt["free_after"] = shutil.disk_usage(ROOT).free
            save()
            upload(rp, f"Ultron:{remote}/cleanup-receipt.json")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
