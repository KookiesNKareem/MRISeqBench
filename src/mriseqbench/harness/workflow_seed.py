"""Hash-verified, explicit seed libraries for the supplied-sequence control."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def load_seed(manifest, root, case_id, arm):
    path = Path(manifest)
    if not path.is_absolute():
        path = Path(root) / path
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or document.get("arm") != arm:
        raise ValueError("seed manifest version or authoring arm mismatch")
    seed = document["cases"].get(case_id)
    if not seed or not seed.get("sources"):
        raise ValueError(f"seed sequence and editable sources required for {case_id}")

    def verified(entry):
        source = Path(entry["path"])
        if not source.is_absolute():
            source = path.parent / source
        if source.is_symlink() or not source.is_file():
            raise ValueError("seed input must be a regular, non-symlink file")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"seed SHA-256 mismatch: {source}")
        return data

    sources = {}
    for name, entry in seed["sources"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or any(p.startswith(".") for p in relative.parts)
            or "\\" in name
            or relative.suffix not in {".py", ".jl", ".m"}
            or relative.parts[0]
            in {"bin", "docs", "inputs", "transcript", "feedback", "simulation"}
        ):
            raise ValueError(f"invalid seed source destination: {name}")
        sources[name] = verified(entry)
    sequence = verified(seed["sequence"])
    identity = {
        "sequence": seed["sequence"]["sha256"],
        "sources": {
            name: hashlib.sha256(data).hexdigest() for name, data in sources.items()
        },
    }
    return sequence, sources, identity
