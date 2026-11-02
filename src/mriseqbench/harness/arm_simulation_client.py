"""Standalone client for the selected arm's fixed simulator adapter."""

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def preview_reconstruction(output):
    for preview in output.glob("image_*-magnitude.png"):
        preview.unlink()
    metadata_path = output / "reconstruction.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "completed":
        return
    import h5py
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(output / "reconstruction.h5") as file:
        previews = []
        for entry in metadata["images"]:
            group = file[entry["group"]]
            image = group["image_real"][()] + 1j * group["image_imag"][()]
            # Koma retains x/y/z/echo/coil/repetition axes. Display the central
            # partition and RSS across coils; the complete complex data is saved.
            magnitude = np.sqrt(np.sum(np.abs(image) ** 2, axis=4))
            planes = magnitude[:, :, magnitude.shape[2] // 2, :, :]
            previews.append((entry, planes))
        maximum = max(float(planes.max()) for _, planes in previews) or 1
        for entry, planes in previews:
            frames = planes.shape[2] * planes.shape[3]
            columns = min(frames, 4)
            rows = (frames + columns - 1) // columns
            fig, axes = plt.subplots(
                rows, columns, squeeze=False, figsize=(3 * columns, 3 * rows)
            )
            for axis in axes.flat:
                axis.set_axis_off()
            for echo in range(planes.shape[2]):
                for repetition in range(planes.shape[3]):
                    axis = axes.flat[echo * planes.shape[3] + repetition]
                    axis.imshow(
                        planes[:, :, echo, repetition].T,
                        origin="lower",
                        cmap="gray",
                        vmin=0,
                        vmax=maximum,
                    )
                    axis.set_title(
                        f"ECO={entry['source']['ECO'][echo]}, REP={entry['source']['REP'][repetition]}"
                    )
            fig.suptitle(
                ", ".join(f"{key}={value}" for key, value in entry["labels"].items())
                + f"\nShared magnitude scale 0–{maximum:.3g}"
            )
            fig.tight_layout()
            filename = entry["group"] + "-magnitude.png"
            fig.savefig(output / filename, dpi=100)
            plt.close(fig)
            entry["magnitude_preview"] = filename
    metadata["magnitude_preview_max"] = maximum
    metadata_path.write_text(json.dumps(metadata, indent=2))


def summarize(result, directory):
    """Keep coordinate arrays and complete metadata on disk, not in tool text."""
    reconstruction = json.loads((directory / "reconstruction.json").read_text())
    device = result.get("device", {})
    return {
        **{
            key: result[key]
            for key in (
                "engine",
                "sha256",
                "backend",
                "queue_wait_s",
                "execution_wall_s",
            )
            if key in result
        },
        "device": device.get("device"),
        "signal_shape": device.get("signal_shape"),
        "physio": device.get("physio"),
        "reconstruction": {
            **{
                key: reconstruction[key]
                for key in ("status", "api", "error", "checks")
                if key in reconstruction
            },
            "images": [
                {
                    key: image[key]
                    for key in ("group", "shape", "axes", "labels", "magnitude_preview")
                    if key in image
                }
                for image in reconstruction.get("images", [])
            ],
            "metadata": "simulation/reconstruction.json",
            "complex_images": "simulation/reconstruction.h5"
            if (directory / "reconstruction.h5").is_file()
            else None,
        },
        "full_receipt": "simulation/result.json",
        "output": str(directory),
    }


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "sequence.seq")
    if not source.is_file() or source.stat().st_size > 16777216:
        raise ValueError("expected a Pulseq file <=16 MiB")
    request = urllib.request.Request(
        "http://127.0.0.1:8765/simulate/simulate",
        data=source.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        text = exc.read(16384).decode("utf-8", errors="replace")
        try:
            detail = json.loads(text)
            text = detail.get("stderr") or detail.get("error") or text
        except (ValueError, AttributeError):
            pass
        raise ValueError(f"HTTP {exc.code}: {text}") from exc
    directory = Path("simulation").resolve()
    directory.mkdir(exist_ok=True)
    for filename, content in result.pop("artifacts").items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError("invalid simulator artifact filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(content, validate=True))
    if (directory / "reconstruction.json").is_file():
        preview_reconstruction(directory)
        result["reconstruction"] = json.loads(
            (directory / "reconstruction.json").read_text()
        )
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    summary = (
        summarize(result, directory)
        if (directory / "reconstruction.json").is_file()
        else result
    )
    print(json.dumps({**summary, "output": str(directory)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, urllib.error.URLError) as exc:
        print(f"simulation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
