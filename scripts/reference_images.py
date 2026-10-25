"""Render achieved reference images. Display uses one target-derived scale per case."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def views(array):
    array = np.abs(array)
    if array.ndim == 2:
        return array.T
    # Native 32-partition voxels are four times thicker than in-plane voxels.
    # Nearest-neighbour display preserves that physical aspect ratio.
    through_plane = array.shape[0] // array.shape[2]
    return np.concatenate(
        [
            array[:, :, array.shape[2] // 2].T,
            np.repeat(array[:, array.shape[1] // 2, :].T, through_plane, axis=0),
            np.repeat(array[array.shape[0] // 2, :, :].T, through_plane, axis=0),
        ],
        axis=1,
    )


def note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def collect(directory):
    images = []
    verdict = next(
        (
            directory / name
            for name in ("recovered-verdict.json", "verdict.json")
            if (directory / name).is_file()
        ),
        None,
    )
    report = json.loads(verdict.read_text()) if verdict else None
    if report is None:
        paths = sorted((directory / "evaluation").rglob("images.npz"))
    else:
        paths = [
            Path(frame["artifacts"]["images"])
            for frame in report.get("frames", [report])
            if frame.get("artifacts", {}).get("images")
        ]
        paths = [path if path.is_absolute() else directory / path for path in paths]
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path) as data:
            if "physical" in data:
                images.append((views(data["physical"]), views(data["case_reference"])))
    if not images and (
        report is None or report.get("reconstruction") == "direct_magnetization@1"
    ):
        stages = sorted((directory / "evaluation").rglob("replay-*.npz"))
        if stages:
            with np.load(stages[-1]) as data:
                if "mxy" in data:
                    actual = np.zeros((128, 128))
                    target = np.zeros_like(actual)
                    xy = np.floor(
                        (data["positions_m"][:, :2] / 0.2 + 0.5) * 128
                    ).astype(int)
                    actual[tuple(xy.T)] = np.abs(data["mxy"])
                    target[tuple(xy.T)] = data["target"]
                    images.append((actual.T, target.T))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for root in args.inputs:
        for path in root.glob("*/*/receipt.json"):
            if (path.parent / "recovered-receipt.json").exists():
                path = path.parent / "recovered-receipt.json"
            row = json.loads(path.read_text())
            cases[row["task"], row["hardware"]] = (row, path.parent)
    hardware = ["low_field", "standard_1_5t", "high_performance"]
    groups = {
        "basic": ["gre_t1w", "gre_t2star", "spin_echo_t2w", "bssfp"],
        "readouts": [
            "epi_single_shot",
            "epi_multishot",
            "tse_t2w",
            "grase",
            "propeller_tse",
            "radial_readout",
            "spiral_gre",
            "ute_radial",
        ],
        "sampling": [
            "sense_sampling",
            "grappa_sampling",
            "compressed_sensing",
            "variable_density",
            "partial_fourier",
            "reduced_fov",
            "rfov_epi",
        ],
        "preparation": [
            "adiabatic_inversion",
            "t2_prepared",
            "t1_mapping",
            "t2_mapping",
            "rf_excitation_2d",
        ],
        "volumes": ["stack_of_stars", "caipirinha", "elliptical_kspace"],
    }
    summary = []
    for name, tasks in groups.items():
        heights = [
            0.4 if name == "volumes" or task in {"t1_mapping", "t2_mapping"} else 1
            for task in tasks
        ]
        fig, axes = plt.subplots(
            len(tasks),
            3,
            figsize=(10, 2.5 * sum(heights) + 0.5),
            gridspec_kw={"height_ratios": heights},
            squeeze=False,
        )
        for i, task in enumerate(tasks):
            for j, hw in enumerate(hardware):
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                entry = cases.get((task, hw))
                title = ["0.55 T", "1.5 T", "3 T"][j] if i == 0 else ""
                if j == 0:
                    ax.set_ylabel(task.replace("_", " "), fontsize=9)
                if entry:
                    row, directory = entry
                    images = collect(directory)
                    if images:
                        actual = np.concatenate([pair[0] for pair in images], axis=1)
                        vmax = max(np.max(pair[1]) for pair in images)
                        ax.imshow(
                            actual,
                            origin="lower",
                            cmap="gray",
                            vmin=0,
                            vmax=vmax,
                            interpolation="nearest",
                        )
                        summary.append(
                            {
                                "task": task,
                                "hardware": hw,
                                "status": row["status"],
                                "image_count": len(images),
                                "display_max": float(vmax),
                            }
                        )
                    else:
                        note(ax, "No achieved image")
                    if row["status"] != "passed":
                        title += "\n" + row["status"]
                else:
                    note(ax, "Pending")
                ax.set_title(title, fontsize=9)
        shape = (
            "128 × 128 × 32 · axial / coronal / sagittal"
            if name == "volumes"
            else "128 × 128"
        )
        fig.suptitle("Achieved images · " + name + " · native " + shape, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.out / f"{name}.png", dpi=160, facecolor="white")
        plt.close(fig)
    (args.out / "images.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
