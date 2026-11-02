"""Read-only previews of the images saved by any grading submission."""

import hashlib
import io
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

PLOT_LOCK = threading.Lock()


def saved_images(report, allowed, folder):
    groups = [("Image", report.get("artifacts", {}))]
    groups += [
        (f"Frame {i + 1}", frame.get("artifacts", {}))
        for i, frame in enumerate(report.get("frames", []))
    ]
    groups += [
        (f"Frame {i + 1}", frame)
        for i, frame in enumerate(report.get("frame_artifacts", []))
    ]
    paths, labels = [], []
    for label, artifacts in groups:
        if not artifacts.get("images"):
            continue
        path = Path(artifacts["images"]).resolve()
        if (
            path.is_relative_to(allowed)
            and path.suffix == ".npz"
            and path.is_file()
            and path not in paths
        ):
            paths.append(path)
            labels.append(label)
    if (
        not paths
        and report.get("reconstruction") == "direct_magnetization@1"
        and folder
        and folder.is_relative_to(allowed)
    ):
        replays = [
            p.resolve()
            for p in (folder / "physical/rf_excitation").glob("replay-*.npz")
            if p.stem.split("-")[-1].isdigit() and p.resolve().is_relative_to(allowed)
        ]
        if replays:
            paths = [max(replays, key=lambda p: int(p.stem.split("-")[-1]))]
            labels = ["RF excitation"]
    return paths, labels


def simulations(directory):
    """Only immutable simulator snapshots; never borrow another candidate's image."""
    allowed = (directory / "control/simulations").resolve()
    records = []
    submissions = []
    for receipt in (directory / "control/submissions").glob("attempt-*/receipt.json"):
        try:
            value = json.loads(receipt.read_text())
            submissions.append(
                (
                    int(value["attempt"]),
                    value.get("accepted_at_unix_s", receipt.stat().st_mtime),
                )
            )
        except (OSError, ValueError):
            continue
    per_try = {}
    for folder in sorted(allowed.glob("[0-9]*")):
        if not folder.name.isdigit() or not folder.resolve().is_relative_to(allowed):
            continue
        metadata = folder / "output/reconstruction.json"
        try:
            result = json.loads(metadata.read_text()) if metadata.is_file() else {}
            candidate = folder / "input/sequence.seq"
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            started = candidate.stat().st_mtime
            grading_try = 1 + max(
                (attempt for attempt, accepted in submissions if accepted <= started),
                default=0,
            )
            per_try[grading_try] = per_try.get(grading_try, 0) + 1
            local_try = per_try[grading_try]
            images = result.get("images", [])
            has_image = (
                result.get("status") == "completed"
                and bool(images)
                and (folder / "output/reconstruction.h5").is_file()
            )
            records.append(
                {
                    "simulation": int(folder.name),
                    "grading_try": grading_try,
                    "simulation_try": local_try,
                    "simulation_label": f"S{grading_try}.{local_try}",
                    "sha256": digest,
                    "status": result.get("status", "No reconstruction saved"),
                    "has_image": has_image,
                    "image_count": len(images) if has_image else 0,
                    "error": result.get("error"),
                    "started": started,
                    "finished": metadata.stat().st_mtime
                    if metadata.exists()
                    else folder.stat().st_mtime,
                }
            )
        except (OSError, ValueError):
            continue
    return records


def simulation_evidence(root, trial_id, number):
    trial = next(
        row
        for row in json.loads((root / "queue.json").read_text())
        if row["trial_id"] == trial_id
    )
    directory = (
        root / "trials" / trial_id / f"runner-attempt-{trial['runner_attempt']:03d}"
    )
    number = int(number)
    record = next(row for row in simulations(directory) if row["simulation"] == number)
    folder = directory / "control/simulations" / f"{number:04d}"
    path = (folder / "output/reconstruction.h5").resolve()
    if not path.is_relative_to((directory / "control/simulations").resolve()):
        raise ValueError("Invalid simulation image path")
    reconstruction = (
        json.loads((folder / "output/reconstruction.json").read_text())
        if (folder / "output/reconstruction.json").is_file()
        else {}
    )
    entries = reconstruction.get("images", []) if record["has_image"] else []
    checks = {}
    for request in sorted((directory / "control").glob("request-*"), reverse=True):
        candidate = request / "sequence.seq"
        if (
            candidate.is_file()
            and (request / "result.json").is_file()
            and hashlib.sha256(candidate.read_bytes()).hexdigest() == record["sha256"]
        ):
            checks = json.loads((request / "result.json").read_text())
            break
    current = directory / "workspace/sequence.seq"
    current_match = (
        current.is_file()
        and hashlib.sha256(current.read_bytes()).hexdigest() == record["sha256"]
    )
    images = [
        {
            "index": i,
            "group": entry["group"],
            "label": entry["group"]
            + (
                " · "
                + ", ".join(
                    f"{key}={value}" for key, value in entry.get("labels", {}).items()
                )
                if entry.get("labels")
                else ""
            ),
        }
        for i, entry in enumerate(entries)
    ]
    metadata = {
        "trial_id": trial_id,
        "arm": trial["arm"],
        "task": trial.get("task")
        or trial["case_id"].split("--", 1)[0].rsplit("-v", 1)[0],
        "status": record["status"],
        "simulation": number,
        "simulation_label": record["simulation_label"],
        "grading_run": int(trial["repeat"]) + 1,
        "source": "simulation",
        "sha256": record["sha256"],
        "current_candidate": current_match,
        "images": images,
        "reason": record.get("error")
        or (None if images else "No reconstructed image was saved by this simulator call."),
        "criteria": [],
        "failed_checks": [
            check["name"].replace("_", " ")
            for check in checks.get("checks", [])
            if check.get("passed") is False
        ],
        "check_status": checks.get("status"),
        "image_note": f"Simulator diagnostic · SHA {record['sha256'][:12]} · "
        + (
            "current candidate"
            if current_match
            else "earlier candidate; the file was edited afterward"
        )
        + ". This image is ungraded and has no judge target comparison.",
    }
    return metadata, [path] * len(images)


@lru_cache(maxsize=48)
def render_simulation_png(path, stamp, group):
    import h5py
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with PLOT_LOCK, h5py.File(path) as file:
        dataset = file[group]
        entries = json.loads(
            Path(path).with_name("reconstruction.json").read_text()
        ).get("images", [])
        entry = next(item for item in entries if item["group"] == group)
        labels = entry.get("source", {})
        shape = dataset["image_real"].shape
        middle = shape[2] // 2
        image = (
            dataset["image_real"][:, :, middle, :, :, :]
            + 1j * dataset["image_imag"][:, :, middle, :, :, :]
        )
        magnitude = np.sqrt(np.sum(np.abs(image) ** 2, axis=3))
        frames = magnitude.shape[2] * magnitude.shape[3]
        columns = min(frames, 4)
        rows = (frames + columns - 1) // columns
        maximum = max(float(magnitude.max()), 1e-15)
        figure = Figure(figsize=(3 * columns, 3 * rows), dpi=110, facecolor="#111827")
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, columns, squeeze=False)
        for axis in axes.flat:
            axis.set_axis_off()
        for echo in range(magnitude.shape[2]):
            for repetition in range(magnitude.shape[3]):
                axis = axes.flat[echo * magnitude.shape[3] + repetition]
                axis.imshow(
                    magnitude[:, :, echo, repetition].T,
                    origin="lower",
                    cmap="gray",
                    vmin=0,
                    vmax=maximum,
                )
                axis.set_title(
                    f"ECO={labels.get('ECO', list(range(magnitude.shape[2])))[echo]} · REP={labels.get('REP', list(range(magnitude.shape[3])))[repetition]}",
                    color="#eef2ff",
                    fontsize=10,
                )
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", facecolor=figure.get_facecolor())
        return output.getvalue()


def evidence(root, trial_id, submission=None):
    trial = next(
        row
        for row in json.loads((root / "queue.json").read_text())
        if row["trial_id"] == trial_id
    )
    directory = (
        root / "trials" / trial_id / f"runner-attempt-{trial['runner_attempt']:03d}"
    )
    allowed = (directory / "control/submissions").resolve()
    if submission is None:
        result = json.loads((directory / "result.json").read_text())
        report = result.get("final_verdict") or result
        submissions = result.get("submissions", [])
        last = submissions[-1] if submissions else {}
        submission = last.get("attempt")
        folder = Path(last["directory"]).resolve() if last.get("directory") else None
    else:
        submission = int(submission)
        if submission < 1:
            raise ValueError("Invalid submission number")
        folder = (allowed / f"attempt-{submission:03d}").resolve()
        if not folder.is_relative_to(allowed):
            raise ValueError("Invalid submission directory")
        report = (
            json.loads((folder / "result.json").read_text())
            if (folder / "result.json").is_file()
            else {"status": "grading"}
        )
        if (
            not (folder / "result.json").is_file()
            and not (folder / "receipt.json").is_file()
        ):
            raise ValueError("No such grading submission")
    paths, labels = saved_images(report, allowed, folder)
    target_only = False
    if not paths and trial["case_id"].startswith("rf_excitation_2d-v"):
        target = root / "runtime/repo/benchmark/references/rf_excitation_2d/target.npz"
        if target.is_file():
            paths, labels, target_only = [target], ["RF target only"], True
    failures = list(
        dict.fromkeys(
            c["name"].replace("_", " ")
            for group in ("checks", "application_checks", "image_checks")
            for c in report.get(group, [])
            if c.get("passed") is False
        )
    )
    criteria = {}
    for group in ("checks", "image_checks", "application_checks"):
        for check in report.get(group, []):
            detail = check.get("detail")
            if (
                isinstance(detail, dict)
                and "value" in detail
                and ("min" in detail or "max" in detail)
            ):
                criteria[check["name"]] = {
                    "name": check["name"].replace("_", " "),
                    "passed": check.get("passed"),
                    "value": detail["value"],
                    "min": detail.get("min"),
                    "max": detail.get("max"),
                }
    reason = None
    if target_only:
        reason = "Canonical target magnitude only. No simulated response was saved; the submission failed before RF replay."
    elif not paths:
        reason = (
            "Judge replay is still running; no reconstruction has been saved yet."
            if report.get("status") == "grading"
            else "Image reconstruction was skipped after sequence or mechanism checks failed."
            if report.get("sequence_valid") is False
            else "No judge image was saved for this submission."
        )
    metadata = {
        "trial_id": trial_id,
        "arm": trial["arm"],
        "task": trial["case_id"].split("-v")[0],
        "status": report.get("status"),
        "submission": submission,
        "grading_run": int(trial["repeat"]) + 1,
        "images": [{"label": label, "index": i} for i, label in enumerate(labels)],
        "reason": reason,
        "criteria": list(criteria.values()),
        "failed_checks": failures[:4],
        "image_note": reason
        if target_only
        else (
            "Target and simulated transverse magnetization magnitudes on the sampled water disc."
            if report.get("reconstruction") == "direct_magnetization@1"
            else "Magnitude of the saved mean reconstruction. Ordinary image NRMSE includes its sampling noise; the judge requires <=2% estimated numerical image error before a verdict. Timing, hardware and task-specific checks must also pass."
        ),
    }
    if (not paths or target_only) and folder and (folder / "sequence.seq").is_file():
        digest = hashlib.sha256((folder / "sequence.seq").read_bytes()).hexdigest()
        records = [
            record
            for record in simulations(directory)
            if record["has_image"] and record["sha256"] == digest
        ]
        if records:
            references = paths if target_only else []
            simulated, paths = simulation_evidence(
                root, trial_id, records[-1]["simulation"]
            )
            images = simulated["images"] + [
                {"index": len(paths) + i, "label": "Canonical RF target"}
                for i in range(len(references))
            ]
            paths += references
            metadata.update(
                images=images,
                source="simulation",
                sha256=digest,
                reason=None,
                check_status=simulated.get("check_status"),
                image_note=f"Simulator diagnostic for this exact submission · SHA {digest[:12]}. "
                + (
                    "Judge replay is still running. "
                    if report.get("status") == "grading"
                    else "Judge image replay was skipped. "
                )
                + "This is an ungraded reconstruction.",
            )
    return metadata, paths


@lru_cache(maxsize=48)
def render_png(path, stamp):
    import numpy as np

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl-cache"))
    with PLOT_LOCK:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        with np.load(path, allow_pickle=False) as arrays:
            target_only = "target_mxy_magnitude" in arrays
            excitation = "mxy" in arrays
            if target_only:
                reference = arrays["target_mxy_magnitude"]
                physical = None
            elif excitation:
                points = arrays["positions_m"]
                x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
                reference = np.ma.masked_all((len(x), len(y)))
                physical = np.ma.masked_all(reference.shape)
                ix, iy = (
                    np.searchsorted(x, points[:, 0]),
                    np.searchsorted(y, points[:, 1]),
                )
                reference[ix, iy] = np.abs(arrays["target"])
                physical[ix, iy] = np.abs(arrays["mxy"])
            else:
                reference = np.abs(
                    arrays["case_reference"]
                    if "case_reference" in arrays
                    else arrays["reference"]
                ).squeeze()
                physical = np.abs(arrays["physical"]).squeeze()
        note = (
            "Canonical RF target · simulated response unavailable"
            if target_only
            else "Saved judge magnitudes"
        )
        if np.ma.isMaskedArray(reference):
            note += " · unsampled area masked"
        if reference.ndim == 3:
            index = reference.shape[2] // 2
            note += f" · central slice {index + 1}/{reference.shape[2]}"
            reference, physical = reference[:, :, index], physical[:, :, index]
        if (
            reference.ndim != 2
            or not np.isfinite(reference).all()
            or (
                physical is not None
                and (
                    reference.shape != physical.shape or not np.isfinite(physical).all()
                )
            )
        ):
            raise ValueError("Saved images cannot be displayed as a finite 2D slice")
        maximum = max(
            float(reference.max()),
            float(physical.max()) if physical is not None else 0,
            1e-15,
        )
        figure = Figure(figsize=(8.4, 3.2), dpi=110, facecolor="#111827")
        FigureCanvasAgg(figure)
        panels = [("Target magnitude", reference)]
        if physical is not None:
            panels += [
                (
                    "Simulated |Mxy|" if excitation else "Reconstruction magnitude",
                    physical,
                ),
                ("Magnitude error", np.abs(physical - reference)),
            ]
        axes = figure.subplots(1, len(panels), squeeze=False).ravel()
        for axis, (title, values) in zip(axes, panels):
            axis.set_facecolor("#1f2937")
            axis.imshow(
                values.T,
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=maximum,
                interpolation="nearest",
            )
            axis.set_title(title, color="#eef2ff", fontsize=11)
            axis.set_axis_off()
        figure.subplots_adjust(
            left=0.02, right=0.98, top=0.84, bottom=0.17, wspace=0.12
        )
        figure.text(
            0.5,
            0.06,
            f"{note} · shared scale 0–{maximum:.3g}",
            ha="center",
            color="#b8c5d8",
            fontsize=9,
        )
        output = io.BytesIO()
        figure.savefig(output, format="png", facecolor=figure.get_facecolor())
        return output.getvalue()
