"""Read-only localhost view of the existing Luna campaign."""

import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from judge_images import (
    evidence,
    render_png,
    render_simulation_png,
    saved_images,
    simulation_evidence,
    simulations,
)

ROOT = None  # Selected explicitly by --campaign in main.
UI = Path(__file__).resolve().parent
LABELS = {
    "j0_koma_base": "PulseqBase.jl",
    "j1_pulseq_julia": "Pulseq.jl",
    "j2_pulseq_julia_unitful": "Pulseq.jl + Unitful",
    "j5_pulseq_julia_unitful_koma": "Pulseq.jl + Unitful + Koma",
    "p0_pypulseq": "PyPulseq",
    "p1_pypulseq_mrzero": "PyPulseq + MRzero",
    "j3_koma_base_koma": "PulseqBase.jl + Koma",
    "j4_pulseq_julia_koma": "Pulseq.jl + Koma",
    "p2_pypulseq_koma": "PyPulseq + komamripy",
}
FILES = {
    "before-after-language.csv": "Julia/Python gap before/after",
    "before-after-trials.csv": "Before/after trials",
    "before-after-arms.csv": "Before/after by arm",
    "failure-mode-comparison.csv": "Failure-mode comparison",
    "classified-problems.csv": "Classified problems",
    "action-comparison.csv": "Response action comparison",
    "trial-results.csv": "Trial results",
    "matched-comparisons.csv": "Matched comparisons",
    "response-actions.csv": "Response actions",
    "response-usage.csv": "Token usage",
    "encountered-problems.csv": "Encountered problems",
    "gre-trace-review.json": "GRE trace review",
    "runtime/Metal-grading-qualification/comparison.json": "Metal / Float64 comparison",
    "runtime/three-graders-qualification/summary.json": "Three-grader throughput qualification",
    "runtime/ten-graders-qualification/summary.json": "Ten-grader throughput qualification",
    "runtime/mcp-latency-qualification/dispatch-breakdown.json": "Grading MCP latency comparison",
    "runtime/author-sysimages/qualification.json": "Julia author sysimage qualification",
    "migrations/metal-grading-20261030T055455Z/CPU-reference-reports/trial-results.csv": "CPU reference results (7 completed)",
}
CACHE = {}


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def read_csv(name):
    path = ROOT / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def number(value):
    return int(value) if value not in ("", None) else None


def task_labels(case):
    parts = case.split("--")
    return parts[0].rsplit("-v", 1)[0], parts[-1].removeprefix("hardware-").rsplit(
        "-v", 1
    )[0]


def trace(path):
    if not path.exists():
        return [], None
    signature = (path.stat().st_mtime_ns, path.stat().st_size)
    if path not in CACHE or CACHE[path][0] != signature:
        commands, message = [], None
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item", {})
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "agent_message"
            ):
                message = item.get("text")
            if (
                event.get("type") == "item.started"
                and item.get("type") == "command_execution"
            ):
                commands.append(item.get("command", ""))
        CACHE[path] = signature, commands, message
    return CACHE[path][1:]


def process_table():
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,%cpu=,rss=,etime=,command="],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    processes = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 5)
        if len(fields) != 6:
            continue
        pid, parent, cpu, rss, elapsed, command = fields
        processes.append(
            {
                "pid": int(pid),
                "parent": int(parent),
                "cpu": float(cpu),
                "rss": int(rss),
                "elapsed": elapsed,
                "command": command,
            }
        )
    return processes


def elapsed_seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    seconds = 0
    for component in clock.split(":"):
        seconds = seconds * 60 + int(component)
    return int(days) * 86400 + seconds


def snapshot():
    now = time.time()
    queue = read_json(ROOT / "queue.json", [])
    status = read_json(ROOT / "status.json", {})
    campaign = read_json(ROOT / "campaign.json", {})
    stats = {row["trial_id"]: row for row in read_csv("trial-results.csv")}
    processes = process_table()
    pool = read_json(ROOT / "runtime/metal-grader-service/ENABLED.json", {})
    members = pool.get("workers", [pool]) if pool else []
    shared = [
        read_json(Path(member["socket"]).with_name("status.json"), member)
        for member in members
    ]
    shared_pids = {member["worker_pid"] for member in shared if "worker_pid" in member}
    graders = [
        p
        for p in processes
        if str(ROOT) in p["command"]
        and "mriseqbench.evaluation.worker --mode evaluation " in p["command"]
    ]
    grading_ids = {p["pid"] for p in graders}
    children = [
        p for p in processes if p["parent"] in grading_ids or p["pid"] in shared_pids
    ]
    active, recent, submissions, simulator_records = [], [], [], []
    arms = {
        key: {
            "id": key,
            "label": label,
            "simulation": "mrzero"
            if key == "p1_pypulseq_mrzero"
            else "komamripy"
            if key == "p2_pypulseq_koma"
            else "koma"
            if key
            in (
                "j3_koma_base_koma",
                "j4_pulseq_julia_koma",
                "j5_pulseq_julia_unitful_koma",
            )
            else "none",
            "total": 0,
            "completed": 0,
            "passed": 0,
            "failed": 0,
            "unresolved": 0,
            "uncalibrated": 0,
            "mean_tokens": None,
            "mean_requests": None,
            "_tokens": [],
            "_requests": [],
        }
        for key, label in LABELS.items()
        if any(trial["arm"] == key for trial in queue)
    }
    for trial in queue:
        arm = arms[trial["arm"]]
        arm["total"] += 1
        state = trial["state"]
        data = stats.get(trial["trial_id"], {})
        directory = (
            ROOT
            / "trials"
            / trial["trial_id"]
            / f"runner-attempt-{trial.get('runner_attempt', 1):03d}"
        )
        task, scanner = task_labels(trial["case_id"])
        tokens, requests = (
            number(data.get("total_tokens")),
            number(data.get("model_responses")),
        )
        simulated = simulations(directory)
        simulator_records.extend(
            {
                **record,
                "trial_id": trial["trial_id"],
                "label": arm["label"],
                "task": task,
                "scanner": scanner,
                "repeat": trial["repeat"],
            }
            for record in simulated
        )
        for receipt in (directory / "control/submissions").glob(
            "attempt-*/receipt.json"
        ):
            report_path = receipt.parent / "result.json"
            report = read_json(report_path, {})
            info = read_json(receipt, {})
            judge_image = bool(
                saved_images(
                    report,
                    (directory / "control/submissions").resolve(),
                    receipt.parent.resolve(),
                )[0]
            )
            simulator_image = any(
                record["has_image"] and record["sha256"] == info.get("sha256")
                for record in simulated
            )
            submissions.append(
                {
                    "trial_id": trial["trial_id"],
                    "label": arm["label"],
                    "task": task,
                    "scanner": scanner,
                    "repeat": trial["repeat"],
                    "attempt": info.get(
                        "attempt", int(receipt.parent.name.split("-")[-1])
                    ),
                    "status": report.get("status", "grading"),
                    "has_image": judge_image or simulator_image,
                    "image_source": "judge"
                    if judge_image
                    else "simulation"
                    if simulator_image
                    else None,
                    "passed": (
                        "unresolved"
                        if report.get("numerically_converged") is False
                        else "uncalibrated"
                        if report.get("status") == "uncalibrated"
                        else report.get("status") == "passed"
                    )
                    if report
                    else None,
                    "started": info.get("accepted_at_unix_s", receipt.stat().st_mtime),
                    "finished": report_path.stat().st_mtime
                    if report_path.exists()
                    else receipt.stat().st_mtime,
                }
            )

        if state == "completed":
            result = read_json(directory / "result.json", {})
            passed = (result.get("final_verdict") or {}).get("status") == "passed"
            arm["completed"] += 1
            unresolved = (result.get("final_verdict") or result).get(
                "numerically_converged"
            ) is False
            uncalibrated = (result.get("final_verdict") or result).get(
                "status"
            ) == "uncalibrated"
            arm[
                "unresolved"
                if unresolved
                else "uncalibrated"
                if uncalibrated
                else "passed"
                if passed
                else "failed"
            ] += 1
            if tokens is not None:
                arm["_tokens"].append(tokens)
            if requests is not None:
                arm["_requests"].append(requests)
            recent.append(
                {
                    "trial_id": trial["trial_id"],
                    "label": arm["label"],
                    "task": task,
                    "scanner": scanner,
                    "repeat": trial["repeat"],
                    "passed": "unresolved"
                    if unresolved
                    else "uncalibrated"
                    if uncalibrated
                    else passed,
                    "tokens": tokens,
                    "requests": requests,
                    "finished": result.get(
                        "completed_at_unix_s", trial.get("finished_at_unix_s")
                    ),
                }
            )
        if state != "running":
            continue
        cli = next(
            (
                p
                for p in processes
                if trial["trial_id"] in p["command"]
                and " exec " in p["command"]
                and "-m gpt-6-luna" in p["command"]
            ),
            None,
        )
        grading = next((p for p in graders if trial["trial_id"] in p["command"]), None)
        receipts = sorted(
            (directory / "control/submissions").glob("attempt-*/receipt.json")
        )
        pending = [
            p.parent for p in receipts if not (p.parent / "result.json").exists()
        ]
        phase = (
            "grading"
            if grading
            else "queued_for_grading"
            if pending
            else "authoring"
            if cli
            else "preparing"
        )
        grade = None
        if pending:
            attempt = pending[-1]
            progress_paths = list(attempt.glob("physical/**/progress.json"))
            progress_path = (
                max(progress_paths, key=lambda p: p.stat().st_mtime)
                if progress_paths
                else None
            )
            progress = read_json(progress_path, {}) if progress_path else {}
            replay_files = list(attempt.glob("physical/**/replays/*.npy"))
            latest_activity = max(
                [p.stat().st_mtime for p in replay_files]
                + (
                    [progress_path.stat().st_mtime]
                    if progress_path
                    else [attempt.stat().st_mtime]
                )
            )
            workers = [p for p in children if grading and p["parent"] == grading["pid"]]
            grade = {
                "attempt": int(attempt.name.split("-")[-1]),
                "elapsed_s": elapsed_seconds(grading["elapsed"])
                if grading
                else max(0, now - (attempt / "receipt.json").stat().st_mtime),
                "cpu_percent": sum(p["cpu"] for p in workers) if grading else 0,
                "pid": grading["pid"] if grading else None,
                "samples": progress.get("current_samples"),
                "replay_files": len(replay_files),
                "last_activity_s": max(0, now - latest_activity),
            }
        commands, message = trace(directory / "codex.jsonl")
        active.append(
            {
                "id": trial["trial_id"],
                "label": arm["label"],
                "task": task,
                "scanner": scanner,
                "repeat": trial["repeat"],
                "phase": phase,
                "elapsed_s": max(0, now - trial.get("started_at_unix_s", now)),
                "requests": requests,
                "tokens": tokens,
                "last_message": message,
                "last_action": commands[-1] if commands else None,
                "grade": grade,
                "cli_pid": cli["pid"] if cli else None,
            }
        )
    for arm in arms.values():
        for key in ("tokens", "requests"):
            values = arm.pop("_" + key)
            arm["mean_" + key] = sum(values) / len(values) if values else None
    comparisons = []
    trial_ids = {
        (row["case_id"], str(row["repeat"]), LABELS[row["arm"]]): row["trial_id"]
        for row in queue
        if row["state"] == "completed"
    }
    for row in read_csv("matched-comparisons.csv"):
        comparisons.append(
            {
                **row,
                "task": task_labels(row["case_id"])[0],
                "julia_trial_id": trial_ids.get(
                    (row["case_id"], row["repeat"], row["julia_arm"])
                ),
                "python_trial_id": trial_ids.get(
                    (row["case_id"], row["repeat"], row["python_arm"])
                ),
                **{
                    key: number(row.get(key))
                    for key in (
                        "julia_tokens",
                        "python_tokens",
                        "julia_responses",
                        "python_responses",
                    )
                },
                **{
                    key: row.get(key) == "True"
                    for key in ("julia_passed", "python_passed")
                },
            }
        )
    counts = dict(Counter(row["state"] for row in queue))
    pool = read_json(ROOT / "grading-pool-status.json", {})
    backend = (
        str(campaign.get("grading_device", "cpu")).title()
        + " "
        + ("Float32" if campaign.get("grading_precision") == "f32" else "Float64")
    )
    if shared:
        backend += f" · {len(shared)} persistent GPU workers · PIDs {', '.join(str(p) for p in sorted(shared_pids))} · {sum(member.get('calls', 0) for member in shared)} engine calls"
    return {
        "relative_comparison": read_csv("before-after-language.csv"),
        "before_after": read_csv("before-after-arms.csv"),
        "failure_modes": read_csv("failure-mode-comparison.csv"),
        "baseline_campaign": campaign.get("baseline_campaign"),
        "now": now,
        "updated": status.get("updated_unix_s"),
        "campaign": {
            "state": status.get("state"),
            "stage": status.get("stage"),
            "total": len(queue),
            "model": campaign.get("model"),
            "grading_backend": backend,
            "simulation_backend": campaign.get("simulation_backend", backend),
            "grading_concurrency": pool.get(
                "configured_slots", campaign.get("grading_concurrency", 1)
            ),
            "epoch": campaign.get("campaign_epoch"),
            "CPU_reference_completed": campaign.get("CPU_reference_completed", 0),
        },
        "counts": counts,
        "passed": sum(a["passed"] for a in arms.values()),
        "failed": sum(a["failed"] for a in arms.values()),
        "unresolved": sum(a["unresolved"] for a in arms.values()),
        "uncalibrated": sum(a["uncalibrated"] for a in arms.values()),
        "active": active,
        "arms": list(arms.values()),
        "simulations": sorted(
            simulator_records, key=lambda row: row["finished"], reverse=True
        ),
        "submissions": sorted(submissions, key=lambda r: r["finished"], reverse=True),
        "recent": sorted(recent, key=lambda r: r["finished"] or 0, reverse=True)[:20],
        "comparisons": comparisons,
        "resources": {
            "grading_jobs": len(graders),
            "grading_waiters": sum(a["phase"] == "queued_for_grading" for a in active),
            "cpu_percent": sum(p["cpu"] for p in children),
            "rss_gib": sum(p["rss"] for p in graders + children) / 1024**2,
        },
        "files": [
            {"label": label, "url": "/reports/" + name}
            for name, label in FILES.items()
            if (ROOT / name).is_file()
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path)
        if route.path in (
            "/api/judge-preview",
            "/judge-image",
            "/api/simulator-preview",
            "/simulator-image",
        ):
            try:
                query = parse_qs(route.query)
                metadata, paths = (
                    simulation_evidence(ROOT, query["trial"][0], query["simulation"][0])
                    if route.path in ("/api/simulator-preview", "/simulator-image")
                    else evidence(
                        ROOT, query["trial"][0], query.get("attempt", [None])[0]
                    )
                )
                if route.path in ("/judge-image", "/simulator-image"):
                    index = int(query.get("image", ["0"])[0])
                    if not 0 <= index < len(paths):
                        raise ValueError("No saved image for this submission")
                    path = paths[index]
                    content, kind = (
                        (
                            render_simulation_png(
                                str(path),
                                path.stat().st_mtime_ns,
                                metadata["images"][index]["group"],
                            )
                            if path.suffix == ".h5"
                            else render_png(str(path), path.stat().st_mtime_ns)
                        ),
                        "image/png",
                    )
                else:
                    metadata["label"] = LABELS[metadata["arm"]]
                    content, kind = json.dumps(metadata).encode(), "application/json"
            except (KeyError, StopIteration, ValueError, OSError) as error:
                self.send_error(404, str(error) or "Trial images unavailable")
                return
        elif self.path == "/api/progress":
            try:
                content, kind = json.dumps(snapshot()).encode(), "application/json"
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                self.send_error(503, str(error))
                return
        elif self.path in ("/", "/index.html"):
            content, kind = (UI / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif (
            self.path.startswith("/reports/")
            and self.path.removeprefix("/reports/") in FILES
        ):
            path = ROOT / self.path.removeprefix("/reports/")
            if not path.exists():
                self.send_error(404)
                return
            content, kind = (
                path.read_bytes(),
                "application/json" if path.suffix == ".json" else "text/csv",
            )
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    ROOT = args.campaign.resolve()
    if not (ROOT / "queue.json").is_file():
        parser.error("--campaign must contain queue.json")
    state = ROOT / "dashboard"
    state.mkdir(exist_ok=True)
    port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    info = {"pid": os.getpid(), "url": f"http://127.0.0.1:{server.server_port}"}
    (state / "server.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    server.serve_forever()
