"""Prepare the public workspace without copying judge code or host credentials."""

import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from ..io import write_json


class BriefDumper(yaml.SafeDumper):
    pass


def sequence_list(dumper, values):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        values,
        flow_style=all(not isinstance(value, (dict, list)) for value in values),
    )


BriefDumper.add_representer(list, sequence_list)


def compact(value):
    """Remove absent optional fields without losing zeroes or false requirements."""
    if isinstance(value, dict):
        return {
            k: compact(v)
            for k, v in value.items()
            if v is not None and (v != [] or k == "effects")
        }
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def task_brief(case):
    """Agent requirements, without catalog metadata or expanded registry records."""
    sequence = {k: v for k, v in case["task"]["sequence"].items() if k != "level"}
    return compact(
        {
            "objective": case["task"]["objective"],
            "sequence": sequence,
            "phantom": case["object"]["phantom"],
            "physics": {"effects": case["physics"]["effects"]},
            "hardware": {
                k: v
                for k, v in case["hardware"].items()
                if k not in ("id", "version", "schema_version", "provenance")
            },
            "seed": case["seed"],
        }
    )


def brief_yaml(case, include_objective=True):
    brief = task_brief(case)
    if not include_objective:
        brief.pop("objective")
    return yaml.dump(brief, Dumper=BriefDumper, sort_keys=False, width=88)


def evaluation_summary(contract):
    if contract["scope"] == "smoke":
        return "Evaluation: hardware/timing preflight."
    if not contract["calibrated"]:
        return "Evaluation: hardware/timing preflight; physical scoring is pending."
    metrics = []
    for metric in contract["metrics"]:
        bounds = " and ".join(
            f"{op} {metric[key]}"
            for key, op in (("min", ">="), ("max", "<="))
            if metric[key] is not None
        )
        metrics.append(f"{metric['name']} {bounds}")
    return f"Evaluation: {contract['reconstruction']}; {', '.join(metrics)}."


def prompt(case):
    task = case["task"]
    # The same resolved contract powers instructions and evaluation.
    return (
        f"# {case['case_id']}\n\n{task['objective'].strip()}\n\n"
        f"Submit `{task['submission']['file']}` in Pulseq format in the working directory.\n"
        "Meet the requirements below; field names include units. Duration is an upper limit.\n"
        "Timing targets use the stated tolerance; max values are upper limits.\n\n"
        "```yaml\n"
        + brief_yaml(case, include_objective=False)
        + "```\n\n"
        + evaluation_summary(case["evaluation"])
        + "\n"
        + "These requirements are also in `task.yaml`; full metadata is in `case.json`.\n"
    )


def prepare_inputs(workspace, case, root, phantom_dir=None):
    assets = {}

    def visit(value):
        if isinstance(value, dict):
            if {"path", "sha256", "format", "shape"} <= value.keys():
                source = (Path(root) / value["path"]).resolve()
                if not source.is_relative_to(Path(root).resolve()):
                    raise ValueError("input asset escapes catalog root")
                target = (
                    Path(workspace)
                    / "inputs"
                    / (value["sha256"] + "".join(source.suffixes))
                )
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != value["sha256"]:
                    raise ValueError(f"input asset changed: {value['path']}")
                assets[value["path"]] = str(target.relative_to(workspace))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(case)
    if phantom_dir is not None:
        metadata = json.loads((Path(phantom_dir) / "artifacts.json").read_text())
        target_dir = Path(workspace) / "inputs"
        target_dir.mkdir(exist_ok=True)
        for filename, expected_hash in metadata["files"].items():
            if Path(filename).name != filename:
                raise ValueError("invalid phantom artifact filename")
            source = Path(phantom_dir) / filename
            target = target_dir / filename
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_hash:
                raise ValueError(f"phantom artifact changed: {filename}")
            assets[filename] = str(target.relative_to(workspace))
    write_json(Path(workspace) / "INPUTS.json", assets)
    return assets


def prepare(workspace, case, experiment, server, phantom_dir=None):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    public_case = {
        **case,
        "hardware": {k: v for k, v in case["hardware"].items() if k != "provenance"},
    }
    write_json(workspace / "case.json", public_case)
    (workspace / "task.yaml").write_text(brief_yaml(case))
    prepare_inputs(workspace, case, server.root, phantom_dir)
    instructions = prompt(case) + (
        f"\n## Execution\n\nTime limit: {experiment.agent.timeout_s:g} seconds.\n"
        f"Feedback mode: {experiment.feedback.mode}; at most {experiment.feedback.max_checks} charged checks.\n"
        "`./bin/lint` runs hardware/timing preflight only when feedback is enabled.\n"
        "`./bin/check` uses the configured feedback mode and check budget.\n"
        "`./bin/submit` records exactly one final sequence and ends the run without showing its verdict.\n"
        "Work and write files only inside this working directory.\n"
        f"Python: `{sys.executable}`; NumPy, h5py and PyPulseq are installed.\n"
        "Input files are listed in `INPUTS.json`; HDF5 files can be read with h5py.\n"
        "For phantom inputs, read `inputs/manifest.json` for companion-field and motion handling.\n"
    )
    if experiment.submission_mode == "submit":
        instructions += "You must run `./bin/submit`; leaving a file without submitting does not count.\n"
    else:
        instructions += (
            "You may submit explicitly or leave `sequence.seq` and exit successfully.\n"
        )
    (workspace / "PROMPT.md").write_text(instructions)
    write_json(
        workspace / ".harness.json",
        {
            "port": server.port,
            "token": server.token,
            "max_submission_bytes": experiment.feedback.max_submission_bytes,
            "timeout_s": experiment.feedback.timeout_s,
        },
    )
    client = workspace / ".harness_client.py"
    client.write_text(Path(__file__).with_name("client.py").read_text())
    directory = workspace / "bin"
    directory.mkdir(exist_ok=True)
    for name in ("lint", "check", "submit"):
        path = directory / name
        path.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(client))} {name}\n"
        )
        path.chmod(0o755)
    return (
        workspace / "PROMPT.md",
        workspace / "case.json",
        workspace / case["task"]["submission"]["file"],
    )


def prepare_pi(workspace, base_url):
    directory = Path(workspace) / ".pi-agent"
    directory.mkdir()
    write_json(
        directory / "models.json", {"providers": {"openrouter": {"baseUrl": base_url}}}
    )
    write_json(directory / "settings.json", {"enableInstallTelemetry": False})
    return {"PI_CODING_AGENT_DIR": str(directory), "PI_OFFLINE": "1"}
