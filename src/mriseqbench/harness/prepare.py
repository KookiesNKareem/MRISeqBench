"""Prepare the public workspace without copying judge code or host credentials."""

import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from ..io import write_json
from ..models import ImagePrecision


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
    research = case["task"].get("prompt_mode") == "research"
    fields = (
        {"matrix", "fov_mm", "duration_s"}
        if research
        else set(case["task"]["sequence"]) - {"level"}
    )
    sequence = {k: v for k, v in case["task"]["sequence"].items() if k in fields}
    return compact(
        {
            "objective": case["task"]["objective"],
            "sequence": sequence,
            "phantom": case["object"]["phantom"],
            "tissue_properties": case["object"].get("tissue_properties"),
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


def evaluation_summary(contract, image_policy=None):
    if contract["scope"] == "smoke":
        return "Evaluation: hardware/timing preflight and final-submission SAR."
    if contract["id"] == "s4":
        criteria = contract["case_image_criteria"]["static"]
        return (
            f"Evaluation (s4@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): "
            "waveform-derived ECG windows, inversion/T2 preparation and balanced bSSFP; "
            "native finite-RF Bloch replay with scanner-specific cardiac tissues, fat and static B0. "
            "Every cardiac image uses BART Cartesian FFT and an independent central-echo target. "
            "Mapping tasks also test myocardial relaxation estimates; no gain, phase or B0 correction is fitted. "
            f"Per-image limits: complex NRMSE <= {criteria['complex_nrmse_max']:g}; "
            f"magnitude NRMSE <= {criteria['magnitude_nrmse_max']:g}; phase RMS <= {criteria['phase_rms_rad_max']:g} rad; "
            f"tissue signal >= {criteria['tissue_signal_fraction_min']:g}; contrast retention >= {criteria['contrast_retention_min']:g}; "
            f"outside energy <= {criteria['outside_energy_fraction_max']:g}. "
            f"Preparation effect >= {contract['preparation_effect_fraction_min']:g} of equilibrium PD; "
            "myocardial echo signal >= 0.01 PD; each inversion/refocusing pulse must invert >=0.90 "
            "of longitudinal magnetization across [-520,70] Hz scaled by scanner field/3 T. "
            f"Myocardial mapping relative error <= {contract['mapping_relative_error_max']:g}; "
            f"numerical relative-error precision {contract['mapping_relative_error_precision']:g}, refined independently as needed. "
            "the Bloch dictionary varies one relaxation parameter with known PD and the other relaxation time. "
            "Final submissions include hardware, timing and SAR checks. "
            + ("Qualification is pending. " if not contract["calibrated"] else "")
        )
    if contract["id"] == "s2":
        qualification = "" if contract["calibrated"] else ", uncalibrated"
        criteria = (contract.get("case_image_criteria") or {}).get(image_policy)
        limits = "; ".join(
            f"{m['name']} "
            + " and ".join(
                f"{op}{m[k]:g}"
                for k, op in (("min", ">="), ("max", "<="))
                if m[k] is not None
            )
            for m in contract["metrics"]
        )
        case_limits = (
            f"Case limits for {image_policy}: complex NRMSE <={criteria['complex_nrmse_max']:g}; "
            f"magnitude NRMSE <={criteria['magnitude_nrmse_max']:g}; phase RMS <={criteria['phase_rms_rad_max']:g} rad; "
            f"minimum tissue signal >={criteria['tissue_signal_fraction_min']:g}; contrast retention >={criteria['contrast_retention_min']:g}; "
            f"outside-object energy <={criteria['outside_energy_fraction_max']:g}. "
            if criteria
            else ""
        )
        return (
            f"Evaluation (s2@{contract['version']}{qualification}): waveform-derived readout identity, physical matrix/FOV coverage, "
            "shot/echo counts and detailed timing targets; final-submission SAR; independent KomaMRI replay "
            "on the fixed cardiac phantom under ideal and declared physics. BART uses Cartesian FFT "
            "or inverse NUFFT with fixed L2 regularization 0.001, at most 50 iterations and direct normal operator. "
            "Declared-physics images use the independent RF phase target; ideal-control imaging is diagnostic only. "
            "Spatial quadrature and time steps adapt until the numerical verdict resolves; trusted grading is checkpointed and excluded from the participant deadline. "
            f"Ideal-image limits: {limits}. "
            + case_limits
            + (
                "Image thresholds are provisional; converged images return uncalibrated. "
                if not contract["calibrated"]
                else ""
            )
            + "See docs/S2_EVALUATION.md for the waveform and image contract."
        )
    if contract["id"] == "s1":
        precision = ImagePrecision.model_validate(contract.get("image_precision") or {})
        case_criteria = (contract.get("case_image_criteria") or {}).get(image_policy)
        case_text = (
            "Overall acceptance requires sequence validity and case-physics image success; ideal-control imaging is diagnostic only. "
            "The case-aware phase target is the independent KomaMRI RF-only response at voxel centres, with fixed material fractions, under declared fields at the echo nearest k-space centre: "
            "Complete RF history and intrinsic fat are retained; spoiled families destroy old transverse coherence before excitation while retaining longitudinal recovery. Receiver phase follows RF coherence, never submitted ADC phase. "
            "No fitted gain, phase, field-map correction or registration is allowed. "
            f"Case limits for {image_policy}: complex NRMSE <={case_criteria['complex_nrmse_max']:g}; "
            f"magnitude NRMSE <={case_criteria['magnitude_nrmse_max']:g}; reference-energy-weighted wrapped phase RMS <={case_criteria['phase_rms_rad_max']:g} rad; "
            f"minimum tissue signal fraction >={case_criteria['tissue_signal_fraction_min']:g}; "
            f"contrast retention >={case_criteria['contrast_retention_min']:g} for tissue pairs whose reference normalized contrast is at least 0.1; "
            f"outside-object energy fraction <={case_criteria['outside_energy_fraction_max']:g}. "
            "Phase is scored only where the independent RF response is defined; signal dropout is checked separately. "
            "Nonlinear magnitude/phase/ROI scores use leading-bias-corrected jackknife estimates and approximate numerical intervals. "
            if case_criteria
            else "Overall acceptance requires sequence validity and ideal-control image success. B0/motion robustness is reported separately. "
        )
        return (
            "Evaluation: waveform-derived Cartesian coverage and sequence identity; KomaMRI finite-RF Bloch replay "
            "on the fixed cardiac phantom with ideal and specified physics; BART Cartesian FFT reconstruction. "
            "Common diagnostics use PD=1, T1=800 ms, T2=80 ms, a 1% signal floor, and 10% T1/T2 "
            "perturbations. Spoiled GRE is compared with perfect spoiling using 1021/2053 "
            "isochromats across one voxel (maximum relative echo error 0.2). Refocusing/T2-star "
            "diagnostics use 65 offsets from -50 to +50 Hz (retention >=0.9 for spin echo, "
            "with the independent response reported for T2-star GRE). T1 weighting requires diagnostic WM/GM >=1.15 "
            "and greater T1 than T2 sensitivity. "
            "Ideal-image criteria: complex NRMSE <=0.2, signal fraction >=0.01, outside-object energy <=0.4; "
            "Image scores use independent randomized spatial quadrature with analytic encoding control variates and approximate 99% numerical intervals. An interval that settles pass/fail permits early exit. Otherwise refinement targets absolute precision: "
            f"normalized squared complex error {precision.normalized_squared_error:g}; signal fraction {precision.signal_fraction:g}; outside-object energy {precision.outside_object_energy_fraction:g}; WM/GM ratio {precision.wm_gm_ratio:g}. "
            f"Case-image precision targets: squared magnitude error {precision.normalized_squared_magnitude_error:g}; squared phase error {precision.normalized_squared_phase_error:g}; tissue signal {precision.min_tissue_signal_fraction:g}; contrast retention {precision.contrast_retention:g}. "
            "Once precise, point estimates determine pass/fail even if an interval crosses a cutoff. Spatial integration and temporal step sizes refine until the numerical verdict resolves; trusted grading is checkpointed and excluded from the participant deadline. Temporal raw-signal NRMSE must be <=0.01. "
            + case_text
            + f"The evaluation contract is s1@{contract['version']}. Echo rephasing must occur inside ADC; KomaMRI must locate the physical echo at the waveform-derived TE."
        )
    if contract["id"] == "s3":
        policy = (contract.get("case_image_criteria") or {}).get(image_policy)
        limits = "; ".join(
            f"{name}={value:g}" for name, value in (policy or {}).items()
        )
        return (
            f"Evaluation (s3@{contract['version']}): waveform-derived sampling, calibration and spatial selection; "
            "independent finite-RF KomaMRI replay with scanner-specific cardiac tissues, fat and static B0. "
            "BART FFT/PICS reconstruction and receive encoding are fixed. Declared-physics images use an independent "
            "full-grid target and RF-only phase reference; ideal-control images are diagnostic. "
            f"Case limits: {limits}. Final submissions include SAR checks. "
            + (
                "Cardiac qualification is pending; image results remain uncalibrated. "
                if not contract["calibrated"]
                else ""
            )
            + "See docs/S3_EVALUATION.md."
        )
    if not contract["calibrated"]:
        return "Evaluation: hardware/timing preflight and final-submission SAR; physical scoring is pending."
    metrics = []
    for metric in contract["metrics"]:
        bounds = " and ".join(
            f"{op} {metric[key]}"
            for key, op in (("min", ">="), ("max", "<="))
            if metric[key] is not None
        )
        metrics.append(f"{metric['name']} {bounds}")
    return f"Evaluation: {contract['reconstruction']}; {', '.join(metrics)}."


def evaluation_overview(case):
    """Keep acceptance semantics inline and detailed scoring available on demand."""
    contract = case["evaluation"]
    if contract["id"] == "s4":
        return (
            f"Evaluation (s4@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): "
            "ECG/preparation/bSSFP waveform validity, SAR and declared-physics image quality; "
            "mapping tasks also require accurate myocardial relaxation estimates. "
            "Native finite-RF Bloch replay and standard Cartesian FFT use independent targets."
        )
    if contract["id"] not in {"s1", "s2"} or contract["scope"] == "smoke":
        return evaluation_summary(
            contract, case.get("image_policy", case.get("object_level"))
        )
    case_imaging = (contract.get("case_image_criteria") or {}).get(
        case.get("image_policy", case.get("object_level"))
    )
    imaging = (
        "images under the declared physics (ideal-control imaging is diagnostic only)"
        if case_imaging
        else "ideal-control images (B0/motion robustness is diagnostic only)"
    )
    return (
        f"Evaluation ({contract['id']}@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): waveform-derived sequence validity, "
        f"SAR and {imaging}, using independent finite-RF Bloch replay."
    )


def prompt(case, *, arm=None):
    task = case["task"]
    research = task.get("prompt_mode") == "research"
    return (
        f"# {case['case_id']}\n\n{task['objective'].strip()}\n\n"
        f"Write `{task['submission']['file']}` in Pulseq format in this working directory.\n"
        + (
            "Choose design parameters that satisfy the named sequence's defining physics and structure.\n"
            if research
            else "Timing targets use the stated tolerance.\n"
        )
        + "Duration and max values are upper limits; hardware and SAR limits apply.\n"
        + (
            "Use the supplied static cardiac phantom with scanner-specific tissue T1/T2, fat and off-resonance. Reconstruction is standard and fixed.\n"
            if case.get("image_policy") == "static"
            else ""
        )
        + (
            "Internet access is enabled; use normal tools as needed.\n"
            if research and arm is None
            else "There is no direct Internet access; use installed tools and documentation.\n"
        )
        + "\n```yaml\n"
        + brief_yaml(case, include_objective=False)
        + "```\n\n"
        + evaluation_overview(case)
        + "\nScoring criteria and SAR model: `EVALUATION.md`. Requirements: `task.yaml`.\n"
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


def prepare(workspace, case, experiment, server, phantom_dir=None, *, arm=None):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    public_case = {
        **case,
        "hardware": {k: v for k, v in case["hardware"].items() if k != "provenance"},
    }
    if case["task"].get("prompt_mode") == "research":
        # Never expose hidden sequence identity checks via a secondary metadata file.
        public_case = task_brief(case)
    write_json(workspace / "case.json", public_case)
    (workspace / "task.yaml").write_text(brief_yaml(case))
    (workspace / "EVALUATION.md").write_text(
        evaluation_summary(
            case["evaluation"], case.get("image_policy", case.get("object_level"))
        )
        + "\n\nSAR uses the hardware sphere load with circular B1+, scales as field_strength_T squared, "
        "and checks peak 360 s averages against the limit and 10 s averages against twice the limit. "
        "Short scans use actual duration without zero padding or repetition. "
        "SAR violations fail a submission. Lint/check exclude SAR; calculate it yourself or use a library.\n"
    )
    prepare_inputs(workspace, case, server.root, phantom_dir)
    python = "python3" if arm else sys.executable
    instructions = prompt(case, arm=arm) + (
        f"\n## Execution\n\nTime budget: {experiment.agent.timeout_s:g} s of agent runtime, excluding submission grading.\n"
        "All other work counts. The harness tracks time automatically.\n"
        "Work and write files only inside this working directory.\n"
        + (
            f"Python: `{python}`; NumPy and h5py are installed.\n"
            if arm
            else f"Python: `{python}`; NumPy, h5py and PyPulseq are installed.\n"
        )
        + "Inputs: `INPUTS.json`; phantom and field definitions: `inputs/manifest.json` when present.\n"
    )
    if arm:
        write_json(workspace / "ARM.json", arm.model_dump(mode="json"))
        instructions += (
            f"\nArm: {arm.id}@{arm.version}. Authoring: {arm.authoring}; language: {arm.language}.\n"
            f"PulseDesigner: {'enabled' if arm.pulse_designer else 'unavailable'}. "
            f"Unitful: {'enabled' if arm.unitful else 'unavailable'}.\n"
            "Only selected authoring libraries are available; package downloads and host runtimes are inaccessible.\n"
        )
        if arm.language == "julia":
            instructions += (
                "Use `./bin/julia script.jl [args...]` or `./bin/julia -e 'Julia code'`.\n"
                "These commands reuse one persistent Julia session, with Revise and the selected authoring libraries already loaded.\n"
                "Package loading and a fixed generic authoring warmup finish before your time budget starts; your code compilation/execution counts.\n"
                "For edits: `./bin/julia -e 'includet(\"design.jl\"); build_sequence()'` once, then edit function definitions and run `./bin/julia -e 'build_sequence()'`.\n"
                "Revise runs before each call. Put computations in functions and call them explicitly; edits to top-level data are not automatically rerun.\n"
                "Session variables persist. Script arguments replace ARGS on each call; stdout/stderr and failures are returned to your tool.\n"
                "A direct `julia --startup-file=no --project=/opt/author` command starts a separate cold process.\n"
            )
        if arm.simulator == "none":
            instructions += "Agent-side simulation is disabled.\n"
        else:
            instructions += (
                f"Simulator: {arm.simulator}, through `./bin/simulate [sequence.seq]`.\n"
                "Simulates supplied Pulseq bytes against public inputs; artifacts: `simulation/`.\n"
                "Simulator libraries are unavailable for authoring.\n"
            )
    if experiment.max_submissions > 1:
        instructions += (
            "\n## Submissions\n\n`./bin/lint`: file/timing checks. `./bin/check`: also RF amplitude, gradients and slew.\n"
            "Both are unlimited; neither consumes submissions or scores SAR/images.\n"
            f"`./bin/submit`: judge `sequence.seq`; up to {experiment.max_submissions} attempts; only the last counts.\n"
            "Wait in the foreground; grading can take minutes. Save each JSON response; resubmitting consumes another attempt.\n\n"
            "- `retry_required`: use the failure feedback to revise and submit again.\n"
            "- `submitted`: run finished; stop.\n\n"
        )
    else:
        instructions += (
            "\n## Submissions\n\n"
            + (
                "Lint/check feedback is disabled.\n"
                if experiment.feedback.mode == "none"
                else "`./bin/lint`: unlimited file/timing checks. "
                f"`./bin/check`: also RF amplitude, gradients and slew; at most {experiment.feedback.max_checks} charged checks. "
                "Neither scores SAR/images.\n"
            )
            + "`./bin/submit`: record the final `sequence.seq` and end the run without showing a verdict.\n"
        )
    if experiment.submission_mode == "submit":
        instructions += "You must run `./bin/submit` before your time expires.\n"
    else:
        instructions += "Submit explicitly, or leave `sequence.seq` and exit successfully for one final evaluation without revision.\n"
    (workspace / "PROMPT.md").write_text(instructions)
    write_json(
        workspace / ".harness.json",
        {
            "port": 8765 if arm else server.port,
            "route_prefix": "/feedback" if arm else "",
            "token": server.token,
            "max_submission_bytes": experiment.feedback.max_submission_bytes,
            "timeout_s": 5 * experiment.feedback.timeout_s,
        },
    )
    client = workspace / ".harness_client.py"
    client.write_text(Path(__file__).with_name("client.py").read_text())
    directory = workspace / "bin"
    directory.mkdir(exist_ok=True)
    for name in ("lint", "check", "submit"):
        path = directory / name
        path.write_text(
            f"#!/bin/sh\nexec {shlex.quote(python)} {shlex.quote(str(client))} {name}\n"
        )
        path.chmod(0o755)
    if arm and arm.language == "julia":
        path = directory / "julia"
        path.write_text('#!/bin/sh\nexec python3 /arm_julia_client.py "$@"\n')
        path.chmod(0o755)
    if arm and arm.simulator != "none":
        simulation_client = workspace / ".simulation_client.py"
        simulation_client.write_text(
            Path(__file__).with_name("arm_simulation_client.py").read_text()
        )
        path = directory / "simulate"
        path.write_text(
            f'#!/bin/sh\nexec python3 {shlex.quote(str(simulation_client))} "$@"\n'
        )
        path.chmod(0o755)
    return (
        workspace / "PROMPT.md",
        workspace / "case.json",
        workspace / case["task"]["submission"]["file"],
    )


def prepare_pi(workspace, base_url, *, offline=True):
    directory = Path(workspace) / ".pi-agent"
    directory.mkdir()
    write_json(
        directory / "models.json", {"providers": {"openrouter": {"baseUrl": base_url}}}
    )
    write_json(directory / "settings.json", {"enableInstallTelemetry": False})
    env = {"PI_CODING_AGENT_DIR": str(directory)}
    if offline:
        env["PI_OFFLINE"] = "1"
    return env
