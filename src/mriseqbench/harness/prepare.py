"""Prepare the public workspace without copying judge code or host credentials."""

import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from ..io import write_json
from . import arm_mcp
from .arm_docs import api_notes


class BriefDumper(yaml.SafeDumper):
    pass


def sequence_list(dumper, values):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        values,
        flow_style=all(not isinstance(value, (dict, list)) for value in values),
    )


BriefDumper.add_representer(list, sequence_list)
BriefDumper.add_representer(
    float, lambda dumper, value: dumper.represent_float(float(f"{value:.6g}"))
)

SAR_NOTICE = (
    "SAR is evaluated only when you submit; a submission that exceeds the scanner's "
    "SAR limit fails, and the feedback will say so."
)
BUDGET_NOTICE = (
    "Outcomes are passed or failed. Grading has no time limit; a verdict that cannot be "
    "resolved at the maximum spatial sampling fails."
)
PREFLIGHT_RULES = (
    "Pulseq rules: raster definitions in the file must equal the hardware rasters; "
    "gradient amplitude and slew limits apply per axis; gradients must be continuous "
    "across blocks and start and end at zero; soft_delay and rf_shim extensions are "
    "not supported."
)


S4_TARGET_EXPLANATION = (
    "Targets freeze each tissue's RF-only response at the frame's central echo, "
    "retaining prior RF, relaxation and fat. Declared B0 supplies the phase target. "
    "Readout transients, blur and B0 signal loss count as image errors. "
    "No gain, phase or registration is fitted. "
)


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
            "tissue_properties": case["object"].get("tissue_properties"),
            "physics": {"effects": case["physics"]["effects"]},
            "hardware": {
                k: v
                for k, v in case["hardware"].items()
                if k not in ("id", "version", "schema_version", "provenance", "sar")
            },
            "seed": case["seed"],
        }
    )


def brief_yaml(case, include_objective=True):
    brief = task_brief(case)
    if not include_objective:
        brief.pop("objective")
    return yaml.dump(brief, Dumper=BriefDumper, sort_keys=False, width=88)


def s2_contract_notes(case):
    """Disclose existing S2 rules, without changing the scientific task."""
    text = (
        "Readout requirements: read docs/S2_EVALUATION.md, supplied in this workspace. "
        "Identity is measured from exported RF/gradient/ADC waveforms, not names or use labels. "
        "RF classification uses on-resonance envelope rotation; a requested flip angle or "
        "a phase-modulated pulse's integrated amplitude is not its measured rotation. "
    )
    if case["task"]["id"] in {"tse_t2w", "propeller_tse", "grase"}:
        text += (
            "Detailed refocused tasks require measured excitation flips 90 +/- 2 degrees "
            "and measured refocusing flips 180 +/- 2 degrees. "
        )
        text += (
            "Every distinct spin echo must retain at least 0.90 of its zero-offset signal "
            "in the gradient-free refocusing assay across -50 to +50 Hz. "
        )
    if case["task"]["sequence"].get("tr_ms"):
        text += (
            "TR is measured between consecutive acquired excitation RF centres. "
            "Terminal recovery counts toward scan duration, not another TR."
        )
    return text.rstrip()


def _image_limits(criteria, *, contrast=True):
    contrast_text = (
        ""
        if not contrast
        else f"contrast retention >={criteria['contrast_retention_min']:g}; "
    )
    return (
        f"complex/magnitude NRMSE <={criteria['complex_nrmse_max']:g}/{criteria['magnitude_nrmse_max']:g}; "
        f"phase RMS <={criteria['phase_rms_rad_max']:g} rad; "
        f"tissue signal >={criteria['tissue_signal_fraction_min']:g}; "
        + contrast_text
        + f"outside-object energy <={criteria['outside_energy_fraction_max']:g}. "
    )


def _metric_limits(metrics):
    return "; ".join(
        metric["name"]
        + " "
        + " and ".join(
            f"{op}{metric[key]:g}"
            for key, op in (("min", ">="), ("max", "<="))
            if metric[key] is not None
        )
        for metric in metrics
    )


def s1_probe_summary(task_id):
    """Required S1 probes for this family; definitions are in docs/S1_EVALUATION.md."""
    text = (
        "Required probes (PD=1, T1=800 ms, T2=80 ms): an echo forms within the ADC at TE; "
        "echo signal >=0.01 PD. "
    )
    if task_id in {"gre_t1w", "gre_t2star"}:
        text += (
            "Spoiling: complex echo error <=0.2 versus perfect spoiling over "
            "1021/2053 voxel isochromats. "
        )
    if task_id == "spin_echo_t2w":
        text += "Refocusing: >=0.90 retention over 65 offsets from -50 to +50 Hz. "
    if task_id == "gre_t1w":
        text += (
            "T1 weighting: WM/GM >=1.15 and T1 sensitivity > T2 sensitivity under 10% "
            "perturbations; WM=(PD 0.7, T1 800 ms, T2 80 ms), GM=(PD 0.8, T1 1300 ms, T2 100 ms). "
        )
    return text


def evaluation_summary(contract, image_policy, *, contrast, task_id):
    if contract["scope"] == "smoke":
        return "Evaluation: Pulseq timing/hardware preflight only; no image or sequence-family qualification."
    kind = contract["id"]
    header = f"Evaluation ({kind}@{contract['version']}): "
    pending = (
        "Qualification is pending; image results remain uncalibrated. "
        if not contract["calibrated"]
        else ""
    )
    if kind == "rf_excitation":
        return (
            header
            + "CUDA finite-RF replay; score transverse magnitude 1 us after sequence end. "
            "Use target.npz (128x128, x-first, positive y upward). Water disc: radius 60 mm, "
            "PD=1, T1=1 s, T2=0.1 s, z=0, FOV=200 mm. Target amplitude 0.5, "
            "Gaussian smoothing sigma=1.25 mm; phase is free. "
            "NRMSE uses all disc spins without fitted gain or registration. "
            "Declared B0 adds to the gradient precession: the off-resonance in Hz at "
            "position r is gradient_hz_per_m . r, with the same sign as gamma_bar * G . r. "
            "Inside signal is mean magnitude/mean target where target>=0.45; outside RMS uses target<=0.005 and divides by 0.5. "
            f"Limits: {_metric_limits(contract['metrics'])}. "
            "Each metric must clear its limit by more than its temporal error estimate, "
            "which must resolve below 0.002. " + BUDGET_NOTICE + " " + pending
        )
    if kind == "s4":
        return (
            header
            + "ECG-triggered prepared bSSFP, CUDA finite-RF replay and BART FFT. "
            "Each segment needs a physio1 input trigger and must fit the specified cardiac ADC window. "
            "Combine consecutive segments into each full Cartesian frame, sampling every point once; "
            "each segment acquires exactly 32 lines, and every ADC must have a positive sample count. "
            "Require alternating RF/ADC phases and balanced gradients. "
            "Each inversion segment has one >=170-degree inversion; TI ends at that segment's nearest-centre echo. "
            "T2 preparation uses separate 90/>=178/>=178/90-degree RF events "
            "(tip tolerance 2 degrees), phases 0,pi/2,pi/2,pi (tolerance 0.01 rad), "
            "and refocusers at 1/4 and 3/4 of the tip-centre interval. "
            "All timing targets (TI, preparation duration and refocuser centres, HS duration, "
            "cardiac window edges, trigger cadence) use +/-2 x the larger RF/block raster (20 us). "
            "Zero-duration T2 preparation has no prep RF. "
            "Adiabatic inversion: sech amplitude and tanh frequency sweep, each within 5% RMS; "
            "duration is the RF waveform span. "
            "Additional preparation RF may use RF-free crusher intervals; the independent oracle spoils only those preparations, retaining inversion and bSSFP readout coherence. "
            "Mapping preparations require >=3 RR recovery between preparation RF centres, "
            "and frames use the listed preparation times in order. "
            + S4_TARGET_EXPLANATION
            + "Every frame must meet: "
            + _image_limits(
                contract["case_image_criteria"]["static"],
                contrast=contrast,
            )
            + "Myocardial echo >=0.01 PD. Each frame with preparation RF needs preparation effect "
            f">={contract['preparation_effect_fraction_min']:g} PD: |myocardial complex RF-only response at "
            "the frame's central echo minus the same replay with that frame's preparation RF removed| / PD; "
            "zero-duration T2-prep frames are exempt. "
            "Each inversion/refocuser must invert >=0.90 Mz across [-520,70] Hz times field/3 T; "
            "the adiabatic inversion must do so at every B1 scale 0.80–1.20 (step 0.05). "
            f"Mapping relative error <={contract['mapping_relative_error_max']:g}, numerical precision "
            f"{contract['mapping_relative_error_precision']:g}; only the requested relaxation time is fitted. "
            + BUDGET_NOTICE
            + " "
            + pending
        )
    if kind in {"s1", "s2", "s3"}:
        reconstruction = {
            "s1": "BART Cartesian FFT",
            "s2": "BART FFT, inverse NUFFT or CG-SENSE (L2=0.001, 50 iterations)",
            "s3": "fixed BART FFT/CG-SENSE",
        }[kind]
        text = (
            header
            + f"waveform identity, physical sampling/FOV and timing; CUDA finite-RF replay with {reconstruction}. "
            "Images include the declared physics, fat and the complete submitted RF history from equilibrium. "
            "Readout bandwidth must be >=500 Hz/pixel: every imaging ADC dwell <= 1/(500 Hz x N), "
            "N = in-plane matrix size (15.625 us at N=128). "
        )
        text += (
            "Declared-physics images determine success; ideal images are diagnostic. Limits: "
            + _image_limits(
                contract["case_image_criteria"][image_policy], contrast=contrast
            )
            + "Targets are independent; no fitted gain, phase, B0 correction or registration. "
        )
        if kind == "s1":
            text += s1_probe_summary(task_id)
        if kind == "s3" and task_id not in {"reduced_fov", "rfov_epi"}:
            text += (
                "Required spoiling probe (water PD=1, T1=800 ms, T2=80 ms): echo signal "
                ">=0.01 PD and complex echo error <=0.2 versus perfect spoiling. "
            )
        text += (
            "Spatial sampling and time steps refine until the numerical verdict resolves. "
            + BUDGET_NOTICE
            + f" Required checks and exact definitions: docs/{kind.upper()}_EVALUATION.md. "
        )
        return text + pending
    raise ValueError(f"unknown evaluator {kind!r}")


def case_evaluation_summary(case):
    from ..evaluation.s1_case import contrast_required

    text = evaluation_summary(
        case["evaluation"],
        case["image_policy"],
        contrast=contrast_required(case),
        task_id=case["task"]["id"],
    )
    if case["task"]["id"] == "rfov_epi":
        nx, ny = case["task"]["sequence"]["matrix"]
        text += (
            " The 2D RF assay uses water with PD=1, T1=0.8 s, T2=0.08 s at z=0, "
            "sampled at the first, central and last EPI line centres. Exact post-RF T2 normalization "
            "requires minimum interior magnitude at RF end >=0.20 PD. Maximum exterior / mean interior "
            "magnitude at each matching echo must be <=0.10, including x-only and y-only stopbands. "
            f"The single EPI shot must acquire {ny // 2} alternating-polarity {nx}-sample lines on the "
            "prescribed even-ky mask. Image encoding starts at the final RF block's end; "
            "effective TE uses the central-ky echo. The independent phase oracle retains RF "
            "selection gradients, removes later imaging gradients, and excludes submitted ADC phase. "
            "Details: docs/RFOV_EPI_EVALUATION.md."
        )
    if case["evaluation"]["id"] == "s4":
        text += " Details: docs/S4_EVALUATION.md."
    elif case["evaluation"]["id"] == "rf_excitation":
        text += " Details: docs/RF_EXCITATION_EVALUATION.md."
    return text


def evaluation_overview(case):
    """Keep acceptance semantics inline and detailed scoring available on demand."""
    contract = case["evaluation"]
    if contract["id"] == "s4":
        return (
            f"Evaluation (s4@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): ECG/preparation/bSSFP validity and every frame's declared-physics image. "
            "Mapping also checks myocardial relaxation error. "
            + S4_TARGET_EXPLANATION
            + "Timing tolerances, trigger timing and ordering rules: docs/S4_EVALUATION.md."
        )
    if contract["id"] not in {"s1", "s2"} or contract["scope"] == "smoke":
        return case_evaluation_summary(case)
    return (
        f"Evaluation ({contract['id']}@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): waveform-derived sequence validity "
        "and images under the declared physics (ideal-control imaging is diagnostic only), "
        "using independent finite-RF Bloch replay. "
        f"Required checks and exact definitions: docs/{contract['id'].upper()}_EVALUATION.md."
    )


def prompt(case):
    task = case["task"]
    return (
        f"# {case['case_id']}\n\n{task['objective'].strip()}\n\n"
        f"Write `{task['submission']['file']}` in Pulseq format in this working directory.\n"
        + (
            "Timing targets use the stated tolerance.\n"
            if any(
                isinstance(v, dict) and "tolerance" in v
                for v in task["sequence"].values()
            )
            else ""
        )
        + "Duration and max values are upper limits; hardware limits apply.\n"
        + SAR_NOTICE
        + "\n"
        + (
            "Use the supplied static cardiac phantom with scanner-specific tissue T1/T2, fat and off-resonance. Reconstruction is standard and fixed.\n"
            if case.get("image_policy") == "static"
            and case["evaluation"]["scope"] != "smoke"
            else ""
        )
        + "There is no direct Internet access; use installed tools and documentation.\n"
        + "\n```yaml\n"
        + brief_yaml(case, include_objective=False)
        + "```\n\n"
        + evaluation_overview(case)
        + ("\n" + s2_contract_notes(case) if case["evaluation"]["id"] == "s2" else "")
        + "\nScoring criteria: `EVALUATION.md`. Requirements: `task.yaml`."
        + (
            " Shared rules: `docs/COMMON_EVALUATION.md`.\n"
            if case["evaluation"]["scope"] != "smoke"
            else "\n"
        )
    )


def copy_verified(source, target, expected, label):
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, target)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"{label} changed: {source.name}")


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
                copy_verified(source, target, value["sha256"], "input asset")
                assets[value["path"]] = str(target.relative_to(workspace))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(case)
    if case["evaluation"]["id"] == "rf_excitation":
        from ..evaluation.excitation_target import save_target

        save_target(Path(workspace) / "target.npz")
        assets["target.npz"] = "target.npz"
    if phantom_dir is not None:
        metadata = json.loads((Path(phantom_dir) / "artifacts.json").read_text())
        for filename, expected_hash in metadata["files"].items():
            target = Path(workspace) / "inputs" / filename
            copy_verified(
                Path(phantom_dir) / filename, target, expected_hash, "phantom artifact"
            )
            assets[filename] = str(target.relative_to(workspace))
    write_json(Path(workspace) / "INPUTS.json", assets)
    return assets


def prepare(workspace, case, experiment, server, phantom_dir=None, *, arm=None):
    from ..evaluation.s1_case import COMPLEX_NRMSE_RULE, contrast_required

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # Ship the public document we tell participants to read. Never copy judge
    # code, reference sequences or the rest of the catalog into the workspace.
    kind = case["evaluation"]["id"]
    if case["evaluation"]["scope"] != "smoke":
        public_docs = workspace / "docs"
        public_docs.mkdir(exist_ok=True)
        filenames = {
            "COMMON_EVALUATION.md",
            "RF_EXCITATION_EVALUATION.md"
            if kind == "rf_excitation"
            else kind.upper() + "_EVALUATION.md",
        }
        if case["task"]["id"] == "rfov_epi":
            filenames.add("RFOV_EPI_EVALUATION.md")
        if case["object"]["phantom"].startswith("cardiac_"):
            filenames.add("CARDIAC_PHANTOM.md")
        for filename in sorted(filenames):
            source = Path(server.root) / "docs" / filename
            shutil.copyfile(source, public_docs / filename)
    public_case = {
        **case,
        "hardware": {
            k: v for k, v in case["hardware"].items() if k not in ("provenance", "sar")
        },
    }
    write_json(workspace / "case.json", public_case)
    (workspace / "task.yaml").write_text(brief_yaml(case))
    (workspace / "EVALUATION.md").write_text(
        case_evaluation_summary(case)
        + ("\n\n" + s2_contract_notes(case) if kind == "s2" else "")
        + "\n\n"
        + SAR_NOTICE
        + " Lint and check never evaluate SAR.\n"
        + PREFLIGHT_RULES
        + (
            " Shared preflight, waveform, metric and numerical rules: docs/COMMON_EVALUATION.md."
            if case["evaluation"]["scope"] != "smoke"
            else ""
        )
        + "\n"
        + (
            ""
            if kind == "rf_excitation" or case["evaluation"]["scope"] == "smoke"
            else "\nPhase conventions: the independent reference chooses its receiver phase from RF coherence. "
            "The actual submitted signal uses your submitted ADC receiver phase; the grader does not "
            "repair or fit that phase. RF and ADC phase settings must produce the intended demodulation.\n"
            "Tissue-response and contrast diagnostics retain the complete excitation history, starting "
            "from equilibrium. They evaluate the acquired echoes, including transients; they do not "
            "assume that a steady state has already been reached. Any preparation/conditioning pulses "
            "must be present in the submitted waveform and obey the same timing, duration and SAR limits.\n"
            "\nImage metrics (exact definitions: docs/COMMON_EVALUATION.md): a voxel is oracle-valid when "
            "its RF-only signal without declared B0 exceeds 1e-6 of peak and its declared-physics signal is "
            ">=1% of that. "
            + COMPLEX_NRMSE_RULE
            + "Magnitude NRMSE uses all voxels, normalized by the target norm. Phase RMS is target-energy "
            "weighted on oracle-valid support voxels above 5% of target peak. Tissue ROIs keep voxels whose "
            "material occupancy, and that of both neighbours along every axis, exceeds 95% (else occupancy "
            ">75%). Tissue signal is the minimum ROI mean-magnitude "
            "ratio to its target, using tissues whose target ROI mean exceeds 5% of target peak. "
            "Outside energy is the reconstructed energy outside object support divided by total image energy. "
            + (
                "Contrast retention is the minimum ratio of signed (a-b)/(a+b) tissue contrast to target contrast, "
                "for target contrast magnitude >=0.1. "
                if contrast_required(case)
                else ""
            )
            + "The phase oracle must be valid over at least half the target energy and its temporal "
            "phase chord must settle to <=0.01; image replays refine time steps until NRMSE <=0.01 between "
            "step and half step. A verdict that cannot be resolved at the maximum spatial sampling fails.\n"
        )
    )
    prepare_inputs(workspace, case, server.root, phantom_dir)
    python = "python3" if arm else sys.executable
    instructions = prompt(case) + (
        f"\n## Execution\n\nTime budget: {experiment.agent.timeout_s:g} s of agent runtime, excluding submission grading.\n"
        "All other work counts. The harness tracks time automatically.\n"
        "Check charged time, remaining seconds and submission allowance with `./bin/status`; this "
        "does not require a sequence file or consume a submission. Check after exploration and simulation.\n"
        + "Use the budget for design and validation; three submissions are a scarce allowance, "
        "not a target to exhaust. Prepare a validated initial candidate within roughly the first half "
        "when practical, and reserve time to diagnose feedback and validate corrections. "
        "Do not rush a submission merely because time or attempts remain.\n"
        "Editing source alone does not change submitted sequence bytes.\n"
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
        (workspace / "ARM_API.md").write_text(api_notes(arm))
        instructions += (
            f"\nArm: {arm.id}@{arm.version}. Authoring: {arm.authoring}; language: {arm.language}.\n"
            f"PulseDesigner: {'enabled' if arm.pulse_designer else 'unavailable'}. "
            f"Unitful: {'enabled' if arm.unitful else 'unavailable'}.\n"
            "Only selected authoring libraries are available; package downloads and host runtimes are inaccessible.\n"
            "Package API and unit conventions: `ARM_API.md`.\n"
        )
        if arm.tool_interface == "julia_mcp":
            instructions += arm_mcp.instructions(arm)
        elif arm.language == "julia":
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
                "Simulation artifacts are diagnostics, not a benchmark verdict or an independent reference image.\n"
                "Simulator execution consumes agent time; waiting for a free simulator slot does not. A full replay can "
                "exceed the remaining allowance, especially for long echo trains or 3D acquisitions; "
                "check ./bin/status before and after simulation. Completing a replay does not check "
                "waveform identity, SAR or all image acceptance conditions.\n"
                "Simulator libraries are unavailable for authoring.\n"
            )
    instructions += (
        "\n## Submissions\n\n`./bin/lint`: file/timing checks. `./bin/check`: also RF amplitude, gradients and slew.\n"
        "Both are unlimited; neither consumes submissions or scores SAR/images.\n"
        f"`./bin/submit`: judge `sequence.seq`; up to {experiment.max_submissions} attempts; only the last counts.\n"
        "Wait in the foreground; grading can take minutes. Full responses are saved automatically "
        "under `feedback/` before output is printed; `feedback/latest-submit.json` retains the latest "
        "submission response. Output lists failed sequence, application and image checks with their "
        "measured details; SAR is reported only as within or over the scanner limit. "
        "Use `./bin/submit --full` for full output. "
        "`./bin/feedback` retrieves the last disclosed response without grading or consuming an attempt. "
        "Resubmitting consumes another attempt.\n\n"
        "- `retry_required`: use the failure feedback to revise and submit again.\n"
        "- `submitted`: run finished; stop.\n\n"
    )
    instructions += (
        "## Design, validate, then submit\n"
        "\n"
        "Your goal is a sequence that meets the task's acceptance criteria. The three submissions are opportunities to recover from mistakes, not a quota to exhaust.\n"
        "\n"
        "1. Read the task, EVALUATION.md and available authoring API notes before designing. Identify the physical requirements, units, timing, coverage, RF/gradient limits and relevant image/application criteria; SAR is checked only at submission. Check the generated waveform rather than assuming requested library parameters produced it.\n"
        "2. Regenerate sequence.seq after source edits. Run ./bin/lint and ./bin/check and read their complete responses. These check file/timing/hardware validity; they do not evaluate SAR, spoiling or image acceptance. Use the available public simulator to investigate signal and encoding when useful, within the remaining agent budget. Inspect its artifacts and explain what they support and what they cannot establish.\n"
        "3. Run ./bin/submit in the foreground. After retry_required, read the complete saved feedback/latest-submit.json using the file-read tool, continuing with offsets if needed. Avoid head/grep or name-only summaries that omit measured values, thresholds, conditions or other failing check groups. A clipped response is a cue to open the saved receipt.\n"
        "4. Before another submission, state each failed criterion, its measured value and required bound. Explain the likely physical cause, the specific change you will make and the effect you expect. Fix the mechanism rather than making an unexplained parameter tweak. Regenerate the sequence and inspect the relevant evidence, including previously passing criteria that the change could damage. Lower error is progress only; it is not a pass while the required bound is still missed.\n"
        "5. Treat contradictory or noisy simulation results as a diagnostic question. Check RF/ADC phase conventions, units, sampling and reconstruction assumptions before changing the sequence merely to make a plot look cleaner. A completed simulation is not proof that the benchmark criterion passes. Separate what was measured from what remains a hypothesis; do not claim a validation you did not perform.\n"
        "6. Check ./bin/status and use the remaining time deliberately. Keep concise working notes linking failures, corrections and evidence. Prefer a supported correction over rapidly spending the remaining attempts. If validation is limited by time or tools, state the limitation honestly and choose the best-supported candidate within the original budget. Stop when the harness returns submitted.\n"
        "\n"
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
    for name in ("lint", "check", "submit", "status", "feedback"):
        path = directory / name
        path.write_text(
            f'#!/bin/sh\nexec {shlex.quote(python)} {shlex.quote(str(client))} {name} "$@"\n'
        )
        path.chmod(0o755)
    if arm and arm.language == "julia":
        path = directory / "julia"
        path.write_text(
            arm_mcp.DISABLED_JULIA
            if arm.tool_interface == "julia_mcp"
            else '#!/bin/sh\nexec python3 /arm_julia_client.py "$@"\n'
        )
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


PI_ROUTES = {
    "openai-responses": "/responses",
    "openai-completions": "/chat/completions",
}


def pi_model(model, api_proxy):
    """Pinned Pi descriptor; Pi 0.87 would otherwise guess an API for unknown ids."""
    models = json.loads(Path(__file__).with_name("pi_models.json").read_text())
    if model not in models or model.startswith("_"):
        raise ValueError(f"no Pi model descriptor for {model!r} in pi_models.json")
    descriptor = dict(models[model])
    endpoint_max_output_tokens = descriptor.pop("maxOutputTokens")
    route = PI_ROUTES[descriptor["api"]]
    if route not in api_proxy.allowed_routes:
        raise ValueError(
            f"Pi sends {model} to {route}, which api_proxy.allowed_routes does not allow"
        )
    return {
        "id": model,
        "reasoning": True,
        "input": ["text"],
        # Pi judges truncated responses against this; it must match the cap
        # the proxy enforces, or the endpoint's maximum when there is none.
        "maxTokens": api_proxy.max_output_tokens or endpoint_max_output_tokens,
        **descriptor,
    }


def prepare_pi(workspace, base_url, agent, api_proxy):
    directory = Path(workspace) / ".pi-agent"
    directory.mkdir()
    descriptor = pi_model(agent.model, api_proxy)
    write_json(
        directory / "models.json",
        {
            "providers": {
                "openrouter": {
                    "baseUrl": base_url,
                    "api": descriptor["api"],
                    "models": [descriptor],
                }
            }
        },
    )
    write_json(
        directory / "settings.json",
        {
            "enableInstallTelemetry": False,
            "compaction": {"enabled": agent.auto_compaction},
            "cacheWarming": "off",
        },
    )
    return {"PI_CODING_AGENT_DIR": str(directory), "PI_OFFLINE": "1"}
