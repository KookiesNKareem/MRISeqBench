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


S4_TARGET_EXPLANATION = (
    "Target construction: each frame's target uses your submitted RF history, "
    "starting from equilibrium, with finite RF pulses, relaxation and intrinsic fat chemical shift. "
    "Spatial encoding gradients are omitted from this tissue-response calculation. "
    "At the acquired echo nearest k=0, each tissue's complex response is frozen and used "
    "across a complete Cartesian grid at the requested matrix and FOV, reconstructed with BART FFT. "
    "The target does not reproduce your line-by-line changing tissue response. "
    "Your actual image retains the full encoding gradients and changing response at every ADC sample. "
    "Transient-induced k-space modulation, blur, ringing and displacement therefore count as image errors; "
    "using the same RF history for the target does not cancel these artifacts. "
    "Declared B0 contributes an independent RF-only phase reference; encoding artifacts and "
    "B0-induced magnitude losses are not incorporated into the target magnitude. "
    "No gain, phase, B0 or registration correction is fitted. "
    "Conditioning pulses, flip-angle ramping, k-space ordering and readout timing are design choices "
    "for controlling these errors, not additional fixed parameter requirements. "
    "Any conditioning must be in the submitted waveform and satisfy the preparation timing, "
    "cardiac ADC window, duration, signal, preparation-effect and SAR constraints. "
    "The mapping dictionary likewise uses central-echo tissue responses and fixed anatomical partial "
    "volumes, not a fit to your transient-distorted reconstructed images. "
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


def s2_contract_notes(case):
    """Disclose existing S2 rules, without changing the scientific task."""
    if case["evaluation"]["id"] != "s2":
        return ""
    text = (
        "S2 waveform contract: read docs/S2_EVALUATION.md, supplied in this workspace. "
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
            "The TR bound also applies to the final acquired repetition, measured from the "
            "final excitation block's start to sequence end. Appending a cooling delay can "
            "therefore violate TR even when total duration and SAR pass. "
        )
    return text


def evaluation_summary(contract, image_policy=None):
    if contract["scope"] == "smoke":
        return "Evaluation: hardware/timing preflight and final-submission SAR."
    if contract["id"] == "rf_excitation":
        limits = "; ".join(
            f"{m['name']} "
            + " and ".join(
                f"{op}{m[k]:g}"
                for k, op in (("min", ">="), ("max", "<="))
                if m[k] is not None
            )
            for m in contract["metrics"]
        )
        return (
            f"Evaluation (rf_excitation@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): "
            "excitation-only RF/gradient waveforms, hardware/timing and final-submission SAR. "
            "Native finite-RF KomaMRI Bloch replay on the supplied water disc with declared B0; "
            "direct per-spin transverse magnetization 1 us after the authored sequence ends. "
            "The font-free upright AI target is supplied in target.npz, x-first and positive y upward. "
            "PD=1, T1=1 s, T2=0.1 s, radius=60 mm, z=0; 64x64 cell-centred 200 mm FOV. "
            "Target magnitude is 0.5, with Gaussian smoothing sigma=1.25 mm. Phase is unconstrained. "
            "NRMSE uses all disc spins with no fitted gain or registration. Inside signal is the mean "
            "magnitude divided by the mean target where target>=0.45. Outside RMS is divided by 0.5 "
            "where target<=0.005. Temporal refinement must resolve all limits with estimated error<=0.002. "
            f"Limits: {limits}. "
            + (
                "Qualification controls are pending; successful responses remain uncalibrated. "
                if not contract["calibrated"]
                else ""
            )
            + "See docs/RF_EXCITATION_EVALUATION.md."
        )
    if contract["id"] == "s4":
        criteria = contract["case_image_criteria"]["static"]
        return (
            f"Evaluation (s4@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): "
            "waveform-derived ECG windows, inversion/T2 preparation and balanced bSSFP; "
            "ECG requires an input trigger on physio1, not an output pulse on osc0. "
            "A static phantom still uses the physiological trigger timeline. "
            "every acquired echo must obey abs(TE-TR/2) <= max(0.01 ms,0.05*TR), "
            "with TE measured from excitation RF center to readout k-space-center echo. "
            "Prepared inversion frames require exactly one inversion RF event with nominal flip >=170 degrees. "
            "Nonzero T2 preparations require four separate RF events: 90/180/180/90 degrees, "
            "with relative phases (0, pi/2, pi/2, pi) and refocusing centers at one-quarter and three-quarters "
            "of the tip-down-center to tip-up-center interval. Tip-flip tolerance is 2 degrees; "
            "refocusing nominal flips must be >=178 degrees; RF phase tolerances are 0.01 rad. "
            "Preparation timing tolerance is twice the larger RF/block-duration raster. "
            "A zero-duration T2-mapping reference has no preparation RF events. "
            "native finite-RF Bloch replay with scanner-specific cardiac tissues, fat and static B0. "
            "Every cardiac image uses BART Cartesian FFT and an independent central-echo target. "
            + S4_TARGET_EXPLANATION
            + "Mapping tasks also test myocardial relaxation estimates; no gain, phase or B0 correction is fitted. "
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
            "For scans shorter than a configured SAR averaging window, SAR averages over the "
            "actual authored sequence duration, including explicit RF-free recovery delays; "
            "there is no zero-padding to the full window or implicit protocol repetition. "
            "Choose RF energy and protocol duty cycle within the supplied hardware SAR limits "
            "and total-duration cap. "
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
            "Complete RF history and intrinsic fat are retained; spoiled families destroy old transverse coherence before excitation while retaining longitudinal recovery. "
            "The independent reference receiver phase follows RF coherence. Actual acquisition demodulation uses the submitted ADC phase; the grader does not correct it. "
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
            "isochromats across one voxel (maximum relative echo error 0.2). This is the L2 relative "
            "error of the complex echo vector across acquired lines, after removing encoding phase, "
            "with the submitted ADC demodulation retained; it is not a last-echo or magnitude-only comparison. Refocusing/T2-star "
            "diagnostics use 65 offsets from -50 to +50 Hz (retention >=0.9 for spin echo, "
            "with the independent response reported for T2-star GRE). T1 weighting requires diagnostic WM/GM >=1.15 "
            "and greater T1 than T2 sensitivity. "
            "WM/GM is the ratio of mean absolute diagnostic echo signals over acquired lines. "
            "When the object lacks WM/GM materials, the independent common weighting diagnostic uses "
            "WM (PD=0.7, T1=800 ms, T2=80 ms) and GM (PD=0.8, T1=1300 ms, T2=100 ms); "
            "this diagnostic is separate from the cardiac phantom. The excitation history starts at equilibrium "
            "and includes conditioning pulses if submitted; no steady-state initialization is assumed. "
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


def case_evaluation_summary(case):
    text = evaluation_summary(
        case["evaluation"], case.get("image_policy", case.get("object_level"))
    )
    if case["task"]["id"] == "rfov_epi":
        text += (
            " The 2D RF assay uses water with PD=1, T1=0.8 s, T2=0.08 s at z=0, "
            "sampled at the first, central and last EPI line centres. It requires minimum interior magnitude >=0.20 PD and "
            "worst exterior / mean interior magnitude <=0.10, including x-only and y-only stopbands. "
            "The single EPI shot must acquire 16 alternating-polarity 32-sample lines on the "
            "prescribed even-ky mask. Image encoding starts at the final RF block's end; "
            "effective TE uses the central-ky echo. The independent phase oracle retains RF "
            "selection gradients, removes later imaging gradients, and excludes submitted ADC phase."
        )
    return text


def evaluation_overview(case):
    """Keep acceptance semantics inline and detailed scoring available on demand."""
    contract = case["evaluation"]
    if contract["id"] == "s4":
        return (
            f"Evaluation (s4@{contract['version']}{', uncalibrated' if not contract['calibrated'] else ''}): "
            "ECG/preparation/bSSFP waveform validity, SAR and declared-physics image quality; "
            "mapping tasks also require accurate myocardial relaxation estimates. "
            "Native finite-RF Bloch replay and standard Cartesian FFT use independent targets. "
            + S4_TARGET_EXPLANATION
        )
    if contract["id"] not in {"s1", "s2"} or contract["scope"] == "smoke":
        return case_evaluation_summary(case)
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


def prompt(case):
    task = case["task"]
    return (
        f"# {case['case_id']}\n\n{task['objective'].strip()}\n\n"
        f"Write `{task['submission']['file']}` in Pulseq format in this working directory.\n"
        + "Timing targets use the stated tolerance.\n"
        + "Duration and max values are upper limits; hardware and SAR limits apply.\n"
        + (
            "Use the supplied static cardiac phantom with scanner-specific tissue T1/T2, fat and off-resonance. Reconstruction is standard and fixed.\n"
            if case.get("image_policy") == "static"
            else ""
        )
        + "There is no direct Internet access; use installed tools and documentation.\n"
        + "\n```yaml\n"
        + brief_yaml(case, include_objective=False)
        + "```\n\n"
        + evaluation_overview(case)
        + ("\n" + s2_contract_notes(case) if case["evaluation"]["id"] == "s2" else "")
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
    if case["evaluation"]["id"] == "rf_excitation":
        from ..evaluation.excitation_target import save_target

        save_target(Path(workspace) / "target.npz")
        assets["target.npz"] = "target.npz"
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
    # Ship the public document we tell participants to read. Never copy judge
    # code, reference sequences or the rest of the catalog into the workspace.
    if case["evaluation"]["id"] == "s2":
        public_docs = workspace / "docs"
        public_docs.mkdir(exist_ok=True)
        shutil.copyfile(
            Path(server.root) / "docs" / "S2_EVALUATION.md",
            public_docs / "S2_EVALUATION.md",
        )
    public_case = {
        **case,
        "hardware": {k: v for k, v in case["hardware"].items() if k != "provenance"},
    }
    write_json(workspace / "case.json", public_case)
    (workspace / "task.yaml").write_text(brief_yaml(case))
    (workspace / "EVALUATION.md").write_text(
        case_evaluation_summary(case)
        + ("\n\n" + s2_contract_notes(case) if case["evaluation"]["id"] == "s2" else "")
        + "\n\nSAR uses the hardware sphere load with circular B1+, scales as field_strength_T squared, "
        "and checks peak 360 s averages against the limit and 10 s averages against twice the limit. "
        "Short scans use actual duration without zero padding or repetition. "
        "SAR violations fail a submission. Lint/check exclude SAR; calculate it yourself or use a library.\n"
        "The exact homogeneous-sphere formula is SAR_W_per_kg = "
        "sigma * (2*pi*gamma*B0_T)^2 * R_m^2 / (10*rho) * mean(abs(B1plus_T)^2), "
        "with sigma in S/m, rho in kg/m^3 and gamma=42.576e6 Hz/T. "
        "It assumes a quadrature (CP) body coil producing circular B1+. "
        "Pulseq RF samples are in Hz: B1plus_T = RF_Hz/gamma. "
        "For B1plus in microtesla, multiply mean(abs(B1plus_uT)^2) by 1e-12. "
        "Average over the complete authored scan (or the worst sliding configured window), "
        "including explicit RF-free intervals; timing constraints still apply to those intervals.\n"
        + (
            ""
            if case["evaluation"]["id"] == "rf_excitation"
            else "\nPhase conventions: the independent reference chooses its receiver phase from RF coherence. "
            "The actual submitted signal uses your submitted ADC receiver phase; the grader does not "
            "repair or fit that phase. RF and ADC phase settings must produce the intended demodulation.\n"
            "Tissue-response and contrast diagnostics retain the complete excitation history, starting "
            "from equilibrium. They evaluate the acquired echoes, including transients; they do not "
            "assume that a steady state has already been reached. Any preparation/conditioning pulses "
            "must be present in the submitted waveform and obey the same timing, duration and SAR limits.\n"
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
                "Simulation artifacts are diagnostics, not a benchmark verdict or an independent reference image.\n"
                "Simulator execution and simulator queue waits consume agent time. A full replay can "
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
        "measured details, and lists numerically unresolved checks separately. "
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
        "1. Read the task, EVALUATION.md and available authoring API notes before designing. Identify the physical requirements, units, timing, coverage, RF/gradient limits, SAR and relevant image/application criteria. Check the generated waveform rather than assuming requested library parameters produced it.\n"
        "2. Regenerate sequence.seq after source edits. Run ./bin/lint and ./bin/check and read their complete responses. These check file/timing/hardware validity; they do not establish SAR, spoiling or image acceptance. Use the available public simulator to investigate signal and encoding when useful, within the remaining agent budget. Inspect its artifacts and explain what they support and what they cannot establish.\n"
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


def prepare_pi(workspace, base_url, agent=None):
    directory = Path(workspace) / ".pi-agent"
    directory.mkdir()
    write_json(
        directory / "models.json", {"providers": {"openrouter": {"baseUrl": base_url}}}
    )
    write_json(
        directory / "settings.json",
        {
            "enableInstallTelemetry": False,
            "compaction": {"enabled": agent.auto_compaction if agent else False},
            "cacheWarming": "off",
        },
    )
    return {"PI_CODING_AGENT_DIR": str(directory), "PI_OFFLINE": "1"}
