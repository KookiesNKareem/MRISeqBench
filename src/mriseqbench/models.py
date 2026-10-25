"""Strict, versioned authoring models. Units are part of each field name."""

import itertools
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Reference = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*@[1-9][0-9]*$")]
# Capabilities grouped by sequence programming layer.
SEQUENCE_CATEGORIES = {
    "S1": ("spoiled_gre", "spin_echo", "bssfp", "cartesian"),
    "S2": (
        "spiral",
        "radial",
        "stack_of_stars",
        "propeller",
        "ute",
        "epi",
        "multi_shot",
        "tse",
        "rare",
        "grase",
    ),
    "S3": (
        "undersampling",
        "grappa",
        "sense",
        "caipirinha",
        "compressed_sensing",
        "variable_density",
        "poisson_disc",
        "partial_fourier",
        "reduced_fov",
        "elliptical_kspace",
    ),
    "S4": (
        "adiabatic",
        "inversion",
        "t2_preparation",
        "multi_contrast",
        "mapping",
        "triggering",
        "gating",
        "rf_excitation_2d",
    ),
}
Capability = Literal[
    tuple(tag for tags in SEQUENCE_CATEGORIES.values() for tag in tags)
]
Vector = tuple[Finite, Finite] | tuple[Finite, Finite, Finite]
Extent = tuple[Positive, Positive] | tuple[Positive, Positive, Positive]
Grid = tuple[PositiveInt, PositiveInt] | tuple[PositiveInt, PositiveInt, PositiveInt]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Document(Strict):
    schema_version: Literal[1]
    id: Identifier
    version: PositiveInt


class Timing(Strict):
    target: Positive | None = None
    tolerance: Positive | None = None
    max: Positive | None = None

    @model_validator(mode="after")
    def coherent(self):
        if (self.target is None) != (self.tolerance is None):
            raise ValueError("target and tolerance must be specified together")
        if self.target is None and self.max is None:
            raise ValueError("specify target/tolerance or max")
        if self.max is not None and self.target is not None and self.target > self.max:
            raise ValueError("target exceeds max")
        return self


class ReadoutSpec(Strict):
    shots: PositiveInt | None = None
    echo_train_length: PositiveInt | None = None
    interleaves: PositiveInt | None = None
    spokes: PositiveInt | None = None
    turns: Positive | None = None
    blade_width_lines: PositiveInt | None = None
    echo_spacing_ms: Positive | None = None
    center_out: StrictBool | None = None


class SamplingSpec(Strict):
    acceleration: Annotated[float, Field(ge=1)] | None = None
    partial_fourier_fraction: Annotated[float, Field(gt=0.5, le=1)] | None = None
    calibration_lines: Annotated[int, Field(ge=0)] | None = None
    mask_seed: Annotated[int, Field(ge=0)] | None = None
    pattern: (
        Literal[
            "uniform",
            "variable_density",
            "poisson_disc",
            "elliptical",
            "reduced_fov",
        ]
        | None
    ) = None
    caipi_shift: Annotated[int, Field(ge=0)] | None = None
    reduced_fov_fraction: Annotated[float, Field(gt=0, le=1)] | None = None


class TemporalSpec(Strict):
    frames: PositiveInt
    segments_per_frame: PositiveInt = 1
    trigger_source: Literal["ecg"] | None = None
    rr_interval_ms: Positive | None = None
    trigger_delay_ms: NonNegative | None = None
    acquisition_window_ms: Positive | None = None

    @model_validator(mode="after")
    def cardiac_window(self):
        if self.segments_per_frame > 1 and self.trigger_source != "ecg":
            raise ValueError("Segmented cardiac frames require ECG triggering")
        cardiac = (
            self.trigger_source,
            self.rr_interval_ms,
            self.trigger_delay_ms,
            self.acquisition_window_ms,
        )
        if any(value is not None for value in cardiac):
            if any(value is None for value in cardiac):
                raise ValueError(
                    "ECG triggering requires source, RR, trigger delay and acquisition window"
                )
            if self.trigger_delay_ms + self.acquisition_window_ms > self.rr_interval_ms:
                raise ValueError(
                    "cardiac acquisition window must end within the RR interval"
                )
        return self


class ApplicationSpec(Strict):
    inversion_time_ms: Positive | None = None
    inversion_times_ms: list[Positive] = Field(default_factory=list)
    preparation_duration_ms: Positive | None = None
    preparation_durations_ms: list[NonNegative] = Field(default_factory=list)

    @model_validator(mode="after")
    def increasing_preparations(self):
        if any(
            a >= b
            for a, b in zip(
                self.preparation_durations_ms, self.preparation_durations_ms[1:]
            )
        ):
            raise ValueError("preparation durations must strictly increase")
        return self


class SequenceSpec(Strict):
    level: Literal["S1", "S2", "S3", "S4"]
    capabilities: Annotated[list[Capability], Field(min_length=1)]
    matrix: Grid
    fov_mm: Extent
    te_ms: Timing | None = None
    tr_ms: Timing | None = None
    duration_s: Positive
    flip_angle_deg: Annotated[float, Field(gt=0, le=180)] | None = None
    readout: ReadoutSpec | None = None
    sampling: SamplingSpec | None = None
    temporal: TemporalSpec | None = None
    application: ApplicationSpec | None = None

    @model_validator(mode="after")
    def unique_capabilities(self):
        if len(self.matrix) != len(self.fov_mm):
            raise ValueError("matrix and FOV dimensions must match")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate capabilities")
        return self


class Submission(Strict):
    format: Literal["pulseq"] = "pulseq"
    file: Literal["sequence.seq"] = "sequence.seq"


class Task(Document):
    schema_version: Literal[1] = 1
    version: PositiveInt = 1
    objective: Annotated[str, Field(min_length=1)]
    sequence: SequenceSpec
    object: Reference = "cardiac_2d@1"
    hardware: Reference = "high_performance@1"
    evaluation: Reference
    submission: Submission = Field(default_factory=Submission)

    @model_validator(mode="after")
    def cardiac_application_contract(self):
        if self.id == "rf_excitation_2d" and (
            self.sequence.level != "S4"
            or self.evaluation != "rf_excitation@3"
            or self.object != "uniform_disc@1"
            or self.sequence.capabilities != ["rf_excitation_2d"]
            or self.sequence.matrix != (128, 128)
            or self.sequence.fov_mm != (200, 200)
            or any(
                value is not None
                for value in (
                    self.sequence.readout,
                    self.sequence.sampling,
                    self.sequence.temporal,
                    self.sequence.application,
                    self.sequence.flip_angle_deg,
                    self.sequence.te_ms,
                    self.sequence.tr_ms,
                )
            )
        ):
            raise ValueError(
                "AI excitation requires the fixed S4 excitation-only water-disc contract"
            )
        if self.sequence.level == "S4" and self.id != "rf_excitation_2d":
            if not {"triggering", "gating"} <= set(self.sequence.capabilities):
                raise ValueError(
                    "S4 cardiac applications require triggering and gating capabilities"
                )
            if (
                self.sequence.temporal is None
                or self.sequence.temporal.trigger_source != "ecg"
            ):
                raise ValueError(
                    "detailed S4 cardiac applications require an ECG timing contract"
                )
            if {"inversion", "t2_preparation"} & set(self.sequence.capabilities) and (
                "bssfp" not in self.sequence.capabilities
                or "spoiled_gre" in self.sequence.capabilities
            ):
                raise ValueError(
                    "detailed S4 magnetization preparations require bSSFP readouts"
                )
        return self


class Asset(Strict):
    """A content-addressed local array asset."""

    path: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    format: Literal["npz", "nifti", "hdf5"]
    array: Annotated[str, Field(min_length=1)]
    units: Literal["label", "Hz", "relative", "mm"]
    shape: Grid
    spacing_mm: Extent
    origin_mm: Vector

    @model_validator(mode="after")
    def dimensions(self):
        from pathlib import PurePosixPath

        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError(
                "asset paths must be relative to the catalog root without traversal"
            )
        if not (len(self.shape) == len(self.spacing_mm) == len(self.origin_mm)):
            raise ValueError("asset shape, spacing and origin dimensions must match")
        return self


class Object(Document):
    description: str
    level: Literal["P0", "P1"] | None = None
    phantom: Reference


class PhantomDefinition(Strict):
    """Metadata for a fixed Julia constructor in benchmark/objects/registry.json."""

    dimensions: Literal[2, 3]
    grid: Grid
    fov_mm: Extent
    bounds_mm: list[tuple[Finite, Finite]]
    materials: Annotated[list[Identifier], Field(min_length=1)]

    @model_validator(mode="after")
    def dimensions_match(self):
        if any(
            len(value) != self.dimensions
            for value in (self.grid, self.fov_mm, self.bounds_mm)
        ):
            raise ValueError("phantom metadata dimensions must match")
        if any(lo > hi for lo, hi in self.bounds_mm):
            raise ValueError("phantom bounds must be ordered")
        if len(set(self.materials)) != len(self.materials):
            raise ValueError("duplicate phantom materials")
        return self


class B0(Strict):
    kind: Literal["b0"]
    model: Literal["linear"]
    offset_hz: Finite = 0
    gradient_hz_per_m: Vector


class ConstantB0(Strict):
    kind: Literal["b0"]
    model: Literal["constant"]
    offset_hz: Finite


class GaussianB0(Strict):
    kind: Literal["b0"]
    model: Literal["gaussian"]
    offset_hz: Finite = 0
    amplitude_hz: Finite
    center_mm: Vector
    sigma_mm: Extent

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.center_mm) != len(self.sigma_mm):
            raise ValueError("Gaussian center and sigma dimensions must match")
        return self


class MapB0(Strict):
    kind: Literal["b0"]
    model: Literal["map"]
    asset: Asset

    @model_validator(mode="after")
    def units(self):
        if self.asset.units != "Hz":
            raise ValueError("B0 map units must be Hz")
        return self


class Motion(Strict):
    kind: Literal["motion"]
    model: Literal["rigid_sinusoidal"]
    translation_amplitude_mm: Vector
    rotation_amplitude_deg: Finite
    period_s: Positive
    phase_deg: Finite = 0


class Pose(Strict):
    time_s: Annotated[float, Field(ge=0)]
    translation_mm: tuple[Finite, Finite, Finite]
    rotation_xyz_deg: tuple[Finite, Finite, Finite]


class RigidTrajectory(Strict):
    kind: Literal["motion"]
    model: Literal["rigid_keyframes"]
    interpolation: Literal["linear", "previous"]
    poses: Annotated[list[Pose], Field(min_length=2)]

    @model_validator(mode="after")
    def time_order(self):
        times = [pose.time_s for pose in self.poses]
        if times[0] != 0 or any(a >= b for a, b in itertools.pairwise(times)):
            raise ValueError(
                "motion keyframes must start at zero and strictly increase"
            )
        return self


class Deformation(Strict):
    kind: Literal["motion"]
    model: Literal["affine_sinusoidal"]
    strain_amplitude: Vector
    period_s: Positive
    phase_deg: Finite = 0

    @model_validator(mode="after")
    def no_folding(self):
        if any(abs(a) >= 1 for a in self.strain_amplitude):
            raise ValueError("strain magnitude must be below one to avoid folding")
        return self


class TimeVariation(Strict):
    kind: Literal["time_variation"]
    model: Literal["sinusoidal"]
    target: Literal["b0_offset", "proton_density"]
    # Unit follows target; require exactly one corresponding amplitude.
    amplitude_hz: Finite | None = None
    amplitude_scale: Finite | None = None
    period_s: Positive
    phase_deg: Finite = 0
    material: Identifier | None = None

    @model_validator(mode="after")
    def target_contract(self):
        if self.target == "b0_offset":
            if (
                self.amplitude_hz is None
                or self.amplitude_scale is not None
                or self.material
            ):
                raise ValueError("b0_offset requires amplitude_hz and no material")
        else:
            if self.amplitude_scale is None or self.amplitude_hz is not None:
                raise ValueError("scale variation requires amplitude_scale")
            if abs(self.amplitude_scale) >= 1:
                raise ValueError("scale variation amplitude must remain below one")
            if (self.target == "proton_density") != (self.material is not None):
                raise ValueError("proton_density requires a material")
        return self


B0Effect = Annotated[B0 | ConstantB0 | GaussianB0 | MapB0, Field(discriminator="model")]
MotionEffect = Annotated[
    Motion | RigidTrajectory | Deformation, Field(discriminator="model")
]
Effect = Annotated[B0Effect | MotionEffect | TimeVariation, Field(discriminator="kind")]


class Physics(Document):
    description: str
    level: Literal["P0", "P2", "P3"] | None = None
    reference_field_strength_T: Positive | None = None
    effects: list[Effect]

    @model_validator(mode="after")
    def level_matches(self):
        kinds = [e.kind for e in self.effects]
        if len(set(kinds)) != len(kinds):
            raise ValueError("only one effect per kind is allowed")
        expected = (
            "P3"
            if {"motion", "time_variation"}.intersection(kinds)
            else ("P2" if kinds else "P0")
        )
        if self.level is not None and self.level != expected:
            raise ValueError(f"effects imply {expected}, not {self.level}")
        return self


class HardwareProvenance(Strict):
    system: str
    source_url: Annotated[str, Field(pattern=r"^https://\S+$")]
    accessed_on: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    notes: str


class SARModel(Strict):
    """Fixed benchmark load, independent of the imaging phantom geometry."""

    model: Literal["homogeneous_sphere_v1"]
    radius_m: Positive
    conductivity_S_per_m: Positive
    density_kg_per_m3: Positive
    whole_body_limit_W_per_kg: Positive


class Hardware(Document):
    provenance: HardwareProvenance | None = None
    field_strength_T: Positive
    sar: SARModel
    max_gradient_mT_per_m: Positive
    max_slew_T_per_m_per_s: Positive
    max_b1_uT: Positive
    rf_dead_time_us: Positive
    rf_ringdown_time_us: Positive
    adc_dead_time_us: Positive
    gradient_raster_us: Positive
    rf_raster_us: Positive
    adc_raster_us: Positive
    block_raster_us: Positive


class Metric(Strict):
    name: Identifier
    min: Finite | None = None
    max: Finite | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.min is None and self.max is None:
            raise ValueError("metric needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("metric min exceeds max")
        return self


class ImagePrecision(Strict):
    """Absolute numerical error targets, independent of acceptance cutoffs."""

    normalized_squared_error: Positive = 0.0008
    signal_fraction: Positive = 0.005
    outside_object_energy_fraction: Positive = 0.005
    wm_gm_ratio: Positive = 0.02
    normalized_squared_magnitude_error: Positive = 0.001
    normalized_squared_phase_error: Positive = 0.002
    min_tissue_signal_fraction: Positive = 0.01
    contrast_retention: Positive = 0.02


class CaseImageCriteria(Strict):
    """S1/S2/S3 case-physics quality requirements; all errors have fixed conventions."""

    complex_nrmse_max: Annotated[float, Field(gt=0, lt=2)]
    magnitude_nrmse_max: Annotated[float, Field(gt=0, lt=1)]
    phase_rms_rad_max: Annotated[float, Field(gt=0, lt=3.141593)]
    tissue_signal_fraction_min: Annotated[float, Field(gt=0, le=1)]
    contrast_retention_min: Annotated[float, Field(gt=0, le=1)]
    outside_energy_fraction_max: Annotated[float, Field(gt=0, lt=1)]


class Evaluation(Document):
    scope: Literal["benchmark", "smoke"]
    calibrated: StrictBool
    reconstruction: Literal[
        "bart_cartesian_fft@1",
        "bart_readout@1",
        "bart_sampling_sense@1",
        "direct_magnetization@1",
        "none",
    ]
    metrics: list[Metric]
    spatial_samples: Annotated[int, Field(ge=4096, le=1048576)] | None = None
    spatial_replicates: Annotated[int, Field(ge=4, le=32)] | None = None
    spatial_max_samples: Annotated[int, Field(ge=4096, le=1048576)] | None = None
    spatial_seed: Annotated[int, Field(ge=0, le=2147483616)] | None = None
    image_precision: ImagePrecision | None = None
    temporal_refinement: Literal[0, 1] | None = None
    case_image_criteria: dict[Literal["static"], CaseImageCriteria] | None = None
    # Optional per-task (family) overrides of the static case image criteria.
    family_image_criteria: dict[str, CaseImageCriteria] | None = None
    preparation_effect_fraction_min: Annotated[float, Field(gt=0, le=1)] | None = None
    mapping_relative_error_max: Annotated[float, Field(gt=0, lt=1)] | None = None
    mapping_relative_error_precision: Annotated[float, Field(gt=0, lt=1)] | None = None

    @model_validator(mode="after")
    def contract(self):
        imaging = self.id in {"s1", "s2", "s3", "s4"}
        expected = {
            "s1": "bart_cartesian_fft@1",
            "s2": "bart_readout@1",
            "s3": "bart_sampling_sense@1",
            "s4": "bart_cartesian_fft@1",
            "rf_excitation": "direct_magnetization@1",
        }
        if self.id in expected and self.reconstruction != expected[self.id]:
            raise ValueError(f"{self.id} requires {expected[self.id]}")
        if imaging != (self.case_image_criteria is not None):
            raise ValueError("s1-s4 (only) require static case image criteria")
        if self.family_image_criteria is not None and not imaging:
            raise ValueError("family image criteria require s1-s4")
        if not imaging and any(
            value is not None
            for value in (
                self.spatial_samples,
                self.spatial_replicates,
                self.spatial_max_samples,
                self.spatial_seed,
                self.image_precision,
                self.temporal_refinement,
            )
        ):
            raise ValueError("spatial/numerical settings require s1-s4")
        for value in (self.spatial_samples, self.spatial_max_samples):
            if value is not None and value & (value - 1):
                raise ValueError("spatial sample counts must be powers of two")
        if (self.spatial_max_samples or 1 << 20) < (self.spatial_samples or 32768):
            raise ValueError("spatial_max_samples must be >= spatial_samples")
        s4 = (
            self.preparation_effect_fraction_min,
            self.mapping_relative_error_max,
            self.mapping_relative_error_precision,
        )
        if (self.id == "s4") != all(v is not None for v in s4) or (
            self.id != "s4" and any(v is not None for v in s4)
        ):
            raise ValueError(
                "preparation and mapping criteria are required by (only) s4"
            )
        if self.id == "s4" and s4[2] >= s4[1]:
            raise ValueError(
                "mapping numerical precision must be below its acceptance limit"
            )
        if self.scope == "benchmark" and (
            not self.metrics or self.reconstruction == "none"
        ):
            raise ValueError("benchmark evaluation needs reconstruction and metrics")
        if self.scope == "smoke" and (
            self.metrics or self.reconstruction != "none" or self.calibrated
        ):
            raise ValueError(
                "smoke evaluation cannot declare physical metrics or calibration"
            )
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric names")
        return self


class SuiteEntry(Strict):
    task: Reference
    objects: Annotated[list[Reference], Field(min_length=1)] | None = None
    hardware: Annotated[list[Reference], Field(min_length=1)] | None = None
    physics: Annotated[list[Reference], Field(min_length=1)]


class Suite(Document):
    cases: Annotated[list[SuiteEntry], Field(min_length=1)]


class Backend(Strict):
    command: Annotated[list[str], Field(min_length=1)]
    timeout_s: Positive | None = None


class SandboxConfig(Strict):
    backend: Literal["auto", "macos", "bubblewrap", "docker", "none"] = "auto"
    read_paths: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)
    network_ports: list[Annotated[int, Field(ge=1, le=65535)]] = Field(
        default_factory=list
    )


class FeedbackConfig(Strict):
    max_submission_bytes: PositiveInt = 16777216
    timeout_s: Positive = 300


class ApiProxyConfig(Strict):
    provider_endpoint: str | None = None
    # None omits the parameter upstream (required for models that reject it,
    # since routing demands that providers support every sent parameter).
    temperature: Annotated[float, Field(ge=0, le=2)] | None = 0.6
    top_p: Annotated[float, Field(gt=0, le=1)] | None = 0.95
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = (
        "medium"
    )
    # None sends the endpoint's own maximum (recorded per model in
    # harness/pi_models.json, which Pi's descriptor states as well): omitting
    # the field would let the provider apply a smaller default.
    max_output_tokens: PositiveInt | None = None
    upstream_url: str = "https://openrouter.ai/api/v1"
    key_env: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")] = (
        "OPENROUTER_API_KEY"
    )
    allowed_routes: list[str] = Field(
        default_factory=lambda: ["/chat/completions", "/responses", "/models"]
    )
    timeout_s: Positive = 300
    max_request_bytes: PositiveInt = 8388608

    @model_validator(mode="after")
    def safe_upstream(self):
        from urllib.parse import urlsplit

        url = urlsplit(self.upstream_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "API upstream must be HTTPS without credentials, query or fragment"
            )
        if not self.allowed_routes or any(
            not route.startswith("/") or "?" in route or "#" in route or ".." in route
            for route in self.allowed_routes
        ):
            raise ValueError(
                "API routes must be nonempty absolute paths without traversal"
            )
        return self


class Agent(Strict):
    uses_model: StrictBool = True
    pi_version: str = "0.87.0"
    auto_compaction: StrictBool = False
    adapter: Literal["command", "pi"] = "command"
    command: Annotated[list[str], Field(min_length=1)]
    timeout_s: Positive = 60
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    environment: dict[str, str] = Field(default_factory=dict)
    model: str = ""


class Arm(Document):
    language: Literal["python", "julia"]
    authoring: Literal["pypulseq", "koma_base", "pulseq_julia"]
    pulse_designer: StrictBool = False
    unitful: StrictBool = False
    simulator: Literal["none", "mrzero", "komamripy", "koma"] = "none"
    # cli: shell tools and ./bin/julia; julia_mcp: Julia only via julia_eval
    # (pinned aplavin/julia-mcp), with file tools and no shell.
    tool_interface: Literal["cli", "julia_mcp"] = "cli"

    @model_validator(mode="after")
    def consistent_tools(self):
        languages = {
            "pypulseq": "python",
            "koma_base": "julia",
            "pulseq_julia": "julia",
        }
        if self.language != languages[self.authoring]:
            raise ValueError("authoring package and language disagree")
        if self.pulse_designer != (self.authoring == "pulseq_julia"):
            raise ValueError("PulseDesigner is enabled only for pulseq_julia")
        if self.unitful and self.authoring != "pulseq_julia":
            raise ValueError("Unitful requires pulseq_julia")
        allowed = {
            "python": {"none", "mrzero", "komamripy"},
            "julia": {"none", "koma"},
        }
        if self.simulator not in allowed[self.language]:
            raise ValueError("simulator is incompatible with arm language")
        if self.tool_interface == "julia_mcp" and (
            self.language != "julia" or self.unitful
        ):
            raise ValueError("julia_mcp requires a Julia arm without Unitful")
        return self


class ArmRuntime(Strict):
    # Tags are resolved to immutable local image IDs before any case is run.
    author_image: Annotated[str, Field(min_length=1, pattern=r"^[^\s]+$")]
    simulator_image: Annotated[str, Field(min_length=1, pattern=r"^[^\s]+$")] | None = (
        None
    )
    probe_timeout_s: Positive = 120
    simulation_timeout_s: Positive | None = None
    julia_heap_size_hint_mb: Annotated[int, Field(ge=32)] | None = None
    author_memory_mb: Annotated[int, Field(ge=256)] = 2048


class Concurrency(Strict):
    agents: Annotated[int, Field(ge=1, le=64)] = 1
    graders: Annotated[int, Field(ge=1, le=64)] = 1
    # Campaign-wide agent simulator executions sharing the selected GPU.
    simulators: Annotated[int, Field(ge=1, le=64)] = 4


class Experiment(Document):
    suite: Reference
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    repeats: Annotated[int, Field(gt=0, le=1000)] = 1
    max_submissions: Annotated[int, Field(ge=1, le=3)] = 3
    agent: Agent
    backend: Backend | None = None
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    api_proxy: ApiProxyConfig | None = None
    submission_mode: Literal["file", "submit"] = "submit"
    arm: Reference | None = None
    arm_runtime: ArmRuntime | None = None
    concurrency: Concurrency = Field(default_factory=Concurrency)

    @model_validator(mode="after")
    def arm_isolation(self):
        if self.agent.uses_model and self.api_proxy is None:
            raise ValueError("model runs require an api_proxy")
        if (self.arm is None) != (self.arm_runtime is None):
            raise ValueError("arm and arm_runtime must be configured together")
        if self.arm is None and self.agent.sandbox.backend == "docker":
            raise ValueError("docker isolation requires an arm and arm_runtime")
        if self.arm is not None:
            if self.agent.sandbox.backend not in {"auto", "docker"}:
                raise ValueError("restricted arms require auto or docker isolation")
            if (
                self.agent.sandbox.read_paths
                or self.agent.sandbox.deny_paths
                or self.agent.sandbox.network_ports
            ):
                raise ValueError(
                    "restricted arms cannot add host read grants or network ports"
                )
            for name in self.agent.environment:
                if name in {"PATH", "HOME", "TMP", "TEMP", "TMPDIR"} or name.startswith(
                    ("PYTHON", "JULIA", "MATLAB", "LD_", "DYLD_")
                ):
                    raise ValueError(
                        f"restricted arm cannot override runtime variable {name}"
                    )
        return self


MODELS = {
    "tasks": Task,
    "objects": Object,
    "physics": Physics,
    "hardware": Hardware,
    "evaluators": Evaluation,
    "suites": Suite,
    "experiments": Experiment,
    "arms": Arm,
}
