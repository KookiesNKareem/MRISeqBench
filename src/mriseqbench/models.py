"""Strict, versioned authoring models. Units are part of each field name."""

import itertools
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Reference = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*@[1-9][0-9]*$")]
# Capabilities grouped by sequence programming layer.
SEQUENCE_CATEGORIES = {
    "S1": ("gre", "spoiled_gre", "spin_echo", "bssfp", "cartesian"),
    "S2": (
        "spiral",
        "rosette",
        "concentric_rings",
        "radial",
        "stack_of_stars",
        "propeller",
        "ute",
        "zte",
        "epi",
        "single_shot",
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
        "wave_caipi",
        "sms",
        "compressed_sensing",
        "variable_density",
        "poisson_disc",
        "partial_fourier",
        "asymmetric_echo",
        "reduced_fov",
        "elliptical_kspace",
    ),
    "S4": (
        "kt_blast",
        "kt_sense",
        "kt_pca",
        "low_rank_sparse",
        "interleaved",
        "golden_angle",
        "view_sharing",
        "twist",
        "keyhole",
        "cine",
        "real_time",
        "dynamic",
    ),
    "S5": (
        "tailored_rf",
        "cest",
        "tone",
        "adiabatic",
        "spectral_spatial",
        "ptx",
        "inversion",
        "t2_preparation",
        "magnetization_transfer",
        "diffusion",
        "asl",
        "fat_saturation",
        "multi_echo",
        "multi_contrast",
        "mpm",
        "mrf",
        "mapping",
        "triggering",
        "gating",
        "navigator",
        "motion_correction",
    ),
}
Capability = Literal[
    tuple(tag for tags in SEQUENCE_CATEGORIES.values() for tag in tags)
]
Vector = tuple[Finite, Finite] | tuple[Finite, Finite, Finite]
Extent = tuple[Positive, Positive] | tuple[Positive, Positive, Positive]
Grid = (
    tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    | tuple[
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
    ]
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Document(Strict):
    schema_version: Literal[1]
    id: Identifier
    version: Annotated[int, Field(gt=0)]


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
    shots: Annotated[int, Field(gt=0)] | None = None
    echo_train_length: Annotated[int, Field(gt=0)] | None = None
    interleaves: Annotated[int, Field(gt=0)] | None = None
    spokes: Annotated[int, Field(gt=0)] | None = None
    dwell_us: Positive | None = None
    bandwidth_hz_per_pixel: Positive | None = None
    turns: Positive | None = None
    rings: Annotated[int, Field(gt=0)] | None = None
    rosette_lobes: Annotated[int, Field(gt=0)] | None = None
    blade_width_lines: Annotated[int, Field(gt=0)] | None = None
    echo_spacing_ms: Positive | None = None
    center_out: StrictBool | None = None


class SamplingSpec(Strict):
    acceleration: Annotated[float, Field(ge=1)] | None = None
    partial_fourier_fraction: Annotated[float, Field(gt=0.5, le=1)] | None = None
    calibration_lines: Annotated[int, Field(ge=0)] | None = None
    multiband_factor: Annotated[int, Field(gt=0)] | None = None
    mask_seed: Annotated[int, Field(ge=0)] | None = None
    pattern: (
        Literal[
            "uniform",
            "variable_density",
            "poisson_disc",
            "elliptical",
            "asymmetric_echo",
            "reduced_fov",
        ]
        | None
    ) = None
    caipi_shift: Annotated[int, Field(ge=0)] | None = None
    asymmetric_echo_fraction: Annotated[float, Field(gt=0.5, le=1)] | None = None
    reduced_fov_fraction: Annotated[float, Field(gt=0, le=1)] | None = None


class TemporalSpec(Strict):
    frames: Annotated[int, Field(gt=0)]
    frame_duration_ms: Positive
    ordering: Literal["sequential", "interleaved", "golden_angle"]
    view_sharing_window: Annotated[int, Field(gt=0)] | None = None
    center_fraction: Annotated[float, Field(gt=0, le=1)] | None = None


class ApplicationSpec(Strict):
    inversion_time_ms: Positive | None = None
    inversion_times_ms: list[Positive] = Field(default_factory=list)
    preparation_duration_ms: Positive | None = None
    saturation_offset_hz: Finite | None = None
    diffusion_b_s_per_mm2: Annotated[float, Field(ge=0)] | None = None
    labeling_duration_ms: Positive | None = None
    post_label_delay_ms: Positive | None = None
    flip_angle_schedule_deg: list[Annotated[float, Field(gt=0, le=180)]] = Field(
        default_factory=list
    )
    tr_schedule_ms: list[Positive] = Field(default_factory=list)
    echo_times_ms: list[Positive] = Field(default_factory=list)

    @model_validator(mode="after")
    def schedule_lengths(self):
        if (
            self.flip_angle_schedule_deg
            and self.tr_schedule_ms
            and len(self.flip_angle_schedule_deg) != len(self.tr_schedule_ms)
        ):
            raise ValueError("flip angle and TR schedule lengths must match")
        if any(a >= b for a, b in zip(self.echo_times_ms, self.echo_times_ms[1:])):
            raise ValueError("echo times must strictly increase")
        return self


class SequenceSpec(Strict):
    level: Literal["S1", "S2", "S3", "S4", "S5"]
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
    version: Annotated[int, Field(gt=0)] = 1
    objective: Annotated[str, Field(min_length=1)]
    sequence: SequenceSpec
    object: Reference = "tissue_discs@1"
    hardware: Reference = "standard@1"
    evaluation: Reference = "image_fidelity@1"
    submission: Submission = Field(default_factory=Submission)


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
    level: Literal["P0", "P1"]
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


class B1(Strict):
    kind: Literal["b1"]
    model: Literal["linear"]
    center_scale: Positive
    gradient_scale_per_m: Vector


class ConstantB1(Strict):
    kind: Literal["b1"]
    model: Literal["constant"]
    scale: Positive


class GaussianB1(Strict):
    kind: Literal["b1"]
    model: Literal["gaussian"]
    baseline_scale: Positive
    amplitude_scale: Finite
    center_mm: Vector
    sigma_mm: Extent

    @model_validator(mode="after")
    def dimensions_and_scale(self):
        if len(self.center_mm) != len(self.sigma_mm):
            raise ValueError("Gaussian center and sigma dimensions must match")
        if self.baseline_scale + min(0, self.amplitude_scale) <= 0:
            raise ValueError("Gaussian B1 must remain positive")
        return self


class MapB1(Strict):
    kind: Literal["b1"]
    model: Literal["map"]
    asset: Asset

    @model_validator(mode="after")
    def units(self):
        if self.asset.units != "relative":
            raise ValueError("B1 map units must be relative")
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
    target: Literal["b0_offset", "b1_scale", "proton_density"]
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
                raise ValueError(
                    "proton_density requires a material; b1_scale does not"
                )
        return self


B0Effect = Annotated[B0 | ConstantB0 | GaussianB0 | MapB0, Field(discriminator="model")]
B1Effect = Annotated[B1 | ConstantB1 | GaussianB1 | MapB1, Field(discriminator="model")]
MotionEffect = Annotated[
    Motion | RigidTrajectory | Deformation, Field(discriminator="model")
]
Effect = Annotated[
    B0Effect | B1Effect | MotionEffect | TimeVariation, Field(discriminator="kind")
]


class Physics(Document):
    description: str
    level: Literal["P0", "P2", "P3", "P4"]
    effects: list[Effect]

    @model_validator(mode="after")
    def level_matches(self):
        kinds = [e.kind for e in self.effects]
        if len(set(kinds)) != len(kinds):
            raise ValueError("only one effect per kind is allowed")
        expected = (
            "P4"
            if {"motion", "time_variation"}.intersection(kinds)
            else ("P3" if len(kinds) > 1 else "P2" if kinds else "P0")
        )
        if self.level != expected:
            raise ValueError(f"effects imply {expected}, not {self.level}")
        return self


class Hardware(Document):
    field_strength_T: Positive
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


class Evaluation(Document):
    scope: Literal["benchmark", "smoke"]
    calibrated: StrictBool
    reconstruction: Literal["cartesian_fft@1", "none"]
    metrics: list[Metric]

    @model_validator(mode="after")
    def contract(self):
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
    physics: Annotated[list[Reference], Field(min_length=1)]


class Suite(Document):
    cases: Annotated[list[SuiteEntry], Field(min_length=1)]


class Backend(Strict):
    command: Annotated[list[str], Field(min_length=1)]
    timeout_s: Positive = 300


class SandboxConfig(Strict):
    backend: Literal["auto", "macos", "bubblewrap", "none"] = "auto"
    read_paths: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)
    network_ports: list[Annotated[int, Field(ge=1, le=65535)]] = Field(
        default_factory=list
    )


class FeedbackConfig(Strict):
    mode: Literal["none", "preflight", "evaluation"] = "none"
    max_checks: Annotated[int, Field(ge=0)] = 0
    max_submission_bytes: Annotated[int, Field(gt=0)] = 16777216
    timeout_s: Positive = 300


class ApiProxyConfig(Strict):
    upstream_url: str = "https://openrouter.ai/api/v1"
    key_env: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")] = (
        "OPENROUTER_API_KEY"
    )
    allowed_routes: list[str] = Field(
        default_factory=lambda: ["/chat/completions", "/responses", "/models"]
    )
    timeout_s: Positive = 300
    max_request_bytes: Annotated[int, Field(gt=0)] = 8388608

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
    adapter: Literal["command", "pi"] = "command"
    command: Annotated[list[str], Field(min_length=1)]
    timeout_s: Positive = 60
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    environment: dict[str, str] = Field(default_factory=dict)
    model: str = ""


class Experiment(Document):
    suite: Reference
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    repeats: Annotated[int, Field(gt=0, le=1000)] = 1
    agent: Agent
    backend: Backend | None = None
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    api_proxy: ApiProxyConfig | None = None
    submission_mode: Literal["file", "submit"] = "file"


MODELS = {
    "tasks": Task,
    "objects": Object,
    "physics": Physics,
    "hardware": Hardware,
    "evaluators": Evaluation,
    "suites": Suite,
    "experiments": Experiment,
}
