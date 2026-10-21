# S2 evaluation

Contract: `s2@4`. Requalification on GPU pending after the SAR (quadrature) and receiver-phase/role changes. This public document
is supplied as `docs/S2_EVALUATION.md` in S2 participant workspaces. Exact task
parameters and hardware/SAR limits are in `task.yaml`; image bounds and applicable
checks are also summarized in `EVALUATION.md`.

S2 covers EPI, radial, spiral, stack-of-stars,
PROPELLER, UTE, TSE and GRASE. Waveform checks verify actual trajectories,
physical FOV/matrix coverage, shot/echo structure and prescribed timing.

## Waveform identity and timing

Names, sequence definitions and RF `use` labels do not establish identity.
The judge measures the exported RF, gradient and ADC waveforms:

- Nominal RF rotation is the Bloch rotation of an initially longitudinal spin
  under the complex RF envelope on resonance, without gradients or RF frequency
  offsets. Rotation at least 135 degrees classifies a refocusing event; otherwise
  it is an excitation. A library's requested flip, or the integral of RF
  magnitude, can differ from this rotation for phase-modulated RF.
- Detailed TSE, PROPELLER and GRASE tasks require excitation rotations within
  **90 +/- 2 degrees** and refocusing rotations within **180 +/- 2 degrees**.
  These fixed flip tolerances do not apply in research prompt mode. Refocused
  trains still need actual refocusing events and a measured rephasing response.
- TE is measured from the excitation's magnitude-weighted RF center to the
  actual k-space-center echo. For Cartesian echo trains and PROPELLER, the
  prescribed TE is the effective central-k-space echo, not the first echo.
- Inter-excitation TRs use RF centers. The final acquired repetition instead
  runs from its excitation **block start to sequence end**. A terminal cooling
  delay counts in this final TR, as well as scan duration and SAR averaging.
- Each TSE/PROPELLER echo must have a refocusing pulse before it. Its coherence
  time residual must be within the larger of 20 microseconds and half the
  smallest ADC dwell. GRASE needs at least two distinct spin-echo intervals per
  shot, each containing at least two gradient echoes straddling its spin echo
  within that tolerance. Task shot counts, echo-train lengths and echo spacing
  apply; prescribed echo spacing has a 0.05 ms tolerance.
- EPI uses alternating readout polarity within each shot, measured along x:
  EPI and GRASE readouts run along x, with y as the phase-encode axis. Detailed multishot
  EPI traverses ky in positive strides equal to the number of shots (in
  reciprocal-FOV units), rather than arbitrary centric ordering.
- Full radial readouts traverse both sides of k=0. In reciprocal-FOV units,
  each spoke must extend below `-Nx/2 + 0.75` and above `Nx/2 - 0.75`.
  Per-sample steps are at most 1.05 and transverse line error at most 0.05.
  UTE or explicitly center-out readouts instead start within radius 0.6,
  reach radius at least `Nx/2 - 0.75`, and have nondecreasing absolute radial
  coordinate to tolerance 0.05. UTE additionally requires nonselective RF of
  duration at most 0.5 ms and TE at most 1 ms, plus any tighter task TE bound.
- Stack-of-stars requires the full Cartesian partition set
  `-Nz/2,...,Nz/2-1`, integer kz to tolerance 0.01 and constant kz during each
  spoke to tolerance 0.01. The required spoke count and angular coverage apply
  **separately in every partition**. There is one readout per excitation.
- Full radial spokes and PROPELLER blade orientations cover **pi radians**;
  opposite orientations are equivalent. UTE center-out spokes cover **2*pi**.
  After sorting angles modulo the applicable period, every circular gap must
  be greater than `0.1*period/count` and at most `1.6*period/count`, with count
  at least two. Thus 16 full PROPELLER blades evenly spaced over 2*pi contain
  only eight distinct orientations and fail.
- Each PROPELLER blade consists of straight lines with transverse variation
  at most 0.05, spanning at least `-Nx/2+0.75` to `Nx/2-0.75`. Absolute sample
  steps must be 1 +/- 0.02. Adjacent sorted line offsets are 1 +/- 0.05 and
  their mean is within 0.6 of zero. The task specifies lines per blade.
- An Archimedean spiral starts within radius 0.6, reaches `Nx/2-0.75`, has
  sample steps at most 1.1 and at least Nx samples per interleaf. Radius is
  nondecreasing to tolerance 0.05 for at least 95% of steps. The RMS residual
  of a linear radius-versus-unwrapped-angle fit is at most 0.1 times maximum
  radius. Winding is measured after excluding radius <=0.5; it must be at
  least one turn and within 0.5 of any prescribed turn count. Interleaf
  angles obey the same circular-gap rule over 2*pi, with one readout per
  excitation.
- Non-Cartesian matrix/FOV acceptance additionally requires every requested
  grid node within the radius `min(Nx,Ny)/2-1` disk to be within 1.5 samples
  of an acquired node. This coverage check does not replace family-specific
  extent, orientation or partition requirements.

Mechanism checks precede image grading. The gradient-free refocusing assay
requires at least **0.90 signal retention at every distinct spin echo** across
65 uniformly spaced offsets from -50 to +50 Hz, relative to the zero-offset
response (PD=1, T1=0.8 s, T2=0.08 s). Non-refocused families require spoiling
relative error at most 0.20 against the ideal spoiled response on two voxel
quadratures. The mean ideal central-echo signal must be at least 0.01 PD.

## SAR and validation tools

SAR assumes a quadrature (circularly polarized, CP) body coil transmitting
into the hardware's homogeneous sphere:

```text
SAR_W_per_kg = sigma * (2*pi*gamma*B0_T)^2 * R_m^2 / (10*rho)
               * mean(abs(B1plus_T)^2)
gamma = 42.576e6 Hz/T
B1plus_T = Pulseq_RF_Hz / gamma
```

Here sigma is conductivity in S/m, rho is density in kg/m^3 and R is sphere
radius in meters. If B1+ is in microtesla, multiply its mean square by 1e-12.
The Pulseq RF amplitude is the co-rotating B1+ of the CP coil; no further
polarization factor applies. RF power includes the complete exported
complex RF envelope. The judge checks the maximum 360 s average against the
hardware limit and maximum 10 s average against twice that limit. Scans shorter
than a window use their actual complete duration, with explicit RF-free
intervals included and no implicit repetition or zero-padding. Adding such
intervals must still obey final TR and duration bounds.

`./bin/lint` checks file/timing validity; `./bin/check` also checks RF amplitude,
gradient and slew limits. Neither checks SAR, sequence-family identity,
spoiling/refocusing mechanisms or image acceptance. Successful preflight is
not a benchmark pass. Read every failed criterion and its measured bounds in
the saved submission feedback before revising.

## Independent image acceptance

Finite-RF KomaMRI replay uses [static cardiac anatomy](CARDIAC_PHANTOM.md),
scanner-specific T1/T2, fat and off-resonance. Reconstruction is fixed Cartesian
BART FFT or BART inverse NUFFT (periodic k-space, Tikhonov λ=0.001,
50 iterations maximum, direct forward/adjoint normal operator). The independent
full-grid target holds finite-RF tissue response at the central echo, including
intrinsic fat. The RF-only phase oracle adds declared B0; receiver errors,
readout-induced fat displacement and blur remain errors. Complete Cartesian
partitions with identical transverse nodes use the equivalent BART partition
FFT plus batched 2D inverse NUFFT; other geometries use the full NUFFT.

Declared-physics image limits:

| Metric | Limit |
|---|---:|
| Complex NRMSE | ≤ 0.45 |
| Magnitude NRMSE | ≤ 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal fraction | ≥ 0.5 |
| Contrast retention | ≥ 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Ideal-control images are diagnostic. Acquisition design must meet these limits
with standard reconstruction: no fitted gain, phase, registration or B0-aware
correction. Receiver tuning must preserve the actual RF/ADC coherence.

Magnitude agreement alone is insufficient: a global pi/2 receiver-phase error
can fail complex NRMSE and phase RMS while leaving the magnitude image intact.

Receiver phase convention (S1–S3, shared): targets use the excitation's FID
convention for every echo. After excitation with RF phase p the matching ADC
phase is p. A refocusing pulse with RF phase q maps the echo to ADC phase
2q - (previous phase) + pi. So 90x-180y (CPMG) echoes use ADC phase 0, and
90x-180x echoes use pi. A wrong receiver phase (e.g. pi off) fails complex
NRMSE and phase RMS even when the magnitude image is correct.
The target uses the complete RF history from equilibrium, frozen at the
representative central echo; it does not reproduce line-by-line transients.
An approximate real-valued or instantaneous-RF diagnostic is not that target.

Participant simulation and its queue waits consume agent time. Full-volume or
long-train replays may exceed the remaining allowance. Simulation artifacts
are diagnostics; a completed simulator call does not run every identity,
mechanism, SAR or fixed-reference image check.

Spatial quadrature and time steps adapt until the numerical verdict resolves.
Unresolved numerical results remain explicit. Passing sequences return `passed`;
invalid sequences fail.
Final submissions also undergo SAR evaluation.

Under the separately declared `bounded_conservative_v1` run policy, failure to
establish a pass within the cumulative 600 s frozen-submission execution limit
is `failed` with `failure_kind=grading_budget_exhausted`. That identifies a
compute-limit outcome, not a measured physical threshold violation. Native
grading and its permit queue are excluded from participant time. Infrastructure
crashes or resource-guard stops require preserved recovery, not a fabricated
model-quality verdict.

Saved artifacts include waveform checks, physical/ideal/target images, metrics,
submitted-byte hashes and numerical provenance. The YAML defines image bounds.
