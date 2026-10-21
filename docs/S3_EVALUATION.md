# S3 evaluation

Contract: [`s3@3`](../benchmark/evaluators/s3.yaml). Requalification on GPU pending after the SAR (quadrature) and receiver-phase/role changes.

S3 covers parallel imaging, compressed sensing, variable density, partial
Fourier, reduced FOV and elliptical k-space. Waveform checks
verify acquired sampling masks, calibration regions, support and spatial
selection.

[`rfov_epi@1`](../benchmark/tasks/rfov_epi.yaml) adds single-shot EPI following a
2D selective excitation, with independent stopband assays in x and y. Encoding
starts at the end of the RF-containing block. Its phase oracle retains the
excitation's selection gradients while removing subsequent imaging gradients;
the submitted ADC phase never defines the target. See [rFOV EPI](RFOV_EPI_EVALUATION.md)
for its reference and qualification controls.

Sampling identity is waveform-derived. Every ADC event is one straight readout
along x with Nx samples at unit kx steps (reciprocal-FOV units, lattice error at
most 0.01); y (and z in 3D) are phase-encode axes. Zero-based grid index `i`
corresponds to `k = (i - N/2 + offset)/FOV`, where the fractional lattice
offset per axis is measured from the samples (zero or one-half in practice).
Each grid node is acquired at most once, and detailed tasks must
reproduce the prescribed mask exactly.

KomaMRI replays [static cardiac anatomy](CARDIAC_PHANTOM.md) with
scanner-specific T1/T2, fat and off-resonance. GRAPPA-style, SENSE, CAIPI,
compressed-sensing and variable-density sampling use the fixed receive model and
BART PICS L2 SENSE (λ=0.001, 50 iterations). Remaining families use BART FFT.
Images are compared against an independently encoded
full-grid target holding finite-RF tissue response at the central echo, including
intrinsic fat. The RF-only phase oracle adds declared B0; the submitted sampling
mask, receiver errors, readout-induced fat displacement and blur cannot define the target.

Declared-physics image limits:

| Metric | Limit |
|---|---:|
| Complex NRMSE | ≤ 0.45 |
| Magnitude NRMSE | ≤ 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal fraction | ≥ 0.5 |
| Contrast retention | ≥ 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Ideal-control imaging is diagnostic. Reconstruction parameters and receive
encoding stay fixed. No fitted gain, phase, registration or B0-aware correction
is introduced for a candidate.

Spatial quadrature and time-step refinement account for reconstruction-amplified
integration error. Unresolved numerical results remain explicit. Passing sequences
return `passed`; structural failures fail. Final submissions also undergo SAR evaluation
(quadrature/CP body coil; formula in [S2](S2_EVALUATION.md#sar-and-validation-tools)).

Artifacts record sampling checks, receive/reconstruction policy, image arrays,
metrics, submitted-byte hashes and numerical provenance. Image limits are in YAML.
