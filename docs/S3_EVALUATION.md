# S3 evaluation

Contract: [`s3@2`](../benchmark/evaluators/s3.yaml). Qualified on all three scanner profiles.

S3 covers parallel imaging, compressed sensing, variable density, partial
Fourier, reduced FOV and elliptical k-space. Waveform checks
verify acquired sampling masks, calibration regions, support and spatial
selection.

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
return `passed`; structural failures fail. Final submissions also undergo SAR evaluation.

Artifacts record sampling checks, receive/reconstruction policy, image arrays,
metrics, submitted-byte hashes and numerical provenance. Image limits are in YAML.
