# S2 evaluation

Contract: [`s2@3`](../benchmark/evaluators/s2.yaml). Qualified on all three scanner profiles.

S2 covers EPI, radial, spiral, stack-of-stars,
PROPELLER, UTE, TSE and GRASE. Waveform checks verify actual trajectories,
physical FOV/matrix coverage, shot/echo structure and prescribed timing.

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

Spatial quadrature and time steps adapt until the numerical verdict resolves.
Unresolved numerical results remain explicit. Passing sequences return `passed`;
invalid sequences fail.
Final submissions also undergo SAR evaluation.

Saved artifacts include waveform checks, physical/ideal/target images, metrics,
submitted-byte hashes and numerical provenance. The YAML defines image bounds.
