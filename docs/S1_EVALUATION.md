# S1 evaluation

Contract: [`s1@4`](../benchmark/evaluators/s1.yaml). Requalification on GPU pending after the SAR (quadrature) and receiver-phase/role changes.

S1 covers Cartesian GRE, spin echo and bSSFP. Waveform-derived
checks verify sequence identity, physical FOV/matrix coverage, echo formation,
TE/TR, spoiling, refocusing and gradient balance as required by the task.

Replay uses finite-RF Bloch simulation in KomaMRI and BART Cartesian FFT
reconstruction. Every case uses [static cardiac anatomy](CARDIAC_PHANTOM.md)
with scanner-specific T1/T2, intrinsic fat shift and declared off-resonance.
Independent full-grid targets hold the finite-RF tissue response at the central
echo, including intrinsic fat. The RF-only phase oracle adds declared B0;
submitted receiver errors, readout-induced fat displacement and blur remain errors.

Receiver phase convention (S1–S3, shared): targets use the excitation's FID
convention for every echo. After excitation with RF phase p the matching ADC
phase is p. A refocusing pulse with RF phase q maps the echo to ADC phase
2q - (previous phase) + pi. So 90x-180y (CPMG) echoes use ADC phase 0, and
90x-180x echoes use pi. A wrong receiver phase (e.g. pi off) fails complex
NRMSE and phase RMS even when the magnitude image is correct.

Declared-physics image limits:

| Metric | Limit |
|---|---:|
| Complex NRMSE | ≤ 0.45 |
| Magnitude NRMSE | ≤ 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal fraction | ≥ 0.5 |
| Contrast retention | ≥ 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Ideal-control images are diagnostic. Fixed mechanism probes separately check
RF coherence, refocusing and weighting. No fitted gain, phase, registration or
B0-aware reconstruction is permitted.

Spatial quadrature and temporal integration refine until the numerical verdict
resolves. Unresolved numerical results remain explicit. Passing sequences return
`passed`; structural and hardware failures still fail. SAR (quadrature/CP body coil;
formula in [S2](S2_EVALUATION.md#sar-and-validation-tools)) is evaluated on final submissions.

Artifacts contain submitted-byte hashes, waveform checks, image arrays,
metrics and numerical provenance. Image limits are defined in the versioned YAML.
