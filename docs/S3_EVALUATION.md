# S3 evaluation

S3 checks the sampling mask, spatial selection and achieved image. Numeric
requirements, scanner-specific tissues and fields are supplied in `task.yaml`,
`case.json` and `inputs/`. Preflight, SAR, outcomes, RF classification,
k-space and ADC timing, lattice tolerances, excitation assignment, echo-in-ADC,
TE/TR, receiver phase, image-metric definitions and numerical acceptance are
defined in [COMMON_EVALUATION.md](COMMON_EVALUATION.md) and apply here.

## Sampling identity

Sampling is waveform-derived; Pulseq labels are ignored. Imaging k follows
the common rules; `rfov_epi` instead starts at the end of its RF block (see
`RFOV_EPI_EVALUATION.md`).

- Every ADC event is one straight readout along x with exactly Nx samples at
  unit kx steps (±0.01 sample); y and z each vary by at most 0.01 sample over
  the whole line. Either polarity is allowed and every readout must bracket
  kx=0. y (and z in 3D) are phase-encode axes.
- S3 uses the fixed centred grid: the lattice offset is the circular mean of
  the fractional sample positions, in (−0.5, 0.5], and is never recentred on
  the acquired extent. Lattice residual ≤0.01 sample; 2D |kz| ≤ 0.01/min(FOV).
  Within 0.01 sample of ±0.5 the two representatives index the same nodes one
  grid point apart; the one that keeps every sample on the grid is used (then
  the one matching more prescribed nodes, then the circular mean itself).
  A set reaching +N/2 is the same N-point Nyquist grid when wrapping +N/2 to
  the −N/2 bin equals shifting the whole set down by s samples (e.g. ky
  −63…+64, or every second line −62…+64): the offset then becomes +s and
  reconstruction is exact.
- Every ADC event must have a positive sample count.
- Grid indices use the common convention `k = (i - N/2 + offset)/FOV`; with
  zero offset and N=128, i=0 is k=−64/FOV, i=64 is k=0 and i=127 is k=+63/FOV.
- The acquired node set must equal the prescribed mask exactly: no missing,
  extra, duplicate or out-of-grid nodes.

Mask semantics per task:

| Task | Prescribed mask (zero-based indices) |
|---|---|
| `sense_sampling` | every second ky (even iy, R=2) plus ACS iy 60..67 added on top: 68 lines |
| `grappa_sampling` | every third ky (iy ≡ 0 mod 3, R=3) plus ACS iy 52..75 added on top: 59 lines |
| `caipirinha` | `acceleration: 4` = Ry×Rz = 2×2. Retain even iz; in retained partition j=iz/2 retain iy ≡ j (mod 2). Calibration: all (iy, iz) pairs with iy 60..67 and iz 12..19, including odd iz |
| `compressed_sensing`, `variable_density` | nominal R=3 is a total budget of ceil(128/3)=43 lines, the 8 ACS lines (iy 60..67) included; listed explicitly in `task.yaml` |
| `partial_fourier` | iy 32..127 (k = −32/FOV … +63/FOV) |
| `elliptical_kspace` | ((iy−64)/64)² + ((iz−16)/16)² ≤ 1, boundary included |
| `reduced_fov`, `rfov_epi` | even iy (every second line), full 128-point x readouts |

## Timing and RF classification

Requested TE/TR bounds (common definitions) apply to every acquired echo
(`rfov_epi` uses its central-ky echo as effective TE). A requested flip has ±1°
tolerance. All families except `reduced_fov` must contain no refocusers
(≥135°).

Gradient-echo trains are allowed: one excitation may acquire several readout
lines, provided each echo meets the TE bounds and the spoiling and image checks
below.

`reduced_fov` requires exactly one refocuser between each readout's excitation
and its echo, centred at the excitation/echo midpoint within ±20 µs (twice the larger of the
block and RF rasters).

## Spoiling probe

`sense_sampling`, `grappa_sampling`, `caipirinha`, `compressed_sensing`,
`variable_density`, `partial_fourier` and `elliptical_kspace` are spoiled GRE.
A water probe (PD=1, T1=0.8 s, T2=0.08 s) at the readout-centre sample of every
line must give mean finite-RF central-echo signal ≥0.01 PD, and its Bloch replay over
1021- and 2053-point Halton quadratures of one image voxel must have complex
error ≤0.20 versus perfect spoiling. The voxel is fov/matrix on each axis:
1.5625 × 1.5625 mm in 2D (no z extent) and 1.5625 × 1.5625 × 6.25 mm for the
3D tasks, so spoiler moments on any axis count by their dephasing across that
axis's voxel. The probe uses RF-derived receiver phase to isolate
spoiling; achieved images retain the submitted ADC phase.

## Spatial selection (`reduced_fov`)

The water probe is replayed at the readout-centre samples of the first, central
and last lines, without B0. Mean interior magnitude must be ≥0.01 PD and
maximum exterior / mean interior magnitude ≤0.10 at each of those echoes.
Probe voxels are centred at x=z=0: nine interior y positions span −35..35 mm;
nine positions per exterior side span ±65..±96 mm. Each probe is one image
voxel (1.5625 × 1.5625 mm in x, y; the 2D object has no z extent). Its complex
signal is averaged over a uniform n × n isochromat grid after removing the
nominal encoding phase of that sample, then scored by magnitude. Two coprime
grids (24 and 31 points per axis) are evaluated; if their leakage ratios differ
by more than 0.01 (or interior means by more than 1%), the pair is refined to
48/61 and then 96/127. Each grid aliases dephasing only at multiples of its own
point count, so agreement certifies the voxel integral; the worse of the two
values is scored, and an unresolved pair fails selection.
Exterior suppression may therefore come from y-selective excitation, or from a
y-selective refocuser whose unrefocused exterior signal is dephased by in-plane
(x or y) crushers within the voxel, as in the textbook inner-volume design.
Crushers on z do not dephase a 2D object. `rfov_epi` uses its own probes,
including declared B0; see `RFOV_EPI_EVALUATION.md`.

## Image replay and reconstruction

KomaMRI replays the supplied static cardiac anatomy with scanner-specific T1/T2,
fat and declared off-resonance. Reconstruction is fixed per task:

| Task | Receive model | Reconstruction |
|---|---|---|
| `sense_sampling`, `compressed_sensing` | 4 trigonometric coils (2D) | CG-SENSE |
| `grappa_sampling` | 4 trigonometric coils (2D) | ESPIRiT-calibrated CG-SENSE |
| `caipirinha` | 16 trigonometric coils (3D) | CG-SENSE |
| `variable_density` | 8 Gaussian y coils | CG-SENSE |
| `partial_fourier`, `elliptical_kspace` | single uniform coil | zero-filled FFT |
| `reduced_fov`, `rfov_epi` | single uniform coil | FFT; central half of phase FOV kept |

CG-SENSE is BART PICS with an L2 penalty only (λ=0.001, 50 iterations) and the
known receive maps. `grappa_sampling` is auto-calibrated instead: one-map
ESPIRiT sensitivities (BART `ecalib -m 1`, 6×6 kernel, calibration region at
most 24×24, BART default threshold and crop) are estimated from the fully
sampled central k-space block of your own data, and the same PICS solve uses
those maps. Too few or misplaced calibration lines therefore degrade the image;
no GRAPPA kernel is fitted and the known maps are not used. Its time-step
convergence check compares the ESPIRiT-calibrated reconstructions of the coarse
and half-step acquisitions (normalized by the target); the known-map linear
projection is also computed and the larger value is used. `compressed_sensing` uses no sparsity prior. Partial
Fourier uses no homodyne or phase-constrained recovery.

Trigonometric maps are products over phase axes of
`(1 + 0.8*cos(2*pi*r/FOV - theta))/sqrt(4*(1 + 0.8^2/2))`, with each theta
independently in `{0, pi/2, pi, 3*pi/2}` (four coils in 2D, sixteen in 3D), with
unit summed squared sensitivity. Gaussian maps have centres
`(j+0.5)*FOVy/8 - FOVy/2`, j=0..7, sigma `FOVy/16`, normalized to unit summed
squared sensitivity. Maps are fixed in object coordinates.

## Image metrics

Images are compared against an independently encoded full-grid target holding
finite-RF tissue response at the central echo, including intrinsic fat. The
RF-only phase oracle adds declared B0; submitted masks, receiver errors,
readout-induced fat displacement and blur cannot define the target.
Receiver phase follows the common FID convention.

Declared-physics image limits are set per task:

| Task | Complex NRMSE ≤ | Magnitude NRMSE ≤ | Phase RMS ≤ (rad) | Tissue signal ≥ | Contrast retention ≥ | Outside energy ≤ |
|---|---:|---:|---:|---:|---:|---:|
| `sense_sampling` | 0.23 | 0.21 | 0.17 | 0.66 | 0.66 | 0.05 |
| `grappa_sampling` | 0.23 | 0.21 | 0.18 | 0.66 | 0.66 | 0.05 |
| `caipirinha` | 0.23 | 0.21 | 0.13 | 0.66 | 0.66 | 0.05 |
| `compressed_sensing` | 0.23 | 0.20 | 0.17 | 0.66 | 0.66 | 0.05 |
| `variable_density` | 0.23 | 0.21 | 0.17 | 0.66 | 0.66 | 0.05 |
| `partial_fourier` | 0.23 | 0.20 | 0.16 | 0.66 | 0.66 | 0.05 |
| `elliptical_kspace` | 0.35 | 0.30 | 0.21 | 0.62 | 0.71 | 0.05 |
| `reduced_fov` | 0.34 | 0.33 | 0.25 | 0.59 | — | 0.05 |
| `rfov_epi` | 0.37 | 0.35 | 0.27 | 0.63 | — | 0.05 |

Each limit is the larger of 1.5× the worst reference value and 1.2× the worst
value of a held-out ensemble of valid alternative designs (readout bandwidth
from the 500 Hz/pixel floor to about 1560 Hz/pixel, forward/reverse/centric
ordering, block or sinc excitation, alternative RF/gradient spoiling, z
spoiling in 3D, inner-volume reduced FOV) across the three hardware profiles,
with a 0.05 floor for outside energy, and never looser than complex 0.45 /
magnitude 0.35 / phase 0.35 rad / tissue 0.5 / contrast 0.5. A +0.6 rad receiver
phase error, missing spoiling or missing calibration fail these limits. The
exact sampling mask remains the primary sampling gate.

Spatial-selection tasks (`reduced_fov`, `rfov_epi`) assess selected-region signal
and rejected exterior without requiring a particular fat/water contrast ordering.
All other image and selection checks apply. Ideal images are diagnostic.
Reconstruction and receive encoding stay fixed; no gain, phase, registration or
B0-aware correction is fitted.
