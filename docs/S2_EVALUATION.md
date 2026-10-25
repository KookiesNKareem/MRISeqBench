# S2 evaluation

Numerical precision settings are supplied in `case.json`. Task dimensions,
counts, timing and hardware limits come from `task.yaml`.

Preflight, SAR, RF classification, k-space and ADC timing, excitation
assignment, Cartesian lattice tolerances, TR, receiver phase, image-metric
definitions and numerical acceptance are defined in
[COMMON_EVALUATION.md](COMMON_EVALUATION.md) and apply here. k-space positions
below are in reciprocal-FOV units ("samples"); with Nx=128 the full-extent bound
is `Nx/2 − 0.75 = 63.25`. Radial, spiral, PROPELLER and stack-of-stars must
cover the requested matrix, angles and partitions; reconstructing onto a larger
grid does not supply missing data.

## TE rules

- **TE** runs from the excitation's magnitude-weighted RF centre to the echo:
  - Cartesian (EPI, TSE, GRASE): the time each readout's readout-axis k
    crosses 0, linearly interpolated between its first and last sample. The prescribed TE
    applies to the readout containing the sample nearest k=0 (effective TE).
  - PROPELLER: the same interpolated crossing along each line; effective TE
    applies as for Cartesian.
  - Radial, UTE, spiral, stack-of-stars: the time of the acquired sample nearest
    k=0 (in kx, ky). The prescribed TE applies to **every** readout.
  - Every readout must have a positive TE.
- **Gradient-echo families** (EPI, radial, UTE, spiral, stack-of-stars) may
  contain no ≥135° RF anywhere; acquired excitations must lie in (0.1°, 135°).

## Cartesian families (EPI, TSE, GRASE)

Every ADC event needs at least 2 samples (`readout_samples`); shorter
readouts fail the structure before any trajectory check.

- Each ADC is one line of exactly Nx=128 samples along one axis, with
  |Δk| = 1 ± 0.01 per sample and ≤0.01 motion on the other axes. No
  oversampling, ramp sampling or multi-echo ADCs. EPI and GRASE must read
  along x (`readout_axis`: |Δkx| = 1 ± 0.01 per sample, ≤0.01 ky/kz motion
  within each readout) and phase-encode along y; readout polarity is the sign
  of x travel.
- All samples lie on the common Cartesian lattice (common rules: residual
  ≤0.01, offset ≤0.51 sample or +1, 2D |kz| ≤ 0.01/min(FOV)). epi_multishot, tse_t2w and grase must
  acquire every node of the 128 × 128 grid (index `n = ky + 64`, n = 0…127).
- **epi_single_shot:** exactly one acquiring excitation. Other RF
  (e.g. CHESS fat saturation followed by a spoiler) is allowed only if it ends
  before that excitation starts; no RF may follow it, and a second acquired
  excitation fails. Coherence left by preparation RF is judged physically: the
  spoiling probe and the image replay include it while the target spoils it
  ideally, so an unspoiled preparation fails. One shot, alternating
  polarity, ky monotonic in one direction. Index `n ↔ ky = (n − 64)/FOVy`; the
  set of acquired ky indices must equal the prescribed set exactly (n = 1, 4,
  …, 127), with no repeated sample.
- **epi_multishot:** 16 shots; within each shot ≥2 readouts with alternating
  polarity, and ky changes by exactly ±16 lines (±0.02) between readouts,
  monotonic in one direction.
- **grase:** polarity alternates across all eight readouts of a shot,
  including across refocusing pulses (see refocused trains).

## Refocused trains (TSE, PROPELLER, GRASE)

- Excitation rotations 90 ± 2°; every refocusing rotation 180 ± 2°.
- Shot count = acquired excitations; each shot has exactly the declared echo
  train length (≥2) and every echo has ≥1 refocusing pulse before it.
- Spin-echo times follow CPMG timing: starting from the excitation centre, each
  refocusing centre t_r maps the echo time t to `2·t_r − t`. Tolerance
  τ = max(20 µs, half the ADC dwell).
  - TSE/PROPELLER: each echo's k-centre time is within τ of its spin-echo time.
  - GRASE: per shot, echoes grouped by preceding refocusing count form exactly
    4 groups of exactly 2; in each group one echo is at or before and one at or
    after the spin echo (within τ).
- Echo spacing (consecutive echo k-centre times in a shot) is within
  0.05 ms of the prescribed value.

## Non-Cartesian families

| Task | Requirements |
|---|---|
| radial_readout | Each readout one straight spoke (perpendicular deviation ≤0.05), steps ≤1.05, **two-sided through k=0**: endpoints below −63.25 and above +63.25. Spoke angle = first→last sample direction, taken modulo π. Exactly 202 spokes; one readout per excitation. |
| ute_radial | Straight centre-out spokes (deviation ≤0.05, steps ≤1.05): first sample within 0.6 of k=0, \|k\| nondecreasing (≥−0.05 per step), reaching ≥63.25. Angles modulo 2π; exactly 404 spokes; one readout per excitation. TE ≤1 ms and within target for every spoke. Excitation is nonselective: every gradient axis is zero (≤1e-6 Hz/m) throughout each excitation pulse. |
| stack_of_stars | Each spoke as in radial_readout (two-sided, extent ±63.25). kz is constant within each spoke (≤0.01) and on an integer partition (≤0.01); the partition set is exactly −16…15. In **each partition**: 202 spokes with angles modulo π. One readout per excitation. |
| spiral_gre | Per interleaf: ≥128 samples, steps ≤1.1, first sample within 0.6 of k=0, last sample radius ≥63.25, radius nondecreasing (≥−0.05) on ≥95% of steps. Excluding radius ≤0.5: winding ≥1 turn and 8 ± 0.5 turns; a linear radius-versus-unwrapped-angle fit has RMS ≤0.1·max radius. Exactly 16 interleaves; start angles (first sample beyond radius 0.5) modulo 2π; one readout per excitation. |
| propeller_tse | Each shot is one blade of exactly 8 lines. Each line: transverse variation ≤0.05, spans ≤−63.25 to ≥+63.25, sample steps 1 ± 0.02. Sorted line offsets are spaced 1 ± 0.05, with \|mean offset\| ≤0.6. 64 blade orientations modulo π. |

**Angular gaps:** after sorting angles modulo the period (π or 2π), every
circular gap must be at most `1.6·period/count`. Use uniform spacing over the
period: golden-angle or doubled-period schemes (e.g. 64 blades over 2π) fail.

**Coverage:** every integer grid node within radius `min(Nx,Ny)/2 − 1 = 63`
in kx–ky (all kz for stack-of-stars) must be within 1.5 samples of an acquired
sample; 2D tasks also need |kz| ≤ 0.05 throughout.

## Mechanism checks

These diagnostics use PD=1, T1=800 ms, T2=80 ms.

- Mean ideal signal at the readout centre samples ≥0.01 PD.
- Gradient-echo families: spoiling error ≤0.20 against ideal spoiling on two
  quadratures of 1021/2053 deterministic isochromats in one voxel with
  per-axis widths FOV/matrix (2D: zero z extent; stack-of-stars:
  1.5625 × 1.5625 × 6.25 mm). This sampling volume is separate from the
  phantom.
- Refocused trains: with gradients zeroed, each shot's RF must retain ≥90% of
  its zero-offset signal at every CPMG spin-echo time over 65 offsets from −50
  to +50 Hz.
- tse_t2w: at the image reference sample, a 10% T2 increase must change signal
  magnitude more than a 10% T1 increase.

## Image acceptance

CUDA finite-RF Koma replay retains RF history, scanner-specific tissue properties,
fat and declared B0 in the [static cardiac anatomy](CARDIAC_PHANTOM.md).
Reconstruction uses fixed BART FFT or GPU inverse NUFFT (L2=0.001, 50 iterations).
Accelerated EPI uses the four-coil receive model and standard BART CG-SENSE
with the same regularization and iteration count. Its four real receive maps are
`s_j(y) = [1 + 0.8 cos(2πy/FOVy − jπ/2)] / sqrt(4(1+0.8²/2))`, `j=0,1,2,3`,
with unit summed squared sensitivity and y in metres. No B0-aware correction is used.

The independent full-grid target holds each tissue at one complex response:

- **Cartesian (EPI, TSE, GRASE):** the response at the echo that encodes the
  k-space origin: the single earliest acquired sample within 1e-3 of the
  global minimum k radius (for a centred full or uniformly undersampled grid,
  the ky = 0 echo). Its magnetisation history (lines before it, preparation
  RF) sets the target.
- **Radial, UTE, spiral, stack-of-stars, PROPELLER:** the mean complex response
  over every readout whose nearest sample lies within 0.75 samples of the
  in-plane origin (kx, ky), taken at that nearest sample. Every spoke,
  interleaf and stack-of-stars readout (all partitions) qualifies, as does the
  central line of each PROPELLER blade. Each of these readouts carries the
  image's low spatial frequencies, so the image contrast is their mix; the
  first readout's transient does not set the target. Readout-to-readout
  variation (approach to steady state) remains an image error through
  streaks and blur.

For non-Cartesian readouts, tied samples (within 1e-3 of that readout's own
minimum) resolve to the one nearest the readout's temporal centroid. The B0
phase oracle replays exactly the same samples (one RF context per
contributing readout; the single central echo for Cartesian), so the target
response and the phase reference share one selection.
Readout transients, blur and displacement remain image errors.

| Every achieved image must meet | Limit |
|---|---:|
| Complex / magnitude NRMSE | ≤0.45 / ≤0.35 |
| Wrapped phase RMS | ≤0.35 rad |
| Minimum tissue signal / contrast retention | ≥0.5 / ≥0.5 |
| Outside-object energy | ≤0.25 |

No gain, receiver phase or registration is fitted; receiver phase follows the
common FID convention. Ideal-control images are diagnostic.

## Tools and outcomes

`./bin/lint` and `./bin/check` run only the common preflight; a passing
preflight or simulator call is not a pass. Saved feedback identifies failed
checks with their measured values.
