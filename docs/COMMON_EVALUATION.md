# Common evaluation rules (S1–S4)

These rules apply to every S1, S2, S3 and S4 task; each level's document adds
the rules for its families, and any task-specific rule applies as well. The AI
excitation task (`RF_EXCITATION_EVALUATION.md`) uses only the Outcomes and
Pulseq preflight sections.

## Outcomes

Outcomes are `passed` or `failed`. A submission passes only if every preflight
check, every required waveform check, every required probe and every image
limit passes. Grading has no time limit. A verdict that cannot be resolved at
the maximum spatial sampling (`spatial_max_samples` in `case.json`) fails
(`failure_kind=unresolved_at_sample_cap`). Temporal refinement is also capped:
the image replay and the phase oracle halve their timesteps at most 6 times
(gradient step 1 ms → 15.6 µs, RF step 20 µs → 0.31 µs, both divided by
2^`temporal_refinement`); a replay still above its 0.01 temporal gate there fails
(`failure_kind=unresolved_at_timestep_cap`). References need at most 2 halvings.

SAR is evaluated only when you submit; a submission that exceeds the scanner's
SAR limit fails, and the feedback will say so.

## Pulseq preflight

- Raster definitions in the file (gradient, RF, ADC, block) must equal the
  hardware rasters (relative tolerance 1e-9).
- Pulseq timing checks pass with the hardware RF dead/ringdown and ADC dead
  times. Total duration is > 0 and ≤ `duration_s`.
- RF and ADC events are present (AI excitation: RF present, no ADC). All RF
  samples, times, phases and frequencies, ADC timing, phases and modulation,
  and gradient samples are finite.
- Every RF and ADC effective frequency offset,
  |freq_offset + freq_ppm·1e-6·γ·B0|, is ≤ 1 MHz (`frequency_offset`).
- Peak B1 and each gradient axis's peak amplitude and slew rate (per axis,
  not the vector sum) stay within the hardware limits.
- Gradient continuity, tolerance 1e-3 × maximum gradient: on every axis the
  value at the end of each block equals the value at the start of the next;
  the first block starts and the last block ends at zero; a gradient that
  starts after a delay, or ends before its block ends, must start/end at zero;
  no step inside a waveform exceeds the slew limit.
- `soft_delay` and `rf_shim` extensions are rejected.
- The file must be Pulseq 1.4 or 1.5 and readable by the native (KomaMRI) Pulseq reader: `[VERSION]` is
  the three lines `major`, `minor`, `revision`; `[BLOCKS]` rows are eight unsigned integers whose block ids run 1..N in file order; RF, gradient,
  trapezoid and ADC rows have exactly the version's field count, with integer ids, shape ids and ADC
  sample counts; tables are contiguous (nothing after the line that ends them); each shape's
  `num_samples` matches its data and its samples end with a blank line (not end of file); `Nx`/`Ny`/`Nz` definitions are integers. Every integer is plain ASCII
  decimal and fits a signed 64-bit value. Event ids are unique and dense from 1 (gradient and
  trapezoid ids share one sequence; RF, ADC and extension lists each have their own), including
  unused entries; every RF/gradient entry's shapes exist (RF phase has the magnitude's length);
  blocks reference existing events; total decoded RF/gradient samples are at most 2^24.
- Fields read by the native replay but derived differently by inspection must agree: a Pulseq 1.5 RF
  `center` lies within its waveform (default raster: 0 to N·raster; time-shaped: first to last time
  sample, ± half a raster); RF and ADC `freq_ppm`/`phase_ppm` are 0 (use absolute `freq`/`phase`);
  the RF use code is one of `e r i p s o u`; ADCs have no phase-modulation shape; the `ROTATIONS`
  extension is rejected. Hardware dead/ringdown times always come from the task's hardware profile,
  not from `[DEFINITIONS]`.
- A nonzero trapezoid needs rise > 0 and fall > 0. With hardware limits (`/check`, grading), every
  gradient event in the file, used or not, is also checked from its own fields: trapezoid amplitude and
  amplitude/rise, amplitude/fall against max slew; arbitrary gradients' samples × amplitude (and 1.5
  `first`/`last`) against max gradient, and their sample-to-sample slew (`gradient_events`).
- Signed fields cannot offset each other: ADC sample counts are ≥ 0 with dwell > 0 and delay ≥ 0;
  RF and gradient delays and trapezoid rise/flat/fall/delay are ≥ 0; rasters in `[DEFINITIONS]` are
  > 0; shape ids and `num_samples` are ≥ 1; explicit time shapes start at ≥ 0 and never decrease. The half-raster time-shape id −1 is not
  supported (PyPulseq's own deduplicating read/write fails on it); use the default raster (0) or an
  explicit time shape.
- Size limits (counted over every block reference, before any other parsing): at most 2^16 = 65,536
  samples per ADC event, 2^22 = 4,194,304 acquired ADC samples in total, 2^18 = 262,144 blocks, and
  2^24 RF/gradient shape samples summed over all blocks. These are 5–7× the largest reference
  sequences (8,800-sample spiral readout; 827,392 samples and 39,552 blocks for stack-of-stars).
- These native-reader rules run before any other parsing. Extension rows must also be native-readable: `[EXTENSIONS]` rows are four
  integers with declared types and existing specification/next ids; `TRIGGERS` rows are five integers
  (id, type, channel, delay µs, duration µs) with delay and duration ≥ 0; `LABELSET`/`LABELINC` rows are
  `id value label`; `ROTATIONS` rows are an id and four finite numbers; `next_id` links have no cycles. Rows must be contiguous (no blank
  or comment line inside a table) and carry no inline comments.

`./bin/lint` runs these checks except peak B1, gradient amplitude and slew (including
the finite-interval slew of arbitrary-gradient endpoints, `gradient_endpoint_slew`; lint still rejects
instantaneous steps as `gradient_continuity`);
`./bin/check` adds those. Neither evaluates SAR, waveform identity, probes or
images.

## RF events

Checks use the waveforms, never labels, names or file metadata.

- An RF event whose samples are all zero is ignored: it is not an excitation,
  does not reset k-space and does not start a TR.
- **Flip** is the on-resonance rotation of longitudinal magnetisation by the
  complex RF envelope alone (no gradients, no frequency offset). A requested
  flip or a phase-modulated pulse's integrated amplitude is not its rotation.
- **Classification.** Flip ≥ 135° is a refocusing pulse. Every other nonzero
  RF event, including preparation, saturation and dummy pulses, is an
  excitation.
- **Centre** is the |B1|-weighted time centroid of the envelope.
- **Effective RF phase** at the centre is the event's phase offset, plus the
  phase of a constant-phase complex envelope (the phase of its net area), plus
  the frequency offset's phase accrued from pulse start to centre.
  Phase-modulated envelopes contribute only the event phase.

Structural RF quadrature (used for flip, centre and classification) subdivides
explicit complex-linear RF segments into steps of at most 0.25 µs; constant
segments and default-raster holds count one interval per hold. Each distinct
RF envelope may need at most 4,000,000 intervals and all distinct envelopes
together at most 8,000,000; otherwise the structure fails (`rf_quadrature`).
The largest reference uses 302,228.

## k-space and ADC timing

- ADC sample i (zero-based) is at `block start + delay + (i + 0.5)·dwell`.
- Every ADC event needs at least 2 samples (`readout_samples`); a zero-sample
  ADC fails it.
- **Readout bandwidth (S1–S3, `readout_bandwidth`).** Every imaging ADC needs
  bandwidth per pixel `1/(N·dwell) ≥ 500 Hz`, i.e. `dwell ≤ 1/(500 Hz·N)`
  (relative tolerance 1e-9), where N is the in-plane matrix size along the
  readout (the in-plane axis of largest k travel in that ADC). At N = 128 the
  limit is dwell ≤ 15.625 µs. The check reports the measured minimum
  bandwidth per pixel.
- k is the gradient integral since the centre of the most recent excitation.
  A refocusing pulse negates the k accumulated up to its centre. (`rfov_epi`
  starts imaging k at the end of its RF block; see its document.)
- k-space positions in "samples" are k·FOV per axis. Grid index i
  (0…N−1) on an axis maps to `k = (i − N/2 + offset)/FOV`, so i = N/2 is
  k = 0 when offset = 0.
- **Cartesian lattice** (S1, S2 EPI/TSE/GRASE, S3, S4): every sample lies within
  0.01 sample of one lattice with spacing 1/FOV. The per-axis lattice offset
  is the circular mean of the fractional sample positions; S1, S2 and S4 then
  shift the origin by whole samples so the acquired extent is centred on the
  grid, and require |offset| ≤ 0.51 sample, or offset +1 (within 0.01): a band
  −N/2+1…+N/2, e.g. a negated textbook phase-encode table, is the same
  N-point Nyquist grid with the +N/2 line aliased to the −N/2 bin (S3 keeps
  the fixed centred grid; see its document). Every index must lie inside the
  N-point grid. In 2D, |kz| ≤ 0.01/min(FOV) at every sample.
- **Cartesian readout:** each ADC event is one line along the axis with the
  largest k range. It has exactly N samples (N = matrix size on that axis),
  consecutive samples step by 1 ± 0.01 sample on that axis and by ≤ 0.01
  sample on the other axes. No oversampling.

## Excitation assignment, echo, TE and TR

- Each ADC event belongs to the latest excitation whose centre is at or before
  its first sample. That first sample must be at or after the excitation's
  end, no RF event may overlap the ADC, and its last sample must precede the
  next excitation centre (or the end of the sequence).
- **Echo in ADC** (S1, S3, S4; S2 and `rfov_epi` state their own rules): on
  the readout axis, linear interpolation between the first
  and last samples' k must place k = 0 at a fraction in [0, 1] of the ADC
  span. There is no further tolerance, and a one-sample ADC fails. That
  interpolated time is the echo time of the line.
- **TE** runs from the assigned excitation's centre to the echo; it must be
  positive and within the task's target ± tolerance (or ≤ max), with 1 ns
  numerical slack. Families with non-Cartesian TE rules are listed in the S2
  document.
- **TR** is the interval between the centres of consecutive excitations that
  acquire data. Excitations without ADC are skipped. A terminal recovery tail
  counts only toward duration.

## Receiver phase and targets

The independent target uses the excitation FID convention. After excitation
with effective RF phase p the matching ADC phase is p. A refocusing pulse with
phase q maps the previous matching phase a to `2q − a + π` (modulo 2π), so
90x-180y (CPMG) echoes use ADC phase 0 and 90x-180x echoes use π. The grader
does not fit or repair receiver phase, gain, B0 or registration. A wrong
receiver phase fails complex NRMSE and phase RMS even with a correct
magnitude image. Targets and probes retain the complete RF history from
equilibrium, including transients and preparation pulses.

## Image metrics

Let `R` be the nominal complex target (RF-only finite-RF tissue response at
the central echo, including intrinsic fat), `P` the phase oracle's unit
phasor and `I` the reconstructed image. Support is the set of voxels that
contain any phantom material.

- **Phase oracle.** The RF-only tissue response is replayed per voxel twice:
  without and with the declared physics (B0). A voxel is oracle-valid when its
  no-B0 signal exceeds 1e-6 of the peak and its declared-physics signal is at
  least 1% of its own no-B0 signal. Inside valid voxels `P` is the phase of
  their ratio. The oracle must cover ≥ 0.5 of the target energy (sum of |R|²
  over valid voxels / total) and resolve temporally (see below); otherwise
  the submission fails (`phase_reference_not_resolved`).
- **Complex NRMSE** = sqrt(E / ‖R‖²) over the full image. E sums |I − R·P|²
  over voxels that are oracle-valid or outside the support, plus
  (|I| − |R|)² over support voxels that are not oracle-valid (their phase is
  undefined, but their magnitude error still counts). The denominator is the
  full target energy ‖R‖², so invalid voxels never shrink it.
- **Magnitude NRMSE** = ‖|I| − |R|‖ / ‖R‖ over all voxels.
- **Phase RMS** = sqrt(Σ w·angle(I·conj(R·P))² / Σ w), weights w = |R|², over
  voxels that are oracle-valid, inside the support and have |R| ≥ 0.05·max|R|.
  Angles are wrapped to (−π, π].
- **Tissue ROIs.** Per material, voxel occupancy is the material's volume
  fraction in the voxel. The ROI keeps voxels whose occupancy and that of both
  neighbours along every image axis exceed 0.95. If that set is empty, the ROI
  is all voxels with occupancy > 0.75. A tissue counts only if its target ROI
  mean |R| exceeds 5% of max|R|.
- **Minimum tissue signal** = minimum over counted tissues of
  mean|I| / mean|R| on the ROI.
- **Contrast retention** = minimum, over counted tissue pairs (a, b) with
  |target (a−b)/(a+b)| ≥ 0.1, of the signed ratio of image contrast
  (a−b)/(a+b) to target contrast, using ROI means of |I| and |R|.
- **Outside energy** = Σ|I|² outside the support / Σ|I|² over all voxels.

## Numerical acceptance

- **Temporal.** Bloch steps start at 1 ms (gradient) and 20 µs (RF), divided
  by 2^`temporal_refinement` when `case.json` sets it, and halve
  until the NRMSE between a replay and its half-step replay is ≤ 0.01. The
  gate is the larger of the raw-signal NRMSE and the image NRMSE, where the
  image NRMSE is ‖recon(half-step − step)‖ / ‖R‖ through the task's fixed
  reconstruction: inverse NUFFT for S2 radial/UTE/spiral/stack-of-stars/
  PROPELLER, the SENSE/GRAPPA policy for accelerated EPI and S3, and the
  Cartesian FFT (averaging any repeated grid points) for S1, S2 EPI/TSE/GRASE
  and S4 frames. Both NRMSEs are always computed: the raw NRMSE divides by the
  coarse replay's signal norm, the image NRMSE by the independent target norm,
  so neither implies the other. The phase oracle halves its steps
  until the |R|²-weighted RMS chord |P_step − P_half| is ≤ 0.01; a voxel whose
  validity changes counts as chord 2.
- **Spatial.** Each image uses `spatial_replicates` independent randomized
  quadratures of `spatial_samples` spins. Scores carry approximate 99%
  Student-t delete-one jackknife intervals. If the interval decides the
  verdict, it is final: pass if the whole interval is inside the limit, fail if
  the whole interval is outside. Otherwise, once every interval half-width is
  within `image_precision` (`case.json`), the point estimate is compared with
  the limit (limits are inclusive). Magnitude NRMSE and phase RMS use the
  intervals of their squared statistics, and a negative bias-corrected squared
  estimate never passes on its point value. Otherwise the sample count doubles
  up to `spatial_max_samples`; still unresolved there fails.
