# Cardiac S4 evaluation

The four cardiac tasks use 128×128 Cartesian bSSFP images, with four ECG segments per frame.

Preflight, SAR, outcomes, RF classification, k-space and ADC timing, image-metric
definitions and numerical acceptance are defined in
[COMMON_EVALUATION.md](COMMON_EVALUATION.md). S4 also applies these common
waveform checks to every ADC: excitation assignment (`rf_before_adc`), echo in
ADC, Cartesian lattice and readout (residual ≤0.01 sample, offset ≤0.51
sample or +1, unit steps, 2D |kz| ≤ 0.01/min(FOV)), every point of the 128×128 grid
acquired at least once over the whole sequence (`matrix`, `fov_mm`), and one
readout line per excitation.

## Triggers and timeline

- Use `physio1` input triggers in standalone blocks (no RF, gradient or ADC in that block).
  Koma resolves each trigger onto the next R wave (R waves at 0, RR, 2 RR, … using the task's RR)
  and inserts the wait as idle time. Successive triggers must land on distinct R waves.
- Trigger rows use integer microsecond delay/duration ≥ 0 (see the common preflight rules); a file the
  native reader or trigger scheduler rejects fails `physio_trigger_resolution`.
- Write Pulseq file version 1.4 or 1.5. Other versions (e.g. 1.3.x, whose block table references a
  delay library) fail `physio_trigger_resolution`, because trigger waits are inserted as duration-tick blocks.
- The trigger time is the resolved R wave. The trigger event's own `delay` field only delays
  arming (the R wave found is the first one after it); it does not delay acquisition after the R wave.
  Implement the trigger delay with ordinary blocks after the trigger block.
- `/check` runs on the raw file, where trigger waits are zero. At submission the duration limit and
  hardware checks are applied again to the resolved timeline, including every inserted trigger wait.
  RF/gradient/ADC waveform bytes are kept unchanged.
- Timing tolerance is ±2·max(block raster, RF raster) = ±20 µs. It applies to TI, T2-prep duration,
  refocuser centres, HS pulse duration, cardiac window edges and RR cadence.

## Segments and readout

- Every ADC sample must follow a trigger and lie within the cardiac window measured from the most
  recent trigger (`trigger_delay_ms` to `trigger_delay_ms + acquisition_window_ms`, currently 600–800 ms).
  Beats without ADCs (recovery beats) are allowed. The number of beats containing ADCs must equal
  frames × segments; consecutive groups of four such beats form frames 0, 1, 2, …
- Each ADC event is one Cartesian line of exactly 128 samples at unit k-space steps: no readout
  oversampling and no dummy or navigator ADCs. Each frame must acquire every grid point exactly once.
  Each segment acquires exactly 32 lines (128 ky lines / 4 segments), each complete inside its trigger
  window; which lines go in which segment and their order within a segment are free (but see TI
  under inversion tasks).
- Every ADC event must have a positive sample count; a zero-sample ADC fails the structure check.
- bSSFP: for every TR (readout RF start to next readout RF start; the last TR ends one TR after the last readout RF start), the net gradient area on all three
  axes must be ≤0.01 cycles across the FOV (x, y use the task FOV; z uses 0.2 m). RF phases alternate
  by pi; each ADC phase offset must match its excitation's effective RF phase at the pulse centre
  (event phase plus envelope phase plus the frequency offset's accrual from pulse start to centre, as
  in COMMON_EVALUATION.md) within 0.01 rad, and RF/ADC frequency offsets must match within 1e-6 Hz.
  With a nonzero carrier offset the ADC phase therefore carries 2π·Δf·(centre − start) of the RF.
- Preparation RF must lie after the previous segment's last ADC and end at least 20 µs before the
  readout train. RF pulses spaced at the train TR immediately before the first line count as part of the train.
  Conditioning pulses up to the readout flip +2° are allowed. Additional preparation RF must be followed
  by an RF-free crusher interval whose net gradient moment wraps the phase by ≥2 full cycles across
  one image voxel on x or y (|moment| × FOV/128 ≥ 2, i.e. ≥1280 cycles/m at 200 mm FOV); z moments do
  not count, since a single-slice object is not dephased through-plane. This displaces the crushed
  coherence by twice the sampled k-space width, so it is modelled as ideally dephased. Other
  unclassified RF, including RF followed by a weaker moment, fails.
  RF centres are |B1|-weighted centroids of the waveform.

## Inversion tasks

- Each segment has exactly one inversion-class RF event (nominal flip ≥170°).
  TI runs from its RF centre to the echo of the line containing the segment's sample nearest k-space centre.
  TI is measured per segment, to that segment's own line nearest k-space centre, so every segment's
  centre line must sit at the prescribed TI (±20 µs). With linear ky ordering across segments those
  lines fall at different train positions, so use the same per-segment ordering relative to each
  segment's centre line (e.g. centre-out/centric per segment) or adjust each segment's inversion delay.
- `adiabatic_inversion`: the pulse must have a sech amplitude and a tanh frequency sweep, each within
  5% RMS of the fitted shape, with a positive sweep; the shape scale is fitted (no fixed beta or bandwidth).
  The 10 ms duration is the RF waveform span. Adiabaticity is tested functionally: the pulse must
  also invert ≥90% Mz at every B1 scale 0.80, 0.85, …, 1.20 (±20% transmit error) at each B0 offset
  of the inversion assay below.
- `t1_mapping`: frame k must use `inversion_times_ms[k]` in the listed order (100, 400, 1200 ms);
  interleaving TIs across frames fails. Consecutive inversions (every segment) must be at least
  3·RR apart, centre to centre. When TI increases between frames the inversion moves earlier
  relative to its trigger, so that gap may need four RR.

## T2-preparation tasks

- Nonzero T2 preparation uses separate 90/≥178/≥178/90° RF events; tip tolerance is 2°.
  Relative phases are `0, pi/2, pi/2, pi` within 0.01 rad. Duration runs from tip-down centre to
  tip-up centre; refocuser centres lie at 1/4 and 3/4 of that interval. Zero-duration preparation has no prep RF.
- `t2_mapping`: frames are 0, 25, 50 ms in that order. Consecutive nonzero preparations must be at least
  3·RR apart (tip-down centre to tip-down centre); zero-duration segments do not count.

## Image and mechanism checks

CUDA finite-RF replay retains all prior RF, relaxation, fat and declared B0. BART FFT reconstructs each frame.
The independent target freezes RF-only tissue responses at that frame's central echo; B0 supplies its phase.
Additional spoiled preparations use ideal transverse dephasing only at their measured crusher boundaries.
Native image replay retains the actual crusher gradients; inversion and bSSFP readout coherence are preserved.
Readout transients, blur, displacement and B0 magnitude loss remain image errors. No gain, phase or registration is fitted.

| Every frame must meet | Limit |
|---|---:|
| Complex / magnitude NRMSE | ≤0.45 / ≤0.35 |
| Phase RMS | ≤0.35 rad |
| Minimum tissue signal | ≥0.5 |
| Contrast retention (non-mapping tasks) | ≥0.5 |
| Outside-object energy | ≤0.25 |

Myocardial echo signal must be ≥0.01 PD. The preparation effect is
|S_prep − S_no-prep| / PD at the myocardium and the frame's k-space-centre echo, where S is the
complex noise-free RF-only (canonical oracle) response, and S_no-prep replays the same RF history with
the frame's designated preparation pulses (inversion, or the 90/180/180/90 T2-prep block of every
segment) removed. It must be ≥0.02 PD. The requirement (and the per-pulse inversion assay below) does
not apply to a zero-duration T2-preparation frame, which has no preparation RF.
Every inversion pulse and T2-prep refocuser must invert ≥90% Mz at each of 17 B0 offsets spanning
[-520,70] Hz × field/3 T. T1-mapping inversions and T2-prep refocusers are tested at nominal B1
only; the `adiabatic_inversion` pulse is tested on the full 9 × 17 grid of B1 scales 0.80–1.20
(step 0.05) × B0 offsets. A B1 scale multiplies the RF amplitude only; the frequency sweep is unchanged.
This band spans fat and water across the declared B0 map, so T2-preparation
refocusers must invert fat as well as water.
Mapping fits only the requested relaxation time, with known PD, other relaxation time and fixed partial volumes.
The fitted signal is the ROI-mean magnitude of each frame's complex mean over spatial-sampling replicates
(complex averaging avoids magnitude bias); its numerical interval refits each leave-one-out ensemble.
Its relative error must be ≤10%, with numerical precision 0.001.
Mapping tasks assess the requested myocardial relaxation time directly; they do not require fat/myocardium contrast ordering at each preparation time. All global image error, phase, signal and preparation requirements still apply.

All common, sequence, image and application checks must pass.
