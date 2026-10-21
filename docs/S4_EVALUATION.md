# S4 evaluation

S4 also includes [2D RF excitation of AI](RF_EXCITATION_EVALUATION.md), scored by
the dedicated `rf_excitation@2` magnetization evaluator. The cardiac application
contract described below applies to its four ECG-triggered imaging tasks.

Contract: [`s4@2`](../benchmark/evaluators/s4.yaml). Requalification on GPU pending after the SAR (quadrature) and receiver-phase/role changes.

S4 covers ECG-triggered adiabatic inversion, T2 preparation and T1/T2 mapping
with Cartesian bSSFP. The judge derives physio1 triggers, cardiac ADC windows,
preparation timing, RF rotations, full-grid coverage, gradient balance and
RF/receiver coherence from actual Pulseq events and waveforms.

Before native S4 grading, KomaMRIBase `resolve_triggers` with
`CardiacSignal(heart_rate=1000/rr_interval_ms, first_peak=0)` resolves each
standalone physiological trigger to the next R-wave. The resolved idle waits
are inserted into a derived execution file while the original RF, gradient,
ADC, shape and extension libraries remain byte-identical. The submitted file
is retained unchanged. Trigger arming delay and post-trigger duration follow
Koma's semantics; phantom motion is an independent input and is not required.
Hardware/duration checks, SAR, cardiac windows, preparation recovery, targets
and native replay all use that same resolved timeline. Native replay does not
resolve it a second time. A `physio/resolution.json` receipt records the original
and execution hashes, package versions, RR and inserted waits. Offline
`s4_sequence.inspect_sequence` inspects its supplied timeline; the public
evaluation entrypoints perform physiological resolution before calling it.

The readout is explicitly symmetric: in every acquired TR,
`abs(TE - TR/2) <= max(0.01 ms, 0.05*TR)`, measuring TE from excitation RF
center to the readout k-space-center echo. This is a requirement of this task
contract, in addition to gradient balance and RF/receiver coherence.

RF between the previous frame's last ADC and the readout train is classified by
role from the waveform, never from labels:

- **Preparation**: inversion-class RF (nominal flip >= 170 degrees) or, for T2
  tasks, the first consecutive 90 / >=178 / >=178 / 90-degree block.
- **Conditioning**: RF no larger than the train's own flips (+2 degrees), e.g.
  alpha/2-TR/2 catalyzation, ramp-up or alpha/2 flip-back after the previous
  train. Conditioning never counts as preparation.
- **Unclassified**: anything else. Unclassified RF fails the frame.

Every inversion-prepared frame has exactly one inversion-class RF event before
the readout train. TI runs from the inversion RF center to the k-space-center
echo time of the frame's central line, interpolated along its readout exactly as
for TE. Every nonzero T2 preparation is four **separate RF events**: 90-degree
tip-down, two refocusers of at least 178 degrees, and 90-degree tip-up (tip
tolerance 2 degrees). Their relative RF phases are `0, pi/2, pi/2, pi`, within
0.01 rad. Preparation duration runs from tip-down center to tip-up center;
refocusing centers occur at one-quarter and three-quarters of that interval.
Preparation duration, inversion TI and refocusing-center timing tolerances are
twice the larger RF/block-duration raster. The zero-duration T2-mapping
reference has no preparation-class RF. The preparation Bloch effect removes only
preparation-class RF; conditioning stays in both replays. Other physically useful preparation designs are outside
this specific benchmark contract.

Finite-RF KomaMRI replay retains the entire preparation and readout RF history.
Every image and independent target uses BART Cartesian inverse FFT
(`bart_cartesian_fft@1`), including the mapping dictionary basis. Each frame has
a full-grid target holding tissue response at its central echo, including fat. The RF-only
phase oracle adds declared B0. Readout blur and displacement remain errors.

The central-echo target uses the **submitted sequence's** RF history; it is not
the image from the qualification reference sequence. Starting from equilibrium,
finite-RF replay without encoding gradients gives each tissue's complex response
at the acquired echo nearest k=0. That response is held constant across the full
requested Cartesian grid. The actual encoded image, in contrast, retains the
changing tissue response at every ADC sample. Consequently, line-by-line
transients can produce blur, ringing and displacement that fail image acceptance
even when the preparation, balanced gradients and RF/receiver phases are valid.
Including transients in the central-echo calculation does not include their
readout artifacts in the target. Declared B0 supplies the independent RF-only
phase reference; its magnitude loss is not added to the target magnitude.

Conditioning pulses, flip-angle ramps, k-space ordering and readout timing are
design choices for controlling these errors. No particular reference TR, flip
angle, ordering or conditioning count is required. Any conditioning must be
authored explicitly and obey the existing preparation timing, cardiac ADC window,
duration, signal, preparation-effect and SAR constraints. The mapping dictionary
also uses central-echo responses with fixed anatomical partial volumes rather
than fitting readout artifacts in the acquired images.

The default submission/feedback summary lists failed sequence, application and
image checks with their measured details. Numerically unresolved checks are
listed separately. Full disclosed responses are saved in `feedback/` and can be
printed with `./bin/submit --full` or retrieved without another grading call using
`./bin/feedback --full`. Terminal blind receipts do not disclose private verdicts.

| Per-frame image metric | Limit |
|---|---:|
| Complex / magnitude NRMSE | ≤ 0.45 / 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal / contrast retention | ≥ 0.5 / 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Myocardial echo signal must reach 0.01 of equilibrium PD, and preparation must
change it by at least 0.02 PD. Each inversion/refocusing RF pulse must invert
at least 90% of longitudinal magnetization over 17 offsets spanning
−520 to +70 Hz, scaled by scanner field / 3 T. RF labels are not evidence.

Mapping compares standard reconstructed myocardial ROIs against an anatomical
Bloch dictionary including fixed partial volumes and BART FFT response. Only the
requested relaxation time varies; PD and the other relaxation time are known.
No gain, phase, B0 or registration is fitted. Relative error must be ≤10%.
Numerical relative-error precision is 0.001; unresolved intervals trigger doubled
independent spatial quadrature, preserving each stage and the unchanged dictionary.
The method follows [prepared bSSFP dictionary mapping](https://doi.org/10.1002/mrm.29143);
the numeric bounds are benchmark acceptance limits.

Every frame and application check must pass. Unresolved results cannot pass.
Final submissions undergo hardware, timing and SAR checks (quadrature/CP body coil;
formula in [S2](S2_EVALUATION.md#sar-and-validation-tools)). External backends must
report every frame and application outcome plus BART version, executable hash
and actual commands. [Independent controls](../tests/fixtures/s4_sequences.py)
and the [qualification runner](../tests/fixtures/qualify_s4.py) retain submitted-byte,
image and numerical provenance.

### RF reader corrections and feasibility

The reader accepts finite explicit RF time vertices, including dwell times larger
than the declared RF raster. Explicit complex RF vertices use linear interpolation
on their declared support, matching KomaMRIFiles; no endpoint extrapolation is
added. SAR integrates the resulting quadratic power exactly and checks interior
sliding-window maxima. Structural/RF assays subdivide each explicit segment to at
most 0.25 microseconds for midpoint propagation; native replay reads the original
file. Previously supported constant and default-raster sample holds retain their
exact behavior. Zero-duration blocks containing only extensions or no events have
zero RF energy; negative/nonfinite durations and zero-duration physical events
still fail closed.

S4 failure details report the measured TE, TI, flip angles, RF phase and preparation
center errors with their existing limits and failed subconditions. These reader
and diagnostic corrections do not change task contracts or acceptance thresholds.

All four independent full S4 positive controls passed at 3 T/high_performance
with the physiological-trigger grader. This establishes attainability for those
frozen simulator cases. Some structural requirements specify this benchmark's
particular sequence design; alternative useful MRI sequences may fall outside
that contract. Passing these controls does not imply scanner certification.

For a protocol shorter than a configured SAR window, the average uses the actual
authored sequence duration, including explicit RF-free recovery delays. It does
not zero-pad to 360 seconds or insert implicit repetitions. A shorter protocol
can therefore fail the duty-cycle limit despite using less total RF energy than
a passing reference. The supplied total-duration and SAR limits still apply.

TI comparisons retain the inclusive two-raster physical tolerance, with only a
1 picosecond numerical guard for accumulated floating-point arithmetic. The
diagnostic reports this guard separately from the physical tolerance. Reference
TR, excitation flip and dummy-pulse counts remain examples, not additional gates.
