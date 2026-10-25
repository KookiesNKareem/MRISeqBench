# S1 evaluation

Numerical precision settings and image limits are supplied in `case.json`.

S1 covers Cartesian GRE (`gre_t1w`, `gre_t2star`), spin echo (`spin_echo_t2w`)
and bSSFP (`bssfp`). Preflight, SAR, RF classification, k-space, ADC timing,
lattice tolerances, TE/TR, receiver phase, image-metric definitions and
numerical acceptance are defined in [COMMON_EVALUATION.md](COMMON_EVALUATION.md)
and apply here. A submission passes only if every common check, every required
sequence check, every required probe and every image limit below passes.

## Required sequence checks

| Check | Requirement |
|---|---|
| RF before ADC | Excitation assignment as in the common rules: after its excitation ends, before the next excitation centre, no RF overlap |
| Echo in ADC | Each line's k = 0 crossing lies inside its ADC (common rules) |
| Readout samples | Every ADC event has at least 2 samples |
| One line per excitation | Each ADC event follows a different excitation (one readout line per excitation). Unacquired excitations (dummies, preparations) may precede the first acquired excitation but must not sit between acquired excitations |
| Readout | Each ADC event has exactly N samples, N = matrix size along its readout axis (128); consecutive samples step by 1/FOV (±0.01 sample) along that axis and ≤ 0.01/FOV on other axes. No oversampling |
| Grid | All samples within 0.01 sample of one Cartesian lattice (spacing 1/FOV, common offset ≤ 0.51 sample, or +1 for a −63…+64 band); every one of the 128 × 128 grid points acquired. Repeated points are averaged |
| Slice rephasing (2D) | \|kz\| ≤ 0.01/min(FOV) at every ADC sample |
| TE, TR | As defined in the common rules, for every line/repetition |
| Flip angle | Where the task sets one, every acquired excitation within ±1° |
| GRE families | No refocusing pulse anywhere (`gre_t1w`, `gre_t2star`, `bssfp`) |
| Spin echo | Excitation 90 ± 2°; every refocusing pulse 180 ± 2°; exactly one refocusing pulse between each excitation and its echo and per TR; its centre within ±20 µs (2× the larger of block and RF raster) of the midpoint between excitation centre and echo |
| bSSFP balance | For each acquired excitation, the net gradient area from its RF start to the next acquired excitation's RF start, and for the last acquired one to the end of the sequence, is ≤ 0.01 cycles across the FOV on every axis (z uses min(FOV)), so a terminal spoiler fails |
| bSSFP phase cycle | Effective RF phase (common rules) advances by a constant π ± 0.01 rad between consecutive acquired excitations |

## Required probes

Probes use fixed materials independent of the phantom, simulated from
equilibrium with the complete RF/timing history (transients included).
Probe material: PD 1, T1 800 ms, T2 80 ms. "Echo signal" is the sample nearest
k = 0 of each line. The T1-weighting probe instead uses only the **central
echo**: the earliest acquired sample nearest the k-space origin (the ky = 0
echo of a Cartesian grid), which is where the image takes its contrast.

| Probe | Families | Requirement |
|---|---|---|
| Echo formation | all | KomaMRI replay of the first line (257 isochromats along the readout across the FOV, 10 µs steps) peaks inside the ADC window within max(20 µs, half the dwell) of the predicted TE; at least 20 µs of ADC on each side of the echo |
| Usable echo signal | all | Mean echo magnitude ≥ 0.01 of PD, perfect-spoiling reference for GRE |
| Spoiling | `gre_t1w`, `gre_t2star` | Relative complex error ‖actual − ideal‖/‖ideal‖ ≤ 0.2 between the actual echoes and perfect spoiling, for 1021 and 2053 Halton isochromats spread over one voxel (FOV/matrix per in-plane axis; the 2D object has no thickness, so through-plane spoilers/crushers have no effect). Encoding phase removed; receiver phase taken from RF |
| Refocusing | `spin_echo_t2w` | First echo with gradients off over 65 offsets uniformly spanning −50…+50 Hz retains ≥ 0.90 of the on-resonance magnitude |
| T1 weighting | `gre_t1w` | Central-echo magnitude ratio WM/GM ≥ 1.15 (WM: PD 0.7, T1 800 ms, T2 80 ms; GM: PD 0.8, T1 1300 ms, T2 100 ms, perfect spoiling), and the probe material's central-echo change for +10% T1 exceeds that for +10% T2. Lines acquired before the central echo set its magnetisation history; dummy scans are not required if that history already gives T1 weighting |

## Image replay

Replay uses finite-RF Bloch simulation in KomaMRI and BART Cartesian FFT
reconstruction. Every case uses [static cardiac anatomy](CARDIAC_PHANTOM.md)
with scanner-specific T1/T2, intrinsic fat shift and declared off-resonance.
Independent full-grid targets hold the finite-RF tissue response at the central
echo (the earliest acquired sample nearest k = 0, i.e. the ky = 0 echo),
including intrinsic fat. The RF-only phase oracle replays that same echo
(ties within 1e-3 sample of the minimum k radius take the earliest) and adds
declared B0;
submitted receiver errors, readout-induced fat displacement and blur remain errors.

Receiver phase follows the common FID convention (for bSSFP the ADC phase
must follow the alternating RF phase).

Image limits under the declared physics (these decide the verdict):

| Metric | Limit |
|---|---:|
| Complex NRMSE | ≤ 0.45 |
| Magnitude NRMSE | ≤ 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal fraction | ≥ 0.5 |
| Contrast retention | ≥ 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Ideal-control images are reported for information only. No fitted gain, phase,
registration or B0-aware reconstruction is permitted.

Metric definitions and numerical acceptance: [COMMON_EVALUATION.md](COMMON_EVALUATION.md).

Artifacts contain submitted-byte hashes, waveform checks, image arrays,
metrics and numerical provenance.
