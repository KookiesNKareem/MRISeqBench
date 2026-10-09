<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" alt="MRISeqBench" width="800">
  </picture>
</p>

MRI sequence programming tasks spanning sequence complexity and object/physics complexity.
Agents submit Pulseq `sequence.seq` files using versioned KomaMRI phantoms.

## Current task grid

**40 sequence tasks, 1,840 cases** in the [full suite](benchmark/suites/full.yaml),
covering 460 object/physics combinations on each of four hardware profiles.
Each checkmark means every task in that row has cases in that object/physics category.

| Sequence | Tasks | P0 | P1 | P2 | P3 |
|---|---|:---:|:---:|:---:|:---:|
| S1 · Basic | T1/T2* GRE, spin echo, bSSFP, geometry GRE, 3D GRE | ✓ | ✓ | ✓ | ✓ |
| S2 · Readout | Single/multishot EPI, radial, spiral, rosette, rings, stack-of-stars, PROPELLER, UTE, TSE, GRASE | ✓ | ✓ | ✓ | ✓ |
| S3 · Sampling | SENSE, GRAPPA, CAIPIRINHA, compressed sensing, variable density, partial Fourier, asymmetric echo, reduced FOV, elliptical k-space | ✓ | ✓ | ✓ | ✓ |
| S4 · Temporal | Radial/interleaved cine, real-time radial, keyhole, k-t BLAST, k-t SENSE, TWIST | ✓ | ✓ | ✓ | ✓ |
| S5 · Application | Inversion recovery, adiabatic inversion, T2 preparation, multi-echo GRE, T1/T2 mapping, MRF | ✓ | ✓ | ✓ | ✓ |

**P0:** uniform ideal object. **P1:** richer ideal geometry/tissues.
**P2:** B0 inhomogeneity only. **P3:** motion or time-varying properties.

The full suite combines each sequence with compatible uniform and multi-tissue
phantoms, ideal fields, B0 inhomogeneity, motion, and B0 plus motion.
2D and 3D cases use matching phantom and physics definitions.

## Hardware

Gradient amplitude and slew limits are **per axis**. The three sourced profiles use
published field strengths and gradient limits from real systems:

| Profile | System/configuration | B0 (T) | Gradient (mT/m) | Slew (T/m/s) |
|---|---|---:|---:|---:|
| `standard@1` | Existing generic benchmark baseline | 3 | 32 | 80 |
| `low_field@1` | [Siemens Free.Max, B80](https://academy.siemens-healthineers.com/_/en-us/magnetom-free-max-system-overview/) | 0.55 | 26 | 45 |
| `standard_1_5t@1` | [Siemens Sola, XJ](https://www.siemens-healthineers.com/en-us/magnetic-resonance-imaging/0-35-to-1-5t-mri-scanner/magnetom-sola) | 1.5 | 33 | 125 |
| `high_performance@1` | [Siemens Prisma, XR](https://www.siemens-healthineers.com/en-au/magnetic-resonance-imaging/3t-mri-scanner/magnetom-prisma) | 3 | 80 | 200 |

Sources checked October 9, 2026; each sourced YAML records its provenance.
RF amplitude (20 µT), dead times and raster settings are shared benchmark
assumptions, not published specifications for these scanners. Profiles enforce
sequence limits and a synthetic global SAR load; they do not model scanner-specific coils, PNS, gradient
duty cycles, noise or field-dependent tissue relaxation. Phantoms and physics
perturbations retain their explicitly defined properties across hardware profiles.

Suite entries can select `hardware: [low_field@1, standard_1_5t@1, high_performance@1]`
alongside objects and physics. Each combination gets a distinct case ID;
the task's default hardware keeps its original ID. The full suite includes all
four profiles; core and smoke retain their default hardware.

## Submissions and feedback

Agent experiments default to **at most three submissions** with explicit
`./bin/submit` calls. Basic checks are available before any submission:

- `./bin/lint` checks Pulseq parsing, finite waveforms, supported events, rasters,
  timing, RF/ADC presence and the public duration limit.
- `./bin/check` runs those checks and adds peak B1, gradient amplitude and slew limits.

Both may be run repeatedly within the run time limit without using submissions.
Neither computes SAR, simulates the phantom or returns image scores. Submission 1
is blind to those evaluation verdicts. Agents can also run their own checks or
use libraries in their workspace.

Each submission freezes the submitted bytes and runs the full evaluator. A failed
attempt returns `retry_required` with its failure reasons when another attempt
remains, allowing the same agent process to revise and submit again. The run ends
on a pass or after submission 3. Incomplete, uncalibrated and infrastructure-error
verdicts end the run with their actual status; they are not counted as passes or
turned into sequence failures. The agent's time limit covers the entire run,
including submission evaluation.

Every attempt retains its sequence, SHA-256 receipt and verdict under
`control/submissions/attempt-NNN/`. Run reports include all `submissions`, the
`first_submission_status`, and `passed_on_submission`, preserving the blind
result separately from later improvement. The final report uses the last recorded
verdict without reevaluating mutable agent files.

`max_submissions: 3` and `submission_mode: submit` are explicit in the Pi example.
For infrastructure smoke commands, `submission_mode: file` still permits one
implicit submission on successful exit, with no opportunity to revise it.
Setting `max_submissions: 1` retains the legacy single-submission experiment and
its configured basic-check budget. Neither basic command invokes image scoring,
even when a legacy configuration sets `feedback.mode: evaluation`.

## SAR evaluation

Submission evaluation fails sequences exceeding either SAR limit, before
physical image scoring, including when no physical backend is configured.
SAR is excluded from separate lint/check feedback. It is included in failure
feedback **after an attempt has been submitted and recorded**. Agents must estimate
it themselves or use an available library before submitting. The load and limits appear in their
hardware contract; submitted file definitions cannot override them.

`homogeneous_sphere_v1` assumes a fixed, uniformly conducting sphere in a uniform
circularly polarized transmit field. It integrates the actual Pulseq RF envelope
(`signal` in Hz) after converting to B1+ in tesla with the proton gyromagnetic
ratio. For this model:

```text
omega = 2*pi*gamma_Hz_per_T*hardware.field_strength_T
SAR(t) = conductivity * omega^2 * radius^2 / (5*density) * |B1+(t)|^2
```

The coefficient follows from the assumed rotating field
`B(t) = B1+ * (cos(omega*t), sin(omega*t), 0)`,
`E = -(dB/dt cross r)/2`, and volume averaging `sigma*|E|^2/rho`.
The B1+ convention matters when comparing formulas with other RF amplitude
definitions. Squaring the complex envelope makes the estimate phase invariant.

All four profiles explicitly use radius **0.15 m**, conductivity **0.5 S/m**, and
density **1000 kg/m³**. These are benchmark assumptions, not measured scanner or
patient calibration. The synthetic load stays fixed across imaging phantoms;
their FOV, dimensionality and proton density do not define an electrical body model.
Fixed conductivity gives quadratic field-strength scaling: identical RF has four
times the modeled SAR at 3 T versus 1.5 T. At 3 T the coefficient is approximately
**1.45 W/kg per µT²** of time-averaged B1+ power.

The limits are **2 W/kg over any 360 s** and **4 W/kg over any 10 s**, following
the IEC normal-mode whole-body limits reported in the literature. Integration
includes RF delays, inter-pulse gaps and ringdown time as zero RF power, and finds
peak windows without coarse time bins. Scans shorter than a window use their
actual duration, without zero padding or assumed repetition; this is an explicit,
more restrictive benchmark convention. Regular-raster RF and constant RF are
supported; other explicit RF time shapes fail evaluation until their integration
convention is supported. Reports include the load, coefficient, field strength,
window location, B1 RMS, estimated SAR and limit. A relative tolerance of 1e-6
is used at the limit, matching other hardware checks.

This is a synthetic global SAR constraint, not a patient safety certification or
a guaranteed upper bound. It omits local/head/partial-body SAR, tissue-dependent
conductivity, electric-field hotspots, implants and scanner-specific transmit
calibration. Clinical SAR needs validated electromagnetic/coil/body models.

Sources reviewed October 9, 2026:

- [Geethanath, Kabil and Vaughan (2020)](https://doi.org/10.1002/9780470034590.emrstm1633)
  describes Q-matrix/VOP methods for open-source sequence RF assessment.
  [sar4seq](https://github.com/imr-framework/sar4seq) and its
  [Python branch](https://github.com/imr-framework/sar4seq/tree/PySar4seq)
  require electromagnetic Q-matrices. Their coefficients are not transferable
  scanner calibration; no implementation or Q-matrix data is vendored here.
- [Tang and Yamamoto (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9849420/)
  reviews the homogeneous-sphere approximation, field-strength scaling and IEC
  averaging limits. Our rotating-field convention and load are specified above.
- [Pulseq maintainers](https://github.com/pulseq/pulseq/discussions/59)
  distinguish RF power/B1 RMS from absolute SAR, which depends on coil and subject.

## Run

```bash
uv sync --locked --group dev
julia --project=. -e 'using Pkg; Pkg.instantiate()'
uv run mriseqbench validate
uv run mriseqbench list --suite full
uv run mriseqbench describe spiral_gre --physics b0_smooth
uv run mriseqbench describe spiral_gre --hardware low_field
uv run mriseqbench run --experiment experiments/sandbox_smoke.yaml
```

Set the model and suite in [experiments/pi.yaml](experiments/pi.yaml) for agent runs.
Task definitions live in [benchmark/tasks/](benchmark/tasks/), combinations in
[benchmark/suites/](benchmark/suites/), and execution settings in [experiments/](experiments/).

Tasks specify an `id`, `objective`, and `sequence` requirements. Defaults: version 1,
`tissue_discs@1`, `standard@1` hardware, `image_fidelity@1` evaluation, and `sequence.seq`
output; override these in the task YAML. Agents receive a compact brief and `task.yaml`;
the full resolved contract remains in `case.json`.
