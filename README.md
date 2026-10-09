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

**40 sequence tasks, 2,760 cases** in the [full suite](benchmark/suites/full.yaml),
covering 690 object/physics combinations on each of four hardware profiles.
Each checkmark means every task in that row has cases in that object/physics category.

| Sequence | Tasks | P0 | P1 | P2 | P3 | P4 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| S1 · Basic | T1/T2* GRE, spin echo, bSSFP, geometry GRE, 3D GRE | ✓ | ✓ | ✓ | ✓ | ✓ |
| S2 · Readout | Single/multishot EPI, radial, spiral, rosette, rings, stack-of-stars, PROPELLER, UTE, TSE, GRASE | ✓ | ✓ | ✓ | ✓ | ✓ |
| S3 · Sampling | SENSE, GRAPPA, CAIPIRINHA, compressed sensing, variable density, partial Fourier, asymmetric echo, reduced FOV, elliptical k-space | ✓ | ✓ | ✓ | ✓ | ✓ |
| S4 · Temporal | Radial/interleaved cine, real-time radial, keyhole, k-t BLAST, k-t SENSE, TWIST | ✓ | ✓ | ✓ | ✓ | ✓ |
| S5 · Application | Inversion recovery, adiabatic inversion, T2 preparation, multi-echo GRE, T1/T2 mapping, MRF | ✓ | ✓ | ✓ | ✓ | ✓ |

**P0:** uniform ideal object. **P1:** richer ideal geometry/tissues.
**P2:** B0 or B1 perturbation. **P3:** combined B0/B1. **P4:** motion.

The full suite combines each sequence with compatible uniform and multi-tissue
phantoms, ideal fields, B0, B1, combined fields, motion, and fields plus motion.
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
sequence limits; they do not model scanner-specific coils, SAR, PNS, gradient
duty cycles, noise or field-dependent tissue relaxation. Phantoms and physics
perturbations retain their explicitly defined properties across hardware profiles.

Suite entries can select `hardware: [low_field@1, standard_1_5t@1, high_performance@1]`
alongside objects and physics. Each combination gets a distinct case ID;
the task's default hardware keeps its original ID. The full suite includes all
four profiles; core and smoke retain their default hardware.

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
