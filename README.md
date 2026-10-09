# MRISeqBench

MRI sequence programming tasks spanning sequence complexity and object/physics complexity.
Agents submit Pulseq `sequence.seq` files using versioned KomaMRI phantoms.

## Current task grid

**40 sequence tasks, 690 cases** in the [full suite](benchmark/suites/full.yaml).
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

## Run

```bash
uv sync --locked --group dev
julia --project=. -e 'using Pkg; Pkg.instantiate()'
uv run mriseqbench validate
uv run mriseqbench list --suite full
uv run mriseqbench describe spiral_gre --physics b0_smooth
uv run mriseqbench run --experiment experiments/sandbox_smoke.yaml
```

Set the model and suite in [experiments/pi.yaml](experiments/pi.yaml) for agent runs.
Task definitions live in [benchmark/tasks/](benchmark/tasks/), combinations in
[benchmark/suites/](benchmark/suites/), and execution settings in [experiments/](experiments/).
