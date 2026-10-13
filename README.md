<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" alt="MRISeqBench" width="800">
  </picture>
</p>

MRISeqBench evaluates an agent's ability to write expert-quality MRI sequences.
Agents submit Pulseq `sequence.seq` files for physical waveform checks, Bloch
simulation, BART reconstruction and image-quality evaluation.

| Difficulty | Focus |
|---|---|
| S1 · Basic | GRE, spin echo and bSSFP |
| S2 · Readout | EPI, non-Cartesian readouts and echo trains |
| S3 · Sampling | Parallel imaging, partial Fourier and undersampling |
| S4 · Cardiac application | ECG-triggered adiabatic inversion, T2 preparation and T1/T2 mapping with bSSFP readouts |

Every task uses static cardiac anatomy with fat and off-resonance. Each scanner
field strength has its own tissue T1/T2 profile; fat chemical shift and B0
inhomogeneity follow scanner strength. There are no phantom difficulty levels or
motion. All image reconstruction uses standard BART methods with fixed settings. See the
[cardiac phantom](docs/CARDIAC_PHANTOM.md) for inputs and property sources.

The full suite contains 25 tasks across three scanner profiles (75 cases).
All four tiers have native evaluation, qualified across all 75 task/scanner cases.

## Run

```bash
uv sync --locked --group dev
julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash runtimes/bart/install.sh  # pinned BART v1.0.00; dependency notes in script
uv run mriseqbench validate
uv run mriseqbench list --suite full
uv run mriseqbench describe spiral_gre
uv run mriseqbench run --experiment experiments/pi.yaml
```

Agents can use `./bin/lint` for timing checks and `./bin/check` for hardware checks.
These are non-consuming; `./bin/submit` performs full evaluation, with up to three
submissions per task.

Task definitions: [tasks](benchmark/tasks/) · [suites](benchmark/suites/) ·
[experiments](experiments/).
Evaluation: [S1](docs/S1_EVALUATION.md) · [S2](docs/S2_EVALUATION.md) ·
[S3](docs/S3_EVALUATION.md) · [S4](docs/S4_EVALUATION.md). Runtime: [authoring arms](docs/ARMS.md).
Run retention, deduplication and checkpoint recovery: [storage](docs/RUN_STORAGE.md).
