<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" alt="MRISeqBench" width="800">
  </picture>
</p>

MRISeqBench evaluates an agent's ability to write expert-quality MRI sequences.
Agents submit Pulseq `sequence.seq` files for physical waveform checks and Bloch
simulation. Imaging tasks use BART reconstruction and image-quality evaluation;
RF design tasks score spatial magnetization directly.

| Difficulty | Focus |
|---|---|
| S1 · Basic | GRE, spin echo and bSSFP |
| S2 · Readout | EPI, non-Cartesian readouts and echo trains |
| S3 · Sampling | Parallel imaging, partial Fourier, undersampling and 2D RF reduced-FOV EPI |
| S4 · RF design and applications | 2D excitation of AI; ECG-triggered adiabatic inversion, T2 preparation and T1/T2 mapping with bSSFP readouts |

Imaging tasks use static cardiac anatomy with fat and off-resonance. Each scanner
field strength has its own tissue T1/T2 profile; fat chemical shift and B0
inhomogeneity follow scanner strength. There are no phantom difficulty levels or
motion. The 2D RF task uses a uniform water disc, a fixed AI target and static B0.
All image reconstruction uses standard BART methods with fixed settings. See the
[cardiac phantom](docs/CARDIAC_PHANTOM.md) for inputs and property sources.

The full suite contains 27 tasks across three scanner profiles (81 cases).
The 25 existing imaging tasks retain their qualification; the RF suite adds AI
excitation and reduced-FOV EPI across the same three scanner profiles.
Both new tasks have [qualified references and prompts](benchmark/references/README.md),
with all 42 positive/negative reference controls verified on CUDA.

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

Set `agent.model`, `api_proxy.provider_endpoint` and `api_proxy.max_campaign_usd`
in the experiment and provide `OPENROUTER_API_KEY` before running it. Rerunning
with the same `--out` resumes a campaign; `--cases`/`--repeats` select a canary.
Campaigns on Ultron run under systemd: [ops](ops/README.md).

Native Bloch grading requires an NVIDIA CUDA GPU and runs in Float64. Select the
grader GPU with `CUDA_VISIBLE_DEVICES` (prefer its UUID on shared hosts). Missing
or unusable CUDA fails explicitly; grading never silently falls back to CPU.
CPU simulation overrides are rejected. Agent-facing simulators also require CUDA;
their containers receive one GPU selected with `MRISEQBENCH_SIMULATOR_GPU` or
`CUDA_VISIBLE_DEVICES`, and record the actual backend/device. BART inverse NUFFT
requires a CUDA build of the pinned release and uses its GPU implementation with
the same trajectory, regularization, and iteration settings. Existing frozen
campaign copies retain their original evaluator until explicitly revised.
Run the native CUDA execution and no-fallback controls on the selected GPU with
`MRISEQBENCH_TEST_CUDA=1 uv run pytest tests/evaluation/test_koma_gpu.py`.

Agents can use `./bin/lint` for timing checks and `./bin/check` for hardware checks.
These are non-consuming; `./bin/submit` performs full evaluation, with up to three
submissions per task.

Task definitions: [tasks](benchmark/tasks/) · [suites](benchmark/suites/) ·
[experiments](experiments/).
Evaluation: [S1](docs/S1_EVALUATION.md) · [S2](docs/S2_EVALUATION.md) ·
[S3](docs/S3_EVALUATION.md) · [S4](docs/S4_EVALUATION.md) · [2D RF excitation](docs/RF_EXCITATION_EVALUATION.md).
Runtime: [authoring arms](docs/ARMS.md).
Run retention, deduplication and checkpoint recovery: [storage](docs/RUN_STORAGE.md).
