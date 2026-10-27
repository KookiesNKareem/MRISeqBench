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

The first stable source release is **v1.0.0**. Use `git checkout v1.0.0` after
cloning to reproduce this benchmark definition. See the [release notes](CHANGELOG.md)
for scope and compatibility details.

| Difficulty | Focus |
|---|---|
| T1 · Basic | GRE, spin echo and bSSFP |
| T2 · Readout | EPI, non-Cartesian readouts and echo trains |
| T3 · Sampling | Parallel imaging, partial Fourier, undersampling and 2D RF reduced-FOV EPI |
| T4 · RF design and applications | 2D excitation of AI; ECG-triggered adiabatic inversion, T2 preparation and T1/T2 mapping with bSSFP readouts |

Repository configuration labels use **T1–T4** for levels and `T<level>.<task>`
for individual tasks, **H1–H3** for hardware, **S1–S3** for simulators, and **P1** for the
cardiac phantom. See the complete [label mapping](docs/LABELS.md).

Imaging tasks use static cardiac anatomy with fat and field-dependent T1/T2, fat
shift and B0 ([cardiac phantom](docs/CARDIAC_PHANTOM.md)); the 2D RF task uses a
uniform water disc and a fixed AI target. Reconstruction is standard BART with
fixed settings. The full suite has 27 tasks on three scanner profiles (81 cases)
at 128×128 (128×128×32 for volumes), each with a qualified
[reference pulse](benchmark/references/README.md).

## Run

```bash
uv sync --locked --group dev
julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash runtimes/bart/install.sh  # pinned BART v1.0.00; dependency notes in script
uv run mriseqbench validate
uv run mriseqbench list --suite full
uv run mriseqbench describe T2.6 --hardware H3
uv run mriseqbench run --experiment experiments/pi.yaml
```

Set `agent.model` and `api_proxy.provider_endpoint`
in the experiment and provide `OPENROUTER_API_KEY` before running it. Rerunning
with the same `--out` resumes a campaign; `--cases`/`--repeats` select a canary.
For systemd deployment and GPU pools, see [operations](ops/README.md).
For result retention and recovery, see [run storage](docs/RUN_STORAGE.md).

Grading and agent simulators require CUDA (Float64 Bloch replay, CUDA BART);
there is no CPU fallback. Select GPUs by UUID with `CUDA_VISIBLE_DEVICES` and
`MRISEQBENCH_SIMULATOR_GPU`. GPU controls:
`MRISEQBENCH_TEST_CUDA=1 uv run pytest tests/evaluation/test_koma_gpu.py`.

Agents get non-consuming `./bin/lint` (timing) and `./bin/check` (hardware), and
up to three `./bin/submit` evaluations per task. Outcomes are passed or failed
([grading policy](docs/GRADING_POLICY.md)).

Task definitions: [tasks](benchmark/tasks/) · [suites](benchmark/suites/) ·
[experiments](experiments/).
Evaluation: [T1](docs/S1_EVALUATION.md) · [T2](docs/S2_EVALUATION.md) ·
[T3](docs/S3_EVALUATION.md) · [T4](docs/S4_EVALUATION.md) · [2D RF excitation](docs/RF_EXCITATION_EVALUATION.md).
Runtime: [authoring arms](docs/ARMS.md).
Run retention, deduplication and checkpoint recovery: [storage](docs/RUN_STORAGE.md).
