# S3: 2D RF reduced-FOV EPI

[`rfov_epi@1`](../benchmark/tasks/rfov_epi.yaml) combines two-dimensional spatial
excitation with single-shot reduced-FOV EPI. It belongs to S3 because spatial
selection enables the reduced phase FOV. The standalone [AI excitation task](RF_EXCITATION_EVALUATION.md)
belongs to S4 RF design.

The prescribed full grid is 32×32 over 200×200 mm. A 2D RF pulse selects a
160×100 mm central rectangle. The acquisition retains even zero-based ky indices
with full 32-sample x readouts, giving a 32×16 image of the central 100 mm phase
FOV at unchanged 6.25 mm resolution. One shot must acquire all 16 lines with
alternating readout polarity and no refocusing pulse. The fixed BART FFT and
central-half crop include the known sampling-density factor. TE is capped at
80 ms; complete authored duration, including recovery, is capped at 250 ms.

The scanner-specific static cardiac phantom, relaxation, fat, B0 and synthetic
SAR model are unchanged. The anatomy lies inside the selected x extent; y
selection restricts its visible phase extent.

## Physical checks

The judge verifies the Cartesian lattice, exact reduced-FOV mask, one EPI shot,
alternating polarity and RF overlap with gradients in both dimensions. Metadata
and RF use labels do not establish selection. Encoding gradients are integrated
from the end of the final RF-containing block, including authored selection
rewinding. A hard-pulse RF-centre k reset is inappropriate for a shaped 2D pulse.

Native finite-RF Bloch probes (water PD=1, T1=0.8 s, T2=0.08 s, z=0) sample
the first, central and last EPI line centres on a grid of interior points and both x-only
and y-only stopbands under declared B0. Worst exterior magnitude divided by mean
interior magnitude must be ≤0.10; every sampled interior response must remain
≥0.20 PD. Thus a one-dimensional slab pulse cannot pass merely because the
compact anatomy contains no spins outside the selected x extent.

Full cardiac replay and image acceptance use the existing `s3@3` criteria:
complex NRMSE ≤0.45, magnitude NRMSE ≤0.35, wrapped phase RMS ≤0.35 rad,
minimum tissue signal fraction ≥0.5, contrast retention ≥0.5, and outside-object
energy fraction ≤0.25. Numerical quadrature and temporal convergence remain
mandatory. Timing, hardware and final-submission SAR also apply.

The independent full-grid target uses fixed anatomy clipped to the prescribed
phase FOV, holding finite-RF tissue response at the central echo. The phase
oracle retains the actual excitation's RF and selection gradients, relaxation
and B0, while removing subsequent imaging gradients. It never uses the acquired
image or submitted ADC receiver phase. This includes the physical phase of a
shaped spatial excitation without excusing readout distortion or receiver errors.
No gain, phase, B0 correction or registration is fitted. Segmented RF history
remains coherent across subpulses.

## Reference and qualification

[`rfov_epi_sequences.py`](../tests/fixtures/rfov_epi_sequences.py) independently
authors a spatial RF trajectory and solves for RF controls before appending EPI.
Its raster excitation trajectory, fat suppression, RF duration and recovery delay
are reference design choices rather than participant requirements.

[`qualify_rfov_epi.py`](../tests/fixtures/qualify_rfov_epi.py) freezes source,
waveforms, prompts, cases, numerical stages, BART images and verdicts. Controls
include missing spatial selection, one-dimensional encoding, wrong EPI polarity,
full-FOV or missing-line sampling, shifted selection and wrong receiver phase.
The one-dimensional control also undergoes an explicit independent x-stopband
assay even when waveform checks already reject it.

CUDA qualification passed all 24 controls on an NVIDIA TITAN RTX, covering all
three scanner profiles. Positive complex NRMSE was 0.1171–0.2159 and phase RMS
was 0.0413–0.1075 rad. Selection leakage was 0.0466–0.0796. Complete verdicts,
frozen sources and image bundles are in the [reference package](../benchmark/references/README.md)
and its [qualification receipt](../benchmark/references/rfov_epi/qualification_cuda/summary.json).

```bash
uv run mriseqbench describe rfov_epi --hardware high_performance
uv run python -m tests.fixtures.qualify_rfov_epi --output tmp/rfov-qualification
```

Both new tasks are available in `rf_excitation@1` and the full suite. The target
geometry, task and scoring requirements are public; reference sequence bytes and
private grading code are not provided to participants.
