# Reduced-FOV EPI

Select the central 160×100 mm (x×y) rectangle using RF that overlaps both
in-plane encoding gradients: x and y gradients must each be non-zero at some
time while RF is on. Follow it with one alternating-polarity EPI shot from a single
excitation: exactly 64 readout lines of 128 x samples, retaining even zero-based
ky indices of the full 128×128 grid over 200×200 mm. Refocusing pulses (≥135°)
are excluded. The fixed single-coil BART FFT reconstructs the central half of
the phase FOV at 1.5625 mm resolution.

Imaging k starts at the end of the Pulseq block containing the excitation RF.
Gradients inside that block, including any rewinder placed in it, never count
toward imaging k; gradients in later blocks (e.g. a separate rewinder or
prephaser) do. Each line must bracket kx=0. Effective TE runs from the
magnitude-weighted RF centre to the central-ky echo and must be within
`te_ms.max`.

## Selection probes

CUDA Bloch probes use water (PD=1, T1=0.8 s, T2=0.08 s) and declared B0 at the
readout-centre samples of the first, central and last EPI lines. Magnitudes are
multiplied by exp((t − t_RFend)/T2), where t_RFend is the end of the excitation
RF that produced the EPI shot (later RF is ignored); this exactly normalizes
them to that RF end.
Every interior magnitude must be ≥0.20, their mean ≥0.01, and at each of those
echoes maximum exterior / mean interior magnitude must be ≤0.10. All probes
have z=0:

| Probe set | x (mm) | y (mm) |
|---|---|---|
| Interior | −56, 0, 56 | nine points −35..35 |
| y stopband | −56, 0, 56 | nine points per side, ±65..±96 |
| x stopband | −96, −92, 92, 96 | −25, 0, 25 |

Each probe is one image voxel (1.5625 × 1.5625 mm in x, y). Its complex signal
is averaged over a uniform isochromat grid after removing the nominal encoding
phase of that sample and scored by magnitude; the same coprime grid pairs and
convergence rule as `reduced_fov` apply (24/31, refined to 48/61 and 96/127;
the worse value is scored). With no refocusing, exterior suppression must come
from the RF selectivity itself.

## Image

The cardiac image is scored under declared B0, relaxation and fat against an
independent target. Its phase oracle retains RF selection gradients and removes
later imaging gradients; submitted ADC phase never defines the target.
Limits: complex/magnitude NRMSE ≤0.37/0.35, phase RMS ≤0.27 rad, tissue signal
≥0.63, outside energy ≤0.05. Contrast retention is diagnostic for this spatial
selection task. Ideal-control images are diagnostic. No fitted gain, phase,
registration or field correction is used. Common rules, including preflight,
SAR, outcomes, image-metric definitions and numerical acceptance, are in
[COMMON_EVALUATION.md](COMMON_EVALUATION.md); see also [S3](S3_EVALUATION.md)
and the supplied `task.yaml` / `EVALUATION.md`.
