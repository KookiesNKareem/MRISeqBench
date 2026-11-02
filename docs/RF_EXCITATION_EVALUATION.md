# AI excitation image acquisition (task v4, evaluator v5)

Repeat **spatial excitation → one Cartesian GRE line → at least 2 s recovery**
128 times, including recovery after the final line. Each RF module fits in
40 ms; every readout ends within 5 ms of its last RF sample. Acquire exactly
128 samples per x line and every point on the 128×128 Cartesian grid once,
with the existing 200×200 mm FOV. Total duration is at most 262 s.
Use LIN labels and Nx=128, Ny=128, Nz=1 and FOV definitions so the simulator's
label-aware reconstruction can produce the same image grouping in every arm.

The uniform water disc and public AI target are unchanged: radius 60 mm,
PD=1, T1=1 s, T2=0.1 s, now spanning a 5 mm slab; target magnitude 0.5 and 1.25 mm edge smoothing.
The full submitted history is simulated with relaxation and declared B0;
BART reconstructs the acquired Cartesian signal. Compare the full-FOV image
magnitude with the public target, with no fitted gain, registration or phase
constraint. Magnitude NRMSE is 15%, matching the other imaging tasks; inside signal fraction 0.80–1.20;
outside RMS fraction at most 0.10. Numerical refinement uses the same
10/5/2.5/1.25/0.625 µs steps and 0.002 maximum metric uncertainty.
Every reconstructed stage is saved for the human monitor. Agent-visible
simulation images come from the selected simulator, not from judge files.

Local physical qualification used Metal Float32 with temporal refinement:
a spatial-AI candidate reconstructed at 16.10% magnitude NRMSE and passed the previous 20% limit; it fails the current 15% limit;
a uniform-excitation control reconstructed at 193.22% and failed. Both acquired
the complete matrix and numerically converged. Qualification records are in
`benchmark/calibration/rf_excitation_image_v4.json`. This verifies the image path;
it does not qualify a passing control at 15%. The new evaluator remains marked uncalibrated until native-signal controls qualify. Task v3/evaluator v4 preserve the previous image protocol at 20%.
The original no-ADC task v2 and direct-magnetization evaluator v3 remain available
for old manifests. Completed campaigns retain their frozen copies and results.

## Previous no-ADC protocol (task v2, evaluator v3)


Design an excitation-only Pulseq sequence for the upright AI target in the
uniform water disc: radius 60 mm, PD=1, T1=1 s, T2=0.1 s, z=0.
The supplied `target.npz` contains x/y coordinates, the disc mask and target
magnitude on a 128×128 grid over 200×200 mm. x is the first array axis; positive
y points upward. Target magnitude is 0.5 with 1.25 mm Gaussian edge smoothing.
Phase is free.

Use untriggered Pulseq 1.4 or 1.5: no external waiting timeline is supplied.
The judge replays the complete finite RF/gradient history on CUDA in Float64,
with relaxation and declared B0, sampling 1 µs after sequence end. Start from
equilibrium; any rewinding or recovery must also fit in 40 ms. No ADC is
allowed. Meet the timing and hardware limits in the supplied `task.yaml` /
`EVALUATION.md`. SAR is evaluated only when you submit; a submission that exceeds
the scanner's SAR limit fails, and the feedback will say so.

Declared B0 adds to the gradient precession with the same sign: at position r
the off-resonance is `gradient_hz_per_m · r` Hz, so `[1000, 0]` Hz/m acts like an
extra +1000 Hz/m on γ̄Gx (γ̄ = 42.576 MHz/T).

| Metric over disc spins | Limit |
|---|---:|
| Magnitude NRMSE, normalized by target norm | ≤0.20 |
| Mean magnitude / mean target where target ≥0.45 | 0.80–1.20 |
| RMS magnitude / 0.5 where target ≤0.005 | ≤0.10 |

No gain, registration or normalization is fitted.

**Numerical acceptance.** The replay runs with RF and gradient time steps of
10, 5, 2.5, 1.25 and 0.625 µs in turn. From the second step on, each metric's
error estimate is `max(0.0001, 2·|current − previous|)`. The verdict is resolved
when every estimate is ≤0.002 and every metric differs from each of its bounds
(both bounds for the inside-signal range) by more than its estimate; the
metrics at that step then decide pass/fail (bounds inclusive). So a stable
NRMSE of 0.19995 never resolves against 0.20 and fails. A verdict still
unresolved at 0.625 µs fails. Preflight, SAR and outcome rules are in
[COMMON_EVALUATION.md](COMMON_EVALUATION.md); this task has no ADC, so its
waveform and image sections do not apply. Grading has no time limit.
