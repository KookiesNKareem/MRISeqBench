# AI excitation

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
