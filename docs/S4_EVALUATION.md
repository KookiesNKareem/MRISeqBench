# S4 evaluation

Contract: [`s4@1`](../benchmark/evaluators/s4.yaml). Qualified on all three scanner profiles.

S4 covers ECG-triggered adiabatic inversion, T2 preparation and T1/T2 mapping
with Cartesian bSSFP. The judge derives physio1 triggers, cardiac ADC windows,
preparation timing, RF rotations, full-grid coverage, gradient balance and
RF/receiver coherence from actual Pulseq events and waveforms.

Finite-RF KomaMRI replay retains the entire preparation and readout RF history.
Every image and independent target uses BART Cartesian inverse FFT
(`bart_cartesian_fft@1`), including the mapping dictionary basis. Each frame has
a full-grid target holding tissue response at its central echo, including fat. The RF-only
phase oracle adds declared B0. Readout blur and displacement remain errors.

| Per-frame image metric | Limit |
|---|---:|
| Complex / magnitude NRMSE | ≤ 0.45 / 0.35 |
| Wrapped phase RMS | ≤ 0.35 rad |
| Minimum tissue signal / contrast retention | ≥ 0.5 / 0.5 |
| Outside-object energy fraction | ≤ 0.25 |

Myocardial echo signal must reach 0.01 of equilibrium PD, and preparation must
change it by at least 0.02 PD. Each inversion/refocusing RF pulse must invert
at least 90% of longitudinal magnetization over 17 offsets spanning
−520 to +70 Hz, scaled by scanner field / 3 T. RF labels are not evidence.

Mapping compares standard reconstructed myocardial ROIs against an anatomical
Bloch dictionary including fixed partial volumes and BART FFT response. Only the
requested relaxation time varies; PD and the other relaxation time are known.
No gain, phase, B0 or registration is fitted. Relative error must be ≤10%.
Numerical relative-error precision is 0.001; unresolved intervals trigger doubled
independent spatial quadrature, preserving each stage and the unchanged dictionary.
The method follows [prepared bSSFP dictionary mapping](https://doi.org/10.1002/mrm.29143);
the numeric bounds are benchmark acceptance limits.

Every frame and application check must pass. Unresolved results cannot pass.
Final submissions undergo hardware, timing and SAR checks. External backends must
report every frame and application outcome plus BART version, executable hash
and actual commands. [Independent controls](../tests/fixtures/s4_sequences.py)
and the [qualification runner](../tests/fixtures/qualify_s4.py) retain submitted-byte,
image and numerical provenance.
