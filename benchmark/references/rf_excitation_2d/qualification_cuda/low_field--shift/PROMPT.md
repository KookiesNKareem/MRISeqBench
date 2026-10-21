# rf_excitation_2d-v1--static_b0-v1--hardware-low_field-v1

Design a 2D spatially selective RF excitation that draws the upright letters AI in the supplied uniform water disc (radius 60 mm, z=0). Start from equilibrium and target transverse magnetization magnitude 0.5 of equilibrium PD within the letters and zero outside, with a 1.25 mm Gaussian edge smoothing. The canonical target is target.npz (x_mm, y_mm, target_mxy_magnitude; arrays use x as the first axis and positive y points upward). Use RF and in-plane gradients; the trajectory and pulse design method are free. Submit an excitation-only sequence without ADC events. The judge samples magnetization 1 microsecond after the end of your complete authored sequence, preserving all RF, gradients, relaxation and declared B0. Phase is unconstrained. No fitted gain, spatial registration or normalization is applied to your response. The full sequence, including any rewinding or recovery, must fit in 40 ms.

Write `sequence.seq` in Pulseq format in this working directory.
Timing targets use the stated tolerance.
Duration and max values are upper limits; hardware and SAR limits apply.
There is no direct Internet access; use installed tools and documentation.

```yaml
sequence:
  capabilities: [rf_excitation_2d]
  matrix: [64, 64]
  fov_mm: [200.0, 200.0]
  duration_s: 0.04
phantom: uniform_disc@1
physics:
  effects:
  - kind: b0
    model: linear
    offset_hz: 0.0
    gradient_hz_per_m: [183.33333333333334, 0.0]
hardware:
  field_strength_T: 0.55
  sar:
    model: homogeneous_sphere_v1
    radius_m: 0.15
    conductivity_S_per_m: 0.5
    density_kg_per_m3: 1000.0
    whole_body_limit_W_per_kg: 2.0
  max_gradient_mT_per_m: 26.0
  max_slew_T_per_m_per_s: 45.0
  max_b1_uT: 20.0
  rf_dead_time_us: 100.0
  rf_ringdown_time_us: 30.0
  adc_dead_time_us: 10.0
  gradient_raster_us: 10.0
  rf_raster_us: 1.0
  adc_raster_us: 0.1
  block_raster_us: 10.0
seed: 42
```

Evaluation (rf_excitation@1, uncalibrated): excitation-only RF/gradient waveforms, hardware/timing and final-submission SAR. Native finite-RF KomaMRI Bloch replay on the supplied water disc with declared B0; direct per-spin transverse magnetization 1 us after the authored sequence ends. The font-free upright AI target is supplied in target.npz, x-first and positive y upward. PD=1, T1=1 s, T2=0.1 s, radius=60 mm, z=0; 64x64 cell-centred 200 mm FOV. Target magnitude is 0.5, with Gaussian smoothing sigma=1.25 mm. Phase is unconstrained. NRMSE uses all disc spins with no fitted gain or registration. Inside signal is the mean magnitude divided by the mean target where target>=0.45. Outside RMS is divided by 0.5 where target<=0.005. Temporal refinement must resolve all limits with estimated error<=0.002. Limits: excitation_magnitude_nrmse <=0.3; inside_signal_fraction >=0.8; outside_rms_fraction <=0.1. Qualification controls are pending; successful responses remain uncalibrated. See docs/RF_EXCITATION_EVALUATION.md.
Scoring criteria and SAR model: `EVALUATION.md`. Requirements: `task.yaml`.
