# rfov_epi-v1--static_b0-v1--hardware-low_field-v1

Acquire a single-shot reduced-FOV gradient-echo EPI image using a genuine 2D spatially selective RF excitation. Select a central 160 mm extent in x and 100 mm extent in y within the prescribed 200 x 200 mm FOV. The cardiac anatomy fits within the x extent; the reduced phase FOV is the central half in y. Preserve 6.25 mm spatial resolution: acquire 32 x 16 samples, retaining even zero-based ky indices of the prescribed 32 x 32 full-FOV grid, with 32 samples per x readout and alternating readout polarity. Use one excitation/readout shot; do not use a refocusing pulse or reconstruct an undersampled full-FOV image as a substitute for spatial selection. RF must overlap encoding gradients in both x and y. Suppress signal outside the selected rectangle, including x outside 80 mm and y outside 50 mm, and produce coherent signal within it. Diagnostic water spins (PD=1, T1=0.8 s, T2=0.08 s, z=0) must retain at least 0.20 of equilibrium PD at every tested interior point, checked at the first, central and last EPI line centres. The largest exterior magnitude divided by mean interior magnitude must be no more than 0.10. Interior probes cover x=-56,0,56 mm and nine y positions spanning -35 to 35 mm. Stopband probes cover y=-96 to -65 mm and 65 to 96 mm at those x positions, plus x=-96,-92,92,96 mm at y=-25,0,25 mm. The judge verifies the excitation response in both dimensions, reduced-FOV sampling, EPI polarity, and the reconstructed cardiac image under declared B0 and intrinsic fat chemical shift. Use the fixed cropped BART FFT reconstruction. TE at the central-ky echo is an upper limit; choose pulse duration and timing within the supplied hardware and SAR limits. Image encoding is measured from the end of the final RF-containing block, including its selection-gradient rewinding. The phase oracle retains the RF selection gradients and declared fields, and excludes subsequent imaging gradients and your submitted ADC phase.

Write `sequence.seq` in Pulseq format in this working directory.
Timing targets use the stated tolerance.
Duration and max values are upper limits; hardware and SAR limits apply.
Use the supplied static cardiac phantom with scanner-specific tissue T1/T2, fat and off-resonance. Reconstruction is standard and fixed.
There is no direct Internet access; use installed tools and documentation.

```yaml
sequence:
  capabilities: [rf_excitation_2d, epi, cartesian, reduced_fov]
  matrix: [32, 32]
  fov_mm: [200.0, 200.0]
  te_ms:
    max: 80.0
  duration_s: 0.25
  readout:
    shots: 1
    echo_train_length: 16
  sampling:
    pattern: reduced_fov
    reduced_fov_fraction: 0.5
phantom: cardiac_2d_055t@1
tissue_properties:
  fat:
    proton_density: 0.10000000149011612
    T1_s: 0.187
    T2_s: 0.093
    chemical_shift_hz: -79.6198850171713
  myocardium:
    proton_density: 0.10000000149011612
    T1_s: 0.701
    T2_s: 0.058
    chemical_shift_hz: 0.0
  blood:
    proton_density: 0.10000000149011612
    T1_s: 1.122
    T2_s: 0.263
    chemical_shift_hz: 0.0
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

Evaluation (s3@2): waveform-derived sampling, calibration and spatial selection; independent finite-RF KomaMRI replay with scanner-specific cardiac tissues, fat and static B0. BART FFT/PICS reconstruction and receive encoding are fixed. Declared-physics images use an independent full-grid target and RF-only phase reference; ideal-control images are diagnostic. Case limits: complex_nrmse_max=0.45; magnitude_nrmse_max=0.35; phase_rms_rad_max=0.35; tissue_signal_fraction_min=0.5; contrast_retention_min=0.5; outside_energy_fraction_max=0.25. Final submissions include SAR checks. See docs/S3_EVALUATION.md. The 2D RF assay uses water with PD=1, T1=0.8 s, T2=0.08 s at z=0, sampled at the first, central and last EPI line centres. It requires minimum interior magnitude >=0.20 PD and worst exterior / mean interior magnitude <=0.10, including x-only and y-only stopbands. The single EPI shot must acquire 16 alternating-polarity 32-sample lines on the prescribed even-ky mask. Image encoding starts at the final RF block's end; effective TE uses the central-ky echo. The independent phase oracle retains RF selection gradients, removes later imaging gradients, and excludes submitted ADC phase.
Scoring criteria and SAR model: `EVALUATION.md`. Requirements: `task.yaml`.
