# Segmented 256² T1 mapping: one waveform prototype

This is a **new, unqualified diagnostic protocol**, outside the active task catalog. It addresses the short cardiac acquisition window constructively rather than declaring that a 256² prepared image must fit one heartbeat. It does not change or pass the existing `t1_mapping@1`/`s4@1` benchmark.

The written Pulseq waveform acquires three contrasts with eight segments per contrast. Each segment acquires one explicitly declared central-ky navigator followed by32 distinct imaging lines, at TR4.6 ms. The train lasts151.8 ms, within the200 ms cardiac window. Selecting an imaging heartbeat every five RR intervals allows24 segment preparations in120 seconds; the minimum inversion-center separation is4.2 seconds, exceeding3 RR.

Measured from the **written file**:

- 202,752 ADC samples:196,608 imaging samples plus6,144 navigator samples.
- Exactly65,536 unique Cartesian imaging points per contrast after explicitly excluding navigators; no missing points or duplicate imaging points. Even-grid extent is−128…127 reciprocal-FOV samples on both axes.
- All ADC nodes occur601.215–750.965 ms after the selected R wave.
- Navigator k-zero occurs100,400 or1200 ms after its segment inversion center. Actual readout TE is2.295 ms, within0.005 ms ofTR/2.
- Maximum gradient magnitude per axis: x38.525 mT/m, y35.790 mT/m; maximum slew x133.315 T/m/s, y127.823 T/m/s.
- Maximum readout-TR residual gradient area: x0.0002001 cycles/m, y approximately0, from the parsed gradient antiderivatives. RF and receiver phases alternate0/180° within each segment.
- Existing hardware/raster/timing checks pass at the3 T high-performance profile. Synthetic homogeneous-sphere SAR passes: overall0.2894 W/kg and worst10-second average0.3094 W/kg. This is not scanner or local-SAR certification.

`waveform-audit.json` contains the actual checks and sequence SHA. `segment-plan.json` declares the navigator/imaging roles, ky lists, preparation centers and timing for every line; `protocol.yaml` states the proposed new semantics. `build.py` reproduces the waveform and compact audit:

```sh
PYTHONPATH=src .venv/bin/python benchmark/variants/256x256/prototypes/t1_mapping_segmented/build.py
```

The prototype uses a1 ms nonselective180° block inversion and35° block readouts, with no slice profile, adiabatic robustness or catalyzation. Those choices allow geometry/timing examination, not a physiological or clinical-quality claim. Repeated inversion preparations can leave tissue-dependent state. The peripheral ky nodes occur later than the first navigator: identical nominal navigator TI does not mean an entire image is acquired at one TI. Even the imaging central-ky line is acquired after its navigator, so its actual TI is different. A qualified reference must simulate each actual RF/gradient/ADC time and include these transient effects, rather than fit each assembled image to an instantaneous ideal inversion-recovery formula.

The present S4 judge treats each trigger-window train as one complete frame and cannot assemble these eight segments into one image. It must receive an explicit versioned segment/navigator/preparation contract before this prototype can be graded. No native GPU image replay, CPU MRI simulation, T1 ROI fit, high-detail anatomy test, agent inference or paid work was performed. T1 accuracy remains unqualified; known-PD/known-T2 conditional fitting would need an explicit limitation even after signal validation.

Two small construction diagnostics are retained: the first exposed an inversion/ADC half-microsecond raster mismatch; the second used an unnecessarily strict1e−5-grid threshold despite Pulseq text amplitude rounding. The final draft uses raster-compatible10 µs ADC dwell and a1e−3-grid coordinate tolerance (tighter than the current0.01-grid structural tolerance). These are prototype construction fixes, with no runtime/evaluator change or scientific replay.
