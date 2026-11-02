# Native fixed-slab density qualification

These are private judge-calibration inputs, not author workspace assets. Never
copy this folder into an agent sandbox or expose it through the monitor API.

`control.seq` preserves the exact candidate used for the numerical comparison;
`case.json` selects a 5 mm slab and ordinary mean-image scores. The selected
profile uses 5,578,752 actual spins; the denser profile uses 11,157,504. Their
native reconstructed images differ by 1.6482%. The image NRMSE limit is 15%.
This qualifies only this 3 T GRE density control, not the complete suite.

The preserved candidate leaves RF-use metadata undefined, as in the source run.
It is a native Bloch density probe, not an author starter or a label-aware preview
example: the latter requires explicit RF roles. The calibration retains original
bytes and records their hash rather than silently repairing them.

Reproduce with the Julia/BART environment described in `ops/README.md` and:

```sh
uv run python ops/qualify_phantom_density.py \
  --case benchmark/calibration/fixed_slab_gre_3t/case.json \
  --sequence benchmark/calibration/fixed_slab_gre_3t/control.seq \
  --output /absolute/path/to/new-qualification
```

The separate z-spoiling record demonstrates native dephasing in a 5 mm slab
and its absence for the former z=0 plane. No analytical signal corrections,
fitted gains or image registration are used. Keep every new contract marked
uncalibrated until its task/scanner controls have qualified.
