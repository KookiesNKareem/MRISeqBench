# Phantom and claim audit for the proposed 256 × 256 baseline

The proposed 200 mm / 256 baseline has a **0.78125 mm sampling interval**. This
does not by itself demonstrate 0.78125 mm effective spatial resolution. A real
acquisition must encode the required k-space, and the reconstruction must retain
the detail. The current cardiac label asset cannot supply independent submillimetre
anatomical information merely by evaluating it on a denser grid.

This audit changes no task, production phantom, evaluator, result or paper figure.
The separate analytic insert below is geometric truth only. No MRI signal
simulation, model call or paid run was performed here.

## What the current anatomy actually contains

| Item | Verified representation | Meaning |
|---|---|---|
| `cardiac_2d@1` | 90 × 89 labels; 1.5 × 1.5 mm cells | 135 × 133.5 mm support; three spatially constant tissue property classes |
| `cardiac_3d@1` | 90 × 89 × 1090 labels; 1.5 × 1.5 × 0.15 mm cells | Anisotropic representation, not independent 0.15 mm anatomy |
| Source | 3,466,834 occupied cells; stored name contains “static 10x-z” | Original acquisition and segmentation provenance are not documented in the asset |
| Repeated z planes | 111 unique label planes; 107 runs of ten identical planes, two runs of eight and two runs of two | Dense z coordinates largely repeat the same segmentation, rather than adding measured fine-z information |
| Central plane | Zero-based 545 equals planes 540–549 byte-for-byte | Its thin z spacing does not improve its 1.5 mm in-plane detail |

The metadata is in [anatomy.json](../benchmark/objects/anatomy.json). The source
HDF5 attributes only identify name, dimensions, spin count and file format
version; no scanner protocol, original acquired matrix, subject provenance,
segmentation accuracy or license is stored. The read-only repeat-plane audit is
[source-z-grid-audit.json](../output/resolution256-phantom-audit/source-z-grid-audit.json).

[PROVENANCE.md](../benchmark/objects/assets/PROVENANCE.md:1) traces the bytes to
the user-supplied `Phantoms/cardiac_10xz_shortaxis_clean.phantom`; it explicitly
does not infer a third-party license. This establishes the project file lineage,
not permission to redistribute external patient data or an original clinical
acquisition resolution. Establish the original source and rights before a public
release of that anatomy.

[s1_anatomy.py](../src/mriseqbench/evaluation/s1_anatomy.py:1) defines labels as
piecewise constant over physical cells. Its `fraction_phantom` computes exact
cell/image overlap; the Fourier oracle integrates those cells, and Bloch replay
samples positions inside the same cells. These are consistent models of the
declared geometry. Their sharp cell boundaries can have high spatial-frequency
content, but those frequencies describe the assumed block model. They do not
establish independent submillimetre patient anatomy. Interpolation, exact cell
integrals, extra integration samples and a 256 display grid do not change this.

## Independent diagnostic artifact supplied now

Run `.venv/bin/python scripts/build_resolution256_phantom.py` to reproduce:

- [geometry.json](../output/resolution256-phantom-audit/geometry.json): continuous
  axis-aligned rectangles in metres, exact physical dimensions, synthetic PD/T1/T2.
- [geometric-truth.npz](../output/resolution256-phantom-audit/geometric-truth.npz):
  exact 256 and 32 cell-average PD, separate tissue fractions, and a few closed-form
  **spatial-density** Fourier integrals. These are not encoded MRI signals.
- [geometric-truth.png](../output/resolution256-phantom-audit/geometric-truth.png):
  matched-window geometry overview and native-grid detail, visually inspected.
- [geometry-checks.json](../output/resolution256-phantom-audit/geometry-checks.json):
  bounded geometry checks and artifact hashes. No benchmark pass is asserted.

This is newly authored mathematical geometry, with no patient label input or
third-party image. Background and inserts have explicitly invented material
constants; they are diagnostic examples rather than measured cardiac tissue.
It is deliberately separate from the production object catalog.

| Equal line/gap width | Cycles/mm | Pixels per line at 256 | Intended role |
|---|---:|---:|---|
| 3.125 mm | 0.16 | 4 | Readily resolvable positive control |
| 1.5625 mm | 0.32 | 2 | Below-Nyquist positive control |
| 0.78125 mm | 0.64 | 1 | Nyquist-boundary diagnostic; phase/PSF sensitive, not an unconditional pass requirement |
| 0.390625 mm | 1.28 | 0.5 | Above-Nyquist negative control; do not require recovery |

Each group has six bars with an off-grid spatial offset. A long off-grid vertical
edge checks exact partial volume and edge location, not a standardized slanted-edge
MTF claim. Three nonoverlapping contrast/relaxation blocks support later controlled
mapping diagnostics. No claim is made that all 256 grid cells are anatomically
independent measurements.

The generator uses exact rectangle/cell intersection areas, not resampling of
a 32 image. Density Fourier truth is the closed-form rectangle sinc integral,
independent of any acquired trajectory or reconstructed image. At200 mm,
`Δk = 5 cycles/m`; a conventional even 256 Cartesian axis runs −640 to 635 cycles/m.
The geometric checks observed maximum area error 8.7e-19 m², exact fine-to-coarse
cell-average agreement 1.1e-16, and density-Fourier DC integral agreement 8.7e-19.
Downsampling this geometry to32 then nearest expanding loses 28.46% relative L2
PD detail. That last number describes this geometric control only; it is not MRI
reconstruction error or a proposed acceptance threshold.

Before this artifact becomes a benchmark object, the contracts/evaluator work
must implement the continuous material-region model consistently in the GPU
Bloch integration and the independent reference; validate actual 256 acquisition
coverage/timing; test one correct sequence plus a deliberate32-coverage sequence
and a missing-high-frequency control; and set physically meaningful acceptance
criteria. Geometry creation is complete. Acquisition/evaluator qualification is
**not** complete. No paid agent work should resume on this proposal yet.

## Higher-resolution anatomy: concrete next option and limits

For an anatomy track, use data whose acquired in-plane sampling is natively finer
than 0.78125 mm, with documented provenance/rights, and retain its physical scale.
Do not stretch the existing 1.5 mm cardiac labels and call them new anatomy.

A concrete research-data candidate is NYU fastMRI knee: the original
[dataset paper](https://arxiv.org/html/1811.08839v2#S4.SS2) documents 320 × 320
acquisition at 0.5 × 0.5 mm in-plane and 3 mm slices. That is an acquired in-plane
sampling candidate, not a cardiac phantom or a full set of quantitative PD/T1/T2
maps. A contrast-weighted magnitude image must not be relabelled as proton density
or tissue truth. A Bloch phantom would require separately justified segmentation
and tissue parameters; its segmentation uncertainty must be reported. Retain the
native 160 mm support within a 200 mm FOV rather than stretching it to fill 200 mm.

The current official [fastMRI data agreement](https://fastmri.med.nyu.edu/)
requires registration and restricts redistribution, including derived data;
publication-image use also needs attention to its stated permission process.
It is not a drop-in openly redistributable benchmark asset. No data were fetched,
agreement accepted, or license rights asserted. For cardiac-specific claims,
obtain an authorized native-submillimetre cardiac dataset with acquisition metadata
and validated masks rather than silently switching anatomy to knee.

An openly downloadable atlas on a 0.5 mm output grid is not automatically an
independently acquired 0.5 mm patient scan either. For example, the official
[Colin27 high-resolution description](https://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres)
describes averaged scans on a 0.5 mm grid. It is useful model provenance but does
not alone certify effective resolution. This is the same distinction the revised
benchmark must preserve.

## Figure and mapping claim corrections

The existing [reference caption](../paper/reference_images.caption.txt) already
states that the dense 256 views are re-evaluated reference models, that tasks remain
32 × 32, and that source anatomy is1.5 mm. The implementation confirms this:
[build_reference_figure.py](../paper/build_reference_figure.py:159) changes only the
view case matrix and reuses the original RF response/acquisition. Keep this caveat
prominent in any poster; “256 × 256 reference views” by itself can mislead.

Suggested standalone reference-figure caption opening:

> Model-reference illustrations from32 × 32 benchmark tasks (200 mm FOV; nominal
> 6.25 mm task sampling). The displayed 256 × 256 spatial-model views are not
> acquired 256 × 256 images and do not demonstrate high-resolution imaging. The
> source is a 1.5 mm piecewise-constant cardiac tissue-label map.

The selected [agent-output figure README](../output/pdf/agent-generated-examples-20261020/README.md)
already called the T1 example a source image and a single-parameter fit, with
known PD/T2. Its caption, README and provenance now explicitly disclose the radial
and T1 acquisition/evaluation matrices of32 ×32,200 mm FOV and6.25 mm nominal
sampling, plus the1.5 mm source-anatomy cells, because “native samples” without
their size is insufficient. Prior text was snapshotted under
[history](../output/resolution256-phantom-audit/history/20261020T222651Z-agent-examples-text-resolution/).
The [text-only correction receipt](../output/resolution256-phantom-audit/agent-examples-text-correction.json)
verifies that all exports and numerical data remained byte-identical. The embedded
labels in the existing raster/PDF/SVG were not regenerated; use the corrected text
caption when presenting those exports. Their huge PNG export dimensions are
layout pixels, not acquired MRI resolution. Do not replace saved images by
upsampled displays and present them as new sequence performance.

The T1 fit is more constrained than general quantitative clinical mapping.
[s4.py](../src/mriseqbench/evaluation/s4.py:242) builds a fixed tissue/RF dictionary,
varies only the myocardial T1 column, holds PD and T2 fixed, uses the known geometry
and other-tissue reference contribution, then compares mean magnitude over a
known myocardial ROI. [_mapping](../src/mriseqbench/evaluation/s4.py:298) fits one
scalar parameter to those framewise ROI means. It does not estimate an independent
T1 value in every pixel, unknown PD, T2, B1 or B0. Its tight numerical fitting error
therefore does not imply equally accurate general clinical T1 maps. A precise
label is **“constrained myocardial-ROI T1 recovery under known phantom PD/T2”**.

No existing main.pdf, poster or source results were overwritten by this audit.
