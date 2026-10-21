# 256 × 256 revision status

The selected new baseline is a **physical 256 × 256 two-dimensional acquisition**, over 200 mm where the task uses that FOV: **0.78125 mm nominal sampling**. This is a new, held contract overlay. No completed result has been relabelled as a 256 acquisition, and no paid agent campaign is authorized on it yet.

## Paid work is held

The human requested a hold of further paid work. The definitive [hold receipt](../runs/opus55-python-pi-81-ultron-20261020/infra/resolution256-human-hold/completion.json) records zero owned paid worker PIDs, both recovery queues stopped, no DeepSeek11 release, and unchanged source/budgets/financial commitments. Opus stopped at 72 canonical trials (29 passed), with nine interrupted cases still unfinished; those interruptions are not new sequence failures. The old campaign automation is paused. Further paid work requires direct human authorization.

## Work completed by the three subagents

| Area | Concrete evidence | Limit |
|---|---|---|
| Contracts | [23-task overlay](../benchmark/variants/256x256/README.md), 18 strict-schema candidate documents and five blocked protocol records; actual sampling masks and count/timing arithmetic | All candidates remain unqualified; production v1 tasks unchanged |
| Reference acquisition/evaluator | [Actual Cartesian and radial sequences](../output/resolution256-evaluator-audit/README.md), 65,536 and 103,424 acquired ADC samples; waveform/hardware/SAR checks passed | Full image and precision qualification pending; initial GRE physical-spoiling diagnostic failed |
| Phantom | [Independent continuous geometry](RESOLUTION256_PHANTOM.md), exact rectangle/cell overlap and Fourier density oracle, line-pair controls | Geometry truth only; GPU MRI signal/reconstruction qualification pending |

These initial feasibility controls are not exact qualification of every new draft: notably the radial control uses TE 5 ms / 404 spokes, whereas the provisional radial draft specifies TE 3 ms / 403. No universal new ADC-dwell constraint was imposed; the drafts preserve existing bandwidth freedom except the explicitly provisional EPI design.

The 32-acquisition/256-contract negative is rejected by the existing waveform inspector. A larger reconstruction/export grid cannot substitute for the missing high-k acquisition.

## Measured GPU feasibility and reference failure

A single full-trajectory 256 Cartesian CUDA Float64 signal replay took 47.402 seconds including Julia startup, used 24,576 actual spatial nodes, and stayed within 1.30 GiB peak helper memory. No CPU fallback or model/API request occurred. This is feasibility evidence for one workload, not a full evaluator pass.

The first full physical GRE reference attempt took 89.522 seconds and stayed within 1.40 GiB. It stopped before image evaluation: the unchanged `spoiled_gre` mechanism criterion measured relative errors 0.296143 / 0.296154 against a 0.2 limit. Thus no eight-replicate image convergence result exists from that attempt. Reference preparation/spoiling needs correction; this is not an evaluator-timeout diagnosis or an old agent sequence failure. The exact failed reference, diagnostics and archived/retired helper receipts are retained. A fixture-based correction retained 12 ms TR and added 128 dummy RF shots with the established 1.3-cycles/voxel spoiler; it also failed the mechanism check (0.309740 / 0.308753 versus 0.2). Read-only comparison then identified an important protocol mismatch: the established reference uses 24 ms TR. Its 256 imaging shots plus 128 dummy shots take 9.216 seconds. The GRE draft now has a 10-second envelope. The final bounded check restored 24 ms TR, 128 dummy RF shots and the established spoiler moment, with only spoiler duration stretched from 0.8 to 1.0 ms to meet the larger gradient-area hardware limit. ADC coverage remained 65,536 points; acceptance thresholds were unchanged.

That [full eight-replicate check](../output/resolution256-evaluator-audit/qualified-timing-reference-r4/image-summary.json) **passed waveform/physical mechanism validation**, including spoiling errors 0.135117 / 0.134065 below 0.2. It finished on CUDA Float64 in **140.20 seconds**, using **1.66 GiB** peak helper memory, with no CPU fallback. Temporal replay error was 6.97e-5 against 0.01. However the active case failed image phase: 0.427295 rad, with the existing approximate 99% numerical interval [0.388537, 0.462819], above the 0.35-rad limit. Case complex NRMSE 0.371939 and magnitude NRMSE 0.314805 passed their 0.45 / 0.35 limits; the separate ideal complex-image diagnostic also failed at 0.381680 versus 0.2. The existing estimator resolved a failure from its intervals, **not by meeting every score-precision target**. This establishes bounded full-evaluator feasibility for one 256 case, not a qualified positive reference or general transfer of convergence. Residual physical coherence, spatial/oracle modelling and phase-reference consistency are not yet isolated. No old agent result or threshold was changed to explain it away. The successful helper process was archived and retired; its scientific reference outcome remains failed/unqualified.

## Protocol and phantom issues that a matrix edit cannot solve

The current S4 judge requires a full Cartesian frame in one 200 ms cardiac window. Its existing 2.3–4.6 ms TRs cannot acquire 256 lines in that window. The [segmented T1 prototype](../benchmark/variants/256x256/prototypes/t1_mapping_segmented/README.md) acquires three contrasts × eight segments, each with 32 imaging lines plus an explicit timing navigator. The 120-second waveform has TR 4.6 ms, measured TE 2.295 ms, all ADC samples 601.215–750.965 ms after the selected cardiac trigger, and 65,536 unique imaging points per contrast. Hardware/timing and synthetic SAR checks pass. It still needs a versioned segment/contrast/trigger contract and independent signal/fit validation; an assembled contrast is not an instantaneous single-TI image. See [contracts](RESOLUTION256_CONTRACTS.md). RFOV-EPI likewise remains a separate protocol-response qualification problem. No old thresholds were loosened to force either through.

The inherited cardiac labels are 90 × 89 at 1.5 mm in-plane. Their exact piecewise-cell model can be sampled more densely but cannot provide independent submillimetre anatomical truth. The new analytic insert provides independent small-feature geometry; its one-pixel/Nyquist-boundary group is a diagnostic, and its half-pixel group is an above-Nyquist negative, not a demanded positive result. A genuinely finer anatomical track additionally needs documented native data and appropriate material maps/provenance.

The existing T1 example is a constrained myocardial-ROI fit with known PD/T2 and known geometry, not general voxelwise clinical mapping. That caveat and the actual 32 × 32 acquisition now appear prominently in its caption/provenance.

## Current figures

[main.pdf](../output/pdf/arm-comparison-figures/main.pdf) now prominently labels existing results **legacy low-resolution** on both pages. The four existing comparison figures and their numerical values are byte-identical. Both pages were rebuilt, rendered and visually checked; the [receipt](../output/pdf/arm-comparison-figures/resolution-label-update-receipt.json) records hashes and the earlier-version snapshot. It contains no 256 acquisition result.

## Small release gate

Before any new paid dispatch: qualify one genuinely correct 256 reference through physical mechanism, independent image and numerical precision criteria on GPU; reject the deliberately insufficient acquisition; validate independent spatial detail; then qualify additional task families and their prompts against their actual sampling/timing protocols. Keep unresolved reference or numerical failures explicitly unresolved, and do not score them as bad agent sequences. Three-dimensional tasks and direct RF excitation require their own resolution designs.
