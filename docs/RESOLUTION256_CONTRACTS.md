# Draft 256 × 256 two-dimensional contracts

The human selected a **256 × 256 2D baseline** and **holding further paid work**. This draft overlay leaves all active v1 tasks, evaluators and completed results untouched. It contains 23 imaging-task records: 18 schema-loadable `@2` candidates and five blocked protocol redesigns. None is qualified for agent dispatch.

At 200 × 200 mm FOV, the nominal pixel pitch is **0.78125 mm**, k-space grid spacing is **5 cycles/m**, and the even Cartesian grid covers **−640 through +635 cycles/m**. Non-Cartesian radius must reach approximately 640 cycles/m with adequate independent angular/radial coverage. Increasing a reconstruction grid or plot export does not create those samples.

## Inventory and timing changes

Durations below are proposed acquisition envelopes, not measured simulation runtimes or qualification receipts. Arithmetic uses the stated nominal TR; actual RF/ramp/ADC timing, preparation, slew, contrast and image acceptance still need reference validation.

| Task | New acquisition prescription | Draft duration | Qualification limit |
|---|---|---:|---|
| gre_t1w | 256 Cartesian lines plus128 dummyTRs, TR24 ms established reference preparation | 10 s | 6.144 s imaging +3.072 s dummy preparation =9.216 s; waveform/native qualification pending |
| gre_t2star | 256 lines, TE20/TR30 ms | 9 s | 7.68 s; old6 s would be too short |
| spin_echo_t2w | 256 line excitations, TE80/TR1000 ms | 270 s | 256 s; old40 s would be too short |
| bssfp | 256 lines, TE2.5/TR5 ms | 6 s | 1.28 s; balance/phase/contrast required |
| radial_readout | 403 full spokes over180°, TE3/TR12 ms | 8 s | ceil(πN/2), waveform/PSF control required |
| ute_radial | 805 half-spokes over360°, TE0.5/TR6 ms | 8 s | ceil(πN), reach radius640 cycles/m; sample convention requires validation |
| spiral_gre | 32 rotated interleaves ×8 turns, TE5/TR30 ms | 6 s | Product256 ≥ nominal N/2=128 heuristic; actual maximum gap and gradient/slew matter |
| epi_single_shot | One256-line train, provisional TE180 ms,2 µs dwell | 0.5 s | Changed long-TE contrast; no qualified high-resolution EPI claim |
| epi_multishot | Eight interleaved shots ×32 lines, provisional TE30 ms,2 µs dwell | 8 s | At1.2 ms illustrative spacing,38.4 ms train/shot; full waveform timing still required |
| tse_t2w | 32 shots ×8 refocused echoes, TE40/TR1200 ms | 42 s | 38.4 s nominal bound; ordering/contrast required |
| grase | 32 shots ×8 lines, four spin echoes/two gradient echoes each, TE40/TR1000 ms | 36 s | 32 s nominal bound; echo identity required |
| propeller_tse | 51 blades ×8 lines, TE40/TR1000 ms | 55 s | Strip-coverage heuristic ceil(πN/(2×8)); actual PSF required |
| sense_sampling | Even ky plus32 ACS (112–143),144 lines, TE3/TR12 ms | 6 s | 1.728 s; SENSE sensitivity/conditioning qualification |
| grappa_sampling | Same144-line mask, TE3/TR12 ms | 6 s | ACS width32 is proposed, not demonstrated sufficient |
| compressed_sensing | Deterministic86-line1D hard-core mask including32 ACS, seed42 | 6 s | Actual frozen prescription exported; not a2D Poisson-disc acquisition |
| variable_density | Deterministic86-line1D weighted mask including32 ACS, seed42 | 6 s | Actual frozen prescription exported; reconstruction conditioning required |
| partial_fourier | Upper192 ky lines64–255 inclusive, TE3/TR12 ms | 6 s | 2.304 s; phase/reconstruction constraints remain |
| reduced_fov | 256×128 acquired points,100 mm phase FOV, TE80/TR1000 ms | 135 s | 128 s; true inner-volume response/suppression required |
| rfov_epi | Would require256×128, single shot, original 2D-selective response constraints | **Blocked** | Readout/response-floor contract requires requalification |
| adiabatic_inversion | 256-line prepared ECG bSSFP image | **Blocked** | Current one-window frame model does not represent segmented heartbeats |
| t2_prepared | 256-line prepared ECG bSSFP image | **Blocked** | Same temporal/model blocker |
| t1_mapping | Three256-line prepared ECG bSSFP images | **Blocked** | Segment preparation/TI/recovery and fitting assumptions must be versioned |
| t2_mapping | Three256-line prepared ECG bSSFP images | **Blocked** | Segment preparation/recovery and fitting assumptions must be versioned |

The five excluded task files are `pulseq_smoke` (infrastructure smoke), `rf_excitation_2d` (64² excitation response rather than imaging), and three3D tasks (`stack_of_stars`, `caipirinha`, `elliptical_kspace`). They are not silently converted to2D or included in a new23-task agent cohort.

## Explicit redesign blockers

For the four ECG-prepared tasks, the old200 ms acquisition window cannot simply contain256 old-style bSSFP TRs: at previously observed2.3–4.6 ms TR it needs588.8–1177.6 ms. At provisional5 ms TR, at most40 lines fit each200 ms window, so at least7 heartbeats/image are needed before preparation/recovery. Three-image mapping would need at least21 imaging heartbeats plus recovery. These are planning bounds, not a new qualified segmented protocol. The present evaluator treats a trigger train as a whole frame, so a segmented design requires an explicit new frame/segment contract and independent references. Do not invent longer total durations while leaving that semantic mismatch intact.

For RFOV-EPI,256×128 samples replace32×16 while preserving the central100 mm phase FOV. At illustrative1.2 ms echo spacing the train lasts153.6 ms. An elementary T2=80 ms exponential-response screen falls below the original0.20 interior floor after128.76 ms even at unit initial response. This flags a conflict to investigate; it is **not** proof of physical impossibility or an evaluator defect. Exact RF dynamics, first/central/last line-center checks, target normalization and a qualified faster trajectory must be examined before choosing a protocol. Increasing TE alone is insufficient justification.

Single-shot EPI's provisional180 ms TE is a contrast redesign; it may produce severe T2* attenuation/distortion. The2 µs dwell implies approximately58.72 mT/m readout at200 mm FOV, within the selected80 mT/m limit before slew/ramp checks. An assumed1.2 ms spacing is a feasibility estimate, not measured output from a valid waveform. Reference validation may require another TE, acceleration or shot structure; changing one-shot identity requires an explicit new contract.

## Independent spatial truth and release gate

The inherited `cardiac_2d@1` phantom is approximately90×89 at1.5 mm native sampling; using it on a0.78125 mm reconstruction grid does not establish0.78125 mm anatomical detail. These files retain its reference only so draft task schemas/families can be exercised. A genuinely higher-detail object/target and independent small-feature/PSF controls must replace or qualify it before high-resolution claims. The separate phantom audit owns that decision. Mapping fit errors must state known-PD/known-T2 priors and be checked against actual preparation/readout dynamics; a small constrained fitting error alone is not spatial-resolution evidence.

The smallest next validation is one actual256×256 Cartesian reference sequence with measured coverage/timing, GPU signal replay, independent target and a deliberately low-k truncated negative control. Only after those pass should additional families be qualified. No paid agent campaigns or larger old-result reconstructions are implied by this overlay.

## Checks performed

`benchmark/variants/256x256/audit.py` validates all18 candidate documents with the actual strict `Task` model, checks23 inventory rows and unchanged source task hashes, and compares both exported86-line masks against the actual frozen `prescribed_mask` implementation. This is a short local contract check. No scientific simulation, native evaluation, API call, source evaluator edit or paid inference was performed.

## Constructive segmented T1 waveform prototype

`benchmark/variants/256x256/prototypes/t1_mapping_segmented/` now contains one actual120-second Pulseq prototype:8segments×32imaging ky lines per contrast, one declared navigator/segment, three navigator-TIs100/400/1200 ms, TR4.6 ms and151.8 ms per train. The written-file audit passes hardware/timing/synthetic-SAR checks and verifies65,536 unique imaging points per contrast plus all ADC nodes within the600–800 ms window. Minimum inversion-center separation is4.2 seconds. This is a new diagnostic protocol, **not** a released task or a result under S4@1. Native signal dynamics, image/map accuracy and segmented evaluator semantics remain explicitly unqualified; the blocked S4 status in `profile.json` is unchanged.

The GRE T1-weighted draft envelope was corrected from8 to10 seconds after inspecting the established reference preparation:256imagingTRs plus128dummyTRs at24ms consume9.216seconds. The earlier arithmetic covered only6.144seconds of imaging and omitted3.072seconds of dummy preparation. A shorterTR reference is not established by this duration correction; full waveform/native-image qualification remains pending.

Dwell time is not fixed universally by the256 matrix. Drafts preserve any original explicit dwell requirement; only the two EPI proposals add a provisional2µs dwell. The GRE/radial waveform-feasibility controls use12.5µs and therefore must not be presented as evidence for a universal4µs contract. The radial control also usesTE5ms/404spokes, whereas `radial_readout@2` presently proposesTE3ms/403spokes. That control demonstrates feasibility of its own waveform only and **does not qualify the exact radial@2 contract**. The radial draft remains unqualified.
