# Benchmark labels

Repository configuration uses stable labels for tasks, hardware, simulators and phantoms.
The shared mapping is implemented in [`labels.py`](../src/mriseqbench/labels.py).
Use these labels in repository configuration, tables, figures and CLI listings. Labels are
stable: append future tasks within each level instead of renumbering tasks.

| Axis | Labels | Meaning |
|---|---|---|
| Task level | T1–T4 | Basic sequences; readout design; sampling; preparation and RF |
| Individual task | `T<level>.<number>` | Numbering starts at 1 within each level, e.g. T2.6 |
| Hardware | H1–H3 | Scanner limit profiles below |
| Simulator access | S1–S3 | Optional agent simulator below |
| Phantom | P1 | Static cardiac anatomy, including its 2D/3D and tissue-profile variants |

The infrastructure smoke task is unnumbered. T4.5 uses a fixed uniform-disc RF
assay instead of cardiac anatomy; its phantom label is null (shown as `assay`
in the CLI). This does not introduce a second numbered benchmark phantom.
P0/P1/P2 and J0–J7 remain the authoring arm names; `P1 S1` as an arm
means PyPulseq with MRzero, independently of phantom P1.

## Hardware

| Label | Manifest ID | Scanner | Field | Gradient (mT/m) | Slew (T/m/s) |
|---|---|---|---:|---:|---:|
| H1 | `low_field` | MAGNETOM Free.Max / B80 | 0.55 T | 26 | 45 |
| H2 | `standard_1_5t` | MAGNETOM Sola / XJ | 1.5 T | 33 | 125 |
| H3 | `high_performance` | MAGNETOM Prisma / XR | 3 T | 80 | 200 |

## Simulators

| Label | Manifest value | Interface |
|---|---|---|
| S1 | `mrzero` | MRzero / Python |
| S2 | `koma` | KomaMRI / Julia |
| S3 | `komamripy` | KomaMRI through komamripy / Python |

Disabled simulation (`none`) has no S modifier. S labels describe agent-side
access, not the trusted judge, which always uses KomaMRI.
See [authoring arms](ARMS.md) for the full configuration table.

## Tasks

Task numbering includes multi-shot EPI T2.2,
spiral GRE T2.6, PROPELLER TSE T2.7 and 2D RF excitation T4.5.

| Label | Manifest ID |
|---|---|
| T1.1 | `gre_t1w` |
| T1.2 | `gre_t2star` |
| T1.3 | `spin_echo_t2w` |
| T1.4 | `bssfp` |
| T2.1 | `epi_single_shot` |
| T2.2 | `epi_multishot` |
| T2.3 | `tse_t2w` |
| T2.4 | `grase` |
| T2.5 | `radial_readout` |
| T2.6 | `spiral_gre` |
| T2.7 | `propeller_tse` |
| T2.8 | `stack_of_stars` |
| T2.9 | `ute_radial` |
| T3.1 | `sense_sampling` |
| T3.2 | `grappa_sampling` |
| T3.3 | `partial_fourier` |
| T3.4 | `compressed_sensing` |
| T3.5 | `variable_density` |
| T3.6 | `elliptical_kspace` |
| T3.7 | `caipirinha` |
| T3.8 | `reduced_fov` |
| T3.9 | `rfov_epi` |
| T4.1 | `t1_mapping` |
| T4.2 | `t2_mapping` |
| T4.3 | `t2_prepared` |
| T4.4 | `adiabatic_inversion` |
| T4.5 | `rf_excitation_2d` |

## Configuration and compatibility

Suite entries use numbered task and hardware references, with explicit versions:

```yaml
cases:
- task: T2.6@2
  hardware: [H1@1, H2@1, H3@1]
  physics: [static_b0@1]
```

Task definitions use `sequence.level: T1` through `T4`, `object: P1@1`, and
optionally `hardware: H3@1`. P1 resolves to the 2D or 3D cardiac object according
to `sequence.matrix`; field-specific tissue properties are selected as before.
T4.5 retains `object: uniform_disc@1` for its fixed RF assay. Arm definitions
use `simulator: S1`, `S2`, `S3` or `none`; the existing language-compatibility
checks still apply. Manifest filenames and `id` fields retain their descriptive
IDs; task labels are references, separate from manifest version numbers.

Model validation normalizes labels to the existing identifiers before any
runtime contract is created. Resolved cases, agent briefs, agent workspaces,
simulator requests and grading contracts keep their existing representation.
Legacy manifest references and values continue to work. In serialized contracts,
`sequence.level` remains `S1`–`S4`; these are historical task-routing keys.
The repository configuration uses T1–T4 for task levels and S1–S3 for simulators.
Case IDs, evaluator IDs (`s1`–`s4`) and evaluation document filenames retain
their existing spelling for saved contracts and campaign provenance.

For repository analysis and listings, `case_labels(case)` from
`mriseqbench.labels` computes the task, level, hardware and phantom labels
without modifying the contract. Simulator modifiers are available through
`SIMULATOR_LABELS` and `arm_label(arm)`.

`describe` accepts task and hardware labels, optionally with an explicit
manifest version (e.g. `T2.6@2`). Existing manifest references still work:

```bash
uv run mriseqbench list --suite full
uv run mriseqbench describe T2.6 --hardware H3
uv run mriseqbench arms
```

`--cases` and `evaluate --case` continue to use the recorded case IDs printed
by `list`. Labels identify task/hardware/phantom axes; case IDs also identify
the versioned physics and remain the provenance and resume keys.
