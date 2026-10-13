# Authoring and simulation arms

Each arm selects installed authoring tools and optional agent-side simulation.
The trusted judge and submission budget are shared across arms.

| Arm | Authoring | PulseDesigner | Unitful | Simulator |
|---|---|:---:|:---:|---|
| `a2_pypulseq@1` | PyPulseq | — | — | None |
| `a3_koma_base@1` | KomaMRIBase / KomaMRIFiles | No | No | None |
| `a4_pulseq_julia@1` | KomaMRIBase / KomaMRIFiles | Yes | No | None |
| `a5_pulseq_julia_unitful@1` | KomaMRIBase / KomaMRIFiles | Yes | Yes | None |
| `s2_pypulseq_mrzero@1` | PyPulseq | — | — | MRzero |
| `s3_pypulseq_komamripy@1` | PyPulseq | — | — | KomaMRI / komamripy |
| `s4_koma_base_koma@1` | KomaMRIBase / KomaMRIFiles | No | No | KomaMRI |
| `s5_pulseq_julia_koma@1` | KomaMRIBase / KomaMRIFiles | Yes | No | KomaMRI |
| `s6_pulseq_julia_unitful_koma@1` | KomaMRIBase / KomaMRIFiles | Yes | Yes | KomaMRI |

Configure the model, suite and local images in the matching experiment YAML:

```bash
uv run mriseqbench arms
uv run mriseqbench doctor --arm a2_pypulseq \
  --author-image mriseqbench/author-pypulseq:local
uv run mriseqbench run --experiment experiments/pi_a2_pypulseq.yaml
```

Restricted arms require local Docker images and run offline with read-only
runtime files. The host catalog, judge, credentials and Docker socket are not
mounted. Missing tools or failed audits stop the run; there is no host fallback.
Build recipes: [`runtimes/arms/`](../runtimes/arms/).

Simulation uses a separate container and fixed adapter receiving exact submitted
Pulseq bytes plus public phantom/field inputs. It returns simulation artifacts,
not judge scores, and cannot execute arbitrary agent programs. Use
`./bin/simulate [sequence.seq]` when the selected arm enables simulation.

Julia arms use a persistent process through `./bin/julia`; permitted packages and
generic warmup load before timing begins. Agent compilation and execution count
against the trial budget. Use `includet("design.jl")` for Revise-based edits.

Restricted arms remain offline; report the selected arm when comparing results.
