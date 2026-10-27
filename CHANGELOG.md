# Changelog

## v1.0.0 — 2026-10-26

First tagged source release of MRISeqBench.

- Defines 27 MRI sequence tasks on three hardware profiles (81 full-suite cases),
  with 128×128 imaging grids and 128×128×32 volumes, plus the fixed 2D RF assay.
- Includes independent Pulseq waveform checks, native Float64 CUDA Bloch replay,
  BART reconstruction, task-specific evaluation and reference qualification tools.
- Provides Python and Julia authoring arms, optional simulator access, and Julia
  MCP integration with pinned runtime dependencies.
- Adds stable T/H/S/P configuration labels and CLI aliases. Existing manifest
  identifiers, resolved contracts and case IDs retain their representation.
- Includes resumable campaigns, submission and cost accounting, provider retries,
  context-aware output limits, queued streaming, and optional shared API admission.
- Supports remote offline author containers, verified artifact transfer, live
  simulation capacity controls, and up to 128 concurrent authors. Transport and
  local context-budget failures remain unscored infrastructure outcomes.

Raw campaign data and operational receipts are not release assets.

### Compatibility and validation scope

Install from the tagged repository checkout using `uv sync --locked --group dev`;
the catalog, Julia environment, BART and author images are also required for full
execution. CUDA is required for grading; there is no CPU simulation fallback.
The package version is 1.0.0; individual manifest versions remain independent.

Software regression tests and native CUDA controls validate implementation
behavior. This release does not represent a fresh full reference-qualification
campaign, a new model benchmark campaign, or scanner/in-vivo validation.

Release validation:

- Full regression suite: 615 passed; 10 opt-in CUDA checks skipped in that run.
- Final remote-author, infrastructure, resume and native CUDA checks: 80 passed,
  including eight of the opt-in CUDA checks.
- Full RF-evaluation reader controls: both remaining opt-in CUDA checks passed.
- Catalog manifests validated; Ruff and diff checks passed; source distribution
  and wheel built successfully as version 1.0.0.
