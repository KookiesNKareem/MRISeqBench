# 256 × 256 2D contract drafts — HELD

These files are outside `benchmark/tasks/`, so the active catalog does not load them.
No paid agent runs, simulation runs, result replacements, evaluator edits, or changes to v1 tasks are performed by these tools.

- `tasks/`: 18 schema-loadable **task@2 draft contracts** using the original task IDs/families.
- `blocked/`: five protocol redesign records, deliberately **not** runnable task documents.
- `profile.json`: all 23 imaging tasks, source hashes, sampling/time arithmetic and blockers.
- `sampling-masks.json`: actual deterministic 256-line prescriptions for the two incoherent masks.
- `audit.py`: one local schema/source/mask check; no MRI replay or model call.

From repository root:

```sh
PYTHONPATH=src .venv/bin/python benchmark/variants/256x256/audit.py
```

The first executable reference target is `gre_t1w@2`, followed by simple Cartesian bSSFP/GRE. Schema-loadable does **not** mean a qualifying sequence, image target, high-detail phantom or adequate runtime has been demonstrated. Existing object/evaluator references are a development starting point; their clinical or resolution validity is not inherited. An agent-run suite must not be created until the draft reference and independent spatial controls pass at the actual acquired resolution.

256 over 200 mm means nominal 0.78125 mm pixels, reciprocal spacing 5 cycles/m, and positive Nyquist radius 640 cycles/m (the even Cartesian grid actually runs -640 through +635 cycles/m). Merely reconstructing a 32-point trajectory onto this grid does not satisfy acquisition coverage. A Fourier bin, pixel pitch and demonstrated spatial resolving power are distinct quantities.

See `docs/RESOLUTION256_CONTRACTS.md` for the full inventory and limits.

No ADC dwell is prescribed. Drafts are curated by hand since GPU qualification
(objectives rewritten, two duration envelopes widened, EPI dwell pin removed);
a task that qualifies is promoted to `benchmark/tasks/<id>@2.yaml` and the
`full256` suite. Per-task status: [docs/RESOLUTION256_STATUS.md](../../../docs/RESOLUTION256_STATUS.md).
