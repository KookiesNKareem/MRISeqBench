# Run storage

Completed S1–S4 evaluations finalize storage automatically through `evaluate()`.
Image bundles share read-only SHA-256 blobs in `runs/.artifacts`; their original
paths remain readable. Replay checkpoints are losslessly compressed, verified,
and recorded in `storage-finalization.json` before removal. Interrupted or
unresolved jobs keep restart files. Converged uncalibrated results may be archived
without changing their verdict. Reports, submissions and frozen source stay intact.

Keep `.artifacts` with the runs. Receipts use absolute paths, so restore in the
original checkout. Never edit shared blobs in place or remove referenced blobs;
there is no automatic garbage collection. Use a fresh output directory to regrade.

Migrate stopped campaigns, or explicitly exclude active ones:

```bash
uv run python -m mriseqbench.storage migrate runs --exclude runs/active-campaign
uv run python -m mriseqbench.storage verify runs/.storage-migrations/<migration-id>
```

Migration inventories original files first and verifies their bytes afterward.
Finalization is restartable: durable receipts precede checkpoint removal, and
verification can recover archive indexes from those receipts after interruption.

Restore exact checkpoint bytes and modes from the relevant receipt:

```bash
uv run python -m mriseqbench.storage restore \
  runs/<campaign>/<case>/evaluation/storage-finalization.json
```

Restore refuses to overwrite changed checkpoints. Restored files can be archived
again without duplicating blobs. Evaluations outside `runs` use local `.artifacts`.
