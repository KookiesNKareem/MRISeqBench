# Run storage

Completed built-in S1–S3 evaluations automatically finalize their storage through
the common `evaluate()` entry point. Storage does not change scoring, numerical
precision, result JSON, submission receipts, or frozen source snapshots.

- `runs/.artifacts/sha256/` stores each image bundle once by its SHA-256. Original
  `images.npz` paths remain readable regular files, hard-linked to read-only blobs.
  The evaluator publishes new bundles by atomic replacement, never by modifying
  a shared inode. Evaluations outside a `runs` tree use a local `.artifacts` folder.
- `storage-finalization.json` seals the full returned result and artifact hashes.
  After a terminal verdict, replay `.npy` checkpoints are losslessly compressed
  into `runs/.artifacts/replays/`, verified by decompression, and recorded in the
  receipt before their original replay paths are removed. All checkpoint bytes
  remain recoverable; this is archival rather than evidence deletion.
- Interrupted, unresolved, and evaluator-error jobs keep their restart files.
  Uncalibrated results are finalized only when numerical convergence is explicitly
  confirmed; their verdict remains uncalibrated. Mutable agent files and source
  snapshots are never deduplicated.

Keep `.artifacts` with the runs when backing up or moving them. Receipts currently
contain absolute paths; restore in the original checkout. Shared blobs must not
be edited in place or removed while receipts reference them. There is no automatic
garbage collection of historical evidence.
Use a new output directory when rescoring or reevaluating completed evidence.

## Migrate existing runs

Run migration while campaigns are stopped. If a campaign must remain active,
exclude its directory explicitly:

```bash
uv run python -m mriseqbench.storage migrate runs
# Or: ... migrate runs --exclude runs/active-campaign
```

Migration inventories every original file and symlink before changing storage.
Only image bundles referenced by completed results are deduplicated. Existing
reports, hash receipts, and source snapshots remain byte-for-byte unchanged.
Completed replay checkpoints are archived with their original SHA-256 and size.
No existing evidence is discarded or numerically re-encoded.

The command prints the migration directory under `runs/.storage-migrations/`.
It contains `before.json`, image and checkpoint indexes, and `verification.json`.
Verification reads every original file or its lossless checkpoint archive:

```bash
uv run python -m mriseqbench.storage verify runs/.storage-migrations/<migration-id>
```

Finalization is restartable: blobs and receipts are persisted before cache removal.
If migration is interrupted, its original manifest can still be verified; checkpoint
mappings are recovered from the finalization receipts. Rerunning migration finishes
remaining work without rewriting historical receipts.

## Restore restart checkpoints

```bash
uv run python -m mriseqbench.storage restore \
  runs/<campaign>/<case>/evaluation/storage-finalization.json
```

Use the actual receipt path printed in the migration checkpoint index. Restore
verifies the archive and recreates the exact original bytes and file mode. It
refuses to overwrite a checkpoint with different contents. Another finalization
can archive restored checkpoints again without duplicating their shared blobs.
