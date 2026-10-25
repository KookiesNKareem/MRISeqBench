# Local storage

Keep generated figures and figure source code locally. Raw transcripts, campaign
results, numerical arrays, submitted sequences, frozen run-source snapshots,
logs, and detailed inventories belong on **Ultron**, under
`/ssd-pool/kfareed/MRISeqBench-results/`. Keep only compact storage navigation
and the latest cleanup receipt locally. This policy does not remove tracked
benchmark inputs, references, fixtures, or project source.

Run experiments and result analysis on Ultron. Render figures there when they
depend on raw data, then fetch only the figure files and plotting source. Do not
mirror campaign trees or raw reports to the laptop, including from monitors.

For existing local generated data, use
`python3 scripts/offload_local_results.py` to inspect the inventory, then add
`--apply` to archive it. Run while local writers are idle. The script checks every
remote file by SHA-256 and preserves changed local files before removing any
verified copy. Full manifests and cleanup receipts stay in the remote archive;
`docs/local-storage-receipt.json` records the latest destination. Absolute-path
replay receipts need the original checkout layout when restoring.
