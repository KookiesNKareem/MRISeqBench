# Read-only campaign monitor

Run from the repository with the development environment (Matplotlib required):

```sh
uv run python ops/monitor/server.py --campaign /absolute/path/to/campaign --port 49211
```

The campaign must supply `queue.json`, `status.json`, trial receipts/results and,
for token comparisons, the collected CSV files. The monitor shows simulator and
graded submissions in one progression, grouping simulator calls before each graded
submission as S1.1, S1.2, then S2.1 after submission 1. Candidate hashes bind each
preview to its actual submission. It neither launches agents nor changes grades.
Serve on loopback only; keep this service outside the author filesystem boundary.
`dashboard/server.json` records the process and URL in the campaign directory.

The native pilot's orchestration scripts were its queue manager: they launched
bounded concurrent authors, recovered accepted submissions and collected reports.
That is process management, not another model reasoning about the sequence.
Those campaign-specific launchers are not required by this read-only monitor.

The native subagent sandbox is distinct from the production Docker arm sandbox.
Native temporary-file restrictions do not imply a Docker-arm write failure:
Docker arms already have a private writable `/tmp`.
