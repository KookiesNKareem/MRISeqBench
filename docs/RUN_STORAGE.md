# Run storage

## Location

Figures and plotting source stay on the laptop. Raw transcripts, results, arrays,
submitted sequences, frozen sources and logs belong on
`Ultron:/ssd-pool/kfareed/MRISeqBench-results/`; analyse and render there, then
fetch only figures. `python3 scripts/offload_local_results.py` inventories local
generated data; `--apply` archives it under `archives/laptop-generated-<UTC>/`,
verifies every hash and removes the unchanged local copies. The latest receipt is
`docs/local-storage-receipt.json`.

Historical reference and qualification outputs, along with the retired 256×256
prototypes, are archived under
`Ultron:/ssd-pool/kfareed/MRISeqBench-results/archives/laptop-pruned-20261021/`.
Current reference authors and qualification commands are documented in
[`benchmark/references/README.md`](../benchmark/references/README.md).

## Resume

Rerunning `mriseqbench run` with the same `--out` resumes a campaign; a resume
with a changed experiment, code/catalog hash or arm image is refused.
`--cases <glob>` and `--repeats N` only select trials.

Each trial `<case>--r<k>` is decided from its files. Outcomes are binary:
`passed` or `failed` (with `failure_kind`, e.g. `no_submission`,
`agent_timeout`, `agent_error` for sessions that ended without a submission).
`result.json` is final. An agent session cannot be resumed, so an interrupted
trial is completed from its files only if its session had already ended:
`session.json` records a genuine end (agent exit, deadline or final
submission), or the last accepted submission ended it (a pass, or the final
allowed attempt). Its last submission is then graded, or its recorded verdict
reused. Any other interruption (infrastructure, Docker, model API, trusted
grading or storage failure, including during grading of a non-final attempt)
voids the trial: it reruns in a sibling `<case>--r<k>.retry<n>` with the full
time budget and submission allowance, and the void workspace is kept only for
audit. Infrastructure interruptions are never terminal, however often they
recur or whatever they cost.

There are no spend caps: every selected trial runs to completion under the same
protocol. HTTP 429/5xx are retried up to four times with back-off excluded from
the agent clock; HTTP 402 (provider account empty) stops dispatch with exit 3 and
leaves unfinished trials resumable. Exit codes: 0 all final, 1 resumable trials
remain, 2 configuration error, 3 provider account empty. Campaign
`accounting.json` reports `observed_spent_usd` (the sum of recorded request costs) and `spent_usd`, which is null while any request lacks a recorded cost. `needs_review` lists trials whose verdict cites a judge/native-reader disagreement (`native_consistency`); check these by hand, since the cause may be a judge bug, and regrade if so.

## Accounting

Each trial writes `accounting.json` (also embedded in its result); each campaign
writes an aggregate. `agent_charged_wall_s` excludes only trusted submission
grading; tools, compilation and agent-side simulation count. Token and cost
totals come from the provider's `usage` objects recorded in `api.jsonl`; totals
are `null` when any generation lacks usage (`usage.observed_totals` keeps the
known part).

## Evaluation storage

Completed evaluations deduplicate image bundles into read-only SHA-256 blobs in
`runs/.artifacts` and record them in `storage-finalization.json`. Keep
`.artifacts` with the runs; receipts use absolute paths. Restore exact bytes with
`uv run python -m mriseqbench.storage restore <storage-finalization.json>`.
