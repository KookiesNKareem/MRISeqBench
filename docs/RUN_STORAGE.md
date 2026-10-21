# Run storage

Every new harness trial writes `accounting.json`, also embedded under `accounting`
in its run result. Each campaign writes a root `accounting.json` with aggregate
time and usage. Accounting receipts survive setup/evaluator exceptions without
creating a scientific verdict.

Trial `wall_s` includes preparation, agent execution, evaluation and cleanup.
`setup_wall_s` ends at agent launch; `julia_setup_wall_s` isolates persistent Julia
warmup when applicable. `agent_wall_s` includes grading waits, while
`agent_charged_wall_s` subtracts only trusted submission-grading time overlapping
agent execution. Compilation, tools, preflight and agent-side simulation count.
`grading_wall_s` includes submission grading and post-agent evaluation. These
fields overlap and must not all be added together. Campaign `wall_s` also includes
shared runtime qualification and phantom materialization. Subprocesses save
`*.execution.json` with UTC timestamps, elapsed and charged seconds.

Every model-backed run (`agent.uses_model`, default `true`) requires `api_proxy`,
a pinned `agent.model` and `api_proxy.provider_endpoint`. Deterministic command
fixtures declare `agent.uses_model: false`.

## Resume, recovery and spend

Rerunning `mriseqbench run` with the same `--out` resumes the campaign; one
process may own a directory at a time. `campaign.json` records the experiment
(except `repeats`), a hash of the running code and catalog, and arm image IDs;
a resume with any of these changed is refused. `--cases <glob>` and `--repeats N`
only select trials, so a canary and the full campaign can share a directory.

Each trial `<case>--r<k>` is decided from its files: `result.json` is final.
Frozen submission bytes without a recorded result (an interrupted grade) are
regraded without rerunning the agent; a session cut short this way keeps its
last frozen submission as the verdict and records `resumed`. Any other
interrupted attempt gets a fresh sibling `<case>--r<k>.retry<n>`: freely if it
cost nothing (no provider cost and no successful model response), otherwise at
most `retries_after_spend` times (default 1), after which the trial is finalized
as `infrastructure_interrupted`. Exclude that status from scores; all attempts
remain in the campaign accounting.

`api_proxy.max_campaign_usd` (required for model runs) and `max_trial_usd` cap
provider-reported cost. Once a cap is reached the proxy refuses further model
requests: a trial cap ends that agent (`agent_trial_budget_exhausted`); the
campaign cap, or an upstream HTTP 402, stops dispatch and exits with status 3.
Requests already in flight can overshoot a cap by their own cost. HTTP 429 and
5xx responses are retried by the proxy up to four times with back-off (honouring
`Retry-After`, at most 60 s each); the wait is excluded from the agent clock and
logged as `upstream_retry`. A trial that still fails on the provider is left
resumable. The campaign `accounting.json` reports `spent_usd`.

`mriseqbench run` exits 0 when every selected trial is final, 1 when resumable
trials remain, 2 on configuration errors and 3 on a spend stop. SIGTERM stops
agents, records interruptions and exits; restarted runs resume. Arm containers
carry a per-campaign label and orphans are removed at start. See
[`ops/README.md`](../ops/README.md) for the systemd setup on Ultron.

The proxy records started/progress/finished events in `api.jsonl`, inspecting streaming
SSE or non-streaming JSON without saving response text or credentials. It records
provider-native input/output/total tokens, cached input, cache writes, reasoning
tokens and `cost_credits` from the provider's `usage` object. Costs are reported
credits, not an estimate from a price table. Cache counters are input details and
reasoning is an output detail; neither is added to total tokens again. Repeated
usage events within one API request are cumulative; separate API requests,
including agent retries, are summed. `/models` requests do not count as generations.

`usage.totals` is populated per field only when coverage is complete; absent
optional counters remain `null`. If any generation has missing or interrupted
usage, totals are `null` and `usage.observed_totals` preserves the known portion.
Check `usage.status`, `requests_missing_usage` and `in_flight_requests` before
comparing runs. At shutdown the host revokes agent access and cancels outstanding
upstream streams. Usage that the stream did not report stays missing; it is not
looked up afterwards.

`api.jsonl` also records each effective provider/sampling/reasoning/session policy,
tool-schema hash and observed response-cache status. Accounting reports include
`resolved_providers`, `provider_resolution_complete`, `response_cache_hits` and
`cached_input_fraction`. Missing provider evidence is reported explicitly; a requested endpoint is not relabeled as an observed one.
See [shared runner controls](ARMS.md) for version, cache and concurrency settings.

A terminal routing/provider error before submission interrupts the trial without
a scientific verdict, including when Pi exits with code zero. HTTP errors and
errors inside successful HTTP streams are recorded without saving error text in
the API journal. Safe structured provider names/error codes/types are retained
when supplied, including errors that omit a top-level provider field.
A recovered request followed by normal agent completion remains
eligible for scoring.

OpenRouter automatically includes usage in the final SSE event or JSON response;
no extra request flag is needed. See its
[usage accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

Completed S1–S4 evaluations finalize storage automatically through `evaluate()`.
Image bundles share read-only SHA-256 blobs in `runs/.artifacts`; their original
paths remain readable. Replay checkpoints are losslessly compressed, verified,
and recorded in `storage-finalization.json` before removal. Interrupted or
unresolved jobs keep restart files. Converged uncalibrated results may be archived
without changing their verdict. Reports, submissions and frozen source stay intact.

Keep `.artifacts` with the runs. Receipts use absolute paths, so restore in the
original checkout. Never edit shared blobs in place or remove referenced blobs;
there is no automatic garbage collection. Use a fresh output directory to regrade.

Restore exact checkpoint bytes and modes from the relevant receipt:

```bash
uv run python -m mriseqbench.storage restore \
  runs/<campaign>/<case>/evaluation/storage-finalization.json
```

Restore refuses to overwrite changed checkpoints. Restored files can be archived
again without duplicating blobs. Evaluations outside `runs` use local `.artifacts`.
