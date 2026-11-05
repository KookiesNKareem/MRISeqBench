# Agentic benchmark

Set `agentic: true` in an existing experiment to continue after a successful
submission. Each subsequent user request
comes from that task's `followups` list, which contains a request and a patch to
the sequence specification. All 27 tasks have follow-ups; list length is unrestricted.

The same Pi process, conversation, source files and authoring runtime persist.
Only a pass unlocks the next request. Patches accumulate, and every stage runs
the existing physical/hardware/image grader against the complete current
specification. Requirements persist within their declared tolerances. A later
request can constrain a parameter that the original task left unspecified.
Future requests stay hidden from the agent until their turn.

Follow-ups establish previously free parameters alongside an earlier edit, so
later requests require a physical change. UTE's initial TE tolerance is 0.02 ms;
its follow-ups use 0.95 and 0.98 ms with disjoint tolerances. Changed stage
contracts require fresh reference qualification; earlier receipts do not
qualify the revised stages.

The existing `agent.timeout_s` is the total workflow budget; `max_submissions`
applies per stage. Total and per-stage cost/token usage come from the same API
log, split at user-turn boundaries. Missing usage remains unknown. Results include
stage outcomes and submission counts; unreached stages are marked `not_reached`.
Complete workflow success requires every stage to pass.

Frozen submissions retain their effective grading cases. Completed workflows
recover from checkpoints. If an unfinished conversation is lost, the existing
infrastructure policy reruns the entire workflow in a fresh retry workspace;
an intermediate pass never becomes a complete-workflow pass.

Use the normal CLI and a fresh output directory for agentic experiments:

```sh
uv run mriseqbench validate
uv run mriseqbench describe T1.1 --hardware H1 --stage 2
uv run mriseqbench run --experiment experiments/my_experiment.yaml --out runs/agentic
```

`describe` and `evaluate` accept `--stage N` to inspect or grade an individual
stage (zero is the original task). Omitting it preserves ordinary single-task
behavior. Agentic execution requires explicit submissions and the Pi adapter
for model runs; non-model command adapters remain available for harness tests.

The earlier GPU qualification audit had passing controls for all four stages
of all 27 tasks on the 0.55 T, 1.5 T and 3 T profiles under the former image
limits. Visual review then found ghosting and streaking in passing images.
The stricter image policy (magnitude NRMSE 0.10, complex NRMSE 0.15,
outside-object energy 0.01) supersedes that quality qualification; complete
passing-control coverage must be re-established under the new limits.
In the earlier edit-validation audit, for every workflow the
previous passing pulse is rejected at each next stage, and the original pulse
is rejected at the final cumulative stage (324 rejection checks in total).

UTE and spiral follow-ups change flip angle and TE: the existing non-Cartesian
coverage rules cannot reliably distinguish their earlier larger-FOV requests
from unchanged oversampled acquisitions. The reduced-FOV reference now applies
ideal balanced-crusher coherence selection at each refocuser, retaining finite
RF effects and longitudinal history. Its reference correction was verified with
full-image GPU controls at all three field strengths under the former limits.
Native subagent trials exercise the real stage server and grader,
but do not validate Pi RPC or API cost/token collection.
They are not blind benchmark scores: native actors have host access, and a
stored TSE trial read private reference-image arrays from the judge directory.
Use the isolated Pi runner for scored agent trials; native subagents remain
useful for reference authoring and grading/integration checks.
