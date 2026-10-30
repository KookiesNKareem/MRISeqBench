# Agentic benchmark development

Development branch: `branch/agentic`. Existing task manifests, suites, references
and experiments retain their single-turn behavior. The opt-in
`experiments/pi_agentic.yaml` selects workflows for all 27 tasks in `full@1`:
81 task/hardware workflows, with four stages each (324 stage specifications).

**Status: development, not scientifically qualified.** Every workflow is marked
`draft`. No GPU reference qualification or model-comparison campaign has been
performed for these edited contracts. The run command refuses draft workflows
unless `workflow.allow_unqualified: true` is explicitly set in a development
experiment. There is intentionally no manifest-only switch to call them qualified.

## Contracts

`benchmark/workflows/<task>.yaml` pins the base task version, an optional initial
parameter patch, and three sequential user requests with explicit patches.
Imaging workflows enlarge the in-plane FOV, edit a family-specific parameter,
then restore FOV while retaining the second edit. The excitation-only workflow
reduces target magnitude, translates the target in x, then restores magnitude
while retaining the translation.

Initial patches give otherwise unconstrained parameters explicit baselines so
later edits are meaningful (for example, GRE T1 TR 16 -> 20 ms). These baselines
belong only to the agentic suite. Each effective specification passes the same
Pydantic and geometry validation as the original task. Patches cannot alter the
evaluator, phantom, physics, scanner, task family, or arbitrary contract fields.
RF target amplitude and translation are explicit new sequence fields; default
single-turn RF evaluation retains the canonical target and metrics.

Inheritance means preserving **declared requirements within their tolerances**,
not freezing every incidental feature of the previously generated waveform.
All stages run full physical/hardware/image evaluation. A pass is required before
the next request is disclosed. Failure feedback permits up to three submissions
per stage. A failed stage ends the workflow; later stages are `not_reached`.

Future stages remain in trusted controller memory. Participants see only the
current effective `case.json`, `task.yaml`, `EVALUATION.md`, and user request.
RF target arrays are refreshed from the current trusted specification. Simulator
requests likewise use the current public case. Numeric stage specifications
supersede numeric prose in the original task description; unchanged structural
requirements remain applicable.

## Execution

The Pi adapter uses the pinned [0.87.0 RPC protocol](https://github.com/badlogic/pi-mono/blob/v0.87.0/packages/coding-agent/docs/rpc.md). One Pi process and one
conversation persist across stages. After `stage_passed`, the agent finishes its
response. The driver waits for `agent_settled` (not merely `agent_end`), obtains
the next request, updates public files, and sends a real RPC `prompt` user turn.
A pass at an intermediate stage does not terminate the proxy or author runtime.
Authoring source and runtime state remain in the same workspace.

The example gives each stage 3,600 charged seconds, within a 14,400-second total
allowance. The same existing grading/queue/backoff exclusions apply. Unused stage
time cannot extend a later stage beyond its allowance. Submissions are indexed
both globally and within each stage. Current status reports both stage and total
budget information. Model usage and recorded cost are attributed by host-proxy
request slices at user-turn boundaries; missing usage remains unknown.

Every submission freezes sequence bytes, SHA-256, the exact effective case,
stage identity, and evaluation in trusted `control/submissions/attempt-NNN/`.
`workflow-state.json` checkpoints stage outcomes. Result reports include every
stage, attempts, usage, timing and grading details; `workflow-summary.json`
reports completed workflows, complete passes, and stage denominators.
These are completed-trial denominators: interrupted trials remain unscored and
are tracked by the existing campaign accounting.

A terminal checkpoint can be finalized without rerunning the agent. A lost
nonterminal conversation voids that workflow and reruns it in a fresh retry
workspace under the existing infrastructure policy. It does **not** resume only
the last successful stage, reset an active stage's allowance, or score an
intermediate pass as complete. This version does not support restoring a live
conversation after a controller crash.

For custom non-model test adapters, `POST /next` with `{"after_stage": N}` and
the normal run token advances a passed stage. Repeated delivery is idempotent;
unpassed/future stages are rejected. Model runs currently require the Pi adapter.

## Inspect and develop

From this checkout:

```sh
uv run mriseqbench validate
uv run mriseqbench workflows
uv run mriseqbench describe T1.1 --hardware H1 --workflow --stage 2
```

Run experiments and result analysis on **Ultron**. Put source snapshots, test
artifacts, candidate/reference sequences, logs and results beneath
`/ssd-pool/kfareed/MRISeqBench-results/`. Keep only code, documentation, figures
and compact storage pointers locally. Do not overwrite a main-benchmark output.

For an explicitly unqualified development campaign, copy the example experiment,
pin the model/provider and runtime image, and set `workflow.allow_unqualified`
to true. Then on Ultron:

```sh
uv run mriseqbench run --experiment experiments/my_agentic_dev.yaml \
  --out /ssd-pool/kfareed/MRISeqBench-results/my-agentic-dev
```

Individual stage candidates can be evaluated independently with the normal
`evaluate` command plus `--workflow --stage N` (0 through 3). Pass an existing
base case ID to `--case` and a fresh remote directory to `--out`.

## Qualification still required

Before publishing comparisons, qualify reference solutions for all 324 effective
stage specifications on the exact hardware profile. Unchanged submissions,
metadata-only edits and full rollbacks must fail where their physical contracts
conflict. Review image/contrast acceptance for changed timing/FOV; structural
validation alone is not evidence of feasibility. The test suite includes GRE
waveform and RF-target controls, not an all-family GPU qualification.

## Comparison modes

`workflow.mode` defaults to `generate_edit`. Two additional modes use the same
contracts and graders:

- `supplied_edit`: requires `workflow.seed_manifest`. The controller verifies the
  sequence and editable source hashes, grades the initial stage in full before
  launching any agent, then begins the conversation with the first edit. The
  initial stage is explicitly recorded as `origin: supplied`; it is not a model
  generation result. This mode uses the ordinary per-edit allowance.
- `final_spec`: starts a fresh workspace and conversation with only the final
  effective specification, and grades one stage. Its time cap is the original
  total agent allowance, so its maximum time matches a complete generate/edit
  workflow. Its submission cap is four times the per-stage cap (12 by default),
  matching the total maximum number of submissions in a generate/edit workflow.
  These are matched maximum allowances, not matched actual expenditure.

Keep each mode in a separate campaign/output directory and report modes separately.
A supplied seed library pins the authoring arm and maps base case IDs to inputs:

```json
{
  "schema_version": 1,
  "arm": "p0_pypulseq@1",
  "cases": {
    "<base-case-id>": {
      "sequence": {"path": "gre/sequence.seq", "sha256": "<sha256>"},
      "sources": {
        "design.py": {"path": "gre/design.py", "sha256": "<sha256>"}
      }
    }
  }
}
```

Paths are relative to the seed manifest or absolute on Ultron. The arm reference
must exactly match the experiment (use null for an unarmed test adapter).
Sources may be Python, Julia or MATLAB files; destination paths cannot traverse
parents, replace harness files or use reserved workspace directories. The seed
library must cover the experiment suite. Content hashes enter the campaign
identity, and changing them invalidates resume. No new seed library is bundled:
the inputs must be curated and qualified on Ultron before comparing models.

Do not interpret success conditional on reaching a stage as an unbiased
cross-model editing ranking.

Detailed development/test artifacts are stored on Ultron at
`/ssd-pool/kfareed/MRISeqBench-results/agentic-development-20261029/`.
