# Required policy for new benchmark runs

Use **`bounded_conservative_v1`** for new S1–S4 benchmark submissions and workers.
The user authorized it for also unfinished current grades on 2026-10-18 and
requires reported outcomes to be **passed** or **failed**.

- Enforce the task's declared `spatial_max_samples` per frame and replicate.
- Enforce **600 seconds of cumulative evaluator process execution per frozen
  submission**, across numerical retries and all frames. Resource-permit queue
  time is recorded separately. Startup, native simulation and reconstruction
  count toward the execution limit.
- If a pass cannot be established within either limit, report **failed** with
  `failure_kind=grading_budget_exhausted`. Never publish a third outcome such as
  uncertain or unresolved. Preserve numerical intervals and partial computations
  in diagnostic evidence; do not fabricate physical threshold violations.
- Keep measured physical failures distinct from grading-budget failures.
  Infrastructure crashes and guard stops require preservation/recovery rather
  than being silently counted as model-quality failures.
- Preserve the original agent allowance and submission history. Numerical retries
  cannot reset the grading budget or create a new paid trial.
- Preserve completed results. Changes to an active frozen comparison require an
  explicit scope decision and a recorded evaluator revision.

The case profile is:

```json
{"grading_policy": {"id": "bounded_conservative_v1"}}
```

Use the bounded implementation and finite executor together; adding this field to
an old unbounded evaluator does not implement the policy. The approved source
overlay, offline controls, reference audit and deployment receipts are retained
under `runs/bounded-conservative-grader-policy-proposal-20261018/`. Active Ultron
campaigns select isolated, pinned grader sources through their
`infra/bounded-grading-policy-20261018/` entries. Their original sources remain
immutable for reproduction.

The four stored passing S4 controls fit the proposed limits, but that retrospective
audit is not a universal runtime guarantee for every task or host load. Reports
must identify the policy and distinguish outcomes from different evaluator
revisions; do not attribute a joint prompt, feedback and grading-policy change
solely to the model or any one intervention.
