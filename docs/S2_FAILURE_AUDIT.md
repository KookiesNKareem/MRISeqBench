# S2 failure attribution and corrective changes

This audit separates a measured rejection from its cause. A model can produce
an invalid waveform while missing or incomplete public instructions prevent
it from diagnosing that waveform. Such a case is **mixed**, not evidence that
the judge fabricated a failure and not an uncontaminated model-only result.

Scope: completed DeepSeek Python225, DeepSeek Julia300 and Sol Python75
campaigns on Ultron, plus tool/action traces from all 27 Sol Python and all
27 Sol Julia S2 conversations and focused DeepSeek examples. No paid/native
probe, participant replay or regrading was used. Additional inspection called
only the waveform parser/identity inspector on retained frozen bytes.

## Attribution rules

| Category | Required evidence | Interpretation/action |
|---|---|---|
| Harness/prompt defect | Missing advertised asset; an enforced rule is undisclosed; or feedback omits the failing measurement/bound | Fix delivery/disclosure for future prepared workspaces. Historical trials remain as run. |
| Model design error | Exported waveform or physical response violates the implemented requirement, with evidence identifying the violation | Retain the measured failure. A better prompt does not retroactively make the sequence pass. |
| Mixed | A measured design error coexists with missing requirements or nondiagnostic feedback relevant to that error | Preserve both causes; do not attribute the entire outcome solely to model capability. |
| Evaluator correctness defect | An independent calculation/control contradicts the evaluator under its declared contract | Isolate with a regression/control before changing science; version and qualify any acceptance change. No such wrong verdict was established in the inspected examples. |
| Evaluation compute limit | `failure_kind=grading_budget_exhausted`, with cumulative execution receipt | Report compute-limited failure separately from measured physical rejection. Preserve the original cumulative budget. |
| Infrastructure interruption | Native/device error, guard stop or failed transport, supported by runtime receipts | Not a model-quality result. Preserve state and use the campaign's authorized recovery procedure; do not fabricate a failed physical check. |
| Undetermined/no submission | Terminal timeout or no submission without a fully attributed causal trace | Do not automatically label as an intrinsic model error; inspect tool/runtime/time accounting. |

## What the completed counts establish

The three completed populations contain 216 S2 trials: 37 passed and 179 failed.
Terminal diagnostic categories are 79 sequence/hardware failures, 78 image
failures, 19 timeout/no-submission outcomes and three grading-budget failures.
Of the 78 image failures, 63 include phase rejection. Twenty first submissions
failed at least one SAR check. These counts describe rejection categories,
**not** mutually isolated causal effects of model, prompt, tools or evaluator.
The populations have different models, arms and operational histories.

## Transcript-backed findings

| Evidence | Observed cause | Attribution and correction |
|---|---|---|
| All 27 Sol Julia S2 transcripts and 15 Sol Python S2 transcripts contain direct failed reads of `docs/S2_EVALUATION.md` | `EVALUATION.md` advertises an absent file (`ENOENT`) | Confirmed harness defect. Stage the public S2 document before agent execution and fail preparation if it is unavailable. |
| Sol Python `propeller_tse` repeat1, final frozen attempt003 | Correct refocusing counts and <1.3 ns coherence residuals, but measured excitation 87.717 degrees and refocusing 138.337 degrees | Mixed. The waveform violates detailed 90/180 +/-2 degree rules, but feedback exposed neither the rotation nor its bound. Disclose the rule and measured rotations. |
| Sol Python `grase` repeat0, final frozen attempt003 | Measured excitation 89.148 degrees and refocusing 173.387 degrees; reported paired-echo timing alone does not explain rejection | Mixed. Expose nominal rotation as a separate diagnostic condition. Do not relax the flip bound. |
| Sol TSE/GRASE/PROPELLER transcripts | Underestimated circular-B1+ SAR; cooling tails then violated final TR; longer RF changed bandwidth | Model mistakes compounded by incomplete convention disclosure. Supply the exact SAR formula and final-TR definition before the first submission. |
| Sol Julia `tse_t2w` repeat0 | Attempt002 complex NRMSE 1.496/phase RMS 1.586 rad with magnitude NRMSE 0.257; receiver correction in attempt003 passed at 0.332/0.186 rad with magnitude 0.257 | Direct evidence of a receiver-phase design error. Preserve phase-sensitive scoring; explain the independent RF target and absence of fitted phase correction. |
| DeepSeek Julia J3 `propeller_tse` repeat1 | 16 blade angles gave only eight distinct orientations modulo pi | Measured geometry error, with an underdisclosed angular convention. Report period, circular gap bounds and observed gaps. |
| Sol Python `stack_of_stars` repeat0, final frozen attempt003 | Correct partition set/spoke counts/angular coverage, but some two-sided endpoints reached only about 15 reciprocal-FOV units instead of beyond 15.25 | Mixed. Family-specific extent fails despite separate matrix/FOV coverage passing. Disclose the extent and every family-specific subcondition. |
| Sol Python `stack_of_stars` repeat0 public simulator receipt | Full simulation timed out after 300.005 charged seconds; first frozen grade exhausted its original 600 s execution limit | A costly diagnostic and a compute-limited grade, not proof of a physical threshold violation. Make simulator time/queue charging explicit; retain the compute-limit category and original allowance. |
| Sol Julia `propeller_tse` repeat1 public simulator receipts | Two completed diagnostic calls took 234.879 +229.733 charged seconds | Tool-cost limitation under the existing budget, not an evaluator correctness defect. A completed public replay does not certify all grading predicates. |

Remote evidence roots (full transcripts/scientific artifacts remain on Ultron):

```text
/home/kfareed/mriseqbench/gpt61-sol-python-pi-75-20261019
/home/kfareed/mriseqbench/gpt61-sol-julia-mcp-sim-81-20261019
/home/kfareed/mriseqbench/python-arm-comparison-flash1017-20261018
/home/kfareed/mriseqbench/julia-arm-comparison-flash1017-20261018
```

Per-trial evidence is under `arms/<arm>/campaign/<task>-v1--<physics>-v1--rN/`:
`agent/transcript/*.jsonl`, `control/submissions/attempt-NNN/{sequence.seq,result.json}`,
and `control/simulations.json`. Stack-of-stars uses `static_b0_3d`; other cited
trials use `static_b0`. The completed comparison reports are under
`/ssd-pool/kfareed/MRISeqBench-results/reports/`.

## Fix scope and comparability

Changes are local source changes for newly prepared runs: public document
delivery, explicit existing contract/SAR/phase/time semantics and measured
identity diagnostics. They do not change acceptance predicates, thresholds,
physics, images, reference targets, resource caps or allowances. Mechanism/image
rejections remain distinct from execution-budget and infrastructure failures.
No active workspace, pinned remote source, frozen result or financial ledger is
edited, and no campaign is launched. Old results are not relabeled or regraded.

Future comparisons must record the new source/prompt/document hashes and
describe these instruction/feedback changes. An improvement under the repaired
harness is not a model-only improvement. Estimating the causal effect requires
a separately authorized matched comparison; this repair does not authorize one.

## Local validation

- 103 focused prompt-delivery, SAR-formula and diagnostic regression tests passed.
- 18 existing S2 waveform/structural/routing controls passed.
- 88 existing agent/feedback/RF/S4 delivery tests passed; the real-Pi integration
  test was excluded. No provider call, Docker/GPU probe or native replay was run.
- Before editing the inspector, 32 positive/negative waveform controls were
  captured under evaluator versions 1 and 3. Comparing the original inspector
  with the repaired one retained every named check decision and exactly equal
  k-space, RF/ADC timing/phase, excitation assignment, flips, indices, offsets
  and measurement dictionaries. New feedback also passed strict JSON encoding.
- Ruff and whitespace checks passed for the changed code.
