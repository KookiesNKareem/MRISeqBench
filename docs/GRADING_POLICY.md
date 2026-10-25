# Grading policy

Reported outcomes are only **passed** or **failed**.

- Grading has no time limit. Slow and 3D cases run to completion.
- Numerical refinement stops at the task's `spatial_max_samples`. A verdict that
  is still unresolved there fails with `failure_kind=unresolved_at_sample_cap`;
  the evaluator's own status is kept as `unresolved_status`. No physical threshold
  violation is fabricated.
- Temporal refinement stops after 6 timestep halvings (references need at most
  2). A replay or phase oracle still unconverged there fails with
  `failure_kind=unresolved_at_timestep_cap`. rf_excitation uses its fixed
  five-step schedule (10 µs → 0.625 µs) and fails `numerical_convergence` if it
  does not settle.
- Worker crashes restart in the same checkpoint directory. Persistent
  infrastructure failures interrupt the trial for recovery; they are never
  counted as model failures.
- Agent allowance and submission history are never reset by regrading.

Implementation: `src/mriseqbench/harness/grading.py`.
