"""Public package conventions, without task solutions or judge implementation."""


def api_notes(arm):
    if arm.language == "python":
        return """# Python authoring API

Use `import pypulseq as pp`; package sources and docstrings are installed locally.
`pp.Opts` accepts explicit `grad_unit='mT/m'` and `slew_unit='T/m/s'` for hardware
limits. Event factory gradient amplitudes use Hz/m and areas use cycles/m;
durations are seconds and RF flip angles and phase offsets are radians.
RF and ADC events have separate `phase_offset` fields. The writer serializes the
submitted settings; it does not infer a receiver phase from the RF phase.
Sampled RF and ordinary arbitrary-gradient factories use raster sample centers:
`N` RF samples with `dwell=d` occupy `N*d`, at `(j+1/2)*d` after the
leading-edge delay. Ordinary arbitrary gradients likewise occupy `N*grad_raster_time`.
ADC `duration=N*d` is the acquisition window; samples are centered half a dwell
inside it. These factory durations are different from Julia's raw endpoint spans.
Use the factory's `dwell`/raster settings rather than deriving dwell as duration/(N-1).
RF dead time, ringdown, ADC dead time and gradient ramps must fit their blocks.
Use `seq.duration()[0]` for total duration in seconds. `seq.write(...)` adds
`seq.definitions['TotalDuration']`; a `Duration` definition is not automatic.
Set RF `use='excitation'`, `'refocusing'`, or `'inversion'` explicitly.
Trajectory calculations use RF roles and RF centers; the block-relative RF
center is `rf.delay + pp.calc_rf_center(rf)[0]`. Keep receiver phase separate.
Use `seq.check_timing()` before `seq.write('sequence.seq')`. This checks timing,
not the benchmark's image, tissue-response or SAR acceptance criteria.

## Interactive sequence checks

Inspect the current sequence between edits, before simulation or submission:
`k_adc, k, t_exc, t_ref, t_adc = seq.calculate_kspace()` returns ADC and
full trajectories in cycles/m. Both trajectory arrays are axis-first `(3, N)`;
inspect x as `k_adc[0, :]` (Julia's corresponding array is sample-first).
`ok, errors = seq.check_timing()` returns a boolean and detailed timing errors;
print both, or use `seq.check_timing(print_errors=True)`.
`print(seq.test_report())` prints timing/readout measurements and maximum
gradient/slew values. This report is an analysis, not a hardware-limit verdict.
Event factories check their construction limits; `./bin/check` independently
checks the exported file against the task hardware, as it does for Julia.
`seq.waveforms()` returns per-axis arrays with time in row 0 and gradient
amplitude in row 1 (Hz/m). `seq.plot(plot_now=False, save=True)` saves sequence
plots. In simulation-enabled arms, inspect your own reconstructed images and
raw signal between edits. Submission adds the task-specific acceptance checks.
"""
    session = (
        "`julia_eval` MCP session"
        if arm.tool_interface == "julia_mcp"
        else "`./bin/julia` process"
    )
    text = f"""# Julia authoring API

`using KomaMRIBase, KomaMRIFiles`; use the persistent {session}.
"""
    if arm.pulse_designer:
        text += """
## PulseDesigner helpers

Use the selected helpers to construct events. Import them explicitly:
`using KomaMRIBase.PulseDesigner: make_adc, make_block_pulse, make_trapezoid`.
`using KomaMRIBase` alone does not import those helper names into Main.

`make_*` helpers return MRI events (or documented event tuples); `build_*`
helpers return a `Sequence`. Each call below returns one event:
- `make_block_pulse(flip; duration=T, sys=sys, phase_offset=phase, use=Excitation())`
  returns an RF event. Flip and phase are radians; `T` is pulse duration in seconds. Use
  `dur(rf, sys)` for the block's timing requirements, including delay/ringdown.
- `make_adc(N, d; delay=t0, sys=sys, phase_offset=phase)` returns an ADC event.
  `d` is dwell in seconds, and `t0` is the Pulseq leading dwell-edge delay.
  Alternatively, `make_adc(N; duration=W, sys=sys)` uses `d = W / N`.
  The helper applies ADC dead time and the half-dwell offset internally.
- `make_trapezoid(; area=A, sys=sys)` designs a gradient from total area;
  `make_trapezoid(; flat_area=A, flat_time=T, sys=sys)` specifies the flat top.
  Plain-number areas are cycles/m, amplitudes and `max_grad` are Hz/m, and
  `max_slew` is Hz/m/s. Times are seconds. Returned gradients store SI values.

Pass a separate RF and ADC `phase_offset` when needed. Use returned event
values; do not apply the helper's half-dwell offset a second time.
The raw `ADC.T` field is the first-to-last sample-center span `(N - 1) * d`
for `N > 1`; the Pulseq acquisition window and helper `duration` are `N*d`.
For example, `make_adc(4, 10e-6; sys=sys)` stores `adc.T = 30e-6` for a
`40e-6` acquisition window. Use the helper arguments when constructing events;
do not replace `adc.T` with `N*d`.
"""
    else:
        text += """
## Base event constructors

Construct events directly in SI units:
- `RF(A, T, Δf, delay; ϕ=phase, use=Excitation())`: RF amplitude `A` in tesla,
  pulse duration `T` in seconds, frequency offset `Δf` in Hz, delay in seconds,
  and phase in radians. Set `delay >= sys.limits.RF_dead_time` for export.
  `KomaMRIBase.γ` is in Hz/T.
- `Grad(A, T, rise, fall, delay)`: gradient amplitude in T/m; `T` is flat-top
  duration, with separate rise/fall durations and delay, all in seconds.
  The four-argument overload is `Grad(A, T, rise, delay)`, with `fall=rise`.
  Use all five arguments when specifying rise and fall separately.
- `ADC(N, T, delay, Δf, phase)`: for `N > 1`, `T = (N - 1) * d`, where `d`
  is the dwell. `delay` is the time to the first sample center. For a Pulseq
  leading-edge delay `t0`, use `ADC(N, (N-1)*d, t0+d/2, Δf, phase)`.
  The Pulseq sampling window is `N*d`; for `N == 1`, `T` is the dwell.
  For example, `N=4` and `d=10e-6` require `T=30e-6` for a `40e-6` window.
  RF phase and ADC receiver phase are separate settings.

`dwell(adc, sys)` returns the dwell; `delay(adc, sys)` returns the Pulseq
leading-edge delay. Choose delays that satisfy the scanner's dead times.
"""
    text += """
## Sampled events: ADC, RF and gradients

Raw constructors store endpoint spans. For `N > 1` evenly spaced samples,
`T = (N-1)*d`, not `N*d`; this also applies to sampled `RF` and `Grad`.
- ADC: `ADC(N, (N-1)*d, t0+d/2, frequency, phase)` puts the first center at
  `t0+d/2`; the acquisition window is `N*d`. For `N=1`, raw `ADC.T` is dwell.
- RF: `RF(samples, (N-1)*d, frequency, t0+d/2; use=Excitation())` places RF
  samples at `t0+d/2 + (0:N-1)*d`. Values are tesla; the Pulseq RF window is
  `N*d`. Choose `d` on the RF raster and leading edge `t0` after RF dead time.
  Example: 4,000 samples at 1 us require `T=3.999e-3`, with a 4 ms window.
  `T=4e-3` would create 1.00025 us spacing and fail a 1 us raster check.
- Gradient: `Grad(samples, (N-1)*d, rise, fall, t0)` places stored samples
  from `t0+rise` to `t0+rise+T`. Rise/fall connect `first`/`last` to the
  sampled segment; total occupancy is `t0+rise+T+fall`. For ordinary Pulseq
  center samples use `d=sys.limits.GR_Δt` and half-raster rise/fall, subject
  to the boundary amplitudes and slew limits. Raw gradients use T/m.

"""
    if arm.pulse_designer:
        text += """
PulseDesigner helpers handle their own center/edge conversions:
`make_arbitrary_rf(samples, flip; dwell=d, delay=t0, sys=sys)` returns
`(rf, gz, gzr)` and rescales the waveform to the requested flip; `d` is RF
sample spacing. It does not preserve preoptimized absolute amplitudes.
`make_arbitrary_grad(samples; delay=t0, sys=sys)` returns one gradient;
plain samples use Hz/m and its default spacing is `sys.limits.GR_Δt`.
Do not apply these helpers' half-raster offsets again.
"""
    text += """
## RF roles, centers and gradient axes

Set RF `use=Excitation()`, `Refocusing()`, or `Inversion()` as appropriate.
This applies to `make_*` helpers as well as raw RF constructors. Helpers can
otherwise default to `Undefined()`, which does not reset the integrated
trajectory at excitation. `get_kspace`/`get_Mk` reset moments at excitation
and reverse them at refocusing; `Inversion()` does not mean refocusing.
The block-relative RF center is `rf.delay + rf.center`; `rf.center` is relative
to RF onset, not the block start. Do not add the delay into `rf.center` again.
These metadata affect trajectory/reconstruction; a successful Bloch replay
alone does not verify them. Inspect the ADC trajectory after construction and
again after export with `get_kspace(read_seq("sequence.seq"))`.
Gradients have no RF-style `use` role: assign their physical axes with
`x=gx`, `y=gy`, `z=gz`. Gradient moments plus RF roles/centers determine k-space;
LIN/REP/SET labels describe the acquisition and do not create spatial encoding.

## Scanner, blocks and export

Use `sys = Scanner(limits=HardwareLimits(...))`. Hardware fields use SI units:
`B0` [T], `B1` [T], `Gmax` [T/m], `Smax` [T/m/s]; `RF_dead_time`,
`RF_ring_down_time`, `ADC_dead_time`, and rasters `GR_Δt`, `RF_Δt`, `ADC_Δt`,
`DUR_Δt` are seconds. Set these to the task's hardware limits.

Use `dur(seq)` or `sum(seq.DUR)` for total duration in seconds; do not assume
`seq.DEF["Duration"]` exists. Keep the scanner variable `sys` separately:
`Sequence(sys)` has no `seq.sys` field.
Create `seq = Sequence(sys)` and append blocks with `@addblock seq += ...`.
RF, ADC and duration events are positional; gradients use named axes:
`@addblock seq += (rf, z=gz)`, `@addblock seq += (adc, x=gx)`, and
`@addblock seq += (x=gx, y=gy, z=gz)` for a gradient-only block.
`Duration(T)` requests a fixed block duration and errors if an event exceeds it.
Inferred block durations include RF ringdown and ADC dead time from
`Sequence(sys)`. `dur(event, sys)` reports these timing requirements; it does
not round the final duration to the block raster `sys.limits.DUR_Δt`.
Explicit `Duration(T)` values must satisfy those requirements and the raster.

`write_seq(seq, "sequence.seq"; sys=sys)` validates export timing and hardware
conventions. `read_seq("sequence.seq")` permits round-trip inspection.
`area(grad)` returns T*s/m; multiply by `KomaMRIBase.γ` to obtain cycles/m.
`get_kspace(seq)` returns integrated coordinates in cycles/m, including the
second returned array at ADC sample times. Arrays have sample rows and three
axis columns: index the ADC result as `k_adc[:, 1]` for x, not `k_adc[1, :]`.

## Interactive sequence checks

Inspect the current in-memory sequence before simulation or submission:
`k, k_adc = get_kspace(seq)`; `get_Mk(seq, 0)` (or order 1/2) gives gradient
moments; `check_timing(seq, sys)` and `check_hw_limits(seq, sys)` check timing
and hardware. These public calculations do not require the judge. Print their
results in the persistent session. Keep these helpers available interactively;
`./bin/check` is an additional check of the exported file, not a replacement;
submission adds the task-specific acceptance checks. In simulation-enabled
arms, inspect your own reconstructed images and raw signal between edits.

## Numerical tools

`FFTW` is available in every Julia authoring arm. Use `using FFTW` for
`fft`/`ifft`; use `using Statistics` for `mean`. These are numerical tools,
not additional sequence-authoring helpers.

## API help

The session preloads Julia's REPL documentation support. Print a function's
formatted help to the captured stdout, for example:
"""
    lookup = "KomaMRIBase.PulseDesigner.make_adc" if arm.pulse_designer else "ADC"
    text += f'`show(stdout, MIME("text/plain"), @doc {lookup}); println()`\n'
    text += """Use the same command with `RF`, `Grad`, `HardwareLimits`, `write_seq`, or
`KomaMRIBase.var"@addblock"` for their help. `@doc` takes a literal binding;
inside a loop over function/type values `f`, use
`show(stdout, MIME("text/plain"), Base.Docs.doc(f)); println()`.
Package sources are also available at
`/opt/sources/KomaMRIBase/src/` and `/opt/sources/KomaMRIFiles/src/`.
"""
    if arm.unitful:
        text += "\nUnitful is enabled; explicit supported quantities avoid plain-number unit ambiguity.\n"
    return text
