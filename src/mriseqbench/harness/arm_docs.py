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
RF dead time, ringdown, ADC dead time and gradient ramps must fit their blocks.
Use `seq.check_timing()` before `seq.write('sequence.seq')`. This checks timing,
not the benchmark's image, tissue-response or SAR acceptance criteria.
"""
    text = """# Julia authoring API

`using KomaMRIBase, KomaMRIFiles`; use the persistent `./bin/julia` process.
Package documentation: `/opt/sources/KomaMRIBase/src/` and
`/opt/sources/KomaMRIFiles/src/`. Inspect Julia docstrings with `@doc`.

Plain-number low-level Koma constructors and `HardwareLimits` use SI units:
gradient amplitudes in T/m, RF amplitudes in T, time in seconds, frequency in Hz,
and phase in radians. `area(grad)` returns T*s/m; multiply by `KomaMRIBase.γ`
to obtain cycles/m. `get_kspace(seq)` returns integrated coordinates in cycles/m,
including the second returned array at ADC sample times.

Create `seq = Sequence(sys)` and add blocks with `addblock!(seq, rf, ...)`,
or gradient axes `addblock!(seq; x=gx, y=gy, z=gz)`. Low-level event duration
alone need not include the scanner's RF ringdown or ADC dead time. Use
`dur(event, sys)` and a raster-aligned `Duration(...)` when specifying block
duration. For example, an RF block can use
`addblock!(seq, rf, Duration(ceil_to_raster(dur(rf, sys), sys.limits.DUR_Δt)))`.
`write_seq(seq, "sequence.seq"; sys=sys)` validates export timing and hardware
conventions. `read_seq("sequence.seq")` permits round-trip inspection.

Low-level RF phase is `ϕ`; ADC receiver phase is the fifth `ADC` constructor
argument. They are separate settings and are preserved by Pulseq export.
ADC event time span and Pulseq dwell-window duration differ: inspect
`dwell(adc, sys)`, `delay(adc, sys)` and `dur(adc, sys)` for exported conventions.
"""
    if arm.pulse_designer:
        text += """
## PulseDesigner helpers

Import helpers explicitly:
`using KomaMRIBase.PulseDesigner: make_adc, make_block_pulse, make_trapezoid`.
`using KomaMRIBase` alone does not import those helper names into Main.

Unlike low-level Koma constructors, plain-number `make_trapezoid` inputs use
Pulseq units: `area`/`flat_area` in cycles/m, `amplitude`/`max_grad` in Hz/m,
and `max_slew` in Hz/m/s. Do not divide a cycles/m area by γ before passing it
to `make_trapezoid`. The returned gradient uses SI internally.
`make_block_pulse(flip; duration=..., sys=sys, phase_offset=...)` takes flip and
phase in radians. `make_adc(N, dwell; sys=sys, phase_offset=...)` takes seconds
and a separate receiver phase. Helpers account for dead-time/sample offsets;
use returned event values rather than adding the half-dwell offset again.
"""
    if arm.unitful:
        text += "\nUnitful is enabled; explicit supported quantities avoid plain-number unit ambiguity.\n"
    return text
