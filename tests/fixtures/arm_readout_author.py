"""Native single-readout controls; embedded in the restricted agent container."""

import json
import subprocess
from pathlib import Path

arm = json.loads(Path("ARM.json").read_text())
h = json.loads(Path("case.json").read_text())["hardware"]
control = globals().get("control_kind", "gradient_echo")
assert control in {"gradient_echo", "spin_echo"}
spin_echo = control == "spin_echo"
flip = 90 if spin_echo else 30
te = 0.012 if spin_echo else 0.008
tr = 0.2 if spin_echo else 0.1

if arm["language"] == "python":
    import numpy as np
    import pypulseq as pp

    sys = pp.Opts(
        max_grad=h["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=h["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=h["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=h["adc_dead_time_us"] * 1e-6,
        grad_raster_time=h["gradient_raster_us"] * 1e-6,
        rf_raster_time=h["rf_raster_us"] * 1e-6,
        adc_raster_time=h["adc_raster_us"] * 1e-6,
        block_duration_raster=h["block_raster_us"] * 1e-6,
    )
    seq = pp.Sequence(sys)
    rf = pp.make_block_pulse(
        np.deg2rad(flip), duration=1e-3, system=sys, use="excitation"
    )
    centre = rf.delay + rf.shape_dur / 2
    seq.add_block(rf)
    if spin_echo:
        ref = pp.make_block_pulse(
            np.pi, duration=1e-3, phase_offset=np.pi / 2, system=sys, use="refocusing"
        )
        seq.add_block(
            pp.make_delay(
                round(
                    centre
                    + te / 2
                    - (ref.delay + ref.shape_dur / 2)
                    - seq.duration()[0],
                    9,
                )
            )
        )
        seq.add_block(ref)
    pre = pp.make_trapezoid(
        "x",
        amplitude=-75000,
        flat_time=1e-3,
        rise_time=1e-4,
        fall_time=1e-4,
        system=sys,
    )
    read = pp.make_trapezoid(
        "x",
        amplitude=50000,
        flat_time=3.2e-3,
        rise_time=1e-4,
        fall_time=1e-4,
        system=sys,
    )
    adc = pp.make_adc(32, duration=3.2e-3, delay=1e-4, system=sys)
    seq.add_block(pre)
    seq.add_block(pp.make_delay(round(centre + te - 1.7e-3 - seq.duration()[0], 9)))
    seq.add_block(read, adc)
    seq.add_block(pp.make_delay(round(tr - seq.duration()[0], 9)))
    assert seq.check_timing()[0]
    seq.write("sequence.seq")
else:
    code = f"""
using KomaMRIBase, KomaMRIFiles
@assert isdefined(KomaMRIBase, :PulseDesigner) == {str(arm["pulse_designer"]).lower()}
@assert (Base.find_package("Unitful") !== nothing) == {str(arm["unitful"]).lower()}
sys=Scanner(limits=HardwareLimits(B0={h["field_strength_T"]},
 Gmax={h["max_gradient_mT_per_m"] * 1e-3}, Smax={h["max_slew_T_per_m_per_s"]},
 B1={h["max_b1_uT"] * 1e-6},
 ADC_Δt={h["adc_raster_us"] * 1e-6}, DUR_Δt={h["block_raster_us"] * 1e-6},
 GR_Δt={h["gradient_raster_us"] * 1e-6}, RF_Δt={h["rf_raster_us"] * 1e-6},
 RF_dead_time={h["rf_dead_time_us"] * 1e-6},
 RF_ring_down_time={h["rf_ringdown_time_us"] * 1e-6},
 ADC_dead_time={h["adc_dead_time_us"] * 1e-6}));
"""
    if arm["unitful"]:
        code += f"""
using Unitful
rf=PulseDesigner.make_block_pulse({flip}*π/180; duration=1u"ms", sys, use=Excitation())
ref=PulseDesigner.make_block_pulse(π; duration=1u"ms", phase_offset=(π/2)u"rad", sys, use=Refocusing())
pre=PulseDesigner.make_trapezoid(; amplitude=(-75000/γ)u"T/m", flat_time=1u"ms", rise_time=100u"μs", fall_time=100u"μs", sys)
read=PulseDesigner.make_trapezoid(; amplitude=(50000/γ)u"T/m", flat_time=3.2u"ms", rise_time=100u"μs", fall_time=100u"μs", sys)
adc=PulseDesigner.make_adc(32; duration=3.2u"ms", delay=100u"μs", sys)
"""
    elif arm["pulse_designer"]:
        code += f"""
rf=PulseDesigner.make_block_pulse({flip}*π/180; duration=1e-3, sys, use=Excitation())
ref=PulseDesigner.make_block_pulse(π; duration=1e-3, phase_offset=π/2, sys, use=Refocusing())
pre=PulseDesigner.make_trapezoid(; amplitude=-75000, flat_time=1e-3, rise_time=1e-4, fall_time=1e-4, sys)
read=PulseDesigner.make_trapezoid(; amplitude=50000, flat_time=3.2e-3, rise_time=1e-4, fall_time=1e-4, sys)
adc=PulseDesigner.make_adc(32; duration=3.2e-3, delay=1e-4, sys)
"""
    else:
        code += f"""
rf=RF(({flip}*π/180)/(2π*γ*1e-3), 1e-3, 0.0, sys.limits.RF_dead_time; use=Excitation())
ref=RF(π/(2π*γ*1e-3), 1e-3, 0.0, sys.limits.RF_dead_time; ϕ=π/2, use=Refocusing())
pre=Grad(-75000/γ, 1e-3, 1e-4, 1e-4, 0.0)
read=Grad(50000/γ, 3.2e-3, 1e-4, 1e-4, 0.0)
adc=ADC(32, 31e-4, 1.5e-4)
"""
    code += f"""
seq=Sequence(sys)
centre=sys.limits.RF_dead_time+0.5e-3
addblock!(seq, rf, Duration(dur(rf, sys)))
if {str(spin_echo).lower()}
 addblock!(seq, Duration(round({te}/2-sum(seq.DUR); digits=9)))
 addblock!(seq, ref, Duration(dur(ref, sys)))
end
addblock!(seq, Duration(1.2e-3); x=pre)
addblock!(seq, Duration(round(centre+{te}-1.7e-3-sum(seq.DUR); digits=9)))
addblock!(seq, adc, Duration(3.4e-3); x=read)
addblock!(seq, Duration(round({tr}-sum(seq.DUR); digits=9)))
write_seq(seq, "sequence.seq"; sys, verbose=false)
"""
    subprocess.run(["./bin/julia", "-e", code], check=True)
