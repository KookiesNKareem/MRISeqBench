"""Embedded by real-container tests; uses only the selected native author API."""

import json
import subprocess
from pathlib import Path

arm = json.loads(Path("ARM.json").read_text())
h = json.loads(Path("case.json").read_text())["hardware"]

if arm["language"] == "python":
    import numpy as np
    import pypulseq as pp

    system = pp.Opts(
        max_grad=h["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=h["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=h["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=h["adc_dead_time_us"] * 1e-6,
    )
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_block_pulse(np.pi / 6, duration=1e-3, system=system, use="excitation")
    )
    seq.add_block(pp.make_adc(32, duration=3.2e-3, system=system))
    assert seq.check_timing()[0]
    seq.write("sequence.seq")
else:
    # Each condition exercises its actual API: low-level structs, PulseDesigner,
    # or the native Unitful extension. No foreign Pulseq writer is available.
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
        code += """
using Unitful
rf=PulseDesigner.make_block_pulse(π/6; duration=1u"ms", sys, use=Excitation())
adc=PulseDesigner.make_adc(32; duration=3.2u"ms", sys)
"""
    elif arm["pulse_designer"]:
        code += """
rf=PulseDesigner.make_block_pulse(π/6; duration=1e-3, sys, use=Excitation())
adc=PulseDesigner.make_adc(32; duration=3.2e-3, sys)
"""
    else:
        code += """
rf=RF((π/6)/(2π*γ*1e-3), 1e-3, 0.0, sys.limits.RF_dead_time; use=Excitation())
# Koma ADC stores the span between sample centres, with delay to the first centre.
adc=ADC(32, 31e-4, sys.limits.ADC_dead_time+0.5e-4)
"""
    code += """
seq=Sequence(sys)
addblock!(seq, rf, Duration(dur(rf, sys)))
addblock!(seq, adc, Duration(dur(adc, sys)))
write_seq(seq, "sequence.seq"; sys, verbose=false)
"""
    subprocess.run(["./bin/julia", "-e", code], check=True)
