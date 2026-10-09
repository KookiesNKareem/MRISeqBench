"""A minimal RF/readout sequence for infrastructure testing."""

import argparse
import json
from pathlib import Path

import numpy as np
from pypulseq import Opts, Sequence, make_adc, make_block_pulse, make_trapezoid

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
case = json.loads(Path(args.case).read_text())
h = case["hardware"]
system = Opts(
    max_grad=h["max_gradient_mT_per_m"],
    grad_unit="mT/m",
    max_slew=h["max_slew_T_per_m_per_s"],
    slew_unit="T/m/s",
    rf_dead_time=h["rf_dead_time_us"] * 1e-6,
    rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
    adc_dead_time=h["adc_dead_time_us"] * 1e-6,
)
seq = Sequence(system)
rf = make_block_pulse(np.pi / 6, duration=1e-3, system=system, use="excitation")
gx = make_trapezoid("x", flat_area=32 / 0.2, flat_time=3.2e-3, system=system)
adc = make_adc(32, duration=gx.flat_time, delay=gx.rise_time, system=system)
seq.add_block(rf)
seq.add_block(gx, adc)
ok, errors = seq.check_timing()
if not ok:
    raise RuntimeError(errors)
seq.write(args.output)
