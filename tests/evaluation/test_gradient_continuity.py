import pypulseq as pp
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.preflight import preflight
from tests.conftest import ROOT


@pytest.mark.parametrize("jump", [False, True])
def test_preflight_rejects_gradient_steps_between_blocks(tmp_path, jump):
    (case,) = Catalog(ROOT).cases(Catalog(ROOT).get("suites", "smoke"))
    hw = case["hardware"]
    system = pp.Opts(
        max_grad=hw["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hw["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hw["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hw["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hw["adc_dead_time_us"] * 1e-6,
        adc_raster_time=hw["adc_raster_us"] * 1e-6,
        block_duration_raster=hw["block_raster_us"] * 1e-6,
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(0.5, duration=1e-3, system=system))
    amplitude = 10e-3 * system.gamma
    seq.add_block(
        pp.make_extended_trapezoid(
            "x", times=[0, 0.2e-3, 1e-3], amplitudes=[0, amplitude, amplitude]
        )
    )
    seq.add_block(
        pp.make_extended_trapezoid(
            "x", times=[0, 0.8e-3, 1e-3], amplitudes=[amplitude, amplitude, 0]
        )
    )
    seq.add_block(pp.make_adc(64, duration=1e-3, delay=20e-6, system=system))
    path = tmp_path / "sequence.seq"
    seq.write(str(path))
    if jump:
        # Pulseq refuses to author this, so drop block 3's ramp-down from the
        # written file: gx steps from +10 mT/m to zero at the block boundary.
        lines = path.read_text().splitlines()
        row = next(
            i
            for i, line in enumerate(lines)
            if line.split()[:1] == ["3"] and lines.index("[BLOCKS]") < i
        )
        fields = lines[row].split()
        fields[3] = "0"
        lines[row] = " ".join(fields)
        path.write_text("\n".join(lines) + "\n")
    (check,) = [c for c in preflight(path, case) if c["name"] == "gradient_continuity"]
    assert check["passed"] is not jump


@pytest.mark.parametrize("first", [0.0, 425760.0])
def test_endpoint_metadata_cannot_hide_a_step(first):
    from types import SimpleNamespace as N

    import numpy as np

    from mriseqbench.evaluation.preflight import gradient_jumps

    system = pp.Opts()
    amplitude = 10e-3 * system.gamma
    # An extended gradient whose first vertex is already at 10 mT/m.
    grad = N(
        type="grad",
        delay=0.0,
        first=first,
        last=0.0,
        tt=np.array([0.0, 0.8e-3, 1e-3]),
        waveform=np.array([amplitude, amplitude, 0.0]),
        shape_dur=1e-3,
    )
    block = N(gx=grad, gy=None, gz=None)
    seq = N(
        block_events={1: None}, get_block=lambda _: block, block_durations={1: 1e-3}
    )
    jumps = gradient_jumps(seq, system)
    # Zero metadata misrepresents the waveform; correct metadata is a real step
    # from rest. Either way the discontinuity is reported.
    assert jumps
