"""Generated refocusing controls preserve source clocks at echo boundaries."""

from types import SimpleNamespace

import numpy as np
from pypulseq import Opts, Sequence, make_block_pulse, make_delay

from mriseqbench.evaluation.s1_koma import KomaWorker


def boundary_control():
    system = Opts(
        rf_dead_time=200e-6,
        rf_ringdown_time=50e-6,
        adc_dead_time=10e-6,
        block_duration_raster=10e-6,
    )
    seq = Sequence(system)
    pulses = []
    for number, phase in ((1, 0.3), (3, -0.7)):
        rf = make_block_pulse(
            1.0, duration=1e-3, phase_offset=phase, system=system, use="excitation"
        )
        start = (number - 1) * 2e-3
        pulses.append(
            SimpleNamespace(
                block_number=number,
                start=start + rf.delay,
                end=start + rf.delay + rf.shape_dur,
            )
        )
        seq.add_block(make_delay(2e-3), rf)
        seq.add_block(make_delay(2e-3))
    seq.add_block(make_delay(1e-3))
    return SimpleNamespace(seq=seq, pulses=pulses, adc=np.array([4e-3, 8e-3]))


def test_multi_echo_boundary_centres_and_rf_clocks_survive_serialization(tmp_path):
    acq = boundary_control()
    worker = object.__new__(KomaWorker)
    worker.path, worker.sequence_cache = tmp_path, {}
    output = Sequence(acq.seq.system)
    output.read(worker.sequence(acq, zero_gradients=True))
    np.testing.assert_allclose(output.adc_times()[0], acq.adc, atol=1e-12, rtol=0)
    for number in acq.seq.block_events:
        assert output.block_durations[number] == acq.seq.block_durations[number]
        original, replay = acq.seq.get_block(number).rf, output.get_block(number).rf
        if original is not None:
            np.testing.assert_allclose(
                replay.delay + replay.t,
                original.delay + original.t,
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(replay.signal, original.signal, rtol=1e-6)
            np.testing.assert_allclose(
                replay.phase_offset, original.phase_offset, atol=1e-6, rtol=0
            )
            assert original.dead_time == 200e-6
            assert original.ringdown_time == 50e-6


def test_single_echo_padding_does_not_move_the_echo(tmp_path):
    acq = boundary_control()
    acq.adc = acq.adc[:1]
    worker = object.__new__(KomaWorker)
    worker.path, worker.sequence_cache = tmp_path, {}
    output = Sequence(acq.seq.system)
    output.read(worker.sequence(acq, zero_gradients=True))
    np.testing.assert_allclose(output.adc_times()[0], acq.adc, atol=1e-12, rtol=0)
