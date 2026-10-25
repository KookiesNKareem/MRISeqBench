"""Structurally malformed S4 submissions fail cleanly; they never raise (CPU)."""

from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s4_sequence
from mriseqbench.evaluation.s4_sequence import inspect_sequence
from tests.fixtures import reference128_advanced

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def case():
    catalog = Catalog(ROOT)
    return catalog.resolve(
        catalog.get("tasks", "adiabatic_inversion"), hardware_ref="high_performance"
    )


def drop_adcs(case, tmp_path, keep_in_segment_1):
    seq = reference128_advanced.build(case)
    adc_blocks = [n for n in seq.block_events if seq.block_events[n][5]]
    assert len(adc_blocks) == 128
    for number in adc_blocks[32 + keep_in_segment_1 : 64]:
        events = np.array(seq.block_events[number])
        events[5] = 0
        seq.block_events[number] = events
    path = tmp_path / "malformed.seq"
    seq.write(str(path))
    return path


def failed(acq):
    return {c["name"] for c in acq.checks if not c["passed"]}


@pytest.mark.parametrize("keep", [0, 1])
def test_missing_segment_lines_fail_without_raising(case, tmp_path, keep):
    acq = inspect_sequence(drop_adcs(case, tmp_path, keep), case)
    assert failed(acq)
    assert acq.measurements["frames"] == []


def test_single_line_segment_fails_even_if_grid_checks_were_bypassed(
    case, tmp_path, monkeypatch
):
    # Exercise the per-segment guard directly: pretend the shared Cartesian
    # checks passed, so the one-line segment reaches the bSSFP reductions.
    original = s4_sequence.inspect_cartesian

    def lenient(path, case):
        acq = original(path, case)
        for item in acq.checks:
            item["passed"] = True
        return acq

    monkeypatch.setattr(s4_sequence, "inspect_cartesian", lenient)
    acq = inspect_sequence(drop_adcs(case, tmp_path, 1), case)
    assert "frame_0_segment_1_acquisition" in failed(acq)
    assert acq.measurements["frames"] == []


def t2_case():
    catalog = Catalog(ROOT)
    return catalog.resolve(
        catalog.get("tasks", "t2_prepared"), hardware_ref="high_performance"
    )


def test_zero_sample_adc_fails_structure_without_raising(tmp_path):
    from pypulseq import make_adc, make_delay

    case = t2_case()
    seq = reference128_advanced.build(case)
    adc = make_adc(1, duration=10e-6, system=seq.system)
    seq.add_block(make_delay(20e-6), adc)
    library_id = int(seq.block_events[max(seq.block_events)][5])
    entry = list(seq.adc_library.data[library_id])
    entry[0] = 0.0
    seq.adc_library.update(library_id, None, tuple(entry))
    path = tmp_path / "zero_adc.seq"
    seq.write(str(path), create_signature=False)
    acq = inspect_sequence(path, case)
    assert failed(acq) == {"adc_sample_count"}
    assert acq.measurements["frames"] == []


@pytest.mark.parametrize("segment_lines", [(2, 32, 32, 32), (31, 32, 32, 32)])
def test_each_segment_must_acquire_exactly_32_lines(
    segment_lines, tmp_path, monkeypatch
):
    # Uneven splits with full coverage (e.g. 2/42/42/42) differ from these only
    # in grid coverage; bypass the shared grid checks to isolate the count rule.
    case = t2_case()
    sequence = reference128_advanced.build(case)
    adc_blocks = [n for n in sequence.block_events if sequence.block_events[n][5]]
    assert len(adc_blocks) == 128
    path = tmp_path / "control.seq"
    sequence.write(str(path))
    baseline = inspect_sequence(path, case)
    assert not failed(baseline)
    # Reassign ADC lines between windows by dropping lines from some segments
    # (a segment line count other than 32 must fail even before grid checks).
    drop = []
    start = 0
    for count in segment_lines:
        if count < 32:
            drop.extend(adc_blocks[start + count : start + 32])
        start += 32
    for number in drop:
        events = np.array(sequence.block_events[number])
        events[5] = 0
        sequence.block_events[number] = events
    sequence.write(str(path))
    original = s4_sequence.inspect_cartesian

    def lenient(path, case):
        acq = original(path, case)
        for item in acq.checks:
            item["passed"] = True
        return acq

    monkeypatch.setattr(s4_sequence, "inspect_cartesian", lenient)
    acq = inspect_sequence(path, case)
    assert any(name.endswith("_acquisition") for name in failed(acq)), failed(acq)
    assert acq.measurements["frames"] == []


def test_gradient_after_last_tr_block_start_is_inside_the_balance_window(tmp_path):
    from pypulseq import make_trapezoid

    case = t2_case()
    seq = reference128_advanced.build(case)
    path = tmp_path / "control.seq"
    seq.write(str(path))
    acq = inspect_sequence(path, case)
    frame = acq.measurements["frames"][0]
    last = acq.pulses[frame["segments"][0]["readout_pulse_indices"][-1]]
    tr = float(
        np.median(
            np.diff(
                [
                    acq.pulses[j].start
                    for j in frame["segments"][0]["readout_pulse_indices"]
                ]
            )
        )
    )
    boundary = last.block_start + tr  # former terminal window end
    assert last.start > boundary - tr  # RF delay exists
    clock, target = 0.0, None
    for number, duration in seq.block_durations.items():
        if abs(clock - boundary) < 1e-9:
            target = number
            break
        clock += duration
    assert target is not None
    events = np.array(seq.block_events[target])
    assert not events[1:7].any(), "expected an RF-free delay block after the last TR"
    rf_delay = last.start - last.block_start
    grad = make_trapezoid("x", area=5, duration=rf_delay - 20e-6, system=seq.system)
    events[2] = seq.register_grad_event(grad)
    seq.block_events[target] = events
    seq.write(str(path))
    acq = inspect_sequence(path, case)
    assert "frame_0_balanced_bssfp" in failed(acq)
