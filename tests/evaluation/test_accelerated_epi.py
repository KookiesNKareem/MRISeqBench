"""Accelerated EPI identity and the shared standard SENSE path; no simulation."""

from pathlib import Path

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s3
from mriseqbench.evaluation.s2_sequence import epi_sampling_mask, inspect_sequence
from tests.fixtures.reference128_readouts import build, chess_epi_options, options_for
from tests.fixtures.s2_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]


def case_for(hardware="high_performance"):
    catalog = Catalog(ROOT)
    return catalog.resolve(
        catalog.get("tasks", "epi_single_shot"), hardware_ref=hardware
    )


@pytest.mark.parametrize(
    "hardware,acs_lines,total_lines",
    [
        ("low_field", 0, 43),
        ("standard_1_5t", 0, 43),
        ("high_performance", 0, 43),
        ("high_performance", 8, 48),
    ],
)
def test_single_shot_r3_preserves_native_matrix_and_exact_prescribed_mask(
    tmp_path, hardware, acs_lines, total_lines
):
    case = case_for(hardware)
    case["task"]["sequence"]["sampling"]["calibration_lines"] = acs_lines
    path = tmp_path / "r3.seq"
    options = options_for(case)
    if acs_lines:
        options["echo_time"] = {
            "low_field": 0.052,
            "standard_1_5t": 0.032,
            "high_performance": 0.024,
        }[hardware]
    make_sequence(case, **options).write(str(path))
    acq = inspect_sequence(path, case)
    assert all(item["passed"] for item in acq.checks), acq.checks
    assert acq.measurements["shots"] == 1
    assert len(acq.lines) == total_lines and len(acq.adc) == total_lines * 128
    assert s3.signal_shape(acq) == (4, total_lines * 128)
    assert acq.measurements["reconstruction"] == "bart_sampling_sense@1"
    expected = np.arange(1, 128, 3)
    if acs_lines:
        expected = np.union1d(expected, np.arange(60, 68))
    np.testing.assert_array_equal(np.unique(acq.indices[:, 1]), expected)


@pytest.mark.parametrize(
    "mutation,rejected",
    [
        ("missing_line", "sampling_pattern"),
        ("no_alternation", "epi"),
        ("wrong_fov", "cartesian"),
    ],
)
def test_r3_rejects_wrong_mask_polarity_and_encoding(tmp_path, mutation, rejected):
    case = case_for()
    path = tmp_path / f"{mutation}.seq"
    options = options_for(case)
    if mutation == "no_alternation":
        options["adaptive_epi"] = (
            True  # Keep this wrong-polarity pulse physically timed.
        )
    make_sequence(case, mutation, **options).write(str(path))
    acq = inspect_sequence(path, case)
    assert any(item["name"] == rejected and not item["passed"] for item in acq.checks)


def test_r4_is_not_supported_by_the_existing_receive_model():
    with pytest.raises(ValueError, match="2 or 3"):
        epi_sampling_mask([128, 128], {"acceleration": 4, "calibration_lines": 8})


def _with_extra_pulse(authored, *, before_adc):
    """Copy a sequence, adding an unspoiled 0.1 rad pulse before ADC block n."""
    from pypulseq import Sequence, make_block_pulse, make_delay

    sequence = Sequence(authored.system)
    pulse = make_block_pulse(
        0.1,
        duration=0.0004,
        delay=authored.system.rf_dead_time,
        system=authored.system,
        use="excitation",
    )
    adc_blocks = 0
    for number in authored.block_events:
        block = authored.get_block(number)
        if block.adc is not None:
            if adc_blocks == before_adc:
                sequence.add_block(pulse)
            adc_blocks += 1
        elif before_adc is None and number == next(iter(authored.block_events)):
            sequence.add_block(pulse)
        events = [
            getattr(block, name)
            for name in ("rf", "gx", "gy", "gz", "adc")
            if getattr(block, name) is not None
        ]
        sequence.add_block(make_delay(authored.block_durations[number]), *events)
    return sequence


@pytest.mark.parametrize("hardware", ["low_field", "standard_1_5t", "high_performance"])
def test_chess_preparation_before_the_single_excitation_is_allowed(tmp_path, hardware):
    case = case_for(hardware)
    path = tmp_path / "chess.seq"
    make_sequence(case, **chess_epi_options(case)).write(str(path))
    acq = inspect_sequence(path, case)
    assert all(item["passed"] for item in acq.checks), acq.checks
    epi = next(item for item in acq.checks if item["name"] == "epi")
    assert epi["detail"]["preparation_rf_pulses"] == 1
    assert epi["detail"]["acquiring_excitations"] == 1


def test_unacquired_preparation_rf_is_left_to_the_physical_judges(tmp_path):
    # Structurally a preparation; its unspoiled coherence is scored on GPU by
    # the spoiled_gre probe and the image (see docs/S2_EVALUATION.md).
    case = case_for()
    path = tmp_path / "unspoiled-preparation.seq"
    _with_extra_pulse(build(case), before_adc=None).write(str(path))
    acq = inspect_sequence(path, case)
    epi = next(item for item in acq.checks if item["name"] == "epi")
    assert epi["passed"] and epi["detail"]["preparation_rf_pulses"] == 1


def test_single_shot_rejects_an_extra_excitation_inside_the_echo_train(tmp_path):
    case = case_for()
    path = tmp_path / "extra-excitation.seq"
    _with_extra_pulse(build(case), before_adc=20).write(str(path))
    acq = inspect_sequence(path, case)
    assert any(item["name"] == "epi" and not item["passed"] for item in acq.checks)
