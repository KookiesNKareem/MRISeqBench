"""Refocused shot timing with unacquired CHESS; no signal simulation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_sequence import gradient_integrals
from mriseqbench.evaluation.s2_sequence import inspect_sequence
from tests.fixtures.s2_sequences import make_sequence


@pytest.mark.parametrize("family", ["grase", "tse_t2w", "propeller_tse"])
def test_preparation_keeps_acquired_tr_independent_of_terminal_recovery(
    tmp_path, family
):
    catalog = Catalog(Path(__file__).resolve().parents[2])
    case = catalog.resolve(catalog.get("tasks", family), hardware_ref="low_field")
    sequence = make_sequence(
        case,
        cartesian_flat=0.00096,
        grase_offset=0.0014,
        grase_spin_echo_spacing=0.020,
        split_grase_polarities=True,
        echo_time=0.0386,
        ref_duration=0.0006,
        fat_saturation_pulses=2,
        centric_tse=True,
    )
    path = tmp_path / "prepared.seq"
    sequence.write(str(path))
    acquired = inspect_sequence(path, case)
    assert all(c["passed"] for c in acquired.checks), acquired.checks
    assert len([p for p in acquired.pulses if not p.refocusing]) > 16
    assert (
        acquired.measurements["shots"] == case["task"]["sequence"]["readout"]["shots"]
    )
    np.testing.assert_allclose(
        acquired.tr, case["task"]["sequence"]["tr_ms"]["target"] / 1000, atol=1e-8
    )

    extended_sequence = deepcopy(sequence)
    extended_sequence.add_block(pp.make_delay(0.2))
    extended_sequence.write(str(path))
    extended = inspect_sequence(path, case)
    assert all(c["passed"] for c in extended.checks), extended.checks
    np.testing.assert_allclose(extended.tr, acquired.tr, atol=1e-10)
    assert extended.seq.duration()[0] > acquired.seq.duration()[0]

    # Ending after the complete train does not invent a shorter RF-to-RF TR.
    for block in list(sequence.block_events)[-2:]:
        assert not sequence.get_block(block).rf and not sequence.get_block(block).adc
        sequence.set_block(block, pp.make_delay(10e-6))
    sequence.block_cache.clear()
    assert sequence.check_timing()[0]
    sequence.write(str(path))
    truncated = inspect_sequence(path, case)
    assert all(c["passed"] for c in truncated.checks), truncated.checks
    np.testing.assert_allclose(truncated.tr, acquired.tr, atol=1e-10)

    # Actual RF-to-RF intervals still have to meet the declared TR bounds.
    for interval_change_ms in (-200, 200):
        changed = deepcopy(case)
        changed["task"]["sequence"]["tr_ms"]["target"] += interval_change_ms
        make_sequence(
            changed,
            cartesian_flat=0.00096,
            grase_offset=0.0014,
            grase_spin_echo_spacing=0.020,
            split_grase_polarities=True,
            echo_time=0.0386,
            ref_duration=0.0006,
            fat_saturation_pulses=2,
            centric_tse=True,
        ).write(str(path))
        wrong_interval = inspect_sequence(path, case)
        assert any(
            c["name"] == "tr_ms" and not c["passed"] for c in wrong_interval.checks
        )


def test_paired_crushers_dephase_fids_without_changing_rf_or_spin_echo_encoding(
    tmp_path,
):
    catalog = Catalog(Path(__file__).resolve().parents[2])
    case = catalog.resolve(
        catalog.get("tasks", "tse_t2w"), hardware_ref="high_performance"
    )
    options = {
        "cartesian_flat": 0.00096,
        "ref_duration": 0.0006,
        "fat_saturation_pulses": 2,
        "centric_tse": True,
    }
    acquired = []
    for crushed in (False, True):
        path = tmp_path / f"crushed-{crushed}.seq"
        make_sequence(case, refocusing_crushers=crushed, **options).write(str(path))
        a = inspect_sequence(path, case)
        assert all(c["passed"] for c in a.checks), a.checks
        acquired.append(a)
    before, after = acquired
    np.testing.assert_allclose(after.k, before.k, atol=1e-7, rtol=0)
    np.testing.assert_allclose(after.adc, before.adc, atol=1e-10, rtol=0)
    for original, crushed in zip(before.pulses, after.pulses, strict=True):
        np.testing.assert_array_equal(crushed.signal, original.signal)
        np.testing.assert_allclose(
            [crushed.center, crushed.phase],
            [original.center, original.phase],
            atol=1e-10,
        )
    # The precrusher actually adds at least one cycle/voxel to an RF-created
    # FID; the equality above proves that its paired lobe cancels for spin echoes.
    centre = next(p.center for p in before.pulses if p.refocusing)
    moment = gradient_integrals(after.seq, [centre]) - gradient_integrals(
        before.seq, [centre]
    )
    assert np.linalg.norm(moment) >= 0.99 * 128 / 0.2
