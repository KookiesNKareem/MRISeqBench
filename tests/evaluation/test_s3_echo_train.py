"""Sampling identity permits a GRE echo train without an excitation per line."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1_sequence import inspect_sequence as inspect_base
from mriseqbench.evaluation.s3_sequence import inspect_sequence

ROOT = Path(__file__).resolve().parents[2]


def test_sense_mask_allows_multiple_gradient_echoes_per_excitation(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", "sense_sampling"))
    system = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(np.deg2rad(5), duration=0.0004, system=system))
    read = pp.make_trapezoid("x", flat_area=128 / 0.2, flat_time=0.00128, system=system)
    adc = pp.make_adc(128, duration=read.flat_time, delay=read.rise_time, system=system)
    rewind = pp.make_trapezoid("x", area=-read.area / 2, duration=0.0008, system=system)
    for ky in sorted(set(range(0, 128, 2)) | set(range(60, 68))):
        phase = pp.make_trapezoid(
            "y", area=(ky - 64) / 0.2, duration=0.0008, system=system
        )
        unphase = pp.make_trapezoid(
            "y", area=-(ky - 64) / 0.2, duration=0.0008, system=system
        )
        seq.add_block(rewind, phase)
        seq.add_block(read, adc)
        seq.add_block(rewind, unphase)
    path = tmp_path / "echo_train.seq"
    seq.write(str(path))
    old_checks = {item["name"]: item for item in inspect_base(path, case).checks}
    assert not old_checks["single_line_per_excitation"]["passed"]
    actual = inspect_sequence(path, case)
    assert all(item["passed"] for item in actual.checks), actual.checks
    assert len(actual.lines) == 68
    # Structural freedom does not bypass the separate spoiling/image assays.
    assert "single_line_per_excitation" not in {item["name"] for item in actual.checks}
