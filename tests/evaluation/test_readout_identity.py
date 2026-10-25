"""Physical readout identity controls; no Bloch replay or benchmark references."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s2_sequence
from tests.fixtures.reference128_readouts import options_for
from tests.fixtures.s2_sequences import make_sequence


@pytest.mark.parametrize("hardware", ["low_field", "standard_1_5t", "high_performance"])
@pytest.mark.parametrize("reverse", [False, True])
def test_multishot_epi_accepts_both_physical_interleave_directions(
    tmp_path, hardware, reverse
):
    catalog = Catalog(Path(__file__).resolve().parents[2])
    case = catalog.resolve(catalog.get("tasks", "epi_multishot"), hardware_ref=hardware)
    path = tmp_path / "interleaved.seq"
    make_sequence(case, reverse_epi=reverse, **options_for(case)).write(str(path))
    acquisition = s2_sequence.inspect_sequence(path, case)
    assert all(check["passed"] for check in acquisition.checks), acquisition.checks
    assert len(acquisition.lines) == 128 and len(acquisition.adc) == 128 * 128
    assert len(np.unique(acquisition.indices, axis=0)) == 128 * 128
    for shot in np.unique(acquisition.excitation):
        group = np.flatnonzero(acquisition.excitation == shot)
        ky = [acquisition.indices[acquisition.lines[line][0], 1] for line in group]
        np.testing.assert_array_equal(np.diff(ky), -16 if reverse else 16)


@pytest.mark.parametrize(
    "spin_echoes,gradient_echoes,passed", [(4, 2, True), (2, 4, False)]
)
def test_grase_requires_declared_spin_echo_pairing(
    monkeypatch, spin_echoes, gradient_echoes, passed
):
    catalog = Catalog(Path(__file__).resolve().parents[2])
    case = catalog.resolve(catalog.get("tasks", "grase"))
    case["task"]["sequence"]["readout"]["shots"] = 2
    case["task"]["sequence"]["te_ms"] = {"max": 100}
    pulses, lines, adc, k, centres, excitation = [], [], [], [], [], []
    for shot in range(2):
        start = float(shot)
        pulses.append(SimpleNamespace(center=start, refocusing=False))
        for echo in range(spin_echoes):
            spin_time = start + (echo + 1) * 0.020
            pulses.append(
                SimpleNamespace(center=spin_time - 0.010, refocusing=True, flip=180)
            )
            for ge in range(gradient_echoes):
                centre = spin_time + (ge - (gradient_echoes - 1) / 2) * 0.002
                ids = np.arange(len(adc), len(adc) + 2)
                polarity = (-1) ** ge
                lines.append(ids)
                adc.extend([centre - 0.0001, centre + 0.0001])
                k.extend([[-polarity / 0.2, 0, 0], [polarity / 0.2, 0, 0]])
                centres.append(centre)
                excitation.append(shot)
    acquisition = SimpleNamespace(
        pulses=sorted(pulses, key=lambda pulse: pulse.center),
        lines=lines,
        adc=np.asarray(adc),
        k=np.asarray(k),
        excitation=np.asarray(excitation),
        flips=np.full(len(lines), 90.0),
        checks=[],
        measurements={"echo_times_s": centres},
    )
    monkeypatch.setattr(s2_sequence, "inspect_base", lambda *_: acquisition)
    result = s2_sequence.inspect_sequence("unused.seq", case)
    checks = {check["name"]: check for check in result.checks}
    assert checks["echo_train_length"]["passed"]
    assert checks["epi"]["passed"]
    assert checks["refocused_echo_train"]["passed"] is passed


def test_radial_coverage_allows_close_spokes_when_physical_grid_is_covered():
    count = 202
    step = np.pi / count
    angles = np.arange(count) * step
    angles[1], angles[2] = 0.05 * step, 1.5 * step
    nodes = np.concatenate(
        [
            np.c_[
                np.cos(angle) * np.arange(-64, 64), np.sin(angle) * np.arange(-64, 64)
            ]
            for angle in angles
        ]
    )
    grid = np.indices((128, 128)).reshape(2, -1).T - 64
    disk = np.linalg.norm(grid, axis=1) <= 63
    assert cKDTree(nodes).query(grid[disk])[0].max() <= 1.5
    assert s2_sequence.angular_coverage(angles, np.pi)
    assert not s2_sequence.angular_coverage(np.zeros(count), np.pi)
