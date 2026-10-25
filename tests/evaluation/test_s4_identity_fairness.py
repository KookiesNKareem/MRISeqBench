"""CPU waveform identity: HS shape and balanced, asymmetric cardiac readouts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s4_sequence import _hs_shape, inspect_sequence
from tests.fixtures.s4_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]


def hs_waveform(beta, sweep_hz=50, samples=10000):
    edges = np.linspace(0, 0.01, samples + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    x = 2 * mids / edges[-1] - 1
    amplitude = 1 / np.cosh(beta * x)
    log_cosh = np.logaddexp(beta * x, -beta * x) - np.log(2)
    phase = 2 * np.pi * sweep_hz * edges[-1] / (4 * beta) * log_cosh
    return SimpleNamespace(
        start=0, end=edges[-1], edges=edges, signal=amplitude * np.exp(1j * phase)
    )


@pytest.mark.parametrize("beta", [0.3, 32])
def test_true_hs_shape_has_no_fixed_beta_or_sweep_floor(beta):
    # Identity only. A low-beta HS approaches a constant-amplitude linear chirp;
    # function is enforced separately by the B0 x B1 (0.8-1.2) inversion assay
    # (test_s4_adiabatic_b1.py), which rejects non-adiabatic chirps.
    result = _hs_shape(hs_waveform(beta))
    assert result["valid"], result
    assert result["beta"] == pytest.approx(beta, rel=0.01)
    assert result["sweep_hz"] < 100


def test_wrong_envelope_does_not_pass_hs_identity():
    pulse = hs_waveform(4)
    x = np.linspace(-1, 1, len(pulse.signal))
    pulse.signal *= 0.2 + 0.8 * np.sin(4 * np.pi * x) ** 2
    result = _hs_shape(pulse)
    assert not result["valid"], result
    assert result["sech_amplitude_rms"] > 0.05


def test_fifteen_sample_hs_is_not_rejected_by_sample_count():
    result = _hs_shape(hs_waveform(0.3, samples=15))
    assert result["valid"], result


def test_balanced_bssfp_does_not_require_echo_at_half_tr(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "t2_prepared"), hardware_ref="high_performance"
    )
    sequence = make_sequence(
        case,
        segment_lines=32,
        readout_flat=0.00128,
        gradient_duration=0.001,
        tr_s=0.0046,
        echo_time_s=0.0026,
    )
    path = tmp_path / "asymmetric.seq"
    sequence.write(str(path))
    acquisition = inspect_sequence(path, case)
    assert all(item["passed"] for item in acquisition.checks), acquisition.checks
    bssfp = next(c for c in acquisition.checks if c["name"] == "frame_0_balanced_bssfp")
    for segment in bssfp["detail"]["segments"]:
        te = np.asarray(segment["echo_te_ms"])
        tr = segment["tr_ms"]
        assert np.all(np.abs(te - tr / 2) > max(0.01, 0.05 * tr))
