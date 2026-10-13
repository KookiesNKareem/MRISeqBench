"""Known-good and corrupted controls for the independent waveform checks."""

import importlib.util
import json
import runpy
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from mriseqbench.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "readout_checks", ROOT / "tests/fixtures/arm_readout_checks.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def author(tmp_path, monkeypatch, kind):
    catalog = Catalog(ROOT)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ARM.json").write_text(
        catalog.get("arms", "a2_pypulseq").model_dump_json()
    )
    case = catalog.resolve(
        catalog.get("tasks", "pulseq_smoke"), catalog.get("physics", "ideal")
    )
    (tmp_path / "case.json").write_text(json.dumps(case))
    runpy.run_path(
        str(ROOT / "tests/fixtures/arm_readout_author.py"),
        init_globals={"control_kind": kind},
    )
    return tmp_path / "sequence.seq"


@pytest.mark.parametrize("kind", ["gradient_echo", "spin_echo"])
def test_independent_readout_controls(tmp_path, monkeypatch, kind):
    metrics = checks.validate_readout(author(tmp_path, monkeypatch, kind), kind)
    assert metrics["adc_count"] == 32


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("prephaser", "readout k-space"),
        ("receiver_phase", "ADC frequency/phase"),
        ("refocusing_label", "refocusing metadata"),
        ("refocusing_flip", "RF flip"),
    ],
)
def test_corrupted_controls_are_rejected(tmp_path, monkeypatch, mutation, reason):
    path = author(tmp_path, monkeypatch, "spin_echo")
    seq = pp.Sequence()
    seq.read(str(path))
    if mutation == "refocusing_label":
        # Re-registering an identical RF waveform in PyPulseq can retain its
        # existing use label. Corrupt the serialized label directly instead.
        text = path.read_text()
        start, end = text.index("[RF]"), text.index("[TRAP]")
        rf_section = text[start:end].replace(" r\n", " e\n")
        assert rf_section != text[start:end]
        path.write_text(text[:start] + rf_section + text[end:])
        with pytest.raises(AssertionError, match=reason):
            checks.validate_readout(path, "spin_echo")
        return
    for index in seq.block_events:
        block = seq.get_block(index)
        if mutation == "prephaser" and block.gx is not None and block.gx.amplitude < 0:
            block.gx.amplitude *= 0.5
            block.gx.area *= 0.5
            seq.set_block(index, block.gx)
            break
        if mutation == "receiver_phase" and block.adc is not None:
            block.adc.phase_offset = np.pi / 2
            seq.set_block(index, block.gx, block.adc)
            break
        if (
            mutation.startswith("refocusing")
            and block.rf is not None
            and block.rf.use == "refocusing"
        ):
            block.rf.signal *= 0.5
            seq.set_block(index, block.rf, pp.make_delay(seq.block_durations[index]))
            break
    seq.write(str(path))
    with pytest.raises(AssertionError, match=reason):
        checks.validate_readout(path, "spin_echo")
