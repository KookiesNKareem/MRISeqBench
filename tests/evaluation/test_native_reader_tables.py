"""Preflight mirrors the native (KomaMRIFiles 0.12.0) Pulseq reader (CPU)."""

import re
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.preflight import (
    native_extension_errors,
    native_reader_errors,
    native_table_errors,
    preflight,
)
from tests.fixtures import reference128_advanced

ROOT = Path(__file__).resolve().parents[2]
HUGE = "9" * 39  # overflows Julia Int64


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "t2_prepared"), hardware_ref="high_performance"
    )
    path = tmp_path_factory.mktemp("native") / "control.seq"
    reference128_advanced.build(case).write(str(path))
    return case, path.read_text()


def section_row(text, header):
    lines = text.split("\n")
    index = lines.index(header) + 1
    while not lines[index] or lines[index].startswith("#"):
        index += 1
    return lines[index]


def test_reference_control_is_clean(control):
    _case, text = control
    assert native_table_errors(text) == []
    assert native_extension_errors(text) == []


def test_unused_overflowing_trigger_table_fails_preflight(control, tmp_path):
    # Codex round 9: an appended, unreferenced table with an Int64 overflow.
    case, text = control
    path = tmp_path / "overflow.seq"
    path.write_text(
        text.replace(
            "[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 1 0\n2 1 2 0\n"
        ).replace("1 2 1 0 10\n", f"1 2 1 0 10\n2 2 1 0 {HUGE}\n")
    )
    checks = {c["name"]: c for c in preflight(path, case, hardware_limits=False)}
    failed = {name for name, c in checks.items() if not c["passed"]}
    assert failed & {"native_tables", "pulseq_preflight"}, failed


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("major 1\nminor 5", f"major 1\nminor {HUGE}", 1),
        lambda t: t.replace("major 1\nminor 5", "major 1\nminor 3", 1),
        lambda t: t.replace("major 1\nminor 5", "major 1\n# c\nminor 5", 1),
        lambda t: t.replace("[BLOCKS]\n", f"[BLOCKS]\n1 {HUGE} 0 0 0 0 0 0\n", 1),
        lambda t: t.replace("[BLOCKS]\n", "[BLOCKS]\n# note\n", 1),
        lambda t: t.replace("[BLOCKS]\n", "[BLOCKS]\n1 +1 0 0 0 0 0 0\n", 1),
        lambda t: t.replace("[ADC]\n", f"[ADC]\n{HUGE} 128 10000 60 0 0 0 0 0\n", 1),
        lambda t: t.replace("[ADC]\n", "[ADC]\n7 1e30 10000 60 0 0 0 0 0\n", 1),
        lambda t: t.replace("[ADC]\n", "[ADC]\n7 1.5 10000 60 0 0 0 0 0\n", 1),
        lambda t: t.replace("shape_id 1\n", f"shape_id {HUGE}\n", 1),
        lambda t: t.replace(
            "[SHAPES]\n\nshape_id 1\nnum_samples ",
            "[SHAPES]\n\nshape_id 1\nnum_samples 9",
            1,
        ),
        lambda t: t.replace("[DEFINITIONS]\n", "[DEFINITIONS]\nNx 1.5\n", 1),
        lambda t: t.replace("[DEFINITIONS]\n", f"[DEFINITIONS]\nNy {HUGE}\n", 1),
        lambda t: t.replace(
            "[DEFINITIONS]\n", "[DEFINITIONS]\nBlockDurationRaster\n", 1
        ),
    ],
    ids=[
        "version-overflow",
        "version-1.3",
        "version-comment",
        "block-overflow",
        "block-comment",
        "block-plus-sign",
        "adc-id-overflow",
        "adc-samples-overflow",
        "adc-samples-fraction",
        "shape-id-overflow",
        "shape-count-mismatch",
        "definition-nx-fraction",
        "definition-ny-overflow",
        "definition-raster-missing",
    ],
)
def test_native_unreadable_sections_are_rejected(control, mutate):
    _case, text = control
    mutated = mutate(text)
    assert mutated != text
    assert native_table_errors(mutated)


def test_rows_after_a_table_break_are_rejected(control):
    _case, text = control
    row = section_row(text, "[RF]")
    mutated = text.replace(row + "\n", row + "\n\n" + "9" + row[1:] + "\n", 1)
    assert any("outside its table" in e for e in native_table_errors(mutated))


def test_extension_ids_and_links_are_int64_bounded(control):
    _case, text = control
    for old, new in (
        ("extension TRIGGERS 1", f"extension TRIGGERS {HUGE}"),
        ("[EXTENSIONS]\n1 1 1 0", f"[EXTENSIONS]\n1 1 1 {HUGE}"),
        ("1 2 1 0 10", f"1 {HUGE} 1 0 10"),
    ):
        assert native_extension_errors(text.replace(old, new, 1)), new


def test_preflight_stops_at_native_table_failure(control, tmp_path):
    case, text = control
    path = tmp_path / "blocks.seq"
    path.write_text(text.replace("[BLOCKS]\n", f"[BLOCKS]\n1 {HUGE} 0 0 0 0 0 0\n", 1))
    checks = preflight(path, case, hardware_limits=False)
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


def run_bounded(function, *args, seconds=60):
    import signal

    def timeout(*_):
        raise TimeoutError("preflight did not return")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    try:
        return function(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


LABEL_CYCLE = "\n[EXTENSIONS]\n1 1 1 1\n\nextension LABELSET 1\n1 1 LIN\n"


def test_unreferenced_extension_cycle_fails_without_hanging(control, tmp_path):
    case, text = control
    # Replace the trigger list with a self-linked label list.
    head = text[: text.index("# Format of extension lists")]
    tail = text[text.index("[SHAPES]") :]
    path = tmp_path / "cycle.seq"
    path.write_text(head + tail.rstrip() + "\n" + LABEL_CYCLE)
    assert any("cycle" in e for e in native_extension_errors(path.read_text()))
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


def test_referenced_extension_cycle_does_not_reach_pypulseq(control, tmp_path):
    case, text = control
    path = tmp_path / "cycle.seq"
    path.write_text(text.replace("[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 1 1\n"))
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda t: t.replace(
                "[ADC]\n", f"[ADC]\n{2**40} 128 10000 60 0 0 0 0 0\n", 1
            ),
            "dense",
        ),
        (
            lambda t: t.replace("[ADC]\n", "[ADC]\n9 128 10000 60 0 0 0 0 0\n", 1),
            "dense",
        ),
        (lambda t: t.replace("[TRAP]\n", "[TRAP]\n1 100 10 10 10 0\n", 1), "dense"),
        (
            lambda t: t.replace(
                "[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 1 0\n5 1 1 0\n", 1
            ),
            "dense",
        ),
        (
            lambda t: re.sub(
                r"\[BLOCKS\]\n *1 +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +\d+ +(\d+)\n",
                r"[BLOCKS]\n1 \1 \2 \3 \4 \5 99 \6\n",
                t,
                count=1,
            ),
            "missing ADC",
        ),
        (
            lambda t: t.replace(
                "[SHAPES]\n\nshape_id 1\n", "[SHAPES]\n\nshape_id 1\n", 1
            ).replace("\nshape_id 1\n", "\nshape_id 777\n", 1),
            "missing",
        ),
        (
            lambda t: t.replace(
                "[SHAPES]\n",
                "[SHAPES]\n\nshape_id 9999\nnum_samples 1000000000\n0\n0\n999999998\n",
                1,
            ),
            "exceed",
        ),
        (lambda t: t.replace("major 1\n", "major " + "1" * 5000 + "\n", 1), "VERSION"),
        (
            lambda t: t.replace(
                "[BLOCKS]\n", "[BLOCKS]\n" + "1" * 5000 + " 1 0 0 0 0 0 0\n", 1
            ),
            "BLOCKS",
        ),
        (lambda t: t.replace("[BLOCKS]\n", "[BLOCKS]\n1 ² 0 0 0 0 0 0\n", 1), "BLOCKS"),
    ],
    ids=[
        "adc-id-2**40",
        "adc-id-gap",
        "trap-gradient-duplicate",
        "extension-id-gap",
        "block-missing-adc",
        "rf-missing-shape",
        "huge-compressed-shape",
        "version-5000-digits",
        "block-5000-digits",
        "block-superscript-digit",
    ],
)
def test_native_allocation_and_token_domains(control, tmp_path, mutate, expected):
    case, text = control
    mutated = mutate(text)
    assert mutated != text
    errors = native_table_errors(mutated) or native_extension_errors(mutated)
    assert any(expected in e for e in errors), errors
    path = tmp_path / "mutated.seq"
    path.write_text(mutated)
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


def test_preflight_always_returns_a_verdict_under_fuzzing(submission, tmp_path):
    import json
    import random

    case = json.loads((submission.parent / "case.json").read_text())
    lines = submission.read_text().split("\n")
    rng = random.Random(1729)
    junk = ["nan", "-1", "9" * 30, "²", "1e999", "", "0x10", "1.5", "#", "[BLOCKS]"]
    for trial in range(150):
        mutated = list(lines)
        for _ in range(rng.randint(1, 4)):
            i = rng.randrange(len(mutated))
            action = rng.random()
            if action < 0.5 and mutated[i].split():
                tokens = mutated[i].split()
                tokens[rng.randrange(len(tokens))] = rng.choice(junk)
                mutated[i] = " ".join(tokens)
            elif action < 0.7:
                del mutated[i]
            elif action < 0.85:
                mutated.insert(i, mutated[i])
            else:
                mutated.insert(i, "")
        path = tmp_path / f"fuzz{trial}.seq"
        path.write_text("\n".join(mutated))
        checks = run_bounded(lambda p=path: preflight(p, case, hardware_limits=True))
        assert isinstance(checks, list) and checks


@pytest.mark.parametrize("ending", ["\n", ""], ids=["final-newline", "no-newline"])
def test_shape_data_ending_at_eof_fails_without_hanging(control, tmp_path, ending):
    case, text = control
    unsigned = re.sub(r"(?ms)^\[SIGNATURE\].*", "", text)
    assert unsigned.rstrip().splitlines()[-2:][0].strip()  # shape data is last
    path = tmp_path / "eof.seq"
    path.write_text(unsigned.rstrip() + ending)
    assert any("EOF" in e for e in native_table_errors(path.read_text()))
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]
    # A terminating blank line is the canonical form and still passes.
    path.write_text(unsigned.rstrip() + "\n\n")
    assert native_table_errors(path.read_text()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: [rows[0].replace("1", "0", 1), *rows[1:]],
        lambda rows: [rows[0], rows[0].replace("1", "2", 1), *rows[1:]],
        lambda rows: [rows[0], *rows],
        lambda rows: [rows[1], rows[0], *rows[2:]],
    ],
    ids=["zero-id", "duplicate-id-row", "duplicate-row", "out-of-order"],
)
def test_blocks_ids_must_be_canonical(control, tmp_path, mutate):
    case, text = control
    start = text.index("[BLOCKS]\n") + len("[BLOCKS]\n")
    end = text.index("\n\n", start)
    rows = text[start:end].split("\n")
    rows = [" ".join(row.split()) for row in rows]
    assert rows[0].split()[0] == "1"
    mutated = text[:start] + "\n".join(mutate(rows)) + text[end:]
    assert any("1..N in file order" in e for e in native_table_errors(mutated))
    path = tmp_path / "blocks.seq"
    path.write_text(mutated)
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


CODEX_HUGE_ADC = """[VERSION]
major 1
minor 4
revision 1

[DEFINITIONS]
AdcRasterTime 1e-07
BlockDurationRaster 1e-05
GradientRasterTime 1e-05
RadiofrequencyRasterTime 1e-06

[BLOCKS]
1 14 1 0 0 0 0 0
2 10000002 0 0 0 0 1 0

[RF]
1 100 1 2 0 100 0 0

[ADC]
1 {samples} 100 10 0 0

[SHAPES]

shape_id 1
num_samples 2
1
1

shape_id 2
num_samples 2
0
0

"""


def test_huge_adc_event_fails_before_any_allocation(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "spin_echo_t2w"), hardware_ref="standard_1_5t"
    )
    path = tmp_path / "huge_adc.seq"
    path.write_text(CODEX_HUGE_ADC.format(samples=10**9))
    errors = native_table_errors(path.read_text())
    assert any("ADC event with 1000000000 samples" in e for e in errors), errors
    assert any("acquired ADC samples" in e for e in errors), errors
    checks = run_bounded(lambda: preflight(path, case))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]
    # The same file with an ordinary readout is not rejected by these bounds.
    assert native_table_errors(CODEX_HUGE_ADC.format(samples=128)) == []


@pytest.mark.parametrize(
    "repeats,expected",
    [(2**22 // 128 + 1, "acquired ADC samples"), (2**18 + 1, "blocks exceed")],
)
def test_aggregate_block_and_sample_bounds(repeats, expected):
    rows = "\n".join(f"{i} 2 0 0 0 0 1 0" for i in range(1, repeats + 1))
    text = CODEX_HUGE_ADC.format(samples=128).replace(
        "1 14 1 0 0 0 0 0\n2 10000002 0 0 0 0 1 0", rows
    )
    assert any(expected in e for e in native_table_errors(text))


def test_waveform_samples_count_every_block_reference():
    rows = "\n".join(f"{i} 100002 1 0 0 0 0 0" for i in range(1, 21))
    text = (
        CODEX_HUGE_ADC.format(samples=128)
        .replace("1 14 1 0 0 0 0 0\n2 10000002 0 0 0 0 1 0", rows)
        .replace("num_samples 2\n1\n1\n", "num_samples 1000000\n1\n1\n999998\n")
        .replace("num_samples 2\n0\n0\n", "num_samples 1000000\n0\n0\n999998\n")
    )
    assert any("shape samples over all blocks" in e for e in native_table_errors(text))


def mutate_row(text, header, column, value):
    row = section_row(text, header)
    fields = row.split()
    fields[column] = value
    return text.replace(row + "\n", " ".join(fields) + "\n", 1)


@pytest.mark.parametrize(
    "header,column,value,expected",
    [
        ("[RF]", 5, "1e9", "outside its waveform"),
        ("[RF]", 5, "-50", "outside its waveform"),
        ("[RF]", 7, "3.5", "freq_ppm/phase_ppm"),
        ("[RF]", 8, "1", "freq_ppm/phase_ppm"),
        ("[RF]", 11, "x", "use 'x'"),
        ("[ADC]", 4, "2", "ADC freq_ppm/phase_ppm"),
        ("[ADC]", 8, "1", "phase modulation"),
    ],
    ids=[
        "rf-center-1e9us",
        "rf-center-negative",
        "rf-freq-ppm",
        "rf-phase-ppm",
        "rf-use",
        "adc-ppm",
        "adc-phase-id",
    ],
)
def test_native_only_fields_must_agree_with_inspection(
    control, tmp_path, header, column, value, expected
):
    case, text = control
    mutated = mutate_row(text, header, column, value)
    assert mutated != text
    errors = native_reader_errors(mutated)
    assert any(expected in e for e in errors), errors
    path = tmp_path / "field.seq"
    path.write_text(mutated)
    checks = run_bounded(lambda: preflight(path, case, hardware_limits=False))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


def test_rf_center_at_waveform_edges_is_accepted(control):
    _case, text = control
    row = section_row(text, "[RF]").split()
    assert native_reader_errors(text) == [] and float(row[5]) > 0


def test_rotations_extension_is_rejected(control):
    _case, text = control
    mutated = text.replace(
        "extension TRIGGERS 1\n1 2 1 0 10\n",
        "extension TRIGGERS 1\n1 2 1 0 10\n\nextension ROTATIONS 2\n1 1 0 0 0\n",
    )
    assert any("ROTATIONS" in e for e in native_extension_errors(mutated))


def test_negative_adc_count_cannot_cancel_the_sample_caps(tmp_path):
    # Codex round 14: 10,000 x 65,536 samples offset by one -655,359,999 row.
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "spin_echo_t2w"), hardware_ref="standard_1_5t"
    )
    rows = ["1 14 1 0 0 0 0 0"]
    rows += [f"{i} 658 0 0 0 0 1 0" for i in range(2, 10002)]
    rows += ["10002 2 0 0 0 0 2 0"]
    text = (
        CODEX_HUGE_ADC.format(samples=65536)
        .replace("1 14 1 0 0 0 0 0\n2 10000002 0 0 0 0 1 0", "\n".join(rows))
        .replace(
            "1 65536 100 10 0 0\n", "1 65536 100 10 0 0\n2 -655359999 100 10 0 0\n"
        )
    )
    path = tmp_path / "negative_adc.seq"
    path.write_text(text)
    errors = native_reader_errors(text)
    assert any("ADC samples must be >= 0" in e for e in errors), errors
    checks = run_bounded(lambda: preflight(path, case))
    assert [c["name"] for c in checks if not c["passed"]] == ["native_tables"]


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("1 128 100 10 0 0", "1 128 -100 10 0 0", "dwell > 0"),
        ("1 128 100 10 0 0", "1 128 100 -10 0 0", "delay >= 0"),
        ("1 100 1 2 0 100 0 0", "1 100 1 2 0 -100 0 0", "RF delay"),
        (
            "1 100 1 2 0 100 0 0",
            "1 100 1 2 -2 100 0 0",
            "time shape id -2 is not supported",
        ),
        (
            "RadiofrequencyRasterTime 1e-06",
            "RadiofrequencyRasterTime -1e-06",
            "positive",
        ),
    ],
)
def test_signed_fields_feeding_time_axes_are_bounded(old, new, expected):
    text = CODEX_HUGE_ADC.format(samples=128)
    assert native_reader_errors(text) == []
    mutated = text.replace(old, new, 1)
    assert mutated != text
    assert any(expected in e for e in native_reader_errors(mutated))


def test_trapezoid_times_must_be_nonnegative(control):
    _case, text = control
    mutated = mutate_row(text, "[TRAP]", 3, "-10")
    assert any("trapezoid" in e for e in native_reader_errors(mutated))


def test_decreasing_time_shape_is_rejected():
    text = CODEX_HUGE_ADC.format(samples=128)
    mutated = text.replace("1 100 1 2 0 100 0 0", "1 100 1 2 3 100 0 0").replace(
        "[SHAPES]\n", "[SHAPES]\n\nshape_id 3\nnum_samples 2\n5\n0\n", 1
    )
    assert any(
        "time shape 3 must not decrease" in e for e in native_reader_errors(mutated)
    )


@pytest.fixture(scope="module")
def gre_t2star(tmp_path_factory):
    from tests.fixtures import reference128_readouts

    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t2star"), hardware_ref="high_performance"
    )
    path = tmp_path_factory.mktemp("gre") / "reference.seq"
    reference128_readouts.build(case).write(str(path))
    return case, path.read_text()


def append_trap_block(text, trap):
    """Add one dense [TRAP] row and a final 1 ms block (after the last ADC)."""
    lines = text.split("\n")
    trap_at = lines.index("[TRAP]") + 1
    while lines[trap_at].startswith("#"):
        trap_at += 1
    end = trap_at
    while lines[end].strip():
        end += 1
    gradient_ids = []
    for header in ("[GRADIENTS]", "[TRAP]"):
        if header not in lines:
            continue
        row = lines.index(header) + 1
        while lines[row].startswith("#"):
            row += 1
        while lines[row].strip():
            gradient_ids.append(int(lines[row].split()[0]))
            row += 1
    new_id = max(gradient_ids) + 1
    lines.insert(end, f"{new_id} {trap}")
    blocks_at = lines.index("[BLOCKS]") + 1
    block_end = blocks_at
    while lines[block_end].strip():
        block_end += 1
    last = int(lines[block_end - 1].split()[0])
    lines.insert(block_end, f"{last + 1} 100 0 {new_id} 0 0 0 0")
    text = "\n".join(lines)
    return re.sub(r"(?ms)^\[SIGNATURE\].*", "", text).rstrip() + "\n\n"


@pytest.mark.parametrize(
    "trap,expected",
    [
        ("1e9 0 0 1000 0", "rise > 0 and fall > 0"),
        ("1e9 1000 0 0 0", "rise > 0 and fall > 0"),
        ("1e9 500 0 500 0", None),
        ("2e6 10 0 10 0", None),
    ],
    ids=["missing-rise", "missing-fall", "over-amplitude", "over-slew"],
)
def test_degenerate_or_over_limit_trapezoid_after_last_adc_fails(
    gre_t2star, tmp_path, trap, expected
):
    case, text = gre_t2star
    path = tmp_path / "base.seq"
    path.write_text(text)
    base = run_bounded(lambda: preflight(path, case))
    assert all(c["passed"] for c in base), [c for c in base if not c["passed"]]
    mutated = append_trap_block(text, trap)
    path.write_text(mutated)
    checks = run_bounded(lambda: preflight(path, case))
    failed = [c["name"] for c in checks if not c["passed"]]
    if expected:
        assert failed == ["native_tables"], failed
        assert expected in str(checks[0]["detail"])
    else:
        assert "gradient_events" in failed, failed


def arbitrary_gradient_file(path, waveform):
    import numpy as np
    from pypulseq import (
        Opts,
        Sequence,
        make_arbitrary_grad,
        make_block_pulse,
        make_delay,
    )

    system = Opts(max_grad=1e9, max_slew=1e15, grad_unit="Hz/m", slew_unit="Hz/m/s")
    seq = Sequence(system)
    seq.add_block(make_block_pulse(0.1, duration=1e-3, system=system))
    grad = make_arbitrary_grad(
        "x", np.asarray(waveform, float), first=0, last=0, system=system
    )
    seq.add_block(grad, make_delay(1e-3))
    seq.write(str(path), create_signature=False)
    return path.read_text()


def test_half_raster_time_id_is_rejected_explicitly(tmp_path):
    text = arbitrary_gradient_file(tmp_path / "grad.seq", [0, 100, 0])
    assert native_reader_errors(text) == []
    lines = text.split("\n")
    row = lines.index("[GRADIENTS]") + 1
    while lines[row].startswith("#"):
        row += 1
    fields = lines[row].split()
    fields[5] = "-1"  # time shape id column (Pulseq 1.5)
    lines[row] = " ".join(fields)
    errors = native_reader_errors("\n".join(lines))
    assert any("time shape id -1 is not supported" in e for e in errors), errors


def test_lint_does_not_apply_slew_to_continuous_gradients(gre_t2star, tmp_path):
    case, _text = gre_t2star
    path = tmp_path / "steep.seq"
    arbitrary_gradient_file(path, [0, 400000, 0])
    lint = {c["name"]: c for c in preflight(path, case, hardware_limits=False)}
    assert lint["gradient_continuity"]["passed"], lint["gradient_continuity"]
    assert "gradient_endpoint_slew" not in lint
    full = {c["name"]: c for c in preflight(path, case)}
    assert not full["gradient_endpoint_slew"]["passed"]
    assert full["gradient_continuity"]["passed"]


def duplicate_blocks_section(text):
    """Codex round 18: renumber blocks and prepend a second [BLOCKS] section."""
    start = text.index("[BLOCKS]\n") + len("[BLOCKS]\n")
    end = text.index("\n\n", start)
    rows = [row.split() for row in text[start:end].split("\n")]
    shifted = "\n".join(" ".join([str(int(r[0]) + 1), *r[1:]]) for r in rows)
    body = text[:start] + shifted + text[end:]
    return body.replace("[BLOCKS]\n", "[BLOCKS]\n1 1 0 0 0 0 0 0\n\n[BLOCKS]\n", 1)


@pytest.mark.parametrize("task", ["rf_excitation_2d", "t2_prepared"])
def test_duplicate_sections_fail_full_evaluation_at_preflight(task, tmp_path):
    import re

    from mriseqbench.evaluation import evaluate

    catalog = Catalog(ROOT)
    case = catalog.resolve(catalog.get("tasks", task), hardware_ref="high_performance")
    path = tmp_path / "control.seq"
    reference128_advanced.build(case).write(str(path))
    text = re.sub(r"(?ms)^\[SIGNATURE\].*", "", path.read_text()).rstrip() + "\n\n"
    mutated = duplicate_blocks_section(text)
    errors = native_table_errors(mutated)
    assert any("duplicate section [BLOCKS]" in e for e in errors), errors
    path.write_text(mutated)
    report = run_bounded(lambda: evaluate(path, case, ROOT, tmp_path / "work"))
    assert report["status"] == "failed"
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert "native_tables" in failed, failed


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda t: t.replace("[RF]\n", "[RF]\n\n[RF]\n", 1), "duplicate section [RF]"),
        (
            lambda t: t.replace(
                "extension TRIGGERS 1\n1 2 1 0 10\n",
                "extension TRIGGERS 1\n1 2 1 0 10\n\nextension LABELSET 1\n1 1 LIN\n",
            ),
            "duplicate section extension LABELSET 1",
        ),
        (lambda t: t.replace("[SHAPES]\n", "[SHAPES] \n", 1), "surrounding whitespace"),
        (
            lambda t: t.replace("[SHAPES]\n", "[SHAPES]\n\n[NOTES]\n", 1),
            "unknown section",
        ),
    ],
    ids=["duplicate-rf", "duplicate-extension-id", "header-whitespace", "unknown"],
)
def test_section_headers_must_be_unique_and_known(control, mutate, expected):
    _case, text = control
    mutated = mutate(text)
    assert mutated != text
    assert any(expected in e for e in native_table_errors(mutated))


HIDDEN_BLOCKS = "[SIGNATURE]\nType md5\n[BLOCKS]\n1 100 1 0 0 0 0 0\n\n"


@pytest.fixture(scope="module")
def excitation_control(tmp_path_factory):
    import re

    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "rf_excitation_2d"), hardware_ref="high_performance"
    )
    path = tmp_path_factory.mktemp("rf") / "control.seq"
    reference128_advanced.build(case).write(str(path))
    text = re.sub(r"(?ms)^\[SIGNATURE\].*", "", path.read_text()).rstrip() + "\n\n"
    return case, text + HIDDEN_BLOCKS


@pytest.mark.parametrize("feedback", [True, False], ids=["feedback", "final"])
def test_header_hidden_in_signature_fails_evaluation(
    excitation_control, tmp_path, feedback
):
    from mriseqbench.evaluation import evaluate

    case, text = excitation_control
    assert any("duplicate section [BLOCKS]" in e for e in native_table_errors(text))
    path = tmp_path / "hidden.seq"
    path.write_text(text)
    report = run_bounded(
        lambda: evaluate(path, case, ROOT, tmp_path / "work", feedback=feedback)
    )
    assert report["status"] == "failed"
    assert "native_tables" in [c["name"] for c in report["checks"] if not c["passed"]]


def test_diagnostic_replay_rejects_an_invalid_generated_file(
    excitation_control, tmp_path
):
    from mriseqbench.evaluation.excitation import (
        DiagnosticReplayError,
        diagnostic_acquisition,
    )

    case, text = excitation_control
    path = tmp_path / "hidden.seq"
    path.write_text(text)
    with pytest.raises(DiagnosticReplayError):
        diagnostic_acquisition(path, case, tmp_path)


def test_diagnostic_replay_of_a_valid_reference_is_accepted(
    excitation_control, tmp_path
):
    from mriseqbench.evaluation.excitation import diagnostic_acquisition

    case, text = excitation_control
    path = tmp_path / "valid.seq"
    path.write_text(text[: -len(HIDDEN_BLOCKS)])
    acquisition = diagnostic_acquisition(path, case, tmp_path)
    assert native_reader_errors(Path(acquisition.source_path).read_text()) == []


def labels_around_blocks(text):
    """Valid ordering: [EXTENSIONS] before [BLOCKS]; LABELSET spec right after it."""
    start = text.index("[BLOCKS]\n") + len("[BLOCKS]\n")
    end = text.index("\n\n", start)
    rows = text[start:end].split("\n")
    first = rows[0].split()
    rows[0] = " ".join([*first[:7], "1"])
    blocks = "[BLOCKS]\n" + "\n".join(rows) + "\n\nextension LABELSET 1\n1 1 LIN\n"
    head = text[: start - len("[BLOCKS]\n")]
    return head + "[EXTENSIONS]\n1 1 1 0\n\n" + blocks + text[end:]


def test_diagnostic_rows_stay_in_their_tables_next_to_label_extensions(
    excitation_control, tmp_path
):
    from mriseqbench.evaluation.excitation import diagnostic_acquisition

    case, text = excitation_control
    mutated = labels_around_blocks(text[: -len(HIDDEN_BLOCKS)])
    path = tmp_path / "labels.seq"
    path.write_text(mutated)
    checks = run_bounded(lambda: preflight(path, case))
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    acquisition = diagnostic_acquisition(path, case, tmp_path)
    generated = Path(acquisition.source_path).read_text()
    assert native_reader_errors(generated) == []
    assert "1 1 LIN\n" in generated and "1 1 LIN\n1" not in generated


@pytest.mark.skipif(
    __import__("os").environ.get("MRISEQBENCH_TEST_CUDA") != "1",
    reason="full RF-excitation replay needs CUDA",
)
@pytest.mark.parametrize("feedback", [True, False], ids=["feedback", "final"])
def test_label_extension_ordering_passes_full_evaluation(
    excitation_control, tmp_path, feedback
):
    from mriseqbench.evaluation import evaluate

    case, text = excitation_control
    path = tmp_path / "labels.seq"
    path.write_text(labels_around_blocks(text[: -len(HIDDEN_BLOCKS)]))
    report = evaluate(path, case, ROOT, tmp_path / "work", feedback=feedback)
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert "diagnostic_replay" not in failed and "excitation_only" not in failed
    assert report["status"] == "passed", failed
