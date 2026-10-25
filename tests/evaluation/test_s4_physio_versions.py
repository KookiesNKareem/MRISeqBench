"""S4 trigger resolution fails cleanly on unsupported or unreadable files (CPU)."""

from pathlib import Path

import numpy as np
import pytest
from pypulseq import Sequence

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s4_physio
from mriseqbench.evaluation.s4_physio import (
    InvalidPhysioSequence,
    explicit_waits,
    pulseq_version,
    resolve_sequence,
)
from tests.fixtures import reference128_advanced

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def case_and_text(tmp_path_factory):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "t2_prepared"), hardware_ref="high_performance"
    )
    path = tmp_path_factory.mktemp("s4") / "control.seq"
    reference128_advanced.build(case).write(str(path))
    return case, path.read_text()


def no_worker():
    raise AssertionError("resolver must not run for an unsupported file")


def test_reference_controls_are_a_supported_version(case_and_text):
    assert pulseq_version(case_and_text[1]) in s4_physio.SUPPORTED_VERSIONS


@pytest.mark.parametrize("minor", ["3", "2", None])
def test_unsupported_pulseq_version_is_a_failed_check(
    case_and_text, tmp_path, monkeypatch, minor
):
    case, text = case_and_text
    if minor is None:
        text = text.replace("[VERSION]", "[NOTVERSION]", 1)
    else:
        text = text.replace("minor 5", f"minor {minor}", 1)
    path = tmp_path / "legacy.seq"
    path.write_text(text)
    monkeypatch.setattr(s4_physio, "worker", no_worker)
    with pytest.raises(InvalidPhysioSequence, match="1.4 or 1.5"):
        resolve_sequence(path, case, tmp_path / "physio")


def test_malformed_block_rows_raise_invalid_physio_sequence():
    text = "[BLOCKS]\n1 1 0 0 0 0 0\n"
    with pytest.raises(InvalidPhysioSequence):
        explicit_waits(text, [0.0], 1e-5)
    with pytest.raises(InvalidPhysioSequence):
        explicit_waits("[DEFINITIONS]\n", [0.0], 1e-5)


def test_unreadable_resolved_timeline_is_a_failed_check(
    case_and_text, tmp_path, monkeypatch
):
    case, text = case_and_text
    path = tmp_path / "control.seq"
    path.write_text(text)
    raw = Sequence()
    raw.read(str(path))
    before = np.asarray(list(raw.block_durations.values()))
    monkeypatch.setattr(
        s4_physio,
        "worker",
        lambda: type(
            "W",
            (),
            {
                "call": lambda self, request: {
                    "block_durations_before_s": before,
                    "waits_s": [0.0] * len(before),
                }
            },
        )(),
    )

    # Reference a nonexistent RF event, as a stale delay/event ID would.
    def corrupt(text, waits, raster):
        head, _, rest = text.partition("[BLOCKS]\n")
        first, _, tail = rest.partition("\n")
        fields = first.split()
        fields[2] = "99999"
        return head + "[BLOCKS]\n" + " ".join(fields) + "\n" + tail

    monkeypatch.setattr(s4_physio, "explicit_waits", corrupt)
    with pytest.raises(InvalidPhysioSequence, match="not a readable Pulseq"):
        resolve_sequence(path, case, tmp_path / "physio")


TRIGGER_ROW = "extension TRIGGERS 1\n1 2 1 0 10\n"


@pytest.mark.parametrize(
    "row",
    [
        "1 2 1 nan 10",
        "1 2 1 0 inf",
        "1 2 1 1.5 10",
        "1 2 1 -10 10",
        "1 2 1 0",
        "1 two 1 0 10",
    ],
)
def test_preflight_rejects_trigger_rows_the_native_reader_cannot_read(
    case_and_text, tmp_path, row
):
    from mriseqbench.evaluation.preflight import preflight

    case, text = case_and_text
    assert TRIGGER_ROW in text
    path = tmp_path / "trigger.seq"
    path.write_text(text.replace(TRIGGER_ROW, f"extension TRIGGERS 1\n{row}\n"))
    checks = {c["name"]: c for c in preflight(path, case, hardware_limits=False)}
    # pypulseq itself rejects some rows; the rest must fail the native rule.
    failed = {name for name, c in checks.items() if not c["passed"]}
    assert failed & {"native_tables", "pulseq_preflight"}, checks


@pytest.mark.parametrize(
    "replace,expected",
    [
        (("[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 1 nan\n"), "4 integers"),
        (("[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 7 1 0\n"), "not declared"),
        (("[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 9 0\n"), "spec 9"),
        (("[EXTENSIONS]\n1 1 1 0\n", "[EXTENSIONS]\n1 1 1 4\n"), "next extension"),
        (("1 2 1 0 10\n", "1 2 1 0 10\n\n2 2 1 0 10\n"), "ignored by the native"),
    ],
)
def test_native_extension_table_errors(case_and_text, replace, expected):
    from mriseqbench.evaluation.preflight import native_extension_errors

    text = case_and_text[1]
    assert native_extension_errors(text) == []
    errors = native_extension_errors(text.replace(*replace, 1))
    assert any(expected in error for error in errors), errors


def test_native_reader_rejections_become_failed_checks(
    case_and_text, tmp_path, monkeypatch
):
    case, text = case_and_text
    path = tmp_path / "control.seq"
    path.write_text(text)

    def failing(message):
        def call(self, request):
            raise RuntimeError(message)

        return lambda: type("W", (), {"call": call})()

    monkeypatch.setattr(
        s4_physio,
        "worker",
        failing("KomaMRI: ArgumentError: invalid base 10 digit 'n' in \"nan\""),
    )
    with pytest.raises(InvalidPhysioSequence, match="rejected the file"):
        resolve_sequence(path, case, tmp_path / "physio")
    for infrastructure in (
        "KomaMRI: OutOfMemoryError()",
        "KomaMRI: CUDA error: out of memory",
        "KomaMRI worker exited: julia: not found",
    ):
        monkeypatch.setattr(s4_physio, "worker", failing(infrastructure))
        with pytest.raises(RuntimeError) as raised:
            resolve_sequence(path, case, tmp_path / "physio")
        assert not isinstance(raised.value, InvalidPhysioSequence)
