"""Resolve S4 trigger waits with Koma, preserving authored waveform sections."""

import hashlib
import re
from pathlib import Path

import numpy as np
from pypulseq import Opts, Sequence

from ..io import write_json
from .s1_koma import worker


class InvalidPhysioSequence(ValueError):
    """A submitted trigger layout rejected by the upstream scheduler."""


SUPPORTED_VERSIONS = {(1, 4), (1, 5)}


def pulseq_version(text):
    """(major, minor) from the [VERSION] section, or None if absent."""
    match = re.search(r"(?m)^\[VERSION\][ \t]*\n((?:[^\[].*\n?)*)", text)
    if match is None:
        return None
    fields = dict(
        line.split(None, 1)
        for line in match[1].splitlines()
        if len(line.split()) >= 2 and not line.lstrip().startswith("#")
    )
    try:
        return int(fields["major"]), int(fields["minor"])
    except (KeyError, ValueError):
        return None


def require_supported_version(text):
    """S4 inserts duration-tick idle blocks, which exist only in Pulseq 1.4+.

    In 1.3.x the second BLOCKS column references [DELAYS] instead, so wait
    insertion would be invalid. Same accepted set as the RF excitation replay.
    """
    version = pulseq_version(text)
    if version not in SUPPORTED_VERSIONS:
        found = "missing" if version is None else "{}.{}".format(*version)
        raise InvalidPhysioSequence(
            f"S4 requires Pulseq file version 1.4 or 1.5 (found {found})"
        )


def explicit_waits(text, waits, raster):
    """Insert only idle blocks; RF/gradient/ADC/extension libraries stay exact.

    Upstream adds a wait inside a standalone trigger block. Inserting the same
    idle time immediately before that block is physically equivalent, including
    its arming delay and post-trigger duration. No waveform reserialization or
    locally implemented R-wave scheduler is used here.
    """
    match = re.search(r"(?m)^\[BLOCKS\]\s*\n", text)
    if match is None:
        raise InvalidPhysioSequence("Pulseq BLOCKS section missing")
    end = re.search(r"(?m)^\[", text[match.end() :])
    stop = match.end() + end.start() if end else len(text)
    section = text[match.end() : stop]
    result, index, count = [], 1, 0
    for line in section.splitlines(keepends=True):
        fields = line.split("#", 1)[0].split()
        if not fields or not fields[0].isdigit():
            result.append(line)
            continue
        if len(fields) != 8 or count >= len(waits):
            raise InvalidPhysioSequence("Expected Pulseq 1.4+ block duration format")
        wait = float(waits[count])
        ticks = round(wait / raster)
        if wait < -1e-9 or abs(ticks * raster - wait) > 1e-9:
            raise InvalidPhysioSequence(
                "Resolved cardiac wait is not on the block raster"
            )
        if ticks:
            result.append(f"{index} {ticks} 0 0 0 0 0 0\n")
            index += 1
        # Preserve every original event reference and duration token.
        result.append(str(index) + " " + " ".join(fields[1:]) + "\n")
        index += 1
        count += 1
    if count != len(waits):
        raise ValueError("Koma/Pulseq block count mismatch")
    output = text[: match.end()] + "".join(result) + text[stop:]
    # A signature belongs to the original bytes, which are preserved separately.
    output = re.sub(r"(?ms)^\[SIGNATURE\]\s*\n.*?(?=^\[|\Z)", "", output)
    return output


# Julia exception types that indicate the host, not the submission.
_INFRASTRUCTURE_ERRORS = (
    "OutOfMemoryError",
    "InterruptException",
    "SystemError",
    "IOError",
    "CUDA",
)


def native_candidate_rejection(exc):
    """True if a resolve_physio failure is a property of the submitted bytes.

    The worker reports exceptions caught inside a request as "KomaMRI: <error>"
    (response ok=false); a dead or unresponsive worker uses other messages and
    TimeoutError. resolve_physio only parses the file and schedules triggers
    on the CPU, so a caught exception there is a candidate rejection unless it
    names a resource, I/O, interrupt or CUDA failure.
    """
    message = str(exc)
    return message.startswith("KomaMRI: ") and not any(
        name in message for name in _INFRASTRUCTURE_ERRORS
    )


def resolve_sequence(submission, case, workspace):
    submission, workspace = Path(submission), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    rr_s = case["task"]["sequence"]["temporal"]["rr_interval_ms"] / 1000
    if not np.isfinite(rr_s) or rr_s <= 0:
        raise ValueError("S4 requires a finite positive nominal RR interval")
    text = submission.read_text()
    require_supported_version(text)
    raw = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    raw.read(str(submission))
    channels = set()
    for number in raw.block_events:
        block = raw.get_block(number)
        triggers = [
            e
            for e in (getattr(block, "trigger", None) or {}).values()
            if e.type == "trigger"
        ]
        # Standalone trigger blocks are an upstream scanner restriction, not a
        # requirement that every intervening block be idle.
        if triggers and any(
            getattr(block, k, None) is not None for k in ("rf", "gx", "gy", "gz", "adc")
        ):
            raise InvalidPhysioSequence(
                "Physiological triggers must occupy standalone blocks"
            )
        channels.update(e.channel for e in triggers)
    if len(channels) > 1:
        raise InvalidPhysioSequence(
            "A single CardiacSignal cannot resolve multiple trigger channels"
        )
    if channels != {"physio1"}:
        # Structure inspection rejects missing/wrong ECG events without loading
        # a simulator for a timeline that has no valid cardiac triggers.
        return submission, {
            "resolver": "not_required",
            "reason": "No ECG trigger events",
        }
    try:
        resolution = worker().call(
            {
                "operation": "resolve_physio",
                "sequence": str(submission.resolve()),
                "heart_rate_hz": 1 / rr_s,
                "first_peak_s": 0.0,
            }
        )
    except RuntimeError as exc:
        if not native_candidate_rejection(exc):
            raise
        raise InvalidPhysioSequence(
            f"Native Pulseq reader/trigger scheduler rejected the file: {exc}"
        ) from exc
    before = np.asarray(list(raw.block_durations.values()))
    np.testing.assert_allclose(
        resolution["block_durations_before_s"],
        before,
        atol=1e-9,
        rtol=0,
        err_msg="Koma/Pulseq original block durations differ",
    )
    path = workspace / "resolved.seq"
    path.write_text(
        explicit_waits(text, resolution["waits_s"], raw.block_duration_raster)
    )
    resolved = Sequence(raw.system)
    try:
        resolved.read(str(path))
    except (KeyError, IndexError, ValueError) as exc:
        # Unreadable resolved timeline: a failed verdict, not an interruption.
        raise InvalidPhysioSequence(
            f"Resolved cardiac timeline is not a readable Pulseq file: {exc!r}"
        ) from exc
    np.testing.assert_allclose(
        resolved.adc_times()[0],
        resolution["adc_times_s"],
        atol=1e-9,
        rtol=0,
        err_msg="Koma/Pulseq resolved ADC timeline differs",
    )
    receipt = {
        "resolver": resolution["resolver"],
        "versions": resolution["versions"],
        "heart_rate_hz": 1 / rr_s,
        "rr_interval_s": rr_s,
        "first_peak_s": 0.0,
        "original_path": str(submission.resolve()),
        "original_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "resolved_path": str(path.resolve()),
        "resolved_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_duration_s": float(before.sum()),
        "resolved_duration_s": float(sum(resolved.block_durations.values())),
        "block_waits_s": resolution["waits_s"],
        "inserted_wait_s": float(sum(resolution["waits_s"])),
        "waveform_event_libraries": "Original RF/gradient/ADC/shape/extension section bytes retained",
        "native_and_python_adc_times_match": True,
        "replay": "Explicit resolved delays; no second trigger resolution",
    }
    write_json(workspace / "resolution.json", receipt)
    return path, receipt
