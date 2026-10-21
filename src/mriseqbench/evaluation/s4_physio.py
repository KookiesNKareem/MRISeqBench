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


def explicit_waits(text, waits, raster):
    """Insert only idle blocks; RF/gradient/ADC/extension libraries stay exact.

    Upstream adds a wait inside a standalone trigger block. Inserting the same
    idle time immediately before that block is physically equivalent, including
    its arming delay and post-trigger duration. No waveform reserialization or
    locally implemented R-wave scheduler is used here.
    """
    match = re.search(r"(?m)^\[BLOCKS\]\s*\n", text)
    if match is None:
        raise ValueError("Pulseq BLOCKS section missing")
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
            raise ValueError("Expected Pulseq 1.4+ block duration format")
        wait = float(waits[count])
        ticks = round(wait / raster)
        if wait < -1e-9 or abs(ticks * raster - wait) > 1e-9:
            raise ValueError("Resolved cardiac wait is not on the block raster")
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


def resolve_sequence(submission, case, workspace):
    submission, workspace = Path(submission), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    rr_s = case["task"]["sequence"]["temporal"]["rr_interval_ms"] / 1000
    if not np.isfinite(rr_s) or rr_s <= 0:
        raise ValueError("S4 requires a finite positive nominal RR interval")
    raw = Sequence(Opts(B0=case["hardware"]["field_strength_T"]))
    raw.read(str(submission))
    # Standalone trigger blocks are an upstream scanner restriction, not a
    # requirement that every intervening block be idle.
    for number in raw.block_events:
        block = raw.get_block(number)
        triggers = list((getattr(block, "trigger", None) or {}).values())
        if any(e.type == "trigger" for e in triggers) and any(
            getattr(block, k, None) is not None for k in ("rf", "gx", "gy", "gz", "adc")
        ):
            raise InvalidPhysioSequence(
                "Physiological triggers must occupy standalone blocks"
            )
    channels = {
        e.channel
        for number in raw.block_events
        for e in (getattr(raw.get_block(number), "trigger", None) or {}).values()
        if e.type == "trigger"
    }
    if len(channels) > 1:
        raise InvalidPhysioSequence(
            "A single CardiacSignal cannot resolve multiple trigger channels"
        )
    resolution = worker().call(
        {
            "operation": "resolve_physio",
            "sequence": str(submission.resolve()),
            "heart_rate_hz": 1 / rr_s,
            "first_peak_s": 0.0,
        }
    )
    before = np.asarray(list(raw.block_durations.values()))
    np.testing.assert_allclose(
        resolution["block_durations_before_s"],
        before,
        atol=1e-9,
        rtol=0,
        err_msg="Koma/Pulseq original block durations differ",
    )
    path = workspace / "resolved.seq"
    text = submission.read_text()
    path.write_text(
        explicit_waits(text, resolution["waits_s"], raw.block_duration_raster)
    )
    resolved = Sequence(raw.system)
    resolved.read(str(path))
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
