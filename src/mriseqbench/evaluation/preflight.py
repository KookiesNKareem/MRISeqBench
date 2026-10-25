"""Pulseq hardware and timing checks."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Effective RF/ADC frequency offsets beyond this are not physical transmit or
# receive offsets (references use <= 434 Hz).
MAX_FREQUENCY_OFFSET_HZ = 1e6


def gradient_jumps(seq, system):
    """Instantaneous gradient steps, which Sequence.waveforms() silently drops.

    A block's waveform is joined to the next at the boundary; arbitrary gradients
    carry first/last values, trapezoids start and end at zero.
    """
    tolerance = 1e-3 * system.max_grad
    jumps = []
    carried = {axis: 0.0 for axis in "xyz"}
    for number in seq.block_events:
        block = seq.get_block(number)
        duration = seq.block_durations[number]
        for axis in "xyz":
            grad = getattr(block, "g" + axis, None)
            start, end = 0.0, 0.0
            if grad is not None and grad.type == "grad":
                first, last = float(grad.first), float(grad.last)
                # File metadata cannot hide a step: first/last are vertices at
                # the waveform ends, joined to the samples within slew limits.
                t = np.concatenate(([0.0], grad.tt, [grad.shape_dur]))
                v = np.concatenate(([first], grad.waveform, [last]))
                dt, dv = np.diff(t), np.abs(np.diff(v))
                coincident = dt < system.grad_raster_time / 4
                if np.any(dv[coincident] > tolerance):
                    jumps.append({"block": number, "axis": axis, "at": "endpoints"})
                # A finite-interval change is continuous; it is a hardware slew
                # question, reported separately so lint does not apply it.
                if np.any(
                    dv[~coincident]
                    > system.max_slew * dt[~coincident] * (1 + 1e-6) + tolerance
                ):
                    jumps.append({"block": number, "axis": axis, "at": "endpoint slew"})
                start = first if grad.delay < system.grad_raster_time / 2 else 0.0
                if grad.delay >= system.grad_raster_time / 2 and abs(first) > tolerance:
                    jumps.append({"block": number, "axis": axis, "at": "delay end"})
                ends_at_boundary = (
                    abs(grad.delay + grad.shape_dur - duration)
                    < system.grad_raster_time / 2
                )
                end = last if ends_at_boundary else 0.0
                if not ends_at_boundary and abs(last) > tolerance:
                    jumps.append({"block": number, "axis": axis, "at": "waveform end"})
            if abs(start - carried[axis]) > tolerance:
                jumps.append({"block": number, "axis": axis, "at": "block start"})
            carried[axis] = end
    for axis, value in carried.items():
        if abs(value) > tolerance:
            jumps.append({"block": "end", "axis": axis, "at": "sequence end"})
    return jumps


# Malformed or unsupported submitted sequences fail as checks. Infrastructure
# errors (OSError, MemoryError, ImportError) propagate and interrupt grading.
MALFORMED_SEQUENCE = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    AssertionError,
    ArithmeticError,
    RuntimeError,
    Warning,
)


_INTEGER_TEXT = re.compile(r"[+-]?[0-9]{1,19}$")
_UNSIGNED_TEXT = re.compile(r"[0-9]{1,19}$")
_FLOAT_TEXT = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]{1,4})?$")
_INT64 = (-(2**63), 2**63 - 1)


def _int_ok(token):
    """Julia parse(Int, token): decimal integer text within Int64."""
    return bool(_INTEGER_TEXT.match(token)) and _INT64[0] <= int(token) <= _INT64[1]


def _float_ok(token, *, integral=False):
    """Julia parse(Float64, token), finite; ``integral`` adds a lossless Int()."""
    if len(token) > 64 or not _FLOAT_TEXT.match(token):
        return False
    value = float(token)
    if not np.isfinite(value):
        return False
    return not integral or (value == int(value) and abs(value) < 2**63)


# Row layout (after the id) of the extension specifications the pinned native
# reader decodes (KomaMRIBase 0.15.1). "i": Julia parse(Int), "f": finite
# Float64, "s": label string. Unknown extensions are skipped natively.
_NATIVE_EXTENSIONS = {
    "TRIGGERS": "iiii",
    "LABELSET": "is",
    "LABELINC": "is",
    "ROTATIONS": "ffff",
}


def _token_ok(kind, token):
    if kind == "i":
        return _int_ok(token)
    if kind == "c":
        return len(token) == 1
    if kind == "s":
        return True
    return _float_ok(token, integral=kind == "n")


def _row_ok(layout, fields):
    return len(fields) == len(layout) and all(map(_token_ok, layout, fields))


def native_extension_errors(text):
    """Extension rows the native Pulseq reader would reject or silently drop.

    The native reader parses [EXTENSIONS] and trigger timing as integers and
    stops a section at the first row with a different field count, so such rows
    must fail at preflight rather than inside a simulator worker.
    """
    errors, section, closed = [], None, False
    types, specs, instances, current = {}, {}, [], None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            # The native reader ends a row block at the first blank or comment.
            closed = True
            continue
        fields = line.split()
        if line.startswith("["):
            section, closed = line, False
            continue
        if fields[0] != "extension" and (
            section == "[EXTENSIONS]" or section in _NATIVE_EXTENSIONS
        ):
            if "#" in raw:
                errors.append(f"line {number}: inline comment in an extension row")
                continue
            if closed:
                errors.append(
                    f"line {number}: extension row after a blank/comment line "
                    "is ignored by the native reader"
                )
                continue
        if fields[0] == "extension":
            if fields[1:2] == ["ROTATIONS"]:
                errors.append(
                    f"line {number}: ROTATIONS extension is not supported "
                    "(native replay would rotate gradients that inspection does not)"
                )
            if len(fields) != 3 or not _int_ok(fields[2]):
                errors.append(f"line {number}: malformed extension header")
                section = None
                continue
            section, closed, current = fields[1], False, int(fields[2])
            types[current] = fields[1]
            specs.setdefault(current, set())
            continue
        if section == "[EXTENSIONS]":
            if len(fields) != 4 or not all(map(_int_ok, fields)):
                errors.append(f"line {number}: [EXTENSIONS] row needs 4 integers")
            else:
                instances.append((number, *map(int, fields)))
        elif section in _NATIVE_EXTENSIONS:
            layout = _NATIVE_EXTENSIONS[section]
            if not _row_ok("i" + layout, fields):
                errors.append(
                    f"line {number}: {section} row needs an integer id and "
                    f"fields {layout!r} (i=integer, f=finite number, s=label)"
                )
                continue
            if section == "TRIGGERS" and (int(fields[3]) < 0 or int(fields[4]) < 0):
                errors.append(
                    f"line {number}: trigger delay/duration must be "
                    "nonnegative integer microseconds"
                )
                continue
            specs[current].add(int(fields[0]))
    ids = [row[1] for row in instances]
    if sorted(ids) != list(range(1, len(ids) + 1)):
        # The native decoder allocates max(id) slots and decodes ids 1..max.
        errors.append("[EXTENSIONS] ids must be unique and dense 1..N")
    links = {row[1]: row[4] for row in instances}
    for start in links:
        seen, node = set(), start
        while node:
            if node in seen:
                errors.append(f"extension {start}: next_id chain has a cycle")
                break
            seen.add(node)
            node = links.get(node, 0)
    ids = set(ids)
    for number, _id, kind, ref, following in instances:
        if kind not in types:
            errors.append(f"line {number}: extension type {kind} is not declared")
        elif types[kind] in _NATIVE_EXTENSIONS and ref not in specs[kind]:
            errors.append(f"line {number}: {types[kind]} spec {ref} is missing")
        if following and following not in ids:
            errors.append(f"line {number}: next extension {following} is missing")
    return errors


# Native token layout per row of each event table (Pulseq 1.4 / 1.5): "i" is
# parse(Int); "f" parse(Float64); "n" a Float64 later converted with Int();
# "c" a single character (KomaMRIFiles 0.12.0 ReadPulseq/PulseqEvents).
_NATIVE_TABLES = {
    (1, 4): {
        "[RF]": "ifnnnfff",
        "[GRADIENTS]": "ifnnf",
        "[TRAP]": "ifffff",
        "[ADC]": "inffff",
        "[DELAYS]": "if",
    },
    (1, 5): {
        "[RF]": "ifnnnffffffc",
        "[GRADIENTS]": "ifffnnf",
        "[TRAP]": "ifffff",
        "[ADC]": "inffffffn",
        "[DELAYS]": "if",
    },
}


def _decompressed_length(data):
    """Sample count of a Pulseq run-length shape (pypulseq decompress_shape)."""
    data = np.asarray(data, float)
    markers = np.flatnonzero(np.diff(data) == 0)
    packed = unpacked = 0
    for marker in markers:
        run = marker - packed
        if run < 0:
            continue
        packed += run
        unpacked += run
        if packed + 2 >= len(data):
            return None
        repeat = data[packed + 2]
        if repeat != int(repeat) or repeat < 0:
            return None
        unpacked += int(repeat) + 2
        packed += 3
    return unpacked + max(0, len(data) - packed)


def _rle_runs(data):
    """(literal derivative samples, repeated value or None, repeat count) runs."""
    packed = 0
    for marker in np.flatnonzero(np.diff(data) == 0):
        if marker < packed:
            continue
        yield data[packed:marker], data[marker], int(data[marker + 2]) + 2
        packed = marker + 3
    yield data[packed:], None, 0


def _shape_endpoints(num_samples, data):
    """First and last decoded sample (pypulseq/KomaMRI: RLE of the derivative)."""
    data = np.asarray(data, float)
    if num_samples == len(data):
        return float(data[0]), float(data[-1])
    total = 0.0
    for literal, value, repeat in _rle_runs(data):
        total += float(np.sum(literal))
        if value is not None:
            total += float(value) * repeat
    return float(data[0]), total


def _shape_increasing(num_samples, data):
    """Whether a decoded time shape starts at >= 0 and never decreases."""
    data = np.asarray(data, float)
    if num_samples == len(data):
        return bool(data[0] >= -1e-9 and np.all(np.diff(data) >= -1e-9))
    steps = []
    for literal, value, repeat in _rle_runs(data):
        steps.extend(literal)
        if value is not None and repeat > 0:
            steps.append(value)
    # Time samples are in raster units; 1e-9 absorbs writer round-off only.
    return bool(
        steps and steps[0] >= -1e-9 and all(step >= -1e-9 for step in steps[1:])
    )


_KNOWN_SECTIONS = {
    "[DEFINITIONS]",
    "[VERSION]",
    "[BLOCKS]",
    "[RF]",
    "[GRADIENTS]",
    "[TRAP]",
    "[ADC]",
    "[DELAYS]",
    "[SHAPES]",
    "[EXTENSIONS]",
    "[SIGNATURE]",
}


def native_table_errors(text, parsed=None):
    """Core sections the native reader would reject, overflow or silently drop.

    Mirrors KomaMRIFiles 0.12.0 parse_pulseq_file: VERSION is three exact
    keyed lines; BLOCKS rows (ended by an empty line) hold 8 unsigned Int64
    tokens; event-table rows have exact layouts and end at the first other
    line, after which leftover rows would be ignored; shapes declare a sample
    count consistent with their data; DEFINITIONS Nx/Ny/Nz convert to Int.
    """
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if text.endswith("\n"):
        lines.pop()  # a final newline terminates the last line; it adds none
    errors, version, index = [], None, 0
    tables = {name: [] for name in ("[RF]", "[GRADIENTS]", "[TRAP]", "[ADC]")}
    shapes, blocks, extension_ids, misnumbered = {}, [], [], False
    shape_ends, shape_increasing, rf_raster = {}, {}, 1e-6
    shape_data, grad_raster = {}, 1e-5

    def error(number, message):
        errors.append(f"line {number}: {message}")

    # Header rules scan every raw line, whichever section would consume it, so
    # no section (e.g. [SIGNATURE]) can hide a second table from them.
    seen_sections = set()
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith(("[", "extension")):
            continue
        # PyPulseq strips headers and keeps the last copy of a section; the
        # native reader matches exact lines and merges or overwrites.
        if stripped != line:
            error(number, "section header has surrounding whitespace")
        if stripped.startswith("["):
            key = stripped
            if stripped not in _KNOWN_SECTIONS:
                error(number, f"unknown section {stripped}")
        else:
            parts = stripped.split()
            key = ("extension", parts[2] if len(parts) > 2 else "")
        if key in seen_sections:
            error(number, f"duplicate section {stripped}")
        seen_sections.add(key)
    while index < len(lines):
        number, header = index + 1, lines[index]
        index += 1
        if not header or header.startswith("#"):
            continue
        if header == "[VERSION]":
            values = []
            for key in ("major", "minor", "revision"):
                fields = lines[index].split() if index < len(lines) else []
                index += 1
                if (
                    len(fields) != 2
                    or fields[0] != key
                    or not _int_ok(fields[1])
                    or not 0 <= int(fields[1]) <= 2**32 - 1
                ):
                    error(index, f"[VERSION] needs '{key} <integer>' here")
                    break
                values.append(int(fields[1]))
            else:
                version = tuple(values[:2])
                if version not in _NATIVE_TABLES:
                    error(number, "Pulseq file version must be 1.4 or 1.5")
            continue
        if header == "[DEFINITIONS]":
            while index < len(lines) and lines[index].split():
                fields = lines[index].split()
                index += 1
                key = fields[0]
                if key in (
                    "BlockDurationRaster",
                    "GradientRasterTime",
                    "RadiofrequencyRasterTime",
                    "AdcRasterTime",
                ) and (
                    len(fields) < 2 or not _float_ok(fields[1]) or float(fields[1]) <= 0
                ):
                    error(index, f"{key} needs a finite positive number")
                if len(fields) > 1 and _float_ok(fields[1]):
                    if key == "GradientRasterTime":
                        grad_raster = float(fields[1])
                    elif key == "RadiofrequencyRasterTime":
                        rf_raster = float(fields[1])
                if key in ("Nx", "Ny", "Nz") and (
                    len(fields) != 2 or not _float_ok(fields[1], integral=True)
                ):
                    error(index, f"{key} needs one Int64-representable integer")
            index += 1
            continue
        if header == "[BLOCKS]":
            if version is None:
                error(number, "[VERSION] must precede [BLOCKS]")
            while index < len(lines) and lines[index] != "":
                fields = lines[index].split()
                index += 1
                if len(fields) != 8 or not all(
                    _UNSIGNED_TEXT.match(token) and int(token) <= _INT64[1]
                    for token in fields
                ):
                    error(index, "BLOCKS row needs 8 unsigned Int64 integers")
                else:
                    blocks.append((index, [int(token) for token in fields]))
                    if blocks[-1][1][0] != len(blocks) and not misnumbered:
                        misnumbered = True
                        # Native skips id 0 and keeps duplicates; PyPulseq keys
                        # by id. Only canonical ids give both the same table.
                        error(
                            index,
                            f"BLOCKS ids must be 1..N in file order "
                            f"(expected {len(blocks)})",
                        )
            index += 1
            continue
        layout = (_NATIVE_TABLES.get(version) or {}).get(header)
        if header in ("[RF]", "[GRADIENTS]", "[TRAP]", "[ADC]", "[DELAYS]"):
            if layout is None:
                error(number, f"{header} not readable before a 1.4/1.5 [VERSION]")
                continue
            while index < len(lines):
                fields = lines[index].split()
                index += 1
                if len(fields) != len(layout):
                    break  # the native reader ends the table on this line
                if not _row_ok(layout, fields):
                    error(index, f"{header} row does not match native layout")
                elif header in tables:
                    tables[header].append(
                        (
                            index,
                            [
                                token if kind == "c" else float(token)
                                for kind, token in zip(layout, fields)
                            ],
                        )
                    )
            continue
        if header == "[SHAPES]":
            while index < len(lines):
                line = lines[index]
                if not line:
                    index += 1
                    continue
                fields = line.split()
                if len(fields) != 2 or fields[0] != "shape_id":
                    break
                index += 1
                if not _int_ok(fields[1]):
                    error(index, "shape_id needs an Int64 integer")
                count = lines[index].split() if index < len(lines) else []
                index += 1
                if (
                    len(count) != 2
                    or count[0] != "num_samples"
                    or not _int_ok(count[1])
                ):
                    error(index, "num_samples needs an Int64 integer")
                    break
                data, terminated = [], False
                while index < len(lines):
                    token = lines[index].strip()
                    index += 1
                    if not _float_ok(token):
                        terminated = True
                        break
                    data.append(float(token))
                if not terminated:
                    # PyPulseq reads EOF as another sample and never returns.
                    error(index, "shape data must end with a blank line, not EOF")
                    break
                samples = int(count[1])
                if samples not in (len(data), _decompressed_length(data)):
                    error(index, "num_samples does not match the shape data")
                elif _int_ok(fields[1]):
                    shape_id = int(fields[1])
                    if shape_id in shapes:
                        error(index, f"shape_id {fields[1]} is duplicated")
                    shapes[shape_id] = samples
                    shape_increasing[shape_id] = bool(data) and (
                        _shape_increasing(samples, data)
                    )
                    if shape_id < 1 or samples < 1:
                        error(index, "shape_id and num_samples must be positive")
                    shape_data[shape_id] = data
                    shape_ends[shape_id] = (
                        _shape_endpoints(samples, data) if data else (0.0, 0.0)
                    )
            continue
        if header in ("[EXTENSIONS]", "[SIGNATURE]") or header.startswith("extension"):
            # Contents are validated by native_extension_errors; here only the
            # native row consumption is mirrored so stray rows are detected.
            fields = header[len("extension") :].split()
            known = header.startswith("extension") and fields
            width = (
                4
                if header == "[EXTENSIONS]"
                else len(_NATIVE_EXTENSIONS[fields[0]]) + 1
                if known and fields[0] in _NATIVE_EXTENSIONS
                else None
            )
            while index < len(lines):
                line = lines[index]
                index += 1
                if not line or (width is not None and len(line.split()) != width):
                    break
                if header == "[EXTENSIONS]" and _int_ok(line.split()[0]):
                    extension_ids.append(int(line.split()[0]))
            continue
        if header.startswith("["):
            continue  # unknown section codes are logged and skipped natively
        error(number, "row outside its table is ignored by the native reader")
    if version is None:
        errors.append("[VERSION] section missing or malformed")
    if not errors:
        errors.extend(
            _native_library_errors(version, tables, shapes, blocks, extension_ids)
            + _native_sign_errors(version, tables, shape_increasing)
            + _native_semantic_errors(version, tables, shapes, shape_ends, rf_raster)
        )
    if parsed is not None and not errors:
        parsed.update(
            version=version,
            tables=tables,
            shapes=shapes,
            shape_data=shape_data,
            grad_raster=grad_raster,
        )
    return errors


# The native decoder materializes every library entry (used or not). Bound
# the total decoded RF/gradient samples so a file cannot request an
# allocation far beyond any realistic sequence (largest of 15k corpus files:
# 2.5M samples, i.e. ~7x headroom).
NATIVE_DECODED_SAMPLES_MAX = 2**24


# Allocation bounds for per-sample inspection, ~5-7x the largest of ~15k
# reference/submission files on record (ADC event 8800 samples: spiral_gre at
# 0.55 T; 827392 acquired samples and 39552 blocks: stack_of_stars; 2.4M
# RF/gradient shape samples summed over blocks: reduced-FOV composites).
ADC_SAMPLES_PER_EVENT_MAX = 2**16
ADC_SAMPLES_TOTAL_MAX = 2**22
BLOCKS_MAX = 2**18
WAVEFORM_SAMPLES_TOTAL_MAX = 2**24


def _gradient_columns(version):
    """(amplitude shape, time shape, delay) columns of a [GRADIENTS] row."""
    return (4, 5, 6) if version == (1, 5) else (2, 3, 4)


def _native_library_errors(version, tables, shapes, blocks, extension_ids):
    """Dense ids, resolvable shapes and bounded decoding (KomaMRIFiles 0.12.0).

    decode_pulseq_libraries allocates max(id) entries for gradients
    (GRADIENTS and TRAP share ids), RF, ADC and extension instances, then
    decodes ids 1..max, so ids must be unique and dense. Every decoded RF and
    arbitrary gradient dereferences its shapes; blocks index the decoded
    vectors directly.
    """
    errors = []
    gradients = tables["[GRADIENTS]"] + tables["[TRAP]"]
    counts = {}
    for name, rows in (
        ("gradient", gradients),
        ("RF", tables["[RF]"]),
        ("ADC", tables["[ADC]"]),
        ("extension", [(0, [i]) for i in extension_ids]),
    ):
        ids = sorted(int(row[1][0]) for row in rows)
        counts[name] = len(ids)
        if ids != list(range(1, len(ids) + 1)):
            errors.append(f"{name} ids must be unique and dense 1..N")
    decoded = 0

    def shape(number, kind, shape_id, length=None):
        nonlocal decoded
        if shape_id not in shapes:
            errors.append(f"line {number}: {kind} shape {shape_id} is missing")
            return None
        if length is not None and shapes[shape_id] != length:
            errors.append(f"line {number}: {kind} shape {shape_id} length differs")
        decoded += shapes[shape_id]
        return shapes[shape_id]

    for number, row in tables["[RF]"]:
        amplitude, mag, phase, time = row[1], int(row[2]), int(row[3]), int(row[4])
        if amplitude == 0 or mag == 0:
            continue
        length = shape(number, "RF magnitude", mag)
        shape(number, "RF phase", phase, length)
        if time not in (0, -1):
            shape(number, "RF time", time)
    amp_column, time_column, _delay = _gradient_columns(version)
    for number, row in tables["[GRADIENTS]"]:
        amp, time = int(row[amp_column]), int(row[time_column])
        shape(number, "gradient amplitude", amp)
        if time not in (0, -1):
            shape(number, "gradient time", time)
    if decoded > NATIVE_DECODED_SAMPLES_MAX or any(
        count > NATIVE_DECODED_SAMPLES_MAX for count in shapes.values()
    ):
        errors.append(
            f"decoded RF/gradient samples {decoded} (largest shape "
            f"{max(shapes.values(), default=0)}) exceed {NATIVE_DECODED_SAMPLES_MAX}"
        )
    # Inspection expands every ADC event per block reference into per-sample
    # arrays; bound each event and the total acquired over all blocks.
    adc_samples = {int(row[0]): max(0, int(row[1])) for _number, row in tables["[ADC]"]}
    largest = max(adc_samples.values(), default=0)
    total = sum(adc_samples.get(row[6], 0) for _number, row in blocks)
    rf_length = {
        int(row[0]): shapes.get(int(row[2]), 0) if row[1] != 0 and row[2] != 0 else 0
        for _number, row in tables["[RF]"]
    }
    grad_length = {
        int(row[0]): shapes.get(int(row[amp_column]), 0)
        for _number, row in tables["[GRADIENTS]"]
    }
    waveform = sum(
        rf_length.get(row[2], 0) + sum(grad_length.get(i, 0) for i in row[3:6])
        for _number, row in blocks
    )
    if waveform > WAVEFORM_SAMPLES_TOTAL_MAX:
        errors.append(
            f"{waveform} RF/gradient shape samples over all blocks exceed "
            f"{WAVEFORM_SAMPLES_TOTAL_MAX}"
        )
    if largest > ADC_SAMPLES_PER_EVENT_MAX:
        errors.append(
            f"ADC event with {largest} samples exceeds {ADC_SAMPLES_PER_EVENT_MAX}"
        )
    if total > ADC_SAMPLES_TOTAL_MAX:
        errors.append(
            f"{total} acquired ADC samples (all blocks) exceed {ADC_SAMPLES_TOTAL_MAX}"
        )
    if len(blocks) > BLOCKS_MAX:
        errors.append(f"{len(blocks)} blocks exceed {BLOCKS_MAX}")
    for number, row in blocks:
        names = ("RF", "gradient", "gradient", "gradient", "ADC", "extension")
        for name, value in zip(names, row[2:]):
            if value > counts[name]:
                errors.append(f"line {number}: block references missing {name} {value}")
    return errors[:50]


def _native_sign_errors(version, tables, shape_increasing):
    """Signed fields that feed allocations, aggregates or time axes.

    Negative ADC sample counts, dwells, delays or trapezoid times would let
    signed sums cancel in the caps above while inspection expands absolute
    sizes; time shapes must start at >= 0 and never decrease. Zero ADC
    samples remain allowed here (inspection fails them as readout_samples).
    """
    errors = []
    for number, row in tables["[ADC]"]:
        if row[1] < 0 or row[2] <= 0 or row[3] < 0:
            errors.append(
                f"line {number}: ADC samples must be >= 0, dwell > 0, delay >= 0"
            )
    rf_columns = (4, 6 if version == (1, 5) else 5)
    for kind, rows, (time_column, delay_column) in (
        ("RF", tables["[RF]"], rf_columns),
        ("gradient", tables["[GRADIENTS]"], _gradient_columns(version)[1:]),
    ):
        for number, row in rows:
            time = int(row[time_column])
            if row[delay_column] < 0:
                errors.append(f"line {number}: {kind} delay must be >= 0")
            if time > 0 and not shape_increasing.get(time, False):
                errors.append(
                    f"line {number}: {kind} time shape {time} must not decrease"
                )
            if time < 0:
                # -1 (half raster) breaks PyPulseq's own deduplicating
                # read/write (KeyError) and is unused by every submission.
                errors.append(
                    f"line {number}: {kind} time shape id {time} is not supported "
                    "(use the default raster or an explicit time shape)"
                )
    for number, row in tables["[TRAP]"]:
        if min(row[2:6]) < 0:
            errors.append(f"line {number}: trapezoid rise/flat/fall/delay must be >= 0")
        elif row[1] != 0 and (row[2] <= 0 or row[4] <= 0):
            # PyPulseq drops ramp-less nonzero trapezoids from its waveforms,
            # hiding them from the amplitude and slew checks.
            errors.append(
                f"line {number}: nonzero trapezoid needs rise > 0 and fall > 0"
            )
    return errors[:50]


RF_USE_CODES = set("eripsou")  # KomaMRIBase get_RF_use_from_char


def _native_semantic_errors(version, tables, shapes, shape_ends, rf_raster):
    """Fields the native reader consumes but Python inspection derives itself.

    Native replay keeps the file's RF centre and ignores ppm offsets and ADC
    phase modulation, whereas inspection derives |B1| centroids and applies
    ppm offsets; such fields must be bounded or neutral so both agree.
    """
    if version != (1, 5):
        return []  # 1.4 rows carry none of these fields
    errors = []
    for number, row in tables["[RF]"]:
        amplitude, mag, time = row[1], int(row[2]), int(row[4])
        center, freq_ppm, phase_ppm, use = row[5] * 1e-6, row[7], row[8], row[11]
        if freq_ppm != 0 or phase_ppm != 0:
            errors.append(f"line {number}: RF freq_ppm/phase_ppm must be 0")
        if use not in RF_USE_CODES:
            errors.append(f"line {number}: RF use '{use}' is not one of eripsou")
        if amplitude == 0 or mag == 0 or mag not in shapes:
            continue
        if time == 0:
            start, end = 0.0, shapes[mag] * rf_raster
        elif time == -1:
            start, end = 0.0, shapes[mag] * rf_raster / 2
        elif time in shape_ends:
            start, end = (value * rf_raster for value in shape_ends[time])
        else:
            continue
        slack = rf_raster / 2 + 1e-12
        if not start - slack <= center <= end + slack:
            errors.append(
                f"line {number}: RF center {center:.9g} s lies outside its "
                f"waveform [{start:.9g}, {end:.9g}] s"
            )
    for number, row in tables["[ADC]"]:
        if row[4] != 0 or row[5] != 0:
            errors.append(f"line {number}: ADC freq_ppm/phase_ppm must be 0")
        if int(row[8]) != 0:
            errors.append(f"line {number}: ADC phase modulation is not supported")
    return errors[:50]


def _decoded_shape(num_samples, data):
    from pypulseq.decompress_shape import decompress_shape

    return np.asarray(
        decompress_shape(
            SimpleNamespace(num_samples=num_samples, data=np.asarray(data, float))
        ),
        float,
    )


def gradient_event_errors(parsed, max_grad, max_slew):
    """Amplitude/slew of every gradient event from its own file fields.

    Independent of PyPulseq's waveforms(), which drops or merges some events:
    trapezoid amplitude and ramp slew; arbitrary-gradient samples x amplitude,
    1.5 first/last, and per-sample slew on the default raster or time shape.
    Units: Hz/m and Hz/m/s (as Opts.max_grad / max_slew).
    """
    errors, grad_limit, slew_limit = [], max_grad * (1 + 1e-6), max_slew * (1 + 1e-6)
    tolerance = 1e-3 * max_grad
    for number, row in parsed["tables"]["[TRAP]"]:
        amplitude, rise, fall = abs(row[1]), row[2] * 1e-6, row[4] * 1e-6
        if amplitude > grad_limit:
            errors.append(f"line {number}: trapezoid amplitude exceeds the limit")
        elif amplitude and max(amplitude / rise, amplitude / fall) > slew_limit:
            errors.append(f"line {number}: trapezoid ramp exceeds the slew limit")
    v15 = parsed["version"] == (1, 5)
    raster = parsed["grad_raster"]
    amp_column, time_column, _delay = _gradient_columns(parsed["version"])
    for number, row in parsed["tables"]["[GRADIENTS]"]:
        amplitude = row[1]
        amp_id, time_id = int(row[amp_column]), int(row[time_column])
        values = amplitude * _decoded_shape(
            parsed["shapes"][amp_id], parsed["shape_data"][amp_id]
        )
        first, last = (row[2], row[3]) if v15 else (None, None)
        peak = max(
            [np.max(np.abs(values))] + [abs(v) for v in (first, last) if v is not None]
        )
        if peak > grad_limit:
            errors.append(
                f"line {number}: arbitrary gradient exceeds the amplitude limit"
            )
            continue
        if time_id == 0:
            t = (np.arange(len(values)) + 0.5) * raster
            end = len(values) * raster
        elif time_id > 0:
            t = raster * _decoded_shape(
                parsed["shapes"][time_id], parsed["shape_data"][time_id]
            )
            end = t[-1]
        else:
            continue  # half-raster ids are rejected in native_tables
        if first is not None and time_id == 0:
            # Time-shaped (extended trapezoid) events carry their own endpoints.
            t = np.concatenate(([0.0], t, [end]))
            values = np.concatenate(([first], values, [last]))
        dt, dv = np.diff(t), np.abs(np.diff(values))
        coincident = dt < raster / 4
        if np.any(dv[coincident] > tolerance) or np.any(
            dv[~coincident] > slew_limit * dt[~coincident] + tolerance
        ):
            errors.append(f"line {number}: arbitrary gradient exceeds the slew limit")
    return errors[:50]


def native_reader_errors(text, parsed=None):
    """All native-reader rules; never raises on any input text."""
    try:
        errors = native_table_errors(text, parsed)
        return errors or native_extension_errors(text)
    except MALFORMED_SEQUENCE as exc:  # malformed input fails, never interrupts
        return [f"unparseable Pulseq text: {type(exc).__name__}: {exc}"[:300]]


_OFFSET_FIELDS = ("freq_offset", "phase_offset", "freq_ppm", "phase_ppm")


def preflight(submission, case, *, hardware_limits=True):
    """Check file/timing validity; optionally also enforce public hardware limits."""
    from pypulseq import Opts, Sequence

    hw = case["hardware"]
    system = Opts(
        B0=hw["field_strength_T"],
        max_grad=hw["max_gradient_mT_per_m"],
        grad_unit="mT/m",
        max_slew=hw["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        rf_dead_time=hw["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=hw["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=hw["adc_dead_time_us"] * 1e-6,
        grad_raster_time=hw["gradient_raster_us"] * 1e-6,
        rf_raster_time=hw["rf_raster_us"] * 1e-6,
        adc_raster_time=hw["adc_raster_us"] * 1e-6,
        block_duration_raster=hw["block_raster_us"] * 1e-6,
    )
    seq = Sequence(system)
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    # Native-reader table rules run first: a huge declared shape length or an
    # overflowing integer must fail here, before any reader allocates for it.
    try:
        text = Path(submission).read_text(errors="replace")
    except OSError as exc:
        table_errors = [f"{type(exc).__name__}: {exc}"]
    else:
        parsed = {}
        table_errors = native_reader_errors(text, parsed)
    check(
        "native_tables",
        not table_errors,
        "Pulseq 1.4/1.5 sections must be readable by the native reader: exact "
        "VERSION lines, BLOCKS rows of 8 unsigned Int64 integers, event rows of "
        "the version's field layout with unique dense ids 1..N, resolvable "
        "shapes, contiguous tables, shape sample counts matching their data, "
        "and acyclic extension lists with integer trigger timing"
        + (f"; {table_errors[:5]}" if table_errors else ""),
    )
    if table_errors:
        return checks
    if hardware_limits:
        try:
            event_errors = gradient_event_errors(
                parsed, system.max_grad, system.max_slew
            )
        except MALFORMED_SEQUENCE as exc:
            event_errors = [f"{type(exc).__name__}: {exc}"[:300]]
        check(
            "gradient_events",
            not event_errors,
            "Every gradient event, used or not, within max gradient and slew from "
            "its own fields (independent of the merged waveforms)"
            + (f"; {event_errors[:5]}" if event_errors else ""),
        )
    try:
        seq.read(str(submission))
        for name in (
            "grad_raster_time",
            "rf_raster_time",
            "adc_raster_time",
            "block_duration_raster",
        ):
            check(
                name,
                np.isclose(
                    getattr(seq, name), getattr(system, name), rtol=1e-9, atol=1e-12
                ),
                "Submitted raster must match the hardware profile",
            )
            # File headers cannot relax the task's raster or hardware limits.
            setattr(seq, name, getattr(system, name))
        check("pulseq_parse", True, "Pulseq file parsed")
        timing_ok, errors = seq.check_timing()
        check("timing", timing_ok, str(errors[:10]))
        duration = float(seq.duration()[0])
        check(
            "duration",
            np.isfinite(duration)
            and 0 < duration <= case["task"]["sequence"]["duration_s"],
            {
                "duration_s": duration if np.isfinite(duration) else None,
                "max_s": case["task"]["sequence"]["duration_s"],
            },
        )
        adc_count, rf_count, rf_peak_hz = 0, 0, 0.0
        finite_events = True
        finite_adc = True
        unsupported_events = []
        frequency_hz = 0.0  # Largest effective |RF/ADC frequency offset|.
        gamma_b0 = system.gamma * hw["field_strength_T"]
        for number in seq.block_events:
            block = seq.get_block(number)
            for event in (block.rf, block.adc):
                if event is not None:
                    with np.errstate(over="ignore", invalid="ignore"):
                        offset = abs(
                            float(event.freq_offset)
                            + float(event.freq_ppm) * 1e-6 * gamma_b0
                        )
                    frequency_hz = max(
                        frequency_hz, offset if np.isfinite(offset) else np.inf
                    )
            if block.adc is not None:
                adc_count += block.adc.num_samples
                finite_adc &= all(
                    np.isfinite(getattr(block.adc, name, 0.0)).all()
                    for name in ("dwell", "delay", *_OFFSET_FIELDS)
                )
                modulation = getattr(block.adc, "phase_modulation", None)
                finite_adc &= modulation is None or np.isfinite(modulation).all()
                # A sampled ADC needs a positive dwell; per-sample phase
                # modulation is either absent or one value per sample.
                finite_adc &= bool(float(block.adc.dwell) > 0)
                finite_adc &= modulation is None or np.size(modulation) in (
                    0,
                    int(block.adc.num_samples),
                )
            if block.rf is not None:
                rf_count += 1
                finite_events &= bool(
                    np.isfinite(block.rf.signal).all() and np.isfinite(block.rf.t).all()
                )
                finite_events &= all(
                    np.isfinite(getattr(block.rf, name, 0.0)).all()
                    for name in ("delay", "shape_dur", *_OFFSET_FIELDS)
                )
                rf_peak_hz = max(rf_peak_hz, float(np.max(np.abs(block.rf.signal))))
            # These extensions require a scanner/platform model, absent here.
            for name in ("soft_delay", "rf_shim"):
                if getattr(block, name, None) is not None:
                    unsupported_events.append(name)
        check(
            "supported_events", not unsupported_events, sorted(set(unsupported_events))
        )
        excitation_only = case["evaluation"]["id"] == "rf_excitation"
        check(
            "has_rf_no_adc" if excitation_only else "has_rf_and_adc",
            rf_count > 0 and (adc_count == 0 if excitation_only else adc_count > 0),
            {"adc_samples": adc_count, "rf_events": rf_count},
        )
        check(
            "finite_rf",
            finite_events,
            "RF samples, times, frequency and phase must be finite",
        )
        check(
            "frequency_offset",
            frequency_hz <= MAX_FREQUENCY_OFFSET_HZ,
            {
                "max_abs_hz": frequency_hz if np.isfinite(frequency_hz) else None,
                "limit_hz": MAX_FREQUENCY_OFFSET_HZ,
                "rule": "|freq_offset + freq_ppm*1e-6*gamma*B0| for every RF and ADC",
            },
        )
        check(
            "finite_adc",
            finite_adc,
            "ADC timing, frequency, phase and modulation must be finite; dwell > 0; "
            "phase modulation empty or one value per sample",
        )
        found = gradient_jumps(seq, system)
        jumps = [item for item in found if item["at"] != "endpoint slew"]
        check(
            "gradient_continuity",
            not jumps,
            "Gradients must be continuous across block boundaries and start/end at zero"
            + (f"; jumps: {jumps[:5]}" if jumps else ""),
        )
        if hardware_limits:
            steep = [item for item in found if item["at"] == "endpoint slew"]
            check(
                "gradient_endpoint_slew",
                not steep,
                "Arbitrary-gradient first/last and sample steps within max slew"
                + (f"; {steep[:5]}" if steep else ""),
            )
        b1 = rf_peak_hz / system.gamma * 1e6
        if hardware_limits:
            check(
                "peak_b1",
                np.isfinite(b1) and b1 <= hw["max_b1_uT"] * (1 + 1e-6),
                {"peak_uT": b1 if np.isfinite(b1) else None, "max_uT": hw["max_b1_uT"]},
            )
        for axis, waveform in zip("xyz", seq.waveforms()):
            if waveform.size == 0:
                continue
            t, g = waveform
            valid = np.isfinite(waveform).all() and (np.diff(t) > 0).all()
            check(
                f"finite_gradient_{axis}", valid, "finite samples with increasing time"
            )
            if not valid or not hardware_limits:
                continue
            peak = float(np.max(np.abs(g)) / system.gamma * 1e3)
            slew = (
                float(np.max(np.abs(np.diff(g) / np.diff(t))) / system.gamma)
                if len(t) > 1
                else 0.0
            )
            check(
                f"gradient_{axis}",
                peak <= hw["max_gradient_mT_per_m"] * (1 + 1e-6),
                {"peak_mT_per_m": peak, "max": hw["max_gradient_mT_per_m"]},
            )
            check(
                f"slew_{axis}",
                slew <= hw["max_slew_T_per_m_per_s"] * (1 + 1e-6),
                {"peak_T_per_m_per_s": slew, "max": hw["max_slew_T_per_m_per_s"]},
            )
    except MALFORMED_SEQUENCE as exc:
        check("pulseq_preflight", False, f"{type(exc).__name__}: {exc}")
    return checks
