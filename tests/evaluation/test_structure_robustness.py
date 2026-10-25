"""Malformed or re-expressed waveforms resolve to structure verdicts; no simulation."""

import copy
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import s2
from mriseqbench.evaluation.s1_sequence import FAMILIES as S1_FAMILIES
from mriseqbench.evaluation.s1_sequence import inspect_sequence as inspect_s1
from mriseqbench.evaluation.s2_sequence import FAMILIES as S2_FAMILIES
from mriseqbench.evaluation.s2_sequence import inspect_sequence as inspect_s2
from tests.fixtures.reference128_readouts import build

ROOT = Path(__file__).resolve().parents[2]


def case_for(task, hardware="standard_1_5t"):
    catalog = Catalog(ROOT)
    return catalog.resolve(catalog.get("tasks", task), hardware_ref=hardware)


def inspector(task):
    return inspect_s2 if task in S2_FAMILIES else inspect_s1


def rebuild(seq, edit=lambda events: events, prefix=()):
    """Copy every block (timing preserved) after optional prefix blocks."""
    out = pp.Sequence(seq.system)
    for events in prefix:
        out.add_block(*events)
    for number in seq.block_events:
        block = seq.get_block(number)
        events = [
            copy.deepcopy(getattr(block, name))
            for name in ("rf", "gx", "gy", "gz", "adc")
            if getattr(block, name) is not None
        ]
        out.add_block(pp.make_delay(seq.block_durations[number]), *edit(events))
    return out


def short_adc_sequence(case, samples):
    h = case["hardware"]
    system = pp.Opts(
        B0=h["field_strength_T"],
        rf_dead_time=h["rf_dead_time_us"] * 1e-6,
        rf_ringdown_time=h["rf_ringdown_time_us"] * 1e-6,
        adc_dead_time=h["adc_dead_time_us"] * 1e-6,
    )
    seq = pp.Sequence(system)
    for _ in range(4):
        seq.add_block(
            pp.make_block_pulse(
                np.deg2rad(20),
                duration=0.0004,
                delay=system.rf_dead_time,
                system=system,
            )
        )
        seq.add_block(
            pp.make_trapezoid("x", area=100, duration=0.001, system=system),
            pp.make_adc(samples, duration=samples * 1e-5, delay=0.0002, system=system),
        )
        seq.add_block(pp.make_delay(0.01))
    return seq


@pytest.mark.parametrize("samples", [1, 2])
@pytest.mark.parametrize("task", sorted(S1_FAMILIES | S2_FAMILIES))
def test_short_readouts_fail_structure_without_raising(tmp_path, task, samples):
    case = case_for(task)
    path = tmp_path / "short.seq"
    short_adc_sequence(case, samples).write(str(path))
    acq = inspector(task)(path, case)
    assert not all(item["passed"] for item in acq.checks)
    if samples == 1:
        assert any(
            item["name"] == "readout_samples" and not item["passed"]
            for item in acq.checks
        )


def test_zero_amplitude_rf_is_absent_in_the_judge(tmp_path):
    case = case_for("epi_single_shot")
    authored = build(case)
    silent = pp.make_block_pulse(
        0.1, duration=0.0004, delay=authored.system.rf_dead_time, system=authored.system
    )
    silent.signal = np.zeros_like(silent.signal)
    original, prefixed = tmp_path / "original.seq", tmp_path / "zero-rf.seq"
    rebuild(authored).write(str(original))
    rebuild(authored, prefix=[(silent,)]).write(str(prefixed))
    before, after = inspect_s2(original, case), inspect_s2(prefixed, case)
    # KomaMRI's active_rf also drops all-zero RF, so both engines agree.
    assert len(after.pulses) == len(before.pulses) == 1
    assert all(item["passed"] for item in after.checks), after.checks


def test_stack_of_stars_target_and_phase_use_every_partition(tmp_path, monkeypatch):
    case = case_for("stack_of_stars")
    path = tmp_path / "sos.seq"
    build(case).write(str(path))
    acq = inspect_s2(path, case)
    spec = case["task"]["sequence"]
    target = s2.contrast_indices(acq, spec)
    assert len(target) == len(acq.lines) == 202 * spec["matrix"][2]
    seen = {}
    monkeypatch.setattr(
        s2,
        "shared_phase_reference",
        lambda *args, sample_indices, **kwargs: seen.setdefault("ids", sample_indices),
    )
    s2.phase_reference(acq, case, ROOT, None)
    np.testing.assert_array_equal(seen["ids"], target)


def test_unequal_nearest_origin_distances_keep_every_spoke(tmp_path):
    case = case_for("radial_readout")
    path = tmp_path / "radial.seq"
    build(case).write(str(path))
    acq = inspect_s2(path, case)
    spec = case["task"]["sequence"]
    fov = np.asarray(spec["fov_mm"]) / 1000
    before = s2.reference_indices(acq, spec)
    acq.k = acq.k.copy()
    acq.k[acq.lines[7], 1] += 0.1 / fov[1]  # one spoke 0.1 sample off-centre
    after = s2.reference_indices(acq, spec)
    assert len(before) == len(after) == len(acq.lines) == 202
    np.testing.assert_array_equal(after, s2.contrast_indices(acq, spec))


def bssfp():
    case = case_for("bssfp")
    return case, build(case)


def test_bssfp_terminal_unacquired_pulse_cannot_hide_a_spoiler(tmp_path):
    case, seq = bssfp()
    seq.add_block(
        pp.make_block_pulse(
            np.deg2rad(1),
            duration=0.0004,
            delay=seq.system.rf_dead_time,
            system=seq.system,
        )
    )
    seq.add_block(pp.make_trapezoid("x", area=5000, duration=0.004, system=seq.system))
    path = tmp_path / "terminal.seq"
    seq.write(str(path))
    acq = inspect_s1(path, case)
    assert any(item["name"] == "bssfp" and not item["passed"] for item in acq.checks)


def test_s1_rejects_unacquired_excitation_between_acquired_ones(tmp_path):
    case, seq = bssfp()
    extra = pp.make_block_pulse(
        np.deg2rad(1), duration=0.0004, delay=seq.system.rf_dead_time, system=seq.system
    )
    adc_blocks = [n for n in seq.block_events if seq.get_block(n).adc is not None]
    middle = adc_blocks[len(adc_blocks) // 2]
    out = pp.Sequence(seq.system)
    for number in seq.block_events:
        block = seq.get_block(number)
        events = [
            getattr(block, name)
            for name in ("rf", "gx", "gy", "gz", "adc")
            if getattr(block, name) is not None
        ]
        out.add_block(pp.make_delay(seq.block_durations[number]), *events)
        if number == middle:
            out.add_block(extra)
    path = tmp_path / "between.seq"
    out.write(str(path))
    acq = inspect_s1(path, case)
    item = next(c for c in acq.checks if c["name"] == "single_line_per_excitation")
    assert not item["passed"]
    assert item["detail"]["unacquired_excitations_between_acquired"] == 1


def test_rf_phase_carried_by_the_envelope_is_the_same_rf(tmp_path):
    case, seq = bssfp()

    def carry(events):
        for event in events:
            if event.type == "rf":
                event.signal = event.signal * np.exp(1j * event.phase_offset)
                event.phase_offset = 0.0
        return events

    plain, carried = tmp_path / "plain.seq", tmp_path / "carried.seq"
    rebuild(seq).write(str(plain))
    rebuild(seq, carry).write(str(carried))
    a, b = inspect_s1(plain, case), inspect_s1(carried, case)
    assert all(item["passed"] for item in b.checks), [
        c for c in b.checks if not c["passed"]
    ]
    np.testing.assert_allclose(
        np.exp(1j * np.array([p.phase for p in b.pulses])),
        np.exp(1j * np.array([p.phase for p in a.pulses])),
        atol=1e-5,  # Pulseq stores phase_offset with finite precision
    )


@pytest.mark.parametrize("task", ["grase", "epi_multishot"])
def test_cartesian_trains_must_read_along_x(tmp_path, task):
    case = case_for(task)

    def swap(events):
        for event in events:
            if event.type in {"trap", "grad"} and event.channel in "xy":
                event.channel = "y" if event.channel == "x" else "x"
        return events

    path = tmp_path / "swapped.seq"
    rebuild(build(case), swap).write(str(path))
    acq = inspect_s2(path, case)
    item = next(c for c in acq.checks if c["name"] == "readout_axis")
    assert not item["passed"]


def test_oracle_invalid_foreground_cannot_hide_complex_error():
    from mriseqbench.evaluation.s1_case import case_image_scores

    reference = np.array([1.0, 1.0, 0.0], complex)
    truth = np.array([1.0, -1.0, 0.0], complex)
    rng = np.random.default_rng(0)
    # Independent replicates with tiny quadrature noise around the image.
    images = truth + 1e-9 * rng.standard_normal((8, 3))
    support = np.array([True, True, False])
    valid = np.array([True, False, False])
    values, _, _ = case_image_scores(
        images, reference, support, np.ones(3, complex), valid, {}
    )
    # Magnitudes agree, so the invalid voxel adds no error; the full target
    # norm normalises (sqrt(0/2)), and the valid voxel is exact.
    assert values["complex_image_nrmse"] == pytest.approx(0.0, abs=1e-6)
    assert values["magnitude_image_nrmse"] == pytest.approx(0.0, abs=1e-6)
    # A magnitude error on the invalid voxel now counts against the full norm.
    images[:, 1] = 0.0
    values, _, _ = case_image_scores(
        images, reference, support, np.ones(3, complex), valid, {}
    )
    assert values["complex_image_nrmse"] == pytest.approx(np.sqrt(1 / 2), rel=1e-6)
    assert values["magnitude_image_nrmse"] == pytest.approx(np.sqrt(1 / 2), rel=1e-6)
    # And a valid-voxel error is normalised by the full target energy.
    images[:, 1] = 1.0
    images[:, 0] = -1.0
    values, _, _ = case_image_scores(
        images, reference, support, np.ones(3, complex), valid, {}
    )
    assert values["complex_image_nrmse"] == pytest.approx(np.sqrt(4 / 2), rel=1e-6)


@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.parametrize("task", ["gre_t1w", "radial_readout", "epi_single_shot"])
def test_zero_sample_adc_fails_structure_without_raising(tmp_path, task, all_empty):
    from mriseqbench.evaluation.s1_sequence import MissingExcitation

    case = case_for(task)
    seq = short_adc_sequence(case, 4)
    empty = pp.make_adc(4, duration=4e-5, system=seq.system)
    empty.num_samples = 0
    if all_empty:
        seq = pp.Sequence(seq.system)
        seq.add_block(
            pp.make_block_pulse(
                np.deg2rad(20),
                duration=0.0004,
                delay=seq.system.rf_dead_time,
                system=seq.system,
            )
        )
    seq.add_block(empty)
    path = tmp_path / "zero.seq"
    seq.write(str(path))
    try:
        acq = inspector(task)(path, case)
    except MissingExcitation:
        # Caught by every evaluator and returned as a failed structure check.
        assert all_empty
        return
    assert not all_empty
    item = next(c for c in acq.checks if c["name"] == "readout_samples")
    assert not item["passed"] and item["detail"]["zero_sample_adc_events"] == 1


@pytest.mark.parametrize("task", ["gre_t1w", "radial_readout", "spiral_gre"])
def test_readout_bandwidth_floor_is_500_hz_per_pixel(tmp_path, task):
    case = case_for(task)
    authored = build(case)

    def slow(events):
        for event in events:
            if event.type == "adc":
                event.dwell = 30e-6  # 1 / (128 x 30 us) = 260 Hz/pixel
        return events

    compliant, narrow = tmp_path / "compliant.seq", tmp_path / "narrow.seq"
    rebuild(authored).write(str(compliant))
    rebuild(authored, slow).write(str(narrow))
    ok = next(
        c
        for c in inspector(task)(compliant, case).checks
        if c["name"] == "readout_bandwidth"
    )
    bad = next(
        c
        for c in inspector(task)(narrow, case).checks
        if c["name"] == "readout_bandwidth"
    )
    assert ok["passed"] and ok["detail"]["measured_min_hz_per_pixel"] >= 500
    assert not bad["passed"]
    assert bad["detail"]["measured_min_hz_per_pixel"] == pytest.approx(
        1 / (128 * 30e-6)
    )


def adc_mutated(tmp_path, task, mutate):
    case = case_for(task)

    done = []

    def edit(events):
        for event in events:
            if event.type == "adc" and not done:  # one malformed ADC suffices
                mutate(event)
                done.append(True)
        return events

    path = tmp_path / "mutated.seq"
    rebuild(build(case), edit).write(str(path))
    return path, case


@pytest.mark.parametrize(
    "mutate",
    [
        lambda adc: setattr(adc, "dwell", 0.0),
        lambda adc: setattr(adc, "phase_modulation", np.zeros(3)),
    ],
    ids=["zero_dwell", "short_phase_modulation"],
)
def test_malformed_adc_definition_fails_preflight_and_inspection(tmp_path, mutate):
    from mriseqbench.evaluation.preflight import preflight

    path, case = adc_mutated(tmp_path, "gre_t1w", mutate)
    assert not all(c["passed"] for c in preflight(path, case))
    acq = inspect_s1(path, case)
    samples = next(c for c in acq.checks if c["name"] == "readout_samples")
    assert not samples["passed"] and samples["detail"]["invalid_adc_events"] > 0


def test_straight_spiral_fails_without_a_degenerate_fit(tmp_path):
    case = case_for("spiral_gre")
    system = build(case).system
    seq = pp.Sequence(system)
    for _ in range(16):
        seq.add_block(
            pp.make_block_pulse(
                np.deg2rad(20),
                duration=0.0004,
                delay=system.rf_dead_time,
                system=system,
            )
        )
        seq.add_block(
            pp.make_trapezoid(
                "x", flat_area=128 / 0.2, flat_time=0.00256, system=system
            ),
            pp.make_adc(256, duration=0.00256, delay=0.0001, system=system),
        )
        seq.add_block(pp.make_delay(0.01))
    path = tmp_path / "straight.seq"
    seq.write(str(path))
    acq = inspect_s2(path, case)
    spiral = next(c for c in acq.checks if c["name"] == "spiral_gre")
    assert not spiral["passed"]
    assert spiral["detail"]["trajectories"][0]["archimedean_fit_rms_samples"] is None


@pytest.mark.parametrize("task", ["propeller_tse", "radial_readout", "spiral_gre"])
def test_stationary_readouts_fail_and_serialize(tmp_path, task):
    from mriseqbench.evaluation.s1 import finite_json
    from mriseqbench.io import write_json

    case = case_for(task)

    def still(events):
        return [e for e in events if e.type not in {"trap", "grad"}]

    path = tmp_path / "still.seq"
    rebuild(build(case), still).write(str(path))
    acq = inspect_s2(path, case)
    travel = next(c for c in acq.checks if c["name"] == "readout_travel")
    assert not travel["passed"]
    write_json(tmp_path / "sequence.json", finite_json(acq.measurements))
    write_json(tmp_path / "checks.json", finite_json(acq.checks))


def test_finite_json_nulls_non_finite_values():
    from mriseqbench.evaluation.s1 import finite_json

    assert finite_json({"a": [np.nan, 1.0, np.float64(np.inf)], "b": np.int64(2)}) == {
        "a": [None, 1.0, None],
        "b": 2,
    }


def test_shaped_bssfp_with_envelope_phase_survives_serialization(tmp_path):
    case = case_for("bssfp")
    seq = build(case)

    def shaped(events):
        for event in events:
            if event.type == "rf":
                sinc = pp.make_sinc_pulse(
                    np.deg2rad(case["task"]["sequence"].get("flip_angle_deg") or 40),
                    duration=0.0004,
                    time_bw_product=2,
                    delay=event.delay,
                    system=seq.system,
                )
                event.signal = sinc.signal * np.exp(1j * event.phase_offset)
                event.t, event.shape_dur = sinc.t, sinc.shape_dur
                event.phase_offset = 0.0
        return events

    plain = tmp_path / "plain.seq"
    carried = tmp_path / "carried.seq"

    def shaped_plain(events):
        for event in events:
            if event.type == "rf":
                phase = event.phase_offset
                shaped([event])
                event.signal = event.signal * np.exp(-1j * phase)
                event.phase_offset = phase
        return events

    rebuild(seq, shaped_plain).write(str(plain))
    rebuild(seq, shaped).write(str(carried))
    a, b = inspect_s1(plain, case), inspect_s1(carried, case)
    phase = next(c for c in b.checks if c["name"] == "bssfp")
    reference = next(c for c in a.checks if c["name"] == "bssfp")
    assert (
        phase["detail"]["coherent_phase_cycle"]
        == reference["detail"]["coherent_phase_cycle"]
    )
    assert phase["detail"]["coherent_phase_cycle"]


def test_phase_modulated_envelope_keeps_block_phase_only():
    from mriseqbench.evaluation.s1_sequence import envelope_phase

    t = np.linspace(-1, 1, 200)
    widths = np.full(200, 1e-6)
    assert envelope_phase(np.sinc(2 * t) * np.exp(1j * 0.7), widths) == (
        pytest.approx(0.7)
    )
    chirp = np.exp(1j * 3 * np.pi * t**2)
    assert envelope_phase(chirp, widths) == 0.0


def long_linear_rf_sequence(case):
    system = build(case).system
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(
        0.1, duration=0.0004, delay=system.rf_dead_time, system=system
    )
    rf.signal = np.array([0.1, 0.2, 0.1], complex)
    rf.t = np.array([0.0, 0.55, 1.1])
    rf.shape_dur = 1.1
    seq.add_block(rf)
    seq.add_block(pp.make_adc(128, dwell=1e-5, system=system))
    return seq


@pytest.mark.parametrize("task", ["gre_t1w", "radial_readout"])
def test_rf_quadrature_limit_is_a_failed_structure_verdict(tmp_path, task):
    from mriseqbench.evaluation.s1 import evaluate_s1
    from mriseqbench.evaluation.s2 import evaluate_s2

    case = case_for(task)
    path = tmp_path / "long-rf.seq"
    long_linear_rf_sequence(case).write(str(path))
    judge = evaluate_s2 if task in S2_FAMILIES else evaluate_s1
    report = judge(path, case, ROOT, tmp_path / "work")
    assert report["status"] == "failed"
    assert [c["name"] for c in report["checks"]] == ["rf_quadrature"]


def test_rf_quadrature_limit_fails_s3_inspection(tmp_path):
    from mriseqbench.evaluation.s3_sequence import inspect_sequence as inspect_s3

    case = case_for("sense_sampling")
    path = tmp_path / "long-rf.seq"
    long_linear_rf_sequence(case).write(str(path))
    assert not inspect_s3(path, case).checks[0]["passed"]


def test_delayed_constant_rf_keeps_its_explicit_support(tmp_path):
    """Constant 833.333 Hz over t = 200..800 us is a 180 deg refocuser."""
    case = case_for("spin_echo_t2w")
    system = build(case).system
    rf = pp.make_block_pulse(
        np.pi, duration=0.001, delay=system.rf_dead_time, system=system
    )
    rf.signal = np.array([833.333, 833.333], complex)
    rf.t = np.array([200e-6, 800e-6])
    rf.shape_dur = 0.001
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_block_pulse(
            np.pi / 2, duration=0.0004, delay=system.rf_dead_time, system=system
        )
    )
    seq.add_block(pp.make_delay(0.002))
    seq.add_block(rf)
    seq.add_block(pp.make_adc(128, dwell=1e-5, system=system))
    path = tmp_path / "delayed.seq"
    seq.write(str(path))
    acq = inspect_s1(path, case)
    pulse = acq.pulses[1]
    assert pulse.flip == pytest.approx(180, abs=0.01)
    assert pulse.refocusing
    origin = pulse.block_start + system.rf_dead_time
    assert pulse.start - origin == pytest.approx(200e-6, abs=1e-9)
    assert pulse.center - origin == pytest.approx(500e-6, abs=1e-9)


def test_block_pulse_hold_is_unchanged():
    from types import SimpleNamespace

    from mriseqbench.evaluation.rf_waveform import rf_waveform

    rf = SimpleNamespace(
        signal=np.full(2, 625.0), t=np.array([0, 4e-4]), shape_dur=4e-4
    )
    edges, values, linear = rf_waveform(rf, 1e-6)
    np.testing.assert_allclose(edges, [0, 4e-4])
    assert not linear and values.tolist() == [625.0]
    raster = SimpleNamespace(
        signal=np.full(4, 1.0), t=(np.arange(4) + 0.5) * 1e-6, shape_dur=5e-6
    )
    np.testing.assert_allclose(rf_waveform(raster, 1e-6)[0], [0, 5e-6])


def test_diagnostic_replay_keeps_half_raster_explicit_rf(tmp_path):
    from mriseqbench.evaluation.s1_koma import KomaWorker

    case = case_for("gre_t1w")
    mutated = []

    def half_raster(events):
        for event in events:
            if event.type == "rf" and np.all(event.signal == event.signal[0]):
                duration = float(event.shape_dur)
                event.signal = np.full(
                    2, event.signal[0] * duration / (duration - 1e-6)
                )
                event.t = np.array([0.5e-6, duration - 0.5e-6])
                mutated.append(True)
        return events

    path = tmp_path / "half-raster.seq"
    rebuild(build(case), half_raster).write(str(path))
    assert mutated
    acq = inspect_s1(path, case)
    worker = KomaWorker.__new__(KomaWorker)  # sequence generation only, no Julia
    worker.path, worker.sequence_cache = tmp_path, {}
    replay = pp.Sequence()
    replay.read(worker.sequence(acq, zero_gradients=True))
    original = pp.Sequence()
    original.read(str(path))

    def rf_events(seq):
        clock, out = 0.0, []
        for number in seq.block_events:
            rf = seq.get_block(number).rf
            if rf is not None:
                out.append((round(clock, 9), rf.t.tolist(), rf.signal.tolist()))
            clock += seq.block_durations[number]
        return out

    assert len(rf_events(replay)) == len(acq.pulses) > 0
    assert rf_events(replay) == rf_events(original)[: len(rf_events(replay))]


def test_native_parse_rejection_is_a_candidate_failure(monkeypatch, tmp_path):
    from mriseqbench.evaluation import s1
    from mriseqbench.evaluation.s1_koma import NativeRejection, worker_error

    tagged = worker_error(
        "Native Pulseq reader rejected the file: OverflowError: parse(Int, ...)"
    )
    assert isinstance(tagged, NativeRejection)
    assert tagged.args[0].startswith("KomaMRI: ")  # S4 physio's contract
    for host in (
        "Native Pulseq reader rejected the file: CUDA error: out of memory",
        "OutOfMemoryError()",
        "KomaMRI/Pulseq RF count mismatch",
    ):
        error = worker_error(host)
        assert type(error) is RuntimeError

    case = case_for("gre_t1w")

    def reject(*args, **kwargs):
        raise tagged

    monkeypatch.setattr(s1, "_evaluate_s1", reject)
    report = s1.evaluate_s1(tmp_path / "x.seq", case, ROOT, tmp_path)
    assert report["status"] == "failed"
    assert [c["name"] for c in report["checks"]] == ["native_pulseq_parse"]

    def crash(*args, **kwargs):
        raise RuntimeError("KomaMRI worker exited: ...")

    monkeypatch.setattr(s1, "_evaluate_s1", crash)
    with pytest.raises(RuntimeError, match="worker exited"):
        s1.evaluate_s1(tmp_path / "x.seq", case, ROOT, tmp_path)


REPORT_KEYS = {
    "status",
    "checks",
    "image_checks",
    "metrics",
    "sequence_valid",
    "numerically_converged",
    "evaluation_contract",
    "artifacts",
    "limitations",
}


def judge_for(task):
    from mriseqbench.evaluation.s1 import evaluate_s1
    from mriseqbench.evaluation.s2 import evaluate_s2
    from mriseqbench.evaluation.s3 import evaluate_s3

    level = case_for(task)["sequence_level"]
    return {"S1": evaluate_s1, "S2": evaluate_s2, "S3": evaluate_s3}[level]


def assert_structured_failure(report, case, tmp_path):
    import json

    from mriseqbench.evaluation.evaluate import normalize_verdict

    assert REPORT_KEYS <= report.keys()
    report = normalize_verdict(report, case)
    assert report["status"] == "failed" and report["sequence_valid"] is False
    json.dumps(report, allow_nan=False)


def zero_rf_sequence(case):
    seq = short_adc_sequence(case, 128)
    for number in seq.block_events:
        rf = seq.get_block(number).rf
        if rf is not None:
            seq.rf_library.data[seq.block_events[number][1]] = tuple(
                0.0 if i == 0 else v
                for i, v in enumerate(seq.rf_library.data[seq.block_events[number][1]])
            )
    return seq


@pytest.mark.parametrize("task", ["gre_t1w", "radial_readout", "sense_sampling"])
def test_native_rejection_is_a_complete_failed_verdict(monkeypatch, tmp_path, task):
    from mriseqbench.evaluation import s1
    from mriseqbench.evaluation.s1_koma import worker_error

    def reject(*args, **kwargs):
        raise worker_error("Native Pulseq reader rejected the file: OverflowError")

    monkeypatch.setattr(s1, "_evaluate_s1", reject)
    case = case_for(task)
    report = judge_for(task)(tmp_path / "x.seq", case, ROOT, tmp_path / "work")
    assert_structured_failure(report, case, tmp_path)
    assert any(c["name"] == "native_pulseq_parse" for c in report["checks"])


@pytest.mark.parametrize("task", ["gre_t1w", "radial_readout", "sense_sampling"])
def test_missing_excitation_is_a_complete_failed_verdict(tmp_path, task):
    case = case_for(task)
    path = tmp_path / "zero-rf.seq"
    zero_rf_sequence(case).write(str(path))
    report = judge_for(task)(path, case, ROOT, tmp_path / "work")
    assert_structured_failure(report, case, tmp_path)


def test_s4_missing_excitation_is_a_failed_verdict(tmp_path):
    import json

    from mriseqbench.evaluation.evaluate import normalize_verdict
    from mriseqbench.evaluation.s4 import evaluate_s4

    case = case_for("t1_mapping")
    path = tmp_path / "zero-rf.seq"
    zero_rf_sequence(case).write(str(path))
    report = evaluate_s4(path, case, ROOT, tmp_path / "work", physio_resolution={})
    report = normalize_verdict(report, case)
    assert report["status"] == "failed" and report["sequence_valid"] is False
    json.dumps(report, allow_nan=False)


def test_s4_frames_get_the_complete_schema_on_native_rejection(monkeypatch, tmp_path):
    from mriseqbench.evaluation import s1
    from mriseqbench.evaluation.s1_koma import worker_error

    def reject(*args, **kwargs):
        raise worker_error("Native Pulseq reader rejected the file: OverflowError")

    monkeypatch.setattr(s1, "_evaluate_s1", reject)
    case = case_for("t1_mapping")
    frame = s1.evaluate_s1(tmp_path / "x.seq", case, ROOT, tmp_path / "frame")
    # S4 aggregates these keys from every frame result.
    for key in ("sequence_valid", "numerically_converged", "metrics", "checks"):
        assert key in frame
    assert all(key in frame for key in ("image_checks", "ideal_image_checks"))


@pytest.mark.parametrize(
    "event,field",
    [
        ("rf", "freq_ppm"),
        ("rf", "phase_ppm"),
        ("adc", "freq_ppm"),
        ("adc", "phase_ppm"),
    ],
)
def test_overflowing_ppm_fails_before_simulation(tmp_path, event, field):
    import json

    from mriseqbench.evaluation.s1 import evaluate_s1

    case = case_for("gre_t1w")

    def huge(events):
        for item in events:
            if item.type == event:
                setattr(item, field, 1e308)
        return events

    path = tmp_path / "ppm.seq"
    rebuild(build(case), huge).write(str(path))
    report = evaluate_s1(path, case, ROOT, tmp_path / "work")
    assert report["status"] == "failed"
    assert [c["name"] for c in report["checks"]] == ["finite_derived_values"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["freq_ppm", "phase_ppm"])
def test_rf_excitation_overflowing_ppm_is_a_failed_check(tmp_path, field):
    from mriseqbench.evaluation.excitation import diagnostic_acquisition

    case = case_for("rf_excitation_2d")
    system = build(case_for("gre_t1w")).system
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(
        0.5, duration=0.001, delay=system.rf_dead_time, system=system
    )
    setattr(rf, field, 1e308)
    seq.add_block(rf)
    path = tmp_path / "ppm.seq"
    seq.write(str(path))
    with pytest.raises(ValueError, match="not finite"):
        diagnostic_acquisition(path, case, tmp_path)


def test_frame_replay_does_not_reuse_a_file_with_extra_rf(tmp_path):
    from mriseqbench.evaluation.s1_koma import KomaWorker
    from mriseqbench.evaluation.s4_sequence import frame_acquisition
    from mriseqbench.evaluation.s4_sequence import inspect_sequence as inspect_s4
    from tests.fixtures.reference128_advanced import build as build_advanced

    case = case_for("t2_prepared")
    seq = build_advanced(case)
    seq.add_block(
        pp.make_block_pulse(
            np.deg2rad(17.5),
            duration=0.0002,
            delay=seq.system.rf_dead_time,
            system=seq.system,
        )
    )
    path = tmp_path / "trailing-rf.seq"
    seq.write(str(path))
    acq = inspect_s4(path, case)
    assert all(c["passed"] for c in acq.checks), [
        c for c in acq.checks if not c["passed"]
    ]
    frame = frame_acquisition(acq, acq.measurements["frames"][0])
    assert len(frame.pulses) == len(acq.pulses) - 1 and len(frame.adc) == len(acq.adc)
    worker = KomaWorker.__new__(KomaWorker)
    worker.path, worker.sequence_cache = tmp_path, {}
    replay = worker.sequence(frame, zero_gradients=False)
    assert replay != str(path)
    native = pp.Sequence()
    native.read(replay)
    rf_blocks = [n for n in native.block_events if native.get_block(n).rf is not None]
    assert len(rf_blocks) == len(frame.pulses)
    assert worker.sequence(acq, zero_gradients=False) == str(path)


def test_huge_refocusing_phase_keeps_receiver_phase_finite(tmp_path):
    from mriseqbench.evaluation.s1_image import expected_receiver_phase

    case = case_for("spin_echo_t2w")

    def huge(events):
        for event in events:
            if event.type == "rf" and event.use == "refocusing":
                event.phase_offset = 1e308
        return events

    path = tmp_path / "phase.seq"
    rebuild(build(case), huge).write(str(path))
    acq = inspect_s1(path, case)
    assert any(p.refocusing for p in acq.pulses)
    phase = expected_receiver_phase(acq)
    assert np.isfinite(phase).all() and np.all((0 <= phase) & (phase < 2 * np.pi))


def test_every_s1_report_path_is_strict_json(monkeypatch, tmp_path):
    import json

    from mriseqbench.evaluation import s1

    monkeypatch.setattr(
        s1,
        "_evaluate_s1",
        lambda *a, **k: {"status": "failed", "metrics": {"x": np.nan}},
    )
    report = s1.evaluate_s1(tmp_path / "x.seq", case_for("gre_t1w"), ROOT, tmp_path)
    assert report["metrics"]["x"] is None
    json.dumps(report, allow_nan=False)


def test_native_consistency_mismatch_is_a_candidate_failure(monkeypatch, tmp_path):
    from mriseqbench.evaluation import s1
    from mriseqbench.evaluation.s1_koma import NativeInconsistency, worker_error

    for message in (
        "Native consistency: KomaMRI/Pulseq RF count mismatch",
        "Native consistency: discretized timeline outside a monotone [0, 128] s",
    ):
        assert isinstance(worker_error(message), NativeInconsistency)
    assert type(worker_error("Native consistency: CUDA error")) is RuntimeError
    assert type(worker_error("KomaMRI/Pulseq RF count mismatch")) is RuntimeError

    def reject(*args, **kwargs):
        raise worker_error("Native consistency: KomaMRI/Pulseq RF count mismatch")

    monkeypatch.setattr(s1, "_evaluate_s1", reject)
    case = case_for("gre_t1w")
    report = s1.evaluate_s1(tmp_path / "x.seq", case, ROOT, tmp_path)
    assert [c["name"] for c in report["checks"]] == ["native_consistency"]
    assert_structured_failure(report, case, tmp_path)


@pytest.mark.parametrize("zero_gradients", [False, True])
def test_rf_excitation_diagnostic_passes_the_real_replay_selector(
    tmp_path, zero_gradients
):
    from mriseqbench.evaluation.excitation import diagnostic_acquisition
    from mriseqbench.evaluation.s1_koma import KomaWorker
    from tests.fixtures.reference128_advanced import build as build_advanced

    case = case_for("rf_excitation_2d")
    path = tmp_path / "excitation.seq"
    build_advanced(case).write(str(path))
    acq = diagnostic_acquisition(path, case, tmp_path)
    worker = KomaWorker.__new__(KomaWorker)
    worker.path, worker.sequence_cache = tmp_path, {}
    replay = worker.sequence(acq, zero_gradients=zero_gradients)
    if not zero_gradients:
        assert replay == acq.source_path  # Diagnostic file matches its metadata.
    native = pp.Sequence()
    native.read(replay)
    rf_blocks = [n for n in native.block_events if native.get_block(n).rf is not None]
    assert len(rf_blocks) == len(acq.pulses) > 0
    np.testing.assert_allclose(native.adc_times()[0], acq.adc, atol=1e-9, rtol=0)


def test_cumulative_rf_quadrature_is_capped_before_expansion(tmp_path, monkeypatch):
    from mriseqbench.evaluation import rf_waveform, s1_sequence
    from mriseqbench.evaluation.s1 import evaluate_s1

    case = case_for("gre_t1w")
    system = build(case).system
    seq = pp.Sequence(system)
    for index in range(12):  # 12 distinct 0.99 s linear envelopes
        rf = pp.make_block_pulse(
            0.1, duration=0.0004, delay=system.rf_dead_time, system=system
        )
        rf.signal = np.array([0.1, 0.2 + 0.01 * index], complex)
        rf.t = np.array([0.0, 0.99])
        rf.shape_dur = 0.99
        seq.add_block(rf)
    seq.add_block(pp.make_adc(128, dwell=1e-5, system=system))
    path = tmp_path / "many-envelopes.seq"
    seq.write(str(path))
    expanded = []
    original = rf_waveform.rf_samples
    monkeypatch.setattr(
        s1_sequence,
        "rf_samples",
        lambda *a, **k: expanded.append(1) or original(*a, **k),
    )
    report = evaluate_s1(path, case, ROOT, tmp_path / "work")
    assert [c["name"] for c in report["checks"]] == ["rf_quadrature"]
    # 0.99 s / 0.25 us = 3.96 M intervals each: the third would exceed 8 M.
    assert len(expanded) == 2


def test_effective_rf_phase_overflow_fails_s4_with_strict_json(tmp_path):
    import json

    from mriseqbench.evaluation.evaluate import normalize_verdict
    from mriseqbench.evaluation.preflight import preflight
    from mriseqbench.evaluation.s1_sequence import NonFiniteDerived
    from mriseqbench.evaluation.s4 import evaluate_s4
    from tests.fixtures.reference128_advanced import build as build_advanced

    case = case_for("t2_prepared", "high_performance")

    def far(events):
        for event in events:
            if event.type == "rf":
                event.freq_offset = 1e308
        return events

    path = tmp_path / "far.seq"
    rebuild(build_advanced(case), far).write(str(path))
    item = next(c for c in preflight(path, case) if c["name"] == "frequency_offset")
    assert not item["passed"]
    with pytest.raises(NonFiniteDerived, match="rf_effective_phase_rad"):
        inspect_s1(path, case)
    report = normalize_verdict(
        evaluate_s4(path, case, ROOT, tmp_path / "work", physio_resolution={}), case
    )
    assert report["status"] == "failed"
    json.dumps(report, allow_nan=False)


def test_references_pass_the_frequency_offset_rule(tmp_path):
    from mriseqbench.evaluation.preflight import preflight

    case = case_for("propeller_tse", "high_performance")
    path = tmp_path / "ref.seq"
    build(case).write(str(path))
    item = next(c for c in preflight(path, case) if c["name"] == "frequency_offset")
    assert item["passed"] and item["detail"]["max_abs_hz"] < 1e6


def test_evaluate_boundary_never_returns_non_json(monkeypatch, tmp_path):
    import importlib
    import json

    module = importlib.import_module("mriseqbench.evaluation.evaluate")

    monkeypatch.setattr(
        module,
        "_evaluate",
        lambda *a, **k: {"status": "failed", "metrics": {"x": np.inf}},
    )
    monkeypatch.setattr(module, "finalize_evaluation", lambda *a, **k: None)
    report = module.evaluate(tmp_path / "x.seq", case_for("gre_t1w"), ROOT, tmp_path)
    assert report["metrics"]["x"] is None
    json.dumps(report, allow_nan=False)


def test_repeated_cartesian_samples_gate_on_the_averaged_image():
    from types import SimpleNamespace

    from mriseqbench.evaluation.s1 import reconstructed_temporal_error
    from mriseqbench.evaluation.s1_image import nrmse

    n = 128
    grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1)
    lines = [grid[:, y] for y in range(n)] + [grid[:, 10]] * 900
    indices = np.concatenate(lines)
    acq = SimpleNamespace(indices=indices)

    def averaging_fft(raw, acq, matrix):
        values, counts = np.zeros(matrix, complex), np.zeros(matrix)
        np.add.at(values, tuple(acq.indices.T), raw)
        np.add.at(counts, tuple(acq.indices.T), 1)
        return np.fft.ifft2(values / np.maximum(counts, 1), norm="ortho")

    coarse = np.ones(len(indices), complex)
    refined = coarse.copy()
    refined[(indices[:, 1] == 20)] += 0.2  # a singly acquired line
    reference = averaging_fft(coarse, acq, (n, n))
    raw = nrmse(refined, coarse)
    image = reconstructed_temporal_error(averaging_fft)(
        refined - coarse, acq, (n, n), reference
    )
    assert raw == pytest.approx(0.006238, abs=1e-5) and raw < 0.01
    assert image == pytest.approx(0.017678, abs=1e-5) and image > 0.01


def averaging_fft(raw, acq, matrix):
    values, counts = np.zeros(matrix, complex), np.zeros(matrix)
    np.add.at(values, tuple(acq.indices.T), raw)
    np.add.at(counts, tuple(acq.indices.T), 1)
    return np.fft.ifft2(values / np.maximum(counts, 1), norm="ortho")


def test_unique_cartesian_gate_is_normalised_by_the_target():
    from types import SimpleNamespace

    from mriseqbench.evaluation.s1 import reconstructed_temporal_error
    from mriseqbench.evaluation.s1_image import nrmse

    n = 128
    grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1)
    acq = SimpleNamespace(indices=grid.reshape(-1, 2))
    rng = np.random.default_rng(1)
    target_raw = rng.standard_normal(n * n) + 1j * rng.standard_normal(n * n)
    coarse, refined = 1.3 * target_raw, 1.312 * target_raw
    reference = averaging_fft(target_raw, acq, (n, n))
    raw = nrmse(refined, coarse)
    image = reconstructed_temporal_error(averaging_fft)(
        refined - coarse, acq, (n, n), reference
    )
    assert raw == pytest.approx(0.00923, abs=1e-5) and raw < 0.01
    assert image == pytest.approx(0.012, abs=1e-6) and image > 0.01


def test_s1_defaults_to_the_reconstructed_temporal_gate(monkeypatch, tmp_path):
    from mriseqbench.evaluation import s1

    seen = {}

    def stop(*args, temporal_norm=None, **kwargs):
        seen["norm"] = temporal_norm
        raise RuntimeError("stop")

    original = s1.reconstructed_temporal_error
    monkeypatch.setattr(
        s1,
        "reconstructed_temporal_error",
        lambda r: seen.setdefault("built", True) and original(r),
    )
    case = case_for("gre_t1w")
    path = tmp_path / "ref.seq"
    build(case).write(str(path))
    monkeypatch.setattr(
        s1, "phantom", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        s1.evaluate_s1(path, case, ROOT, tmp_path / "work")
    assert seen.get("built")


def test_invalid_rf_vertices_are_a_structural_failure(tmp_path):
    from mriseqbench.evaluation.s1 import evaluate_s1

    case = case_for("gre_t1w")
    system = build(case).system
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(
        0.5, duration=0.001, delay=system.rf_dead_time, system=system
    )
    rf.signal = np.array([1.0, 0.5, 1.0], complex)
    rf.t = np.array([0, 1000, 1000]) * system.rf_raster_time
    rf.shape_dur = 1000 * system.rf_raster_time
    seq.add_block(rf)
    seq.add_block(pp.make_adc(128, dwell=1e-5, system=system))
    path = tmp_path / "vertices.seq"
    seq.write(str(path))
    report = evaluate_s1(path, case, ROOT, tmp_path / "work")
    assert [c["name"] for c in report["checks"]] == ["rf_vertices"]


def shift_y(scale=-1.0, shift=0.0, fov=0.2):
    """Scale every y trapezoid and add ``shift`` samples to its area."""

    def edit(events):
        for e in events:
            if getattr(e, "channel", None) == "y":
                e.area = scale * e.area + shift / fov
                e.amplitude = e.area / (e.flat_time + (e.rise_time + e.fall_time) / 2)
                e.flat_area = e.amplitude * e.flat_time
        return events

    return edit


def test_s1_whole_sample_band_shift_is_the_same_nyquist_grid(tmp_path):
    # A negated textbook PE table acquires ky = -63..+64: +N/2 aliases to -N/2.
    case = case_for("spin_echo_t2w")
    path = tmp_path / "negated.seq"
    rebuild(build(case), shift_y()).write(str(path))
    acq = inspect_s1(path, case)
    checks = {c["name"]: c for c in acq.checks}
    assert all(checks[n]["passed"] for n in ("cartesian", "fov_mm", "matrix"))
    assert abs(acq.offset[1] - 1) <= 0.01
    q = acq.k[:, 1] * 0.2
    assert set(acq.indices[np.isclose(q, 64, atol=0.01), 1]) == {127}
    detail = checks["cartesian"]["detail"]
    assert detail["acquired_k_index_range"][1][1] == 64
    assert "lattice_offset_limit" in detail
    # Wrong FOV and shifts beyond the one-line wrap stay off the grid.
    wrong = copy.deepcopy(case)
    wrong["task"]["sequence"]["fov_mm"] = [200, 206]
    for shifted, inspected in (
        (shift_y(shift=1.0), case),
        (shift_y(shift=0.6), case),
        (shift_y(), wrong),
    ):
        rebuild(build(case), shifted).write(str(path))
        checks = {c["name"]: c for c in inspect_s1(path, inspected).checks}
        assert not checks["cartesian"]["passed"] and not checks["fov_mm"]["passed"]


def test_s3_reduced_fov_wrapped_band_matches_the_prescribed_mask(tmp_path):
    from mriseqbench.evaluation.s3_sequence import inspect_sequence as inspect_s3
    from tests.fixtures.s3_sequences import make_sequence

    case = case_for("reduced_fov")
    path = tmp_path / "negated.seq"
    seq = make_sequence(case, reduced_design="inner_volume")
    rebuild(seq, shift_y()).write(str(path))
    acq = inspect_s3(path, case)
    checks = {c["name"]: c["passed"] for c in acq.checks}
    assert checks["matrix"] and checks["sampling_pattern"], acq.checks
    assert abs(acq.offset[1] - 2) <= 0.01


def test_s3_lattice_offset_shifts_only_a_wrap_equivalent_band():
    from mriseqbench.evaluation.s3_sequence import lattice_offset

    n = np.array([8, 8])

    def offset(ys, expected_ys):
        q = np.array([(x, y) for x in range(-4, 4) for y in ys], float)
        expected = np.zeros((8, 8), bool)
        expected[:, np.asarray(expected_ys) + 4] = True
        return round(float(lattice_offset(q, n, expected)[1]), 6)

    assert offset(range(-3, 5), range(-4, 4)) == 1
    assert offset(range(-2, 5, 2), range(-4, 4, 2)) == 2
    # Not the same grid modulo N: partial band, odd lines short of +N/2.
    assert offset(range(-1, 5), range(-2, 4)) == 0
    assert offset(range(-3, 4, 2), range(-4, 4, 2)) == 0


def test_shifted_band_reconstructs_as_the_wrapped_edge_line(monkeypatch):
    from types import SimpleNamespace

    from mriseqbench.evaluation import bart, s3

    def numpy_fft(array, *, inverse, axes):
        transform = np.fft.ifftn if inverse else np.fft.fftn
        shifted = np.fft.ifftshift(array, axes)
        return np.fft.fftshift(transform(shifted, axes=axes), axes)

    monkeypatch.setattr(bart, "fft", numpy_fft)
    n = 8
    image = np.random.default_rng(0).normal(size=(n, n)) + 0j
    pos = (np.arange(n) + 0.5) / n - 0.5
    ys = np.arange(-n // 2 + 1, n // 2 + 1)  # -3..+4
    kx, ky = np.meshgrid(np.arange(-n // 2, n // 2), ys, indexing="ij")
    kernel = np.exp(
        -2j
        * np.pi
        * (
            kx.ravel()[:, None, None] * pos[:, None]
            + ky.ravel()[:, None, None] * pos[None, :]
        )
    )
    raw = np.sum(kernel * image, axis=(1, 2))
    # Evaluator mapping: offset +1, every line one bin down.
    shifted = SimpleNamespace(
        indices=np.column_stack([kx.ravel() + n // 2, ky.ravel() - 1 + n // 2]),
        offset=np.array([0.0, 1.0]),
    )
    # Same data with the +N/2 line wrapped to the -N/2 bin (index 0); at
    # half-voxel pixel centres that line carries a sign flip.
    edge = ky.ravel() == n // 2
    wrapped = SimpleNamespace(
        indices=np.column_stack([kx.ravel() + n // 2, (ky.ravel() + n // 2) % n]),
        offset=np.zeros(2),
    )
    assert set(wrapped.indices[edge, 1]) == {0}
    sign = np.where(edge, -1, 1)
    for result in (
        bart.reconstruct_cartesian(raw, shifted, (n, n)),
        bart.reconstruct_cartesian(raw * sign, wrapped, (n, n)),
    ):
        np.testing.assert_allclose(result, image, atol=1e-10)
    grid = np.zeros((n, n), complex)
    grid[tuple(shifted.indices.T)] = raw
    np.testing.assert_allclose(s3.inverse(grid, shifted.offset), image, atol=1e-10)


def test_s1_spoiling_probe_object_has_no_z_extent(monkeypatch, tmp_path):
    # Like the image replay, the 2D object is a plane: a z-only spoiler is inert.
    from mriseqbench.evaluation import s1_probes

    case = case_for("gre_t1w")
    path = tmp_path / "gre.seq"
    build(case).write(str(path))
    acq = inspect_s1(path, case)
    seen = []

    def simulate(a, spins, **kwargs):
        seen.append(np.asarray(spins.positions))
        return np.ones(len(a.adc), complex)

    monkeypatch.setattr(s1_probes, "simulate", simulate)
    monkeypatch.setattr(
        s1_probes,
        "canonical_response",
        lambda a, props, *args, **kwargs: np.ones((len(props), len(a.adc)), complex),
    )
    s1_probes.mechanism_probes(acq, case, None, np.ones((2, 3)))
    ensembles = [p for p in seen if len(p) in (1021, 2053)]
    assert len(ensembles) == 2
    assert all(np.all(p[:, 2] == 0) and np.ptp(p[:, 0]) > 0 for p in ensembles)
