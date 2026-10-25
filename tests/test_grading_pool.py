"""Live pool changes exercise waiting jobs without restarting their holders."""

import fcntl
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from mriseqbench.harness.grading import EvaluationIncomplete, grading_slot
from mriseqbench.harness.grading_pool import GradingSlots, atomic_json, read_gpu_pool

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for scheduler progress"
        time.sleep(0.01)


def pool(tmp_path, count):
    config = tmp_path / "slots.json"
    atomic_json(config, {"slots": count})
    return GradingSlots(
        config, status=tmp_path / "status.json", history=tmp_path / "changes.jsonl"
    )


def test_increase_starts_fifo_waiters_without_releasing_running_grade(tmp_path):
    slots = pool(tmp_path, 1)
    assert slots.acquire()
    entered = []
    release = threading.Event()

    def job(number):
        with grading_slot(slots):
            entered.append(number)
            assert release.wait(5)

    workers = []
    try:
        for number in range(3):
            worker = threading.Thread(target=job, args=(number,))
            worker.start()
            workers.append(worker)
            wait_for(lambda number=number: slots.stats()["waiting"] == number + 1)
        atomic_json(slots.config, {"slots": 2})
        wait_for(lambda: len(entered) == 1)
        assert entered == [0]
        atomic_json(slots.config, {"slots": 3})
        wait_for(lambda: len(entered) == 2)
        assert entered == [0, 1]
        assert slots.stats()["active"] == 3
        # The original grade still holds its slot; capacity alone admitted two.
        slots.release()
        wait_for(lambda: len(entered) == 3)
        assert entered == [0, 1, 2]
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
    assert slots.stats()["active"] == 0


def test_shrink_and_pause_drain_holders_without_oversubscription(tmp_path):
    slots = pool(tmp_path, 3)
    for _ in range(3):
        assert slots.acquire()
    atomic_json(slots.config, {"slots": 1})
    assert not slots.acquire(blocking=False)
    slots.release()
    slots.release()
    assert not slots.acquire(blocking=False)  # One existing holder still drains.
    slots.release()
    assert slots.acquire(blocking=False)
    atomic_json(slots.config, {"slots": 0})
    slots.release()
    assert not slots.acquire(timeout=0.01)
    atomic_json(slots.config, {"slots": 2})
    assert slots.acquire(blocking=False)
    slots.release()


def test_bad_live_update_pauses_until_repaired_and_records_status(tmp_path):
    slots = pool(tmp_path, 1)
    assert slots.acquire()
    slots.config.write_text("{")
    slots.release()  # A broken config never prevents a holder from draining.
    assert not slots.acquire(timeout=0.01)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["config_error"] and status["configured_slots"] == 0
    atomic_json(slots.config, {"slots": 2})
    assert slots.acquire(timeout=1)
    assert slots.stats()["config_error"] is None
    slots.release()
    changes = [
        json.loads(line)
        for line in (tmp_path / "changes.jsonl").read_text().splitlines()
    ]
    assert [change["configured_slots"] for change in changes] == [1, 0, 2]


def test_cancelled_head_does_not_block_the_next_waiter(tmp_path):
    slots = pool(tmp_path, 0)
    cancel = threading.Event()
    outcomes = []

    def cancelled():
        try:
            with grading_slot(slots, cancel):
                outcomes.append("unexpected")
        except EvaluationIncomplete:
            outcomes.append("cancelled")

    head = threading.Thread(target=cancelled)
    head.start()
    wait_for(lambda: slots.stats()["waiting"] == 1)
    cancel.set()
    head.join(5)
    assert outcomes == ["cancelled"] and slots.stats()["waiting"] == 0
    atomic_json(slots.config, {"slots": 1})
    with grading_slot(slots):
        assert slots.stats()["active"] == 1


def test_gpu_capacity_tracks_additions_and_draining_devices(tmp_path, monkeypatch):
    config = tmp_path / "gpus.json"
    atomic_json(config, [{"uuid": "A", "slots": 2}])
    monkeypatch.setenv("MRISEQBENCH_GRADING_GPU_CONFIG", str(config))
    slots = GradingSlots.for_campaign(tmp_path, 1)
    assert slots.stats()["configured_slots"] == 2
    atomic_json(config, [{"uuid": "A", "slots": 2}, {"uuid": "B", "slots": 1}])
    assert slots.stats()["configured_slots"] == 3
    atomic_json(config, [{"uuid": "B", "slots": 0}])
    assert slots.stats()["configured_slots"] == 0


def test_resuming_keeps_the_operator_capacity_instead_of_resetting_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MRISEQBENCH_GRADING_GPU_CONFIG", raising=False)
    first = GradingSlots.for_campaign(tmp_path, 1)
    atomic_json(first.config, {"slots": 4})
    assert GradingSlots.for_campaign(tmp_path, 1).stats()["configured_slots"] == 4


@pytest.mark.parametrize(
    "value",
    [
        {},
        [{"uuid": "A", "slots": True}],
        [{"uuid": "A", "slots": -1}],
        [{"uuid": "A", "slots": 65}],
        [{"uuid": "", "slots": 1}],
        [{"uuid": "A", "slots": 1}, {"uuid": "A", "slots": 1}],
        [{"uuid": "A", "slots": 40}, {"uuid": "B", "slots": 40}],
    ],
)
def test_invalid_gpu_pool_is_rejected_before_start(tmp_path, value):
    config = tmp_path / "gpus.json"
    atomic_json(config, value)
    with pytest.raises(ValueError):
        read_gpu_pool(config)
    with pytest.raises(ValueError):
        GradingSlots(config, gpu=True)


def load_ops(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "ops" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("initial", ["occupied", "paused", "broken"])
def test_waiting_gpu_worker_replans_after_pool_change(tmp_path, monkeypatch, initial):
    worker = load_ops("local_gpu_worker")
    config = tmp_path / "gpus.json"
    atomic_json(config, [{"uuid": "A", "slots": 0 if initial == "paused" else 1}])
    if initial == "broken":
        config.write_text("{")
    locks = tmp_path / "grading-gpu-locks"
    locks.mkdir()
    holder = (locks / "A-0.lock").open("w")
    fcntl.flock(holder, fcntl.LOCK_EX)
    monkeypatch.setenv("MRISEQBENCH_GRADING_GPU_CONFIG", str(config))
    monkeypatch.setattr(
        worker.subprocess, "check_output", lambda *a, **kw: "A, GPU A\nB, GPU B\n"
    )
    calls = []

    def change_pool(_seconds):
        # Remove A altogether; releasing its lock must not dispatch stale work.
        atomic_json(config, [{"uuid": "B", "slots": 1}])
        fcntl.flock(holder, fcntl.LOCK_UN)

    def launch(argv, env):
        calls.append(env["CUDA_VISIBLE_DEVICES"])
        return 0

    monkeypatch.setattr(worker.time, "sleep", change_pool)
    monkeypatch.setattr(worker.subprocess, "call", launch)
    try:
        assert (
            worker.main(["--root", str(ROOT), "--workspace", str(tmp_path / "job")])
            == 0
        )
    finally:
        holder.close()
    assert calls == ["B"]
    receipt = json.loads(
        next((tmp_path / "job").glob("grading-gpu-*.json")).read_text()
    )
    assert receipt["gpu_uuid"] == "B" and receipt["pool_config"] == [
        {"uuid": "B", "slots": 1}
    ]


def test_running_gpu_worker_finishes_on_original_device(tmp_path, monkeypatch):
    worker = load_ops("local_gpu_worker")
    config = tmp_path / "gpus.json"
    atomic_json(config, [{"uuid": "A", "slots": 1}])
    monkeypatch.setenv("MRISEQBENCH_GRADING_GPU_CONFIG", str(config))
    monkeypatch.setattr(
        worker.subprocess, "check_output", lambda *a, **kw: "A, GPU A\nB, GPU B\n"
    )

    def running(argv, env):
        atomic_json(config, [{"uuid": "B", "slots": 1}])
        assert env["CUDA_VISIBLE_DEVICES"] == "A"
        holder = (tmp_path / "grading-gpu-locks/A-0.lock").open("w")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            holder.close()
        return 0

    monkeypatch.setattr(worker.subprocess, "call", running)
    assert worker.main(["--root", str(ROOT), "--workspace", str(tmp_path / "job")]) == 0


def test_operator_update_is_atomic_and_invalid_input_keeps_old_pool(
    tmp_path, monkeypatch
):
    setter = load_ops("set_grading_pool")
    config = tmp_path / "gpus.json"
    atomic_json(config, [{"uuid": "A", "slots": 1}])
    monkeypatch.setattr(
        setter.subprocess, "check_output", lambda *a, **kw: "0, A\n1, B\n"
    )
    assert setter.main(["--config", str(config), "--gpu", "0=2", "--gpu", "1=1"]) == 0
    expected = [{"uuid": "A", "slots": 2}, {"uuid": "B", "slots": 1}]
    assert read_gpu_pool(config) == expected
    for args in (
        ("--gpu", "unknown=1"),
        ("--gpu", "0=-1"),
        ("--gpu", "0=1", "--gpu", "A=1"),
    ):
        with pytest.raises(SystemExit):
            setter.main(["--config", str(config), *args])
        assert read_gpu_pool(config) == expected
