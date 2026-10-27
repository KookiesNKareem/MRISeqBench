import threading
import time

import pytest

from ops.live_simulation_control import LiveSlots, atomic, validate


def configure(root, revision, capacity, admit=True):
    atomic(root / "simulation-qualification.json", {"passed": True})
    atomic(
        root / "simulation-control.json",
        {
            "revision": revision,
            "remote_slots": capacity,
            "admit_new": admit,
            "routing": "runpod",
        },
    )


def test_live_shrink_and_drain_preserve_holders_and_cancel_queued_work(tmp_path):
    configure(tmp_path, 1, 2)
    slots = LiveSlots(tmp_path)
    acquired = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def holder(i):
        assert slots.acquire(lambda: False)
        acquired[i].set()
        assert release[i].wait(5)
        slots.release()

    workers = [threading.Thread(target=holder, args=(i,)) for i in range(2)]
    for worker in workers:
        worker.start()
    assert all(event.wait(3) for event in acquired)
    configure(tmp_path, 2, 1, admit=False)
    cancelled = threading.Event()
    outcome = []
    follower = threading.Thread(
        target=lambda: outcome.append(slots.acquire(cancelled.is_set))
    )
    follower.start()
    time.sleep(0.1)
    assert len(slots.holders) == 2  # Reduction never kills a current simulation.
    release[0].set()
    workers[0].join(3)
    time.sleep(0.3)
    assert outcome == []
    cancelled.set()
    follower.join(3)
    assert outcome == [False]
    release[1].set()
    workers[1].join(3)
    assert not slots.holders and not slots.waiters
    slots.close()


def test_unqualified_routes_and_scientific_knobs_are_rejected():
    control = {"revision": 1, "remote_slots": 8, "admit_new": True, "routing": "runpod"}
    for invalid in [
        {**control, "remote_slots": 9},
        {**control, "max_tokens": 128000},
        {**control, "routing": "unqualified-host"},
        {**control, "remote_slots": True},
    ]:
        with pytest.raises(ValueError):
            validate(invalid)


def test_live_host_addition_uses_distinct_physical_slots_and_keeps_holders(tmp_path):
    configure(tmp_path, 1, 8)
    proof = tmp_path / "qualified.json"
    atomic(proof, {"passed": True, "eight_concurrent_CUDA_slots": True})
    first = {"id": "first", "slots": 8, "qualification": str(proof), "host": {"endpoint": "first"}}
    second = {"id": "second", "slots": 8, "qualification": str(proof), "host": {"endpoint": "second"}}
    atomic(tmp_path / "simulation-hosts.json", {"hosts": [first]})
    slots = LiveSlots(tmp_path)
    release = threading.Event()
    acquired = [threading.Event() for _ in range(16)]

    def holder(i):
        assert slots.acquire(lambda: False)
        acquired[i].set()
        assert release.wait(5)
        slots.release()

    threads = [threading.Thread(target=holder, args=(i,)) for i in range(16)]
    for thread in threads[:8]:
        thread.start()
    assert all(event.wait(3) for event in acquired[:8])
    initial = list(slots.holders.values())
    atomic(tmp_path / "simulation-hosts.json", {"hosts": [first, second]})
    configure(tmp_path, 2, 16)
    for thread in threads[8:]:
        thread.start()
    assert all(event.wait(3) for event in acquired[8:])
    assert len({(h["host_id"], h["slot"]) for h in slots.holders.values()}) == 16
    assert all(h in slots.holders.values() for h in initial)
    atomic(tmp_path / "simulation-hosts.json", {"hosts": [second]})
    time.sleep(0.4)
    assert slots.error and len(slots.holders) == 16
    release.set()
    for thread in threads:
        thread.join(3)
    slots.close()
