"""A timed-out native request must not corrupt the next case's worker protocol."""

import io
import threading
from types import SimpleNamespace

import pytest

from mriseqbench.evaluation import s1_koma


def test_bart_reconstruction_has_no_default_wall_clock_cutoff(monkeypatch):
    import numpy as np

    from mriseqbench.evaluation import bart

    expected = np.arange(4, dtype=np.complex64)

    def completed(command, **kwargs):
        assert kwargs["timeout"] is None
        bart.write_cfl(command[-1], expected)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bart, "executable", lambda: "/reviewed/bart")
    monkeypatch.setattr(bart.subprocess, "run", completed)
    np.testing.assert_array_equal(
        bart.run(["fft", "-i", "1", "{data}", "{out}"], {"data": expected}),
        expected,
    )


def fake_worker(tmp_path, monkeypatch):
    worker = s1_koma.KomaWorker.__new__(s1_koma.KomaWorker)
    worker.lock = threading.Lock()
    worker.path = tmp_path
    worker.process = SimpleNamespace(
        stdin=io.StringIO(), stdout=object(), poll=lambda: None
    )
    closed = []
    worker.close = lambda: closed.append(True)
    monkeypatch.setattr(s1_koma, "_WORKER", worker)
    return worker, closed


def test_expired_budget_does_not_dispatch_or_discard_idle_worker(tmp_path, monkeypatch):
    worker, closed = fake_worker(tmp_path, monkeypatch)
    with s1_koma.simulation_budget(-1), pytest.raises(TimeoutError):
        worker.call({"operation": "simulate"})
    assert worker.process.stdin.getvalue() == ""
    assert not closed
    assert s1_koma._WORKER is worker
    assert s1_koma._DEADLINE.get() is None


def test_inflight_timeout_discards_worker_before_next_case(tmp_path, monkeypatch):
    worker, closed = fake_worker(tmp_path, monkeypatch)

    class NoReply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def register(self, *_):
            pass

        def select(self, timeout):
            assert 0 < timeout <= 1
            return []

    monkeypatch.setattr(s1_koma.selectors, "DefaultSelector", NoReply)
    with s1_koma.simulation_budget(1), pytest.raises(TimeoutError):
        worker.call({"operation": "simulate"})
    assert '"operation": "simulate"' in worker.process.stdin.getvalue()
    assert closed == [True]
    assert s1_koma._WORKER is None
    assert s1_koma._DEADLINE.get() is None


def test_nested_budget_cannot_extend_parent():
    with s1_koma.simulation_budget(1):
        parent = s1_koma._DEADLINE.get()
        for seconds in (None, 10):
            with s1_koma.simulation_budget(seconds):
                assert s1_koma._DEADLINE.get() == parent
            assert s1_koma._DEADLINE.get() == parent
    assert s1_koma._DEADLINE.get() is None


@pytest.mark.parametrize("native_aborts", [True, False])
def test_control_prefetch_stops_on_native_failure_or_deadline(
    tmp_path, monkeypatch, native_aborts
):
    import time

    import numpy as np

    from mriseqbench.evaluation import s1
    from mriseqbench.evaluation.s1_phantom import Spins

    started, stopped = threading.Event(), threading.Event()
    active, finished = set(), set()
    lock = threading.Lock()

    class Control:
        history = 64
        known = np.ones(1)

        def sampled(self, _spins, *, check):
            ident = threading.get_ident()
            with lock:
                active.add(ident)
                if len(active) == 4:
                    started.set()
            try:
                while True:
                    check()
                    time.sleep(0.001)
            finally:
                with lock:
                    finished.add(ident)
                    if finished == active:
                        stopped.set()

    class Replay:
        hits = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            assert started.wait(2)
            if native_aborts:
                raise TimeoutError("native request expired")
            return np.ones(1)

    case = {
        "task": {
            "id": "caipirinha",
            "sequence": {"matrix": [1, 1, 1], "fov_mm": [200] * 3},
        },
        "evaluation": {
            "id": "s3",
            "version": 3,
            "metrics": [],
            "calibrated": True,
            "spatial_samples": 4,
            "spatial_replicates": 8,
        },
        "object": {"phantom": "synthetic"},
        "physics": {"effects": []},
    }
    spins = Spins(
        np.zeros((4, 3)),
        np.zeros(4, int),
        ["water"],
        np.ones((1, 3)),
        1.0,
        (1, 1, 1),
        "mock",
    )
    acq = SimpleNamespace(checks=[], measurements={}, adc=np.zeros(1))
    control = Control()
    monkeypatch.setattr(s1, "phantom", lambda *_: spins)
    monkeypatch.setattr(s1, "sampled_phantom", lambda *_args, **_kwargs: spins)
    monkeypatch.setattr(s1, "ReplayCache", Replay)
    (tmp_path / "unused.seq").write_text("mocked sequence")
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        s1.evaluate_s1(
            tmp_path / "unused.seq",
            case,
            tmp_path,
            tmp_path / "evaluation",
            deadline=t0 + (5 if native_aborts else 0.5),
            inspector=lambda *_: acq,
            probes=lambda *_: ([], {}, np.ones((1, 1))),
            analytic_sampler=lambda *_: np.ones(1),
            reference_builder=lambda *_: np.ones((1, 1, 1)),
            reconstructor=lambda raw, *_: np.reshape(raw, (1, 1, -1)),
            sampler=lambda *_: pytest.fail("unused Fourier control must not run"),
            control_builder=lambda *_: (control, control),
        )
    assert time.monotonic() - t0 < 2
    assert stopped.wait(2), "analytical work leaked past case cancellation"
    assert len(active) == len(finished) == 4
