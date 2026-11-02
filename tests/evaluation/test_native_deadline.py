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
