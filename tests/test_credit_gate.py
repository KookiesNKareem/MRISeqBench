import threading

from mriseqbench.harness.credit_gate import (
    CreditGate,
    ProviderAdmission,
    haiku_reservation,
)


def test_shared_reservations_block_then_release_without_changing_request():
    gate = CreditGate(lambda: 10, margin=2)
    assert gate.acquire(7, lambda: True)
    admitted = threading.Event()
    worker = threading.Thread(
        target=lambda: (gate.acquire(2, lambda: True), admitted.set())
    )
    worker.start()
    assert not admitted.wait(0.1)
    gate.release(7, 0.1)
    assert admitted.wait(2)
    worker.join()
    assert gate.held == 2


def test_credit_wait_cancels_and_long_context_uses_write_tier():
    gate = CreditGate(lambda: 0)
    assert gate.acquire(1, lambda: False) is False
    short = {
        "max_tokens": 128000,
        "context_budget": {"input_token_upper_estimate": 20000},
    }
    long = {
        "max_tokens": 128000,
        "context_budget": {"input_token_upper_estimate": 200000},
    }
    assert haiku_reservation(short) == 0.0665
    assert haiku_reservation(long) == 0.445


def test_auto_topup_admission_does_not_require_worst_case_cash_up_front():
    gate = ProviderAdmission()
    for _ in range(128):
        assert gate.acquire(1000000, lambda: True)
    assert gate.inflight == 128 and gate.capacity == 128
    for _ in range(128):
        gate.release(1000000, 0.01)
    assert gate.inflight == 0 and gate.estimated_held == 0
    assert abs(gate.observed_spend - 1.28) < 1e-10


def test_real_provider_backpressure_reduces_calls_without_killing_holders():
    now = [0.0]
    gate = ProviderAdmission(maximum=4, clock=lambda: now[0])
    for _ in range(4):
        assert gate.acquire(1, lambda: True)
    gate.rejected(10)
    assert gate.capacity == 2 and gate.inflight == 4
    gate.release(1, 0.01)
    gate.release(1, None)
    assert gate.inflight == 2 and gate.unreconciled_requests == 1
    now[0] = 11
    gate.release(1, 0.02)
    assert gate.acquire(1, lambda: True)
    assert gate.capacity == 2 and gate.inflight == 2


def test_default_shared_policy_does_not_query_prepaid_balance(monkeypatch):
    from mriseqbench.harness import credit_gate

    monkeypatch.delenv("MRISEQBENCH_HAIKU_ADMISSION_POLICY", raising=False)
    monkeypatch.setattr(
        credit_gate.urllib.request,
        "urlopen",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("no balance gate")),
    )
    gate = credit_gate.shared_gate("auto-topup-no-credit-fetch-control")
    assert isinstance(gate, ProviderAdmission)
    assert gate.acquire(1000, lambda: True)
    gate.release(1000, 0.01)
