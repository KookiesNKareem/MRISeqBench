"""Operational admission for shared native reservations; never billing data."""

import json
import os
import threading
import time
import urllib.request
from collections import deque


class CreditGate:
    def __init__(self, read_balance, margin=2.0):
        self.read_balance = read_balance
        self.margin = margin
        self.condition = threading.Condition()
        self.balance = self.held = self.spent_since_refresh = self.refreshed = 0.0
        self.waiters = deque()

    def acquire(self, amount, allowed):
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            try:
                while allowed():
                    if time.monotonic() - self.refreshed > 15:
                        self.balance = float(self.read_balance())
                        self.spent_since_refresh = 0.0
                        self.refreshed = time.monotonic()
                    if (
                        self.waiters[0] is ticket
                        and self.held + amount
                        <= self.balance - self.margin - self.spent_since_refresh
                    ):
                        self.held += amount
                        return True
                    self.condition.wait(timeout=1)
                return False
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def release(self, amount, observed_cost):
        with self.condition:
            self.held = max(0.0, self.held - amount)
            self.spent_since_refresh += (
                amount if observed_cost is None else observed_cost
            )
            self.condition.notify_all()

    def wake(self):
        with self.condition:
            self.condition.notify_all()


class ProviderAdmission:
    """FIFO reservations governed by real upstream limits, not guessed cash.

    Estimated maximum costs are telemetry only. Auto-top-up users do not need
    to pre-fund every worker's hypothetical maximum completion for admission.
    """

    policy = "provider_feedback"

    def __init__(self, maximum=128, clock=time.monotonic):
        self.maximum = self.capacity = maximum
        self.clock = clock
        self.condition = threading.Condition()
        self.waiters = deque()
        self.inflight = 0
        self.estimated_held = self.observed_spend = 0.0
        self.unreconciled_requests = 0
        self.blocked_until = 0.0
        self.last_reduction = self.last_growth = float("-inf")

    def acquire(self, amount, allowed):
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            try:
                while allowed():
                    if (
                        self.waiters[0] is ticket
                        and self.inflight < self.capacity
                        and self.clock() >= self.blocked_until
                    ):
                        self.inflight += 1
                        self.estimated_held += amount
                        return True
                    self.condition.wait(timeout=1)
                return False
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def release(self, amount, observed_cost):
        with self.condition:
            self.inflight = max(0, self.inflight - 1)
            self.estimated_held = max(0.0, self.estimated_held - amount)
            if observed_cost is None:
                self.unreconciled_requests += 1
            else:
                self.observed_spend += observed_cost
            now = self.clock()
            if (
                self.capacity < self.maximum
                and now - self.last_reduction >= 60
                and now - self.last_growth >= 5
            ):
                self.capacity += 1
                self.last_growth = now
            self.condition.notify_all()

    def rejected(self, delay):
        with self.condition:
            now = self.clock()
            if now >= self.blocked_until:
                self.capacity = max(1, self.capacity // 2)
            self.last_reduction = now
            self.blocked_until = max(self.blocked_until, now + delay)
            self.condition.notify_all()

    def wake(self):
        with self.condition:
            self.condition.notify_all()


_gates = {}
_lock = threading.Lock()


def shared_gate(key):
    with _lock:
        if key not in _gates:
            if os.environ.get("MRISEQBENCH_HAIKU_ADMISSION_POLICY") != "legacy_balance":
                _gates[key] = ProviderAdmission()
                return _gates[key]

            def balance():
                request = urllib.request.Request(
                    "https://openrouter.ai/api/v1/credits",
                    headers={"Authorization": "Bearer " + key},
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = json.load(response)["data"]
                return data["total_credits"] - data["total_usage"]

            _gates[key] = CreditGate(balance)
        return _gates[key]


def haiku_reservation(effective):
    audit = effective.get("context_budget") or {}
    prompt = audit.get("input_token_upper_estimate", 1000000)
    output = effective.get("max_tokens", 128000)
    multiplier = 5 if prompt > 100000 else 1
    return multiplier * (prompt * 0.125 + output * 0.5) / 1000000
