"""Model routing, refused-request detection and simulator-queue accounting."""

import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from types import SimpleNamespace

import pytest

from mriseqbench.harness import arm_runtime
from mriseqbench.harness.accounting import read_api
from mriseqbench.harness.feedback import FeedbackServer
from mriseqbench.harness.prepare import pi_model, prepare_pi
from mriseqbench.harness.proxy import ApiProxy
from mriseqbench.harness.runner import proxy_refusal
from mriseqbench.models import Agent, ApiProxyConfig


@pytest.mark.parametrize(
    ("model", "route"),
    [
        ("openai/gpt-6.1-sol", "/responses"),
        ("openai/gpt-6-luna", "/responses"),
        ("anthropic/claude-opus-5.5", "/chat/completions"),
        ("anthropic/claude-haiku-5.5", "/chat/completions"),
    ],
)
def test_pi_descriptor_pins_api_and_requires_its_route(tmp_path, model, route):
    config = ApiProxyConfig(allowed_routes=[route], max_output_tokens=16384)
    agent = Agent(command=["pi"], model=model)
    prepare_pi(tmp_path, "http://127.0.0.1:8765/api", agent, config)
    provider = json.loads((tmp_path / ".pi-agent/models.json").read_text())[
        "providers"
    ]["openrouter"]
    (descriptor,) = provider["models"]
    assert descriptor["id"] == model and descriptor["maxTokens"] == 16384
    assert provider["api"] == descriptor["api"]
    other = "/chat/completions" if route == "/responses" else "/responses"
    with pytest.raises(ValueError, match="allowed_routes"):
        pi_model(model, ApiProxyConfig(allowed_routes=[other]))
    # No proxy cap: Pi is told the endpoint's own maximum instead.
    uncapped = pi_model(model, ApiProxyConfig(allowed_routes=[route]))
    assert uncapped["maxTokens"] == 128000 and "maxOutputTokens" not in uncapped


def test_unknown_pi_model_fails_before_launch():
    with pytest.raises(ValueError, match="no Pi model descriptor"):
        pi_model("openai/unknown", ApiProxyConfig())


def test_refused_route_is_journaled_and_fails_loudly(tmp_path):
    log = tmp_path / "api.jsonl"
    config = ApiProxyConfig(allowed_routes=["/responses"])
    with ApiProxy(config, log, model="openai/gpt-6.1-sol", test_key="k") as proxy:
        request = urllib.request.Request(
            proxy.base_url + "/chat/completions",
            data=b'{"model": "openai/gpt-6.1-sol"}',
            headers={"Authorization": "Bearer " + proxy.token},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 403
    assert read_api(log) == []  # no model request, no usage to account
    assert "route not allowed" in proxy_refusal(log)
    assert proxy_refusal(tmp_path / "missing.jsonl") is None


def excluder():
    return SimpleNamespace(
        retry_time_lock=threading.Lock(),
        backoff_seconds=0.0,
        active=Counter(),
        since={},
        excluded_seconds=0.0,
        excluded_by_kind={"grading": 0.0, "simulator_queue": 0.0},
    )


def test_overlapping_excluded_waits_are_credited_once():
    server = excluder()
    with FeedbackServer.excluded(server, "simulator_queue"):
        time.sleep(0.05)
        with FeedbackServer.excluded(server, "grading"):
            time.sleep(0.05)
        assert FeedbackServer.grading_credit(server) >= 0.1
    credit = FeedbackServer.grading_credit(server)
    assert 0.1 <= credit < 0.2  # union of the intervals, not their sum
    assert server.excluded_by_kind["simulator_queue"] >= 0.1
    assert server.excluded_by_kind["grading"] >= 0.05


def test_simulator_queue_wait_is_excluded_and_recorded(tmp_path, monkeypatch):
    server = excluder()
    (tmp_path / "simulation_inputs").mkdir()
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name=None: (["docker", "--rm"], image),
    )
    monkeypatch.setattr(arm_runtime, "configure_simulator_args", lambda args: args)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: True)
    monkeypatch.setattr(
        arm_runtime, "docker_output", lambda args, timeout=120: '{"ExitCode": 1}'
    )
    running = []

    def execute(args, cwd, timeout, prefix, **_kwargs):
        running.append(server.active[None])  # the queue credit has ended
        for suffix in (".stdout.log", ".stderr.log"):
            prefix.with_name(prefix.name + suffix).write_text("")
        return {"status": "failed"}

    monkeypatch.setattr(arm_runtime, "execute", execute)
    slots = arm_runtime.SimulatorSlots(1)
    simulator = arm_runtime.Simulator(
        {"simulator_image": "sim", "arm": {"simulator": "koma"}},
        SimpleNamespace(simulation_timeout_s=10),
        tmp_path,
        tmp_path,
        {},
        slots=slots,
        waiting=lambda: FeedbackServer.excluded(server, "simulator_queue"),
    )
    slots.free.acquire()  # another trial holds the only GPU slot
    threading.Timer(0.2, slots.free.release).start()
    status, body = simulator(b"seq")
    assert status == 422 and running == [0]
    assert body["queue_wait_s"] >= 0.15
    assert server.excluded_by_kind["simulator_queue"] >= 0.15
    assert slots.free.acquire(blocking=False)  # released after execution


def test_concurrent_waits_of_one_kind_are_counted_once():
    server = excluder()
    workers = [
        threading.Thread(target=lambda: _wait(server, "simulator_queue"))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    credit = FeedbackServer.grading_credit(server)
    queue = server.excluded_by_kind["simulator_queue"]
    assert queue <= credit + 1e-3 and queue < 0.15


def _wait(server, kind):
    with FeedbackServer.excluded(server, kind):
        time.sleep(0.1)
