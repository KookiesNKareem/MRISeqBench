import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest

from mriseqbench.harness.proxy import ApiProxy, QuietHTTPServer
from mriseqbench.models import ApiProxyConfig

BUDGET_ERROR = {
    "error": {
        "message": "Retry after in-flight requests settle",
        "code": 402,
        "metadata": {"reason": "in_flight_budget_exhausted"},
    }
}


@pytest.mark.parametrize("kind", ["recovers", "keeps_rejecting", "permanent", "revoke"])
def test_temporary_credit_reservations_do_not_stop_the_campaign(
    tmp_path, monkeypatch, kind
):
    calls = []
    body402 = json.dumps(
        BUDGET_ERROR
        if kind != "permanent"
        else {"error": {"message": "Insufficient account credits", "code": 402}}
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            calls.append(1)
            status = 200 if kind == "recovers" and len(calls) > 1 else 402
            body = b"{}" if status == 200 else body402
            self.send_response(status)
            self.send_header("Retry-After", "120")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{server.server_address[1]}/api/v1", timeout_s=10
    )
    log = tmp_path / "api.jsonl"
    waits = []
    pauses = []

    @contextmanager
    def pause():
        pauses.append(True)
        yield

    try:
        with ApiProxy(config, log, test_key="k", pause=pause) as proxy:

            def wait(seconds):
                waits.append(seconds)
                if kind == "revoke":
                    proxy.revoke()
                return proxy.revoked.is_set()

            monkeypatch.setattr(proxy.revoked, "wait", wait)
            request = urllib.request.Request(
                proxy.base_url + "/models",
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            try:
                response = urllib.request.urlopen(request, timeout=10)
            except urllib.error.HTTPError as exc:
                response = exc
            body = response.read()
            assert proxy.settle(5)
            if kind == "permanent":
                assert response.code == 402 and body == body402
                assert proxy.exhausted == "account" and len(calls) == 1
            else:
                assert proxy.exhausted is None
                assert waits and all(delay == 120 for delay in waits)
                assert len(pauses) == len(waits)
                if kind == "recovers":
                    assert response.code == 200 and body == b"{}" and len(calls) == 2
                    assert proxy.allowed()
                elif kind == "keeps_rejecting":
                    assert response.code == 402 and body == body402
                    assert len(calls) == 5 and len(waits) == 4 and proxy.allowed()
                else:
                    assert response.code == 502 and len(calls) == 1
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        retries = [row for row in rows if row["event"] == "upstream_retry"]
        assert len(retries) == len(waits)
        assert all(row["reason"] == "in_flight_budget_exhausted" for row in retries)
        assert len([row for row in rows if row["event"] == "finished"]) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_budget_classification_requires_the_explicit_provider_reason():
    from mriseqbench.harness.proxy import in_flight_budget_error

    assert in_flight_budget_error(json.dumps(BUDGET_ERROR).encode())
    assert in_flight_budget_error(json.dumps(BUDGET_ERROR["error"]).encode())
    for value in (
        b"",
        b"[]",
        b"null",
        b"{}",
        b"not json",
        b'{"error":null}',
        b'{"error":{"message":"in-flight requests, but account empty"}}',
    ):
        assert not in_flight_budget_error(value)


def upstream_503():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            calls.append(time.monotonic())
            body = b"{}"
            self.send_response(503)
            self.send_header("Retry-After", "60")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def test_revoke_ends_back_off_without_another_send(tmp_path):
    upstream, calls = upstream_503()
    config = ApiProxyConfig.model_construct(  # Plain HTTP for a local upstream.
        upstream_url=f"http://127.0.0.1:{upstream.server_address[1]}/api/v1",
        timeout_s=10,
    )
    paused, waited = [], []

    @contextmanager
    def pause():
        paused.append(time.monotonic())
        yield
        waited.append(time.monotonic() - paused[-1])

    log = tmp_path / "api.jsonl"
    with ApiProxy(config, log, test_key="k", pause=pause) as proxy:
        reply = {}

        def client():
            request = urllib.request.Request(
                proxy.base_url + "/models",
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            try:
                urllib.request.urlopen(request, timeout=30)
            except urllib.error.HTTPError as exc:
                reply["status"] = exc.code

        thread = threading.Thread(target=client)
        thread.start()
        deadline = time.monotonic() + 10
        while not paused and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(paused) == 1  # In back-off (Retry-After: 60) after the 503.
        revoked = time.monotonic()
        proxy.revoke()
        assert proxy.settle(5)
        assert time.monotonic() - revoked < 5
        thread.join(5)
        assert not proxy.allowed()
    upstream.shutdown()
    assert len(calls) == 1 and reply["status"] == 502
    assert waited and waited[0] < 5  # Only the time actually waited is excluded.
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "upstream_retry", "finished"]


def test_stop_event_disallows_new_sends(tmp_path):
    stop = threading.Event()
    config = ApiProxyConfig.model_construct()
    with ApiProxy(config, tmp_path / "api.jsonl", test_key="k", stop_when=stop) as p:
        assert p.allowed()
        stop.set()
        assert not p.allowed() and not p.begin("r", None, "/models")
        assert p.settle(0)


def test_sampling_is_host_owned_and_omittable():
    from mriseqbench.harness.proxy import configure_request

    payload = {"temperature": 1.0, "top_p": 0.1, "top_k": 5, "max_output_tokens": 10}
    configure_request(payload, "openai/gpt-6-luna", ApiProxyConfig(), "/responses")
    assert payload == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 5,
        "max_output_tokens": 128000,
    }
    config = ApiProxyConfig(temperature=None, top_p=None)
    payload = {"temperature": 1.0, "top_p": 0.1, "max_tokens": 10}
    # No configured cap: the endpoint maximum is sent, never an omitted field
    # (OpenRouter then defaulted anthropic/* to 65536, truncating reasoning).
    assert configure_request(payload, "openai/gpt-6-luna", config) == {
        "max_tokens": 128000
    }
    assert configure_request(
        {"max_tokens": 10}, "anthropic/claude-haiku-5.5", config
    ) == {"max_tokens": 128000, "cache_control": {"type": "ephemeral"}}
    assert configure_request({"max_tokens": 10}, "openai/unknown", config) == {}
    config = ApiProxyConfig(temperature=None, top_p=None, max_output_tokens=16384)
    payload = {"max_tokens": 10, "max_completion_tokens": 10}
    assert configure_request(payload, "openai/gpt-6-luna", config, "/responses") == {
        "max_output_tokens": 16384
    }
    assert configure_request({}, "openai/gpt-6-luna", config) == {"max_tokens": 16384}


def test_anthropic_requests_get_one_host_cache_breakpoint():
    from mriseqbench.harness.proxy import configure_request

    payload = {
        "top_k": 5,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "s",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {"role": "user", "content": "u", "cache_control": {"type": "ephemeral"}},
        ],
        "tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}],
    }
    config = ApiProxyConfig(temperature=None, top_p=None)
    configure_request(payload, "anthropic/claude-haiku-5.5", config)
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert "top_k" not in payload and "temperature" not in payload
    assert json.dumps(payload).count("cache_control") == 1


def upstream_once(raw):
    """A raw HTTP upstream that sends `raw` and closes the connection."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def serve():
        conn, _ = listener.accept()
        conn.recv(65536)
        conn.sendall(raw)
        conn.close()
        listener.close()

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def proxied(tmp_path, raw):
    port = upstream_once(raw)
    config = ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{port}/api/v1", timeout_s=10
    )
    log = tmp_path / "api.jsonl"
    with ApiProxy(config, log, test_key="k") as proxy:
        request = urllib.request.Request(
            proxy.base_url + "/models",
            headers={"Authorization": f"Bearer {proxy.token}"},
        )
        try:
            urllib.request.urlopen(request, timeout=10).read()
        except Exception:  # noqa: BLE001, S110 - truncation may surface either way
            pass
        assert proxy.settle(5)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return next(row for row in rows if row["event"] == "finished")


SSE = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"


def test_truncated_content_length_body_is_incomplete(tmp_path):
    body = b'data: {"id": "g", "choices": [{"delta": {"content": "par'
    row = proxied(tmp_path, SSE + b"Content-Length: 9999\r\n\r\n" + body)
    assert row["status"] == 200 and row["response_complete"] is False
    assert row["cancelled"] is False


def test_sse_without_terminal_event_is_incomplete(tmp_path):
    body = b'data: {"id": "g", "choices": []}\n\n'
    row = proxied(tmp_path, SSE + b"Connection: close\r\n\r\n" + body)
    assert row["response_complete"] is False and row["protocol_complete"] is False


def test_sse_with_done_is_complete(tmp_path):
    body = b'data: {"id": "g", "choices": []}\n\ndata: [DONE]\n\n'
    row = proxied(tmp_path, SSE + b"Connection: close\r\n\r\n" + body)
    assert row["response_complete"] is True


def test_responses_terminal_event_completes_stream():
    from mriseqbench.harness.accounting import UsageCapture

    capture = UsageCapture("text/event-stream")
    capture.feed(b'data: {"type": "response.output_text.delta"}\n\n')
    assert not capture.terminal
    capture.feed(b'data: {"type": "response.completed", "response": {}}\n\n')
    capture.finish()
    assert capture.terminal


@pytest.mark.parametrize("end", ["revoke", "stop"])
def test_access_ending_during_connect_sends_nothing(tmp_path, monkeypatch, end):
    import http.client

    upstream, calls = upstream_503()
    config = ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{upstream.server_address[1]}/api/v1",
        timeout_s=10,
    )
    connecting, resume = threading.Event(), threading.Event()
    connect = http.client.HTTPConnection.connect

    def slow_connect(self):
        if self.port == upstream.server_address[1]:  # Not the client's connection.
            connecting.set()
            resume.wait(10)
        connect(self)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", slow_connect)
    stop = threading.Event()
    log = tmp_path / "api.jsonl"
    with ApiProxy(config, log, test_key="k", stop_when=stop) as proxy:
        reply = {}

        def client():
            request = urllib.request.Request(
                proxy.base_url + "/models",
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            try:
                urllib.request.urlopen(request, timeout=30)
            except urllib.error.HTTPError as exc:
                reply["status"] = exc.code

        thread = threading.Thread(target=client)
        thread.start()
        assert connecting.wait(10)
        proxy.revoke() if end == "revoke" else stop.set()
        resume.set()
        thread.join(10)
        assert proxy.settle(5)
    upstream.shutdown()
    assert calls == [] and reply["status"] == 502
