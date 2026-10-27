import copy
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler

import pytest

from mriseqbench.harness.proxy import ContextBudget, configure_request
from mriseqbench.models import ApiProxyConfig


def test_local_budget_rejection_is_durable_infrastructure_event(tmp_path, descriptor):
    import urllib.error

    from mriseqbench.harness.proxy import ApiProxy

    descriptor["test/model"].update(contextWindow=1000, maxOutputTokens=900)
    log = tmp_path / "api.jsonl"
    config = ApiProxyConfig.model_construct(
        reasoning_effort="high",
        provider_endpoint="test",
        max_output_tokens=None,
        allowed_routes=["/chat/completions"],
    )
    with ApiProxy(config, log, model="test/model", test_key="test") as proxy:
        req = urllib.request.Request(
            proxy.base_url + "/chat/completions",
            data=json.dumps(
                {
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "x" * 2000}],
                }
            ).encode(),
            headers={
                "Authorization": "Bearer " + proxy.token,
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req, timeout=5)
        assert error.value.code == 400
        assert proxy.exhausted == "context_budget"
        assert not proxy.allowed()
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows[-1]["event"] == "local_budget_rejected"
    assert rows[-1]["upstream_sent"] is False


@pytest.fixture
def descriptor(monkeypatch):
    from pathlib import Path

    from mriseqbench.harness import proxy

    original = Path.read_text
    metadata = {"test/model": {"contextWindow": 262144, "maxOutputTokens": 235929}}

    def read(path, *args, **kwargs):
        if path == Path(proxy.__file__).with_name("pi_models.json"):
            return json.dumps(metadata)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return metadata


def assert_fits(budget, payload, route="/chat/completions"):
    audit = budget.audit
    key = "max_output_tokens" if route == "/responses" else "max_tokens"
    assert 0 < payload[key] <= audit["configured_output_ceiling"]
    assert (
        payload[key] + audit["input_token_upper_estimate"] + audit["safety_tokens"]
        <= audit["context_window"]
    )


@pytest.mark.parametrize(
    "text_tokens,tool_tokens,native,context",
    [
        (26659, 699, 235929, 262144),
        (26260, 699, 235929, 262144),
        (26268, 699, 235929, 262144),
        (25220, 699, 230400, 256000),
    ],
)
def test_observed_context_rejections_fit_after_budgeting(
    descriptor, text_tokens, tool_tokens, native, context
):
    # Reproduce all four disclosed rejection counts. One-byte text is also a
    # conservative stress case: the reserve cannot assume four chars/token.
    payload = {
        "messages": [{"role": "user", "content": "x" * text_tokens}],
        "tools": [{"description": "t" * tool_tokens}],
    }
    descriptor["test/model"].update(contextWindow=context, maxOutputTokens=native)
    budget = ContextBudget()
    result = budget.fit(payload, "test/model", native, "/chat/completions")
    assert (
        result + text_tokens + tool_tokens + max(2048, (context + 99) // 100) <= context
    )
    assert result < native


@pytest.mark.parametrize("route", ["/chat/completions", "/responses"])
def test_input_and_tools_are_preserved_while_only_output_changes(descriptor, route):
    key = "input" if route == "/responses" else "messages"
    payload = {
        key: [{"role": "user", "content": "MRI 🧠" * 12000}],
        "tools": [{"name": "simulate", "parameters": {"type": "object"}}],
    }
    original = copy.deepcopy(payload)
    budget = ContextBudget()
    configure_request(payload, "test/model", ApiProxyConfig(), route, budget=budget)
    assert payload[key] == original[key]
    assert payload["tools"] == original["tools"]
    assert_fits(budget, payload, route)


def test_successful_usage_anchors_only_identical_prefixes_and_tools(descriptor):
    payload = {
        "messages": [{"role": "user", "content": "a" * 150000}],
        "tools": [{"name": "simulate"}],
    }
    budget = ContextBudget()
    budget.observe(payload, "/chat/completions", {"input_tokens": 40000})
    payload["messages"].append({"role": "tool", "content": "Δ" * 2000})
    configure_request(payload, "test/model", ApiProxyConfig(), budget=budget)
    assert budget.audit["method"] == "provider_prompt_usage_plus_utf8_delta"
    assert budget.audit["input_token_upper_estimate"] >= 50000 + 4000
    assert_fits(budget, payload)
    payload["tools"].append({"name": "new_tool", "description": "x" * 10000})
    configure_request(payload, "test/model", ApiProxyConfig(), budget=budget)
    assert budget.audit["method"] == "serialized_utf8_bound"
    assert_fits(budget, payload)
    payload["messages"][0]["content"] = "changed prefix"
    configure_request(payload, "test/model", ApiProxyConfig(), budget=budget)
    assert budget.audit["method"] == "serialized_utf8_bound"


def test_unknown_usage_never_replaces_a_valid_anchor(descriptor):
    payload = {"messages": [{"role": "user", "content": "x" * 40000}]}
    budget = ContextBudget()
    budget.observe(payload, "/chat/completions", {"input_tokens": 10000})
    anchor = budget.previous
    for usage in [None, {}, {"input_tokens": None}, {"input_tokens": -1}]:
        budget.observe(payload, "/chat/completions", usage)
        assert budget.previous == anchor


def test_native_and_explicit_ceilings_remain_ceilings(descriptor):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    configure_request(payload, "test/model", ApiProxyConfig())
    assert payload["max_tokens"] == 235929
    configure_request(payload, "test/model", ApiProxyConfig(max_output_tokens=12345))
    assert payload["max_tokens"] == 12345


def test_input_exhaustion_is_rejected_without_prompt_truncation(descriptor):
    payload = {"messages": [{"role": "user", "content": "x" * 263000}]}
    original = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="no safe output context budget"):
        configure_request(payload, "test/model", ApiProxyConfig())
    assert payload["messages"] == original["messages"]


def test_anthropic_cache_normalization_keeps_the_usage_prefix_stable(descriptor):
    descriptor["anthropic/test"] = descriptor["test/model"].copy()
    client = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "x" * 40000,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "tools": [{"name": "simulate", "cache_control": {"type": "ephemeral"}}],
    }
    budget = ContextBudget()
    first = copy.deepcopy(client)
    configure_request(first, "anthropic/test", ApiProxyConfig(), budget=budget)
    budget.observe(first, "/chat/completions", {"input_tokens": 10000})
    second = copy.deepcopy(client)
    second["messages"].append({"role": "assistant", "content": "reply"})
    configure_request(second, "anthropic/test", ApiProxyConfig(), budget=budget)
    assert budget.audit["method"] == "provider_prompt_usage_plus_utf8_delta"
    assert "cache_control" not in second["tools"][0]
    assert "cache_control" not in second["messages"][0]["content"][0]
    assert second["cache_control"] == {"type": "ephemeral"}
    assert_fits(budget, second)


def test_proxy_uses_completed_usage_for_the_next_request(descriptor, tmp_path):
    from mriseqbench.harness.proxy import ApiProxy, QuietHTTPServer

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(
                {"usage": {"prompt_tokens": 20000, "completion_tokens": 1}}
            ).encode()
            self.send_response(200)
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
    try:
        with ApiProxy(config, log, model="test/model", test_key="k") as proxy:
            payload = {
                "model": "test/model",
                "messages": [{"role": "user", "content": "x" * 60000}],
            }
            for _ in range(2):
                request = urllib.request.Request(
                    proxy.base_url + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": f"Bearer {proxy.token}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
                assert proxy.settle(5)
                payload["messages"].append({"role": "tool", "content": "result"})
            assert proxy.context_budget.previous is not None
    finally:
        server.shutdown()
    starts = [
        json.loads(line)
        for line in log.read_text().splitlines()
        if json.loads(line)["event"] == "started"
    ]
    assert (
        starts[0]["effective_request"]["context_budget"]["method"]
        == "serialized_utf8_bound"
    )
    assert (
        starts[1]["effective_request"]["context_budget"]["method"]
        == "provider_prompt_usage_plus_utf8_delta"
    )
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
    assert calls[1]["max_tokens"] <= 235929
