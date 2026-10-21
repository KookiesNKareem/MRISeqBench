"""Host-side usage receipts. Missing provider counters are unknown, never zero."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path


def utc_now():
    return datetime.now(UTC).isoformat()


def number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        valid = (
            math.isfinite(value) and value >= 0 and (not integer or int(value) == value)
        )
    except OverflowError:
        return None
    if not valid:
        return None
    return int(value) if integer else value


def normalize_usage(usage):
    """Chat and Responses names describe the same native-token counters.

    Cache reads/writes are input details; reasoning is an output detail. Neither
    is added to the corresponding total a second time.
    """

    def field(*names):
        return next((usage[n] for n in names if n in usage), None)

    prompt = field("input_tokens_details", "prompt_tokens_details") or {}
    completion = field("output_tokens_details", "completion_tokens_details") or {}
    prompt = prompt if isinstance(prompt, dict) else {}
    completion = completion if isinstance(completion, dict) else {}
    inputs = number(field("input_tokens", "prompt_tokens"), integer=True)
    outputs = number(field("output_tokens", "completion_tokens"), integer=True)
    total = number(usage.get("total_tokens"), integer=True)
    if total is None and inputs is not None and outputs is not None:
        total = inputs + outputs
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": total,
        "cached_input_tokens": number(prompt.get("cached_tokens"), integer=True),
        "cache_write_tokens": number(prompt.get("cache_write_tokens"), integer=True),
        "reasoning_tokens": number(completion.get("reasoning_tokens"), integer=True),
        "cost_credits": number(usage.get("cost")),
    }


class UsageCapture:
    """Inspect JSON/SSE incrementally without buffering the response or saving text."""

    limit = 2 * 1024 * 1024

    def __init__(self, content_type):
        self.sse = "text/event-stream" in content_type.lower()
        self.buffer = b""
        self.data = []
        self.data_size = 0
        self.overflow = False
        self.parse_error = False
        self.generation_id = None
        self.model = None
        self.provider = None
        self.usage = None
        self.provider_error = False
        self.provider_error_code = None
        self.provider_error_type = None

    def payload(self, raw):
        if raw.strip() == b"[DONE]":
            return
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            self.parse_error = True
            return
        if not isinstance(value, dict):
            return
        # Responses streaming terminal events put counters inside response.
        if isinstance(value.get("response"), dict):
            value = value["response"]
        if value.get("error") is not None or value.get("status") == "failed":
            self.provider_error = True
            error = value.get("error")
            if isinstance(error, dict):
                self.provider_error_code = number(error.get("code"), integer=True)
                metadata = error.get("metadata") or {}
                if isinstance(metadata, dict):
                    provider = metadata.get("provider_name")
                    error_type = metadata.get("error_type")
                    if isinstance(provider, str):
                        self.provider = provider[:512]
                    if (
                        isinstance(error_type, str)
                        and error_type.replace("_", "").isalnum()
                    ):
                        self.provider_error_type = error_type[:128]
        for key, attr in (
            ("id", "generation_id"),
            ("model", "model"),
            ("provider", "provider"),
        ):
            if isinstance(value.get(key), str):
                setattr(self, attr, value[key][:512])
        if isinstance(value.get("usage"), dict):
            # Streaming counters are cumulative, not additive within a request.
            usage = normalize_usage(value["usage"])
            if self.usage is None:
                self.usage = usage
            else:
                self.usage.update({k: v for k, v in usage.items() if v is not None})

    def event(self):
        if self.data:
            self.payload(b"\n".join(self.data))
        self.data, self.data_size = [], 0

    def feed(self, chunk):
        if self.overflow:
            return
        self.buffer += chunk
        if self.sse:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.removesuffix(b"\r")
                if not line:
                    self.event()
                elif line.startswith(b"data:"):
                    line = line[5:].removeprefix(b" ")
                    self.data_size += len(line)
                    if self.data_size > self.limit:
                        self.overflow = True
                        self.data, self.buffer = [], b""
                        return
                    self.data.append(line)
        if len(self.buffer) > self.limit:
            self.overflow = True
            self.buffer, self.data = b"", []

    def finish(self):
        if self.overflow:
            return
        if self.sse:
            if self.buffer:
                self.feed(b"\n")
            self.event()
        elif self.buffer:
            self.payload(self.buffer)
        self.buffer = b""

    def receipt(self):
        return {
            "generation_id": self.generation_id,
            "model": self.model,
            "provider": self.provider,
            "usage": dict(self.usage) if self.usage is not None else None,
            "usage_parse_error": self.parse_error or self.overflow,
            "provider_error": self.provider_error,
            "provider_error_code": self.provider_error_code,
            "provider_error_type": self.provider_error_type,
        }


def read_api(path):
    """Join started/progress/finished journal rows by request id."""
    requests = {}
    path = Path(path)
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # A partial journal line cannot manufacture usage.
                if not isinstance(row, dict) or row.get("route") not in {
                    "/chat/completions",
                    "/responses",
                }:
                    continue
                key = row.get("request_id", f"legacy-{len(requests)}")
                requests.setdefault(key, {}).update(row)
                requests[key]["request_id"] = key
                if row.get("event") == "finished":
                    requests[key]["transport_finished"] = True
    return list(requests.values())


def usage_complete(row):
    return (
        row.get("transport_finished") is True
        and row.get("response_complete") is True
        and not row.get("usage_parse_error")
        and all(
            (row.get("usage") or {}).get(f) is not None
            for f in ("input_tokens", "output_tokens", "total_tokens")
        )
    )


def summarize_api(path, *, configured=True):
    """Join started/finished records, retaining interrupted requests in the denominator."""
    rows = read_api(path)
    fields = tuple(normalize_usage({}))
    observed = {}
    totals = {}
    for field in fields:
        values = [(row.get("usage") or {}).get(field) for row in rows]
        known = [v for v in values if v is not None]
        observed[field] = (
            sum(known) if known else (0 if configured and not rows else None)
        )
        totals[field] = observed[field] if len(known) == len(rows) else None
    complete = sum(usage_complete(row) for row in rows)
    if complete != len(rows):
        totals = dict.fromkeys(fields)
    return {
        "source": "host_api_proxy" if configured else None,
        "status": "not_configured"
        if not configured
        else "no_requests"
        if not rows
        else "complete"
        if complete == len(rows)
        else "incomplete",
        "model_requests": len(rows),
        "requests_with_complete_usage": complete,
        "requests_missing_usage": len(rows) - complete,
        "in_flight_requests": sum(not row.get("transport_finished") for row in rows),
        "resolved_providers": sorted(
            {row["provider"] for row in rows if row.get("provider")}
        ),
        "provider_resolution_complete": all(row.get("provider") for row in rows),
        "response_cache_hits": sum(
            (row.get("response_cache_status") or "").upper() == "HIT" for row in rows
        ),
        "cached_input_fraction": totals["cached_input_tokens"] / totals["input_tokens"]
        if totals["cached_input_tokens"] is not None and totals["input_tokens"]
        else None,
        "totals": totals,
        "observed_totals": observed,
        "generation_ids": list(
            dict.fromkeys(
                row["generation_id"] for row in rows if row.get("generation_id")
            )
        ),
    }


def summarize_trials(directory):
    """Aggregate trial receipts, including interrupted trials without verdicts."""
    trials = [
        json.loads(p.read_text()) for p in Path(directory).glob("*/accounting.json")
    ]
    usage = [trial["usage"] for trial in trials if "usage" in trial]
    fields = tuple(normalize_usage({}))

    def total(rows, field):
        values = [row.get(field) for row in rows]
        return sum(values) if values and all(v is not None for v in values) else None

    return {
        "trials_started": len(trials),
        "trials_finished": sum(t.get("status") == "finished" for t in trials),
        "agent_wall_s": total(trials, "agent_wall_s"),
        "agent_charged_wall_s": total(trials, "agent_charged_wall_s"),
        "grading_wall_s": total(trials, "grading_wall_s"),
        "usage": {
            "status": "no_trials"
            if not trials
            else "complete"
            if len(usage) == len(trials)
            and all(u["status"] in {"complete", "no_requests"} for u in usage)
            else "incomplete",
            "model_requests": sum(u["model_requests"] for u in usage),
            "requests_missing_usage": sum(u["requests_missing_usage"] for u in usage),
            "totals": {
                field: total([u["totals"] for u in usage], field)
                if len(usage) == len(trials)
                else None
                for field in fields
            },
            "observed_totals": {
                field: sum(values)
                if (
                    values := [
                        u["observed_totals"][field]
                        for u in usage
                        if u["observed_totals"][field] is not None
                    ]
                )
                else None
                for field in fields
            },
        },
    }
