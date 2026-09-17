import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_request_timing: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_timing", default=None
)


def _record_timing(name: str, elapsed_ms: float, **details: Any) -> None:
    """Attach a timing event to the active request, if there is one."""
    collector = _request_timing.get()
    if collector is None:
        return
    collector["events"].append(
        {
            "name": name,
            "elapsed_ms": round(elapsed_ms, 1),
            **details,
        }
    )


def _usage_metadata(response: Any) -> dict[str, int]:
    """Normalize token counters exposed by LangChain model responses."""

    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = (
        response_metadata.get("token_usage")
        or response_metadata.get("usage_metadata")
        or {}
    )

    prompt_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("prompt_token_count")
        or 0
    )
    completion_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("candidates_token_count")
        or 0
    )
    total_tokens = (
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or token_usage.get("total_token_count")
        or int(prompt_tokens) + int(completion_tokens)
    )
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def _record_llm_usage(usage: dict[str, int]) -> None:
    collector = _request_timing.get()
    if collector is None:
        return
    totals = collector["token_usage"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        totals[key] += usage.get(key, 0)


def collected_token_usage(collector: dict[str, Any]) -> dict[str, int]:
    usage = collector.get("token_usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


@contextmanager
def timed_section(name: str, **details: Any) -> Iterator[None]:
    """Measure a non-LLM step within a request."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _record_timing(name, (time.perf_counter() - start) * 1000, **details)


@contextmanager
def request_timing(**details: Any) -> Iterator[dict[str, Any]]:
    """Collect and emit a compact, structured latency record for one request."""
    collector: dict[str, Any] = {
        "request_id": uuid.uuid4().hex[:12],
        "started_at": time.perf_counter(),
        "events": [],
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        **details,
    }
    token = _request_timing.set(collector)
    try:
        yield collector
    finally:
        elapsed_ms = (time.perf_counter() - collector["started_at"]) * 1000
        payload = {
            key: value
            for key, value in collector.items()
            if key != "started_at"
        }
        payload["total_ms"] = round(elapsed_ms, 1)
        print(f"[Request Timing] {json.dumps(payload, ensure_ascii=False, default=str)}")
        _request_timing.reset(token)


def _content_char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_char_count(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return len(value["text"])
        return len(json.dumps(value, ensure_ascii=False, default=str))
    content = getattr(value, "content", None)
    if content is not None:
        return _content_char_count(content)
    return len(str(value))


def prompt_char_count(prompt: Any) -> int:
    return _content_char_count(prompt)


def timed_llm_invoke(model: Any, prompt: Any, label: str) -> Any:
    start = time.perf_counter()
    prompt_chars = prompt_char_count(prompt)
    try:
        response = model.invoke(prompt)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _record_timing(label, elapsed_ms, kind="llm", prompt_chars=prompt_chars, status="failed")
        print(
            f"[LLM Timing] {label}: failed after {elapsed_ms:.1f} ms "
            f"(prompt_chars={prompt_chars})"
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    usage = _usage_metadata(response)
    _record_llm_usage(usage)
    _record_timing(
        label,
        elapsed_ms,
        kind="llm",
        prompt_chars=prompt_chars,
        status="ok",
        **usage,
    )
    print(
        f"[LLM Timing] {label}: {elapsed_ms:.1f} ms "
        f"(prompt_chars={prompt_chars}, total_tokens={usage['total_tokens']})"
    )
    return response
