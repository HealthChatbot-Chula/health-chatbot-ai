import json
import time
from typing import Any


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
        print(
            f"[LLM Timing] {label}: failed after {elapsed_ms:.1f} ms "
            f"(prompt_chars={prompt_chars})"
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[LLM Timing] {label}: {elapsed_ms:.1f} ms "
        f"(prompt_chars={prompt_chars})"
    )
    return response
