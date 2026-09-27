import json
import os
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_active_chat_trace: ContextVar[bool] = ContextVar(
    "active_chat_trace", default=False
)
_reported_errors: set[str] = set()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def langfuse_enabled() -> bool:
    return _env_flag("LANGFUSE_ENABLED")


def langfuse_content_capture_enabled() -> bool:
    """Return whether masked chat content may be sent to Langfuse.

    This is deliberately a separate opt-in from general tracing because this
    application handles health-related conversations.
    """

    return _env_flag("LANGFUSE_CAPTURE_CONTENT")


_CONTENT_MAX_CHARS = 2_000
_SENSITIVE_CONTENT_PATTERNS = (
    # Thai national ID numbers, with or without the usual separators.
    (re.compile(r"(?<!\d)(?:\d[- ]?){12}\d(?!\d)"), "[REDACTED_NATIONAL_ID]"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[REDACTED_EMAIL]"),
    # ISO and common day/month/year date forms can identify a patient when
    # combined with the surrounding conversation. This must run before the
    # phone matcher because its separators resemble phone-number formatting.
    (
        re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
        "[REDACTED_DATE]",
    ),
    # International numbers plus common Thai local mobile/landline forms.
    (
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,3}\)?[ -]?){2,4}\d{3,4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
    (re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE), "[REDACTED_URL]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
)


def sanitize_langfuse_content(value: Any) -> str:
    """Mask common direct identifiers before optional Langfuse capture."""

    content = str(value)
    for pattern, replacement in _SENSITIVE_CONTENT_PATTERNS:
        content = pattern.sub(replacement, content)
    if len(content) > _CONTENT_MAX_CHARS:
        content = f"{content[:_CONTENT_MAX_CHARS]}… [TRUNCATED]"
    return content


def _langfuse_message_role(message: Any) -> str:
    message_type = str(getattr(message, "type", "")).lower()
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }.get(message_type, "user")


def _langfuse_message_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def sanitized_langfuse_messages(prompt: Any) -> list[dict[str, str]]:
    """Return a bounded, role-labelled, masked representation of an LLM prompt."""

    messages = prompt if isinstance(prompt, (list, tuple)) else [prompt]
    return [
        {
            "role": _langfuse_message_role(message),
            "content": sanitize_langfuse_content(_langfuse_message_content(message)),
        }
        for message in messages
    ]


def _report_error(stage: str, exc: BaseException) -> None:
    key = f"{stage}:{type(exc).__name__}:{exc}"
    if key in _reported_errors:
        return
    _reported_errors.add(key)
    print(f"[Langfuse] {stage} failed; tracing disabled for this operation: {exc}")


def _load_langfuse_sdk():
    try:
        from langfuse import get_client, propagate_attributes
    except (ImportError, ModuleNotFoundError) as exc:
        _report_error("SDK import", exc)
        return None
    return get_client, propagate_attributes


class ObservationHandle:
    def __init__(self, observation: Any = None):
        self._observation = observation

    def update(self, **details: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(**details)
        except Exception as exc:
            _report_error("observation update", exc)


def _enter_context(context: Any):
    return context.__enter__()


def _exit_context(context: Any, exc_info: tuple[Any, Any, Any]) -> None:
    try:
        context.__exit__(*exc_info)
    except Exception as exc:
        _report_error("observation close", exc)


@contextmanager
def observe_chat_request(
    *,
    user_id: str | None,
    session_id: str,
    conversation_id: str,
    model: str,
    message_count: int,
    latest_user_chars: int,
    latest_user_message: str,
) -> Iterator[ObservationHandle]:
    """Create one Langfuse trace for a chatbot turn.

    Raw health content is never captured. When explicitly opted in, the root
    trace includes a bounded, masked copy of the latest user message and final
    assistant answer to make trace review useful without exposing common IDs.
    """

    if not langfuse_enabled():
        yield ObservationHandle()
        return

    sdk = _load_langfuse_sdk()
    if sdk is None:
        yield ObservationHandle()
        return

    get_client, propagate_attributes = sdk
    span_context = None
    attributes_context = None
    try:
        client = get_client()
        propagated_attributes = {
            "session_id": session_id,
            "tags": ["health-chat"],
            "metadata": {"conversation_id": conversation_id},
            "trace_name": "health-chat-turn",
        }
        if user_id:
            propagated_attributes["user_id"] = user_id
        # Attribute propagation must be active before the root observation is
        # created. Langfuse uses every observation carrying session_id to build
        # correct session-level metrics, including total token usage.
        attributes_context = propagate_attributes(**propagated_attributes)
        _enter_context(attributes_context)

        trace_input: Any = {
            "message_count": message_count,
            "latest_user_chars": latest_user_chars,
        }
        if langfuse_content_capture_enabled():
            trace_input = [
                {
                    "role": "user",
                    "content": sanitize_langfuse_content(latest_user_message),
                }
            ]

        span_context = client.start_as_current_observation(
            as_type="span",
            name="health-chat-turn",
            input=trace_input,
            metadata={
                "conversation_id": conversation_id,
                "requested_model": model,
                "message_count": message_count,
                "latest_user_chars": latest_user_chars,
                "content_capture": langfuse_content_capture_enabled(),
            },
        )
        span = _enter_context(span_context)
    except Exception as exc:
        _report_error("trace setup", exc)
        if attributes_context is not None:
            _exit_context(attributes_context, (None, None, None))
        if span_context is not None:
            _exit_context(span_context, sys.exc_info())
        yield ObservationHandle()
        return

    token = _active_chat_trace.set(True)
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield ObservationHandle(span)
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _active_chat_trace.reset(token)
        _exit_context(span_context, exc_info)
        _exit_context(attributes_context, exc_info)


def _model_name(model: Any) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


@contextmanager
def observe_generation(
    *,
    name: str,
    model: Any,
    prompt_chars: int,
    prompt: Any,
) -> Iterator[ObservationHandle]:
    """Create a generation only while a chat request trace is active."""

    if not langfuse_enabled() or not _active_chat_trace.get():
        yield ObservationHandle()
        return

    sdk = _load_langfuse_sdk()
    if sdk is None:
        yield ObservationHandle()
        return

    get_client, _ = sdk
    generation_context = None
    try:
        generation_input: Any = {"prompt_chars": prompt_chars}
        if langfuse_content_capture_enabled():
            generation_input = sanitized_langfuse_messages(prompt)
        generation_context = get_client().start_as_current_observation(
            as_type="generation",
            name=name,
            model=_model_name(model),
            input=generation_input,
        )
        generation = _enter_context(generation_context)
    except Exception as exc:
        _report_error("generation setup", exc)
        if generation_context is not None:
            _exit_context(generation_context, sys.exc_info())
        yield ObservationHandle()
        return

    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield ObservationHandle(generation)
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _exit_context(generation_context, exc_info)


def shutdown_langfuse() -> None:
    if not langfuse_enabled():
        return

    sdk = _load_langfuse_sdk()
    if sdk is None:
        return

    get_client, _ = sdk
    try:
        get_client().shutdown()
    except Exception as exc:
        _report_error("shutdown", exc)
