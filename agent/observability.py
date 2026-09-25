import os
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
) -> Iterator[ObservationHandle]:
    """Create one privacy-minimized Langfuse trace for a chatbot turn."""

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
        span_context = client.start_as_current_observation(
            as_type="span",
            name="health-chat-turn",
            input={
                "message_count": message_count,
                "latest_user_chars": latest_user_chars,
            },
            metadata={
                "conversation_id": conversation_id,
                "requested_model": model,
            },
        )
        span = _enter_context(span_context)
        propagated_attributes = {
            "session_id": session_id,
            "tags": ["health-chat"],
            "metadata": {"conversation_id": conversation_id},
            "trace_name": "health-chat-turn",
        }
        if user_id:
            propagated_attributes["user_id"] = user_id
        attributes_context = propagate_attributes(**propagated_attributes)
        _enter_context(attributes_context)
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
        _exit_context(attributes_context, exc_info)
        _exit_context(span_context, exc_info)


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
        generation_context = get_client().start_as_current_observation(
            as_type="generation",
            name=name,
            model=_model_name(model),
            input={"prompt_chars": prompt_chars},
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
