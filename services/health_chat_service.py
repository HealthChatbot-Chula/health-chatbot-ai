import hashlib
import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.dependencies import load_agent_resources
from backend.openai_compat import empty_usage
from backend.schemas import ChatRequest, Message
from agent.llm_timing import (
    collected_token_usage,
    request_timing,
    timed_llm_invoke,
    timed_section,
)


SLOT_FIELD_NAMES = (
    "health_state",
    "profile_metrics",
    "age",
    "gender",
    "underlying_disease",
    "current_medications",
    "current_symptoms",
    "fasting_status",
    "extracted_lab_values",
    "pending_slot",
)

slot_memory_store: dict[str, dict[str, Any]] = {}
session_usage_store: dict[str, dict[str, int]] = {}

HISTORY_WINDOW_TURNS = 3
HISTORY_WINDOW_MESSAGES = HISTORY_WINDOW_TURNS * 2
SUMMARY_MESSAGE_COUNT_FIELD = "_summary_message_count"
SUMMARY_SOURCE_HASH_FIELD = "_summary_source_hash"
SUMMARY_INPUT_CHAR_LIMIT = 8000
SUMMARY_MESSAGE_CHAR_LIMIT = 1000

FAST_PATH_PENDING_SLOTS = {"gender", "age", "fasting_status"}
FASTING_FOLLOWUP_LABS = {
    "FBS",
    "Triglycerides",
}

OPENWEBUI_TASK_MARKERS = (
    "Generate a concise, 3-5 word title",
    "Generate 1-3 broad tags",
    "Suggest 3-5 relevant follow-up questions",
)


def langchain_messages(messages: list[Message]) -> list[HumanMessage | AIMessage | SystemMessage]:
    converted = []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        elif message.role == "system":
            converted.append(SystemMessage(content=message.content))
    return converted


def _message_role(message: BaseMessage) -> str:
    return str(getattr(message, "type", message.__class__.__name__)).lower()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _history_hash(messages: list[BaseMessage]) -> str:
    fingerprint = "\n".join(
        f"{_message_role(message)}::{_message_text(message)}"
        for message in messages
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _split_history_window(
    messages: list[BaseMessage],
    max_window_messages: int = HISTORY_WINDOW_MESSAGES,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    system_messages = [
        message for message in messages if _message_role(message) == "system"
    ]
    chat_messages = [
        message for message in messages if _message_role(message) != "system"
    ]
    if len(chat_messages) <= max_window_messages:
        return system_messages + chat_messages, []
    return (
        system_messages + chat_messages[-max_window_messages:],
        chat_messages[:-max_window_messages],
    )


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    current_size = 0

    # Keep the freshest older context if a very long chat needs a rebuild.
    for message in reversed(messages):
        role = "User" if _message_role(message) == "human" else "Assistant"
        text = " ".join(_message_text(message).split())
        if len(text) > SUMMARY_MESSAGE_CHAR_LIMIT:
            text = f"{text[:SUMMARY_MESSAGE_CHAR_LIMIT]}..."
        line = f"{role}: {text}"
        line_size = len(line) + 1
        if lines and current_size + line_size > SUMMARY_INPUT_CHAR_LIMIT:
            break
        lines.append(line)
        current_size += line_size

    return "\n".join(reversed(lines))


def _summarize_messages(
    existing_summary: str,
    messages_to_summarize: list[BaseMessage],
) -> str:
    if not messages_to_summarize:
        return existing_summary

    load_agent_resources()
    from agent.graph import intent_model

    transcript = _format_messages_for_summary(messages_to_summarize)
    prompt = (
        "Summarize the preceding conversation into compact context for a health chatbot.\n"
        "Keep only important facts: age, sex, laboratory results, conditions, medications, symptoms, "
        "the user's questions or concerns, and important prior answers.\n"
        "Do not give new advice, diagnose, or invent facts absent from the transcript.\n"
        "Write the summary in Thai in no more than eight short bullet points.\n\n"
    )
    if existing_summary:
        prompt += f"Existing summary:\n{existing_summary}\n\n"
    prompt += f"New transcript to incorporate:\n{transcript}"

    response = timed_llm_invoke(
        intent_model,
        [
            SystemMessage(content="You summarize prior Thai health-chat context compactly and neutrally."),
            HumanMessage(content=prompt),
        ],
        "service_history_summary",
    )
    return str(response.content).strip()


def _prepare_windowed_messages_and_summary(
    conversation_key_value: str,
    converted_messages: list[BaseMessage],
) -> tuple[list[BaseMessage], str]:
    current_slots = slot_memory_store.setdefault(conversation_key_value, {})
    windowed_messages, older_messages = _split_history_window(converted_messages)

    existing_summary = str(current_slots.get("summary") or "")
    saved_count = int(current_slots.get(SUMMARY_MESSAGE_COUNT_FIELD) or 0)
    saved_source_hash = str(current_slots.get(SUMMARY_SOURCE_HASH_FIELD) or "")

    if not older_messages:
        print(
            "[History Window] No older history to summarize; "
            f"passing {len(windowed_messages)} messages."
        )
        return windowed_messages, existing_summary

    current_source_hash = _history_hash(older_messages)
    if saved_count == len(older_messages) and saved_source_hash == current_source_hash:
        print(
            "[History Window] Reusing cached summary; "
            f"passing {len(windowed_messages)} messages."
        )
        return windowed_messages, existing_summary

    if 0 < saved_count < len(older_messages):
        previous_source_hash = _history_hash(older_messages[:saved_count])
        if previous_source_hash == saved_source_hash:
            messages_to_summarize = older_messages[saved_count:]
            summary_base = existing_summary
        else:
            messages_to_summarize = older_messages
            summary_base = ""
    else:
        messages_to_summarize = older_messages
        summary_base = "" if saved_count > len(older_messages) else existing_summary

    try:
        updated_summary = _summarize_messages(summary_base, messages_to_summarize)
    except Exception as exc:
        print(
            "[History Window] Summary failed; continuing with cached summary. "
            f"error={exc}"
        )
        return windowed_messages, existing_summary

    current_slots["summary"] = updated_summary
    current_slots[SUMMARY_MESSAGE_COUNT_FIELD] = len(older_messages)
    current_slots[SUMMARY_SOURCE_HASH_FIELD] = current_source_hash

    print(
        "[History Window] Updated summary from older history; "
        f"older={len(older_messages)}, window={len(windowed_messages)}."
    )
    return windowed_messages, updated_summary


def is_openwebui_task(user_text: str) -> bool:
    return any(marker in user_text for marker in OPENWEBUI_TASK_MARKERS)


def conversation_key(req: ChatRequest) -> str:
    """
    Prefer explicit IDs from the caller. If none are available, fall back to a
    stable fingerprint of the first user message plus the optional user field.
    """

    direct_key = req.conversation_id or req.chat_id or req.session_id
    if direct_key:
        return direct_key

    metadata = req.metadata or {}
    metadata_key = (
        metadata.get("conversation_id")
        or metadata.get("chat_id")
        or metadata.get("session_id")
    )
    if metadata_key:
        return str(metadata_key)

    first_user_message = next(
        (message.content for message in req.messages if message.role == "user"),
        "",
    )
    fingerprint_source = f"{req.user or 'anonymous'}::{first_user_message}"
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


def request_extra_fields(req: ChatRequest) -> dict[str, Any]:
    known_fields = set(getattr(req, "__fields__", {}).keys())
    extra_fields = getattr(req, "model_extra", None)
    if isinstance(extra_fields, dict):
        return extra_fields
    return {
        key: value
        for key, value in req.__dict__.items()
        if key not in known_fields
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _dict_at(value: Any, *keys: str) -> dict[str, Any]:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _metrics_to_lab_values(metrics: Any) -> dict[str, Any]:
    if not metrics:
        return {}

    if isinstance(metrics, dict):
        lab_values: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                lab_values[key] = _first_present(
                    value.get("value"),
                    value.get("result"),
                    value.get("numeric_value"),
                    value.get("measurement"),
                )
            else:
                lab_values[key] = value
        return lab_values

    if isinstance(metrics, list):
        lab_values = {}
        for item in metrics:
            if not isinstance(item, dict):
                continue
            name = _first_present(
                item.get("name"),
                item.get("key"),
                item.get("label"),
                item.get("metric"),
                item.get("code"),
            )
            value = _first_present(
                item.get("value"),
                item.get("result"),
                item.get("numeric_value"),
                item.get("measurement"),
            )
            if name is not None and value is not None:
                lab_values[str(name)] = value
        return lab_values

    return {}


def _normalized_request_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    if not raw_state:
        return {}

    from agent.intake import (
        merge_lab_values,
        normalize_age,
        normalize_gender,
        normalize_lab_values,
        normalize_yes_no,
    )

    profile = _dict_at(raw_state, "profile") or _dict_at(raw_state, "patient_profile")
    demographics = _dict_at(raw_state, "demographics")
    profile_metrics = _first_present(
        raw_state.get("profile_metrics"),
        raw_state.get("metrics"),
        raw_state.get("lab_values"),
        raw_state.get("labs"),
        raw_state.get("extracted_lab_values"),
    )
    lab_source_supplied = any(
        key in raw_state
        for key in (
            "profile_metrics",
            "metrics",
            "lab_values",
            "labs",
            "extracted_lab_values",
        )
    )

    normalized: dict[str, Any] = {"health_state": raw_state}
    if profile_metrics is not None:
        normalized["profile_metrics"] = profile_metrics

    age = normalize_age(_first_present(raw_state.get("age"), profile.get("age"), demographics.get("age")))
    if age is not None:
        normalized["age"] = age

    gender = normalize_gender(
        _first_present(raw_state.get("gender"), profile.get("gender"), demographics.get("gender"))
    )
    if gender is not None:
        normalized["gender"] = gender

    fasting_status = normalize_yes_no(
        _first_present(
            raw_state.get("fasting_status"),
            raw_state.get("fasting"),
            raw_state.get("is_fasting"),
        )
    )
    if fasting_status is not None:
        normalized["fasting_status"] = fasting_status

    direct_labs = normalize_lab_values(raw_state.get("extracted_lab_values"))
    metric_labs = normalize_lab_values(_metrics_to_lab_values(profile_metrics))
    labs = merge_lab_values(direct_labs, metric_labs)
    if labs:
        normalized["extracted_lab_values"] = labs
        normalized["pending_slot"] = None
    elif lab_source_supplied:
        normalized["extracted_lab_values"] = {}
        normalized["pending_slot"] = None

    copied_fields = ("underlying_disease", "current_medications", "current_symptoms")
    if "extracted_lab_values" not in normalized:
        copied_fields = (*copied_fields, "pending_slot")

    for field_name in copied_fields:
        if field_name in raw_state and raw_state.get(field_name) is not None:
            normalized[field_name] = raw_state.get(field_name)

    return normalized


def _request_health_state(req: ChatRequest) -> dict[str, Any]:
    raw_state: dict[str, Any] = {}
    if isinstance(req.health_state, dict):
        raw_state.update(req.health_state)

    metadata = req.metadata or {}
    metadata_state = metadata.get("health_state")
    if isinstance(metadata_state, dict):
        raw_state.update(metadata_state)

    for key in ("profile_metrics", "lab_values", "extracted_lab_values"):
        value = getattr(req, key, None)
        if value is not None:
            raw_state[key] = value

    extra_fields = request_extra_fields(req)
    for key in ("profile_metrics", "lab_values", "labs", "extracted_lab_values", "health_state"):
        value = extra_fields.get(key)
        if isinstance(value, dict) and key == "health_state":
            raw_state.update(value)
        elif value is not None:
            raw_state[key] = value

    for key in ("profile_metrics", "lab_values", "labs", "extracted_lab_values"):
        value = metadata.get(key)
        if value is not None and key not in raw_state:
            raw_state[key] = value

    return _normalized_request_state(raw_state)


def _has_authoritative_profile_state(request_state: dict[str, Any]) -> bool:
    return "profile_metrics" in request_state or "extracted_lab_values" in request_state


def _authoritative_profile_instruction() -> str:
    return (
        "\n\nImportant current-health-data instructions:"
        "\n- The structured data below is the latest information confirmed by the user or saved in a profile/form."
        "\n- If the summary or chat history conflicts with structured laboratory or blood-pressure values, treat it as outdated."
        "\n- Use the summary and chat history only to understand intent and context; never replace current structured values with older numbers."
    )


def preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def log_openwebui_request(req: ChatRequest, *, is_webui_task: bool) -> None:
    # TEMP_OPENWEBUI_LOG: remove this function and its call after confirming
    # which OpenWebUI field should be used as the per-chat memory key.
    metadata = req.metadata or {}
    extra_fields = request_extra_fields(req)
    role_counts: dict[str, int] = {}
    for message in req.messages:
        role_counts[message.role] = role_counts.get(message.role, 0) + 1

    first_user_message = next(
        (message.content for message in req.messages if message.role == "user"),
        "",
    )
    last_message = req.messages[-1] if req.messages else None

    print("\n[TEMP_OPENWEBUI_LOG] Incoming /v1/chat/completions request")
    print(f"[TEMP_OPENWEBUI_LOG] is_webui_task={is_webui_task}")
    print(f"[TEMP_OPENWEBUI_LOG] model={req.model!r} user={req.user!r}")
    print(
        "[TEMP_OPENWEBUI_LOG] explicit_ids="
        f"conversation_id={req.conversation_id!r}, "
        f"chat_id={req.chat_id!r}, session_id={req.session_id!r}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] metadata="
        f"{json.dumps(metadata, ensure_ascii=False, default=str)}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] extra_fields="
        f"{json.dumps(extra_fields, ensure_ascii=False, default=str)}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] messages="
        f"count={len(req.messages)}, roles={role_counts}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] first_user_sha256="
        f"{hashlib.sha256(first_user_message.encode('utf-8')).hexdigest() if first_user_message else None}"
    )
    if last_message:
        print(
            "[TEMP_OPENWEBUI_LOG] last_message="
            f"role={last_message.role!r}, preview={preview_text(last_message.content)!r}"
        )
    print(f"[TEMP_OPENWEBUI_LOG] selected_memory_key={conversation_key(req)!r}")


def initial_slot_state(
    conversation_key_value: str,
    request_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    saved_slots = slot_memory_store.get(conversation_key_value, {})
    if request_state:
        saved_slots = _merge_slot_state(saved_slots, request_state)
    return {field_name: saved_slots.get(field_name) for field_name in SLOT_FIELD_NAMES}


def _merge_slot_state(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge partial profile updates without losing prior conversation labs."""

    merged = {**existing, **incoming}
    if "extracted_lab_values" in incoming:
        from agent.intake import merge_lab_values

        merged["extracted_lab_values"] = merge_lab_values(
            existing.get("extracted_lab_values"),
            incoming.get("extracted_lab_values"),
        ) or {}

    # Profile forms commonly submit only fields changed in this turn. Preserve
    # the rest of a dictionary-shaped profile for answer context as well.
    old_metrics = existing.get("profile_metrics")
    new_metrics = incoming.get("profile_metrics")
    if isinstance(old_metrics, dict) and isinstance(new_metrics, dict):
        merged["profile_metrics"] = {**old_metrics, **new_metrics}

    return merged


def save_slot_state(conversation_key_value: str, graph_result: dict[str, Any]) -> None:
    current_slots = slot_memory_store.setdefault(conversation_key_value, {})
    for field_name in SLOT_FIELD_NAMES:
        if field_name in graph_result:
            current_slots[field_name] = graph_result.get(field_name)
    if "summary" in graph_result:
        current_slots["summary"] = graph_result.get("summary") or ""


def health_state_snapshot(conversation_key_value: str) -> dict[str, Any]:
    current_slots = slot_memory_store.get(conversation_key_value, {})
    snapshot = {
        field_name: current_slots.get(field_name)
        for field_name in SLOT_FIELD_NAMES
        if current_slots.get(field_name) is not None
    }
    summary = current_slots.get("summary")
    if summary:
        snapshot["summary"] = summary
    return snapshot


def needs_fasting_followup(slot_state: dict[str, Any]) -> bool:
    labs = slot_state.get("extracted_lab_values") or {}
    return any(name in FASTING_FOLLOWUP_LABS for name in labs)


def followup_question_for_slot(pending_slot: str) -> Optional[str]:
    if pending_slot == "gender":
        return (
            "เพศของผู้ที่เป็นเจ้าของผลตรวจคือชายหรือหญิงครับ? "
            "ข้อมูลนี้ช่วยให้เทียบช่วงอ้างอิงได้เหมาะสมขึ้น"
        )

    if pending_slot == "age":
        return "ผู้ที่เป็นเจ้าของผลตรวจอายุเท่าไรครับ?"

    if pending_slot == "fasting_status":
        return "ผลเลือดชุดนี้ตรวจหลังงดอาหารหรือไม่ครับ? กรุณาตอบว่าใช่หรือไม่ใช่"

    return None


def next_followup_slot(slot_state: dict[str, Any]) -> Optional[str]:
    labs = slot_state.get("extracted_lab_values") or {}
    if not labs:
        return None

    if slot_state.get("gender") is None:
        return "gender"

    if slot_state.get("age") is None:
        return "age"

    if slot_state.get("fasting_status") is None and needs_fasting_followup(slot_state):
        return "fasting_status"

    return None


def pending_slot_fast_response(
    conversation_key_value: str,
    latest_user_message: str,
) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    slot_state = initial_slot_state(conversation_key_value)
    pending_slot = slot_state.get("pending_slot")

    if pending_slot not in FAST_PATH_PENDING_SLOTS:
        return None

    from agent.slot_filling_graph import _updates_from_pending_slot

    slot_updates = _updates_from_pending_slot(slot_state, latest_user_message)
    parsed_slot_answer = any(
        field_name in slot_updates
        for field_name in ("gender", "age", "fasting_status")
    )
    if not parsed_slot_answer:
        followup_question = followup_question_for_slot(str(pending_slot))
        if not followup_question:
            return None
        response_metadata = slot_response_metadata({"pending_slot": pending_slot})
        print(
            "[Slot Fast Path] Could not parse pending slot "
            f"{pending_slot!r}; repeating fixed question without Gemini."
        )
        return followup_question, response_metadata

    if slot_updates.get("pending_slot") is not None:
        return None

    updated_state = {**slot_state, **slot_updates}
    next_pending_slot = next_followup_slot(updated_state)

    if not next_pending_slot:
        save_slot_state(conversation_key_value, updated_state)
        return None

    followup_question = followup_question_for_slot(next_pending_slot)
    if not followup_question:
        return None

    updated_state["pending_slot"] = next_pending_slot
    save_slot_state(conversation_key_value, updated_state)
    response_metadata = slot_response_metadata({"pending_slot": next_pending_slot})

    print(
        "[Slot Fast Path] Parsed pending slot "
        f"{pending_slot!r}; asking next slot {next_pending_slot!r} without Gemini."
    )

    return followup_question, response_metadata


def slot_response_metadata(graph_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    pending_slot = graph_result.get("pending_slot")
    choices_by_slot: dict[str, list[dict[str, str]]] = {
        "gender": [
            {"label": "ชาย", "value": "ชาย"},
            {"label": "หญิง", "value": "หญิง"},
        ],
        "fasting_status": [
            {"label": "ใช่", "value": "ใช่"},
            {"label": "ไม่ใช่", "value": "ไม่ใช่"},
        ],
    }

    if not pending_slot:
        return None

    metadata: dict[str, Any] = {"pending_slot": pending_slot}
    choices = choices_by_slot.get(str(pending_slot))
    if choices:
        metadata["choices"] = choices

    return metadata


def run_chat_completion(
    req: ChatRequest,
    converted_messages: list[HumanMessage | AIMessage | SystemMessage],
    latest_user_message: str,
    *,
    is_webui_task: bool,
) -> tuple[str, dict[str, int], Optional[dict[str, Any]]]:
    conversation_key_value = conversation_key(req)
    with request_timing(
        route="chat_completion",
        model=req.model,
        stream_requested=bool(req.stream),
        message_count=len(converted_messages),
        latest_user_chars=len(latest_user_message),
        openwebui_task=is_webui_task,
        conversation_key=conversation_key_value,
    ) as timing:
        assistant_content, _, metadata = _run_chat_completion_with_timing(
            req,
            converted_messages,
            latest_user_message,
            is_webui_task=is_webui_task,
            timing=timing,
        )
        request_usage = collected_token_usage(timing)
        session_usage = session_usage_store.setdefault(
            conversation_key_value,
            empty_usage(),
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            session_usage[key] += request_usage[key]

        response_metadata = metadata or {}
        response_metadata["token_usage"] = {
            "request": dict(request_usage),
            "session": dict(session_usage),
        }
        timing["session_token_usage"] = dict(session_usage)
        print(
            "[Token Usage] "
            f"conversation={conversation_key_value!r} "
            f"request={request_usage} session={session_usage}"
        )
        return assistant_content, request_usage, response_metadata


def _run_chat_completion_with_timing(
    req: ChatRequest,
    converted_messages: list[HumanMessage | AIMessage | SystemMessage],
    latest_user_message: str,
    *,
    is_webui_task: bool,
    timing: dict[str, Any],
) -> tuple[str, dict[str, int], Optional[dict[str, Any]]]:
    usage_data = empty_usage()

    if is_webui_task:
        print("\n[Interceptor] Open WebUI automated task detected. Bypassing LangGraph.")
        timing["path"] = "openwebui_automated_task"
        with timed_section("agent_resource_load"):
            _, chat_model = load_agent_resources()
        response = timed_llm_invoke(
            chat_model,
            converted_messages,
            "openwebui_automated_task",
        )
        assistant_content = response.content
        timing["response_chars"] = len(str(assistant_content))

        meta = getattr(response, "usage_metadata", {}) or {}
        usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
        usage_data["completion_tokens"] = meta.get("output_tokens", 0)
        usage_data["total_tokens"] = meta.get("total_tokens", 0)
        return assistant_content, usage_data, None

    print("\n[Interceptor] Normal user message detected. Routing to LangGraph.")
    conversation_key_value = conversation_key(req)
    request_state = _request_health_state(req)
    if request_state:
        slot_memory_store[conversation_key_value] = _merge_slot_state(
            slot_memory_store.get(conversation_key_value, {}),
            request_state,
        )
        if _has_authoritative_profile_state(request_state):
            slot_memory_store[conversation_key_value]["pending_slot"] = None

    fast_response = pending_slot_fast_response(conversation_key_value, latest_user_message)
    if fast_response:
        timing["path"] = "slot_fast_path"
        assistant_content, response_metadata = fast_response
        metadata = response_metadata or {}
        health_state = health_state_snapshot(conversation_key_value)
        if health_state:
            metadata["health_state"] = health_state
        return assistant_content, usage_data, metadata or None

    windowed_messages, summary = _prepare_windowed_messages_and_summary(
        conversation_key_value,
        converted_messages,
    )
    if _has_authoritative_profile_state(request_state):
        summary = f"{summary}{_authoritative_profile_instruction()}"

    with timed_section("agent_resource_load"):
        graph, _ = load_agent_resources()
    timing["path"] = "langgraph"
    timing["history_window_messages"] = len(windowed_messages)
    timing["summary_chars"] = len(summary)
    with timed_section("graph_invoke"):
        result = graph.invoke({
            "messages": windowed_messages,
            "steps": [],
            "current_node": "",
            "intent": None,
            **initial_slot_state(conversation_key_value, request_state),
            "summary": summary,
        })
    save_slot_state(conversation_key_value, result)

    assistant_msg = result["messages"][-1]
    assistant_content = assistant_msg.content
    timing["response_chars"] = len(str(assistant_content))

    meta = getattr(assistant_msg, "usage_metadata", {}) or {}
    usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
    usage_data["completion_tokens"] = meta.get("output_tokens", 0)
    usage_data["total_tokens"] = meta.get("total_tokens", 0)

    metadata = slot_response_metadata(result) or {}
    citations = result.get("citations") or []
    if citations:
        metadata["citations"] = citations
    health_state = health_state_snapshot(conversation_key_value)
    if health_state:
        metadata["health_state"] = health_state
    return assistant_content, usage_data, metadata or None
