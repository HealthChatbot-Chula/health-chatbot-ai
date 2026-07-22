import hashlib
import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.dependencies import load_agent_resources
from backend.openai_compat import empty_usage
from backend.schemas import ChatRequest, Message
from agent.llm_timing import timed_llm_invoke


SLOT_FIELD_NAMES = (
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

HISTORY_WINDOW_TURNS = 3
HISTORY_WINDOW_MESSAGES = HISTORY_WINDOW_TURNS * 2
SUMMARY_MESSAGE_COUNT_FIELD = "_summary_message_count"
SUMMARY_SOURCE_HASH_FIELD = "_summary_source_hash"
SUMMARY_INPUT_CHAR_LIMIT = 8000
SUMMARY_MESSAGE_CHAR_LIMIT = 1000

FAST_PATH_PENDING_SLOTS = {"gender", "age", "fasting_status"}
FASTING_FOLLOWUP_LABS = {
    "FBS",
    "Glucose",
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
        "สรุปบทสนทนาก่อนหน้าเพื่อส่งต่อให้ health chatbot ใช้เป็นบริบทแบบสั้นมาก\n"
        "ให้เก็บเฉพาะข้อเท็จจริงที่สำคัญ เช่น อายุ เพศ ผลตรวจ โรคประจำตัว ยาที่ใช้ อาการ "
        "คำถาม/ความกังวลของผู้ใช้ และคำตอบสำคัญที่เคยให้ไป\n"
        "ห้ามให้คำแนะนำใหม่ ห้ามวินิจฉัย และห้ามแต่งข้อมูลที่ไม่มีใน transcript\n"
        "ตอบเป็นภาษาไทย ไม่เกิน 8 bullet สั้น ๆ\n\n"
    )
    if existing_summary:
        prompt += f"สรุปเดิม:\n{existing_summary}\n\n"
    prompt += f"ข้อความใหม่ที่ต้องรวมเข้า summary:\n{transcript}"

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

    updated_summary = _summarize_messages(summary_base, messages_to_summarize)
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


def initial_slot_state(conversation_key_value: str) -> dict[str, Any]:
    saved_slots = slot_memory_store.get(conversation_key_value, {})
    return {field_name: saved_slots.get(field_name) for field_name in SLOT_FIELD_NAMES}


def save_slot_state(conversation_key_value: str, graph_result: dict[str, Any]) -> None:
    current_slots = slot_memory_store.setdefault(conversation_key_value, {})
    for field_name in SLOT_FIELD_NAMES:
        if field_name in graph_result:
            current_slots[field_name] = graph_result.get(field_name)
    if "summary" in graph_result:
        current_slots["summary"] = graph_result.get("summary") or ""


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
    usage_data = empty_usage()

    if is_webui_task:
        print("\n[Interceptor] Open WebUI automated task detected. Bypassing LangGraph.")
        _, chat_model = load_agent_resources()
        response = timed_llm_invoke(
            chat_model,
            converted_messages,
            "openwebui_automated_task",
        )
        assistant_content = response.content

        meta = getattr(response, "usage_metadata", {}) or {}
        usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
        usage_data["completion_tokens"] = meta.get("output_tokens", 0)
        usage_data["total_tokens"] = meta.get("total_tokens", 0)
        return assistant_content, usage_data, None

    print("\n[Interceptor] Normal user message detected. Routing to LangGraph.")
    conversation_key_value = conversation_key(req)
    fast_response = pending_slot_fast_response(conversation_key_value, latest_user_message)
    if fast_response:
        assistant_content, response_metadata = fast_response
        return assistant_content, usage_data, response_metadata

    windowed_messages, summary = _prepare_windowed_messages_and_summary(
        conversation_key_value,
        converted_messages,
    )

    graph, _ = load_agent_resources()
    result = graph.invoke({
        "messages": windowed_messages,
        "steps": [],
        "current_node": "",
        "intent": None,
        **initial_slot_state(conversation_key_value),
        "summary": summary,
    })
    save_slot_state(conversation_key_value, result)

    assistant_msg = result["messages"][-1]
    assistant_content = assistant_msg.content

    meta = getattr(assistant_msg, "usage_metadata", {}) or {}
    usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
    usage_data["completion_tokens"] = meta.get("output_tokens", 0)
    usage_data["total_tokens"] = meta.get("total_tokens", 0)

    return assistant_content, usage_data, slot_response_metadata(result)
