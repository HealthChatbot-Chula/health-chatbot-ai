"""Semantic, per-turn intent classification for graph routing."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .intake import (
    classify_interaction_intent,
    extract_lab_values,
    latest_user_text,
    parse_intake_updates,
)
from .llm_timing import timed_llm_invoke
from .state import AgentState


TURN_INTENTS = {
    "greeting",
    "capability_question",
    "out_of_scope",
    "lab_interpretation",
    "medication_safety",
    "urgent_red_flag",
    "general_health",
    "slot_answer",
    "unknown",
}

DIRECT_RESPONSE_INTENTS = {
    "greeting",
    "capability_question",
    "out_of_scope",
    "unknown",
}

_SYSTEM_PROMPT = """You classify the latest message sent to a Thai health chatbot.
Return only JSON in this exact shape: {"intent": "one_allowed_value"}.

Allowed values:
- greeting: greetings, thanks, acknowledgements, or casual pleasantries
- capability_question: asks what the chatbot can do or which conditions it supports
- out_of_scope: clearly asks about a non-health topic
- lab_interpretation: asks about, reports, or follows up on health measurements or lab results
- medication_safety: asks whether to start, stop, increase, decrease, or change medication
- urgent_red_flag: describes potentially urgent symptoms or asks whether urgent care is needed
- general_health: an in-scope health question that is not specifically about a lab result
- slot_answer: directly answers a question requesting age, sex, fasting status, or a lab value
- unknown: meaning or requested action is unclear

Classify the latest message, not the user's stored medical data. Recent context is supplied only
to resolve short follow-ups such as "แล้วอันนี้สูงไหม". A prior lab discussion must not turn a
greeting, acknowledgement, or unrelated question into lab_interpretation.

Examples:
- "มอนิ่งงงง" -> greeting
- "ขอบคุณนะ" -> greeting
- "ช่วยเรื่องอะไรได้บ้าง" -> capability_question
- "LDL 178 สูงไหม" -> lab_interpretation
- "แล้วค่าที่ส่งไปต่ำไหม" after a lab discussion -> lab_interpretation
- "ควรหยุดยาไหม" -> medication_safety
- "เขียนโค้ด Python ให้หน่อย" -> out_of_scope
"""


def _recent_context(state: AgentState, limit: int = 2) -> str:
    messages = state.get("messages", [])
    latest_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            latest_index = index
            break

    if latest_index <= 0:
        return ""

    lines: list[str] = []
    for message in messages[max(0, latest_index - limit):latest_index]:
        role = "User" if getattr(message, "type", None) == "human" else "Assistant"
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_intent_response(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return "unknown"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unknown"
    intent = str(payload.get("intent", "")).strip()
    return intent if intent in TURN_INTENTS else "unknown"


def _fallback_intent(text: str) -> str:
    """Use current-turn rules only; never infer intent from saved lab values."""

    intent = classify_interaction_intent(text, {})
    if intent in {"medication_safety", "urgent_red_flag", "general_info"}:
        return "greeting" if intent == "general_info" else intent
    return "unknown"


def classify_turn_intent_node(state: AgentState) -> dict[str, str]:
    latest_message = latest_user_text(state.get("messages", []))
    if not latest_message:
        return {"intent": "unknown"}

    pending_slot = state.get("pending_slot")
    if pending_slot:
        pending_updates = parse_intake_updates(state, latest_message, "")
        if any(
            field in pending_updates
            for field in ("age", "gender", "fasting_status", "extracted_lab_values")
        ):
            print(f"[turn_intent] intent=slot_answer pending_slot={pending_slot}")
            return {"intent": "slot_answer"}

    # Keep a deterministic current-turn override for the two safety-critical
    # routes. All conversational and clinical semantics still go through the
    # model; saved health state is deliberately absent from this check.
    safety_intent = classify_interaction_intent(latest_message, {})
    if safety_intent in {"medication_safety", "urgent_red_flag"}:
        print(f"[turn_intent] intent={safety_intent} safety_override=true")
        return {"intent": safety_intent}

    current_labs = extract_lab_values(latest_message, "") or {}
    glucose_value = current_labs.get("Glucose") or current_labs.get("FBS")
    if "Potassium" in current_labs or (
        glucose_value is not None and glucose_value >= 300
    ):
        print("[turn_intent] intent=urgent_red_flag current_lab_override=true")
        return {"intent": "urgent_red_flag"}

    prompt_parts = []
    recent_context = _recent_context(state)
    if recent_context:
        prompt_parts.append(f"Recent context:\n{recent_context}")
    if pending_slot:
        prompt_parts.append(f"The chatbot is waiting for this field: {pending_slot}")
    prompt_parts.append(f"Latest user message:\n{latest_message}")

    try:
        from .models import intent_model

        response = timed_llm_invoke(
            intent_model,
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(prompt_parts)),
            ],
            "turn_intent_classification",
        )
        intent = _parse_intent_response(response.content)
    except Exception as exc:
        intent = _fallback_intent(latest_message)
        print(f"[turn_intent] classification failed; fallback={intent}. error={exc}")

    print(f"[turn_intent] intent={intent}")
    return {"intent": intent}
