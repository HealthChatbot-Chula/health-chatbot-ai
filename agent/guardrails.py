import json
import re

from langchain_core.messages import AIMessage

from .constants import (
    INPUT_HEALTH_TERMS,
    INPUT_NON_HEALTH_TERMS,
    OUTPUT_REVIEW_INTENTS,
    OUTPUT_REVIEW_PATTERNS,
)
from .followups import _append_structured_followup, _structured_followup_question_for_slot
from .llm_timing import timed_llm_invoke
from .models import chat_model
from .slot_filling_graph import (
    _extract_lab_values_from_text,
    _looks_like_greeting,
    _looks_like_scope_question,
    _mentioned_lab_topics,
    _updates_from_pending_slot,
)
from .state import AgentState


def _looks_like_health_query(text: str, state: AgentState) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False

    has_non_health_term = any(term in normalized for term in INPUT_NON_HEALTH_TERMS)
    has_health_term = any(term in normalized for term in INPUT_HEALTH_TERMS)
    if has_non_health_term and not has_health_term:
        return False

    if _looks_like_greeting(normalized) or _looks_like_scope_question(normalized):
        return True

    if any(term in normalized for term in INPUT_HEALTH_TERMS):
        return True

    if _mentioned_lab_topics(normalized):
        return True

    if _extract_lab_values_from_text(text, ""):
        return True

    if state.get("extracted_lab_values") and len(normalized) <= 160:
        return True

    return False


def _input_guardrail_fast_decision(
    text: str,
    state: AgentState,
) -> tuple[str, str]:
    normalized = text.strip().lower()
    if not normalized:
        return "BLOCK", "empty message"

    if _looks_like_health_query(text, state):
        return "ALLOW", "health keyword/lab/greeting matched"

    has_non_health_term = any(term in normalized for term in INPUT_NON_HEALTH_TERMS)
    has_health_term = any(term in normalized for term in INPUT_HEALTH_TERMS)
    if has_non_health_term and not has_health_term:
        return "BLOCK", "clear non-health topic matched"

    # Fail open for ambiguous messages to avoid blocking health users and to keep
    # the intake path LLM-free. Slot/routing rules will ask fixed follow-ups.
    return "ALLOW", "ambiguous message allowed without LLM"


def _input_rejection_payload(state: AgentState, reason: str) -> dict:
    print(f"    🚫 BLOCKED: {reason}")
    rejection_msg = AIMessage(
        content=(
            "ขออภัยครับ ผมช่วยได้เฉพาะเรื่องสุขภาพและการแปลผลตรวจสุขภาพเท่านั้น\n"
            "หากมีคำถามเกี่ยวกับอาการ ผลแลป หรือโรคที่เกี่ยวข้อง ยินดีช่วยเสมอครับ"
        )
    )
    return {
        "messages": [rejection_msg],
        "blocked": True,
        "current_node": "guardrail_input",
        "steps": state.get("steps", []) + ["guardrail_input_blocked"],
    }


def guardrail_input_node(state: AgentState):
    """
    ตรวจสอบ input ของ user ก่อนเข้าระบบหลัก
    - ALLOW: คำถามสุขภาพ, แปรผลแลป, การทักทาย, บริบทอื่นๆที่เกี่ยวข้อง
    - BLOCK: ไม่เกี่ยวกับสุขภาพเลย เช่น เกม, การเมือง, ความบันเทิง
    """
    state["current_node"] = "guardrail_input"
    state["steps"].append("guardrail_input")

    last_user_message = state["messages"][-1].content

    print("[1.5] 🛡️ INPUT GUARDRAIL: Checking relevance...")

    pending_slot_updates = _updates_from_pending_slot(state, last_user_message)
    if pending_slot_updates:
        print(
            "    ✅ ALLOWED: user is answering the pending slot "
            f"{state.get('pending_slot')!r}"
        )
        return {
            "blocked": False,
            "current_node": "guardrail_input",
            "steps": state.get("steps", []) + ["guardrail_input_pending_slot_allowed"],
        }

    action, reason = _input_guardrail_fast_decision(last_user_message, state)
    if action == "BLOCK":
        return _input_rejection_payload(state, reason)

    print(f"    ✅ ALLOWED WITHOUT LLM: {reason}")
    return {
        "blocked": False,
        "current_node": "guardrail_input",
        "steps": state.get("steps", []) + ["guardrail_input_fast_allowed"],
    }


def route_after_input_guardrail(state: AgentState):
    """Route หลัง input guardrail: ถ้าถูก block ให้จบเลย"""
    if state.get("blocked", False):
        return "blocked"
    return "continue"


def _requires_output_guardrail(state: AgentState, content: str) -> tuple[bool, str]:
    intent = state.get("intent")
    if intent in OUTPUT_REVIEW_INTENTS:
        return True, f"intent={intent}"

    normalized = " ".join(str(content).lower().split())
    for pattern in OUTPUT_REVIEW_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return True, f"matched pattern={pattern}"

    return False, "low-risk response"


def guardrail_output_node(state: AgentState):
    """
    ตรวจสอบ output ก่อนส่งให้ user
    """
    state["current_node"] = "guardrail_output"
    state["steps"].append("guardrail_output")

    last_ai_message = state["messages"][-1].content

    print("[3] 🛡️ OUTPUT GUARDRAIL: Checking safety rules...")
    needs_review, review_reason = _requires_output_guardrail(state, last_ai_message)
    if not needs_review:
        print(f"    ✅ SKIPPED: {review_reason}")
        return {
            "steps": state.get("steps", []) + ["guardrail_skipped_low_risk"],
        }

    print(f"    🔎 REVIEW REQUIRED: {review_reason}")

    guard_prompt = (
        "You are a senior-nurse safety editor. Review the AI answer against the rules below and return JSON only.\n\n"
        "--- Review rules ---\n"
        "1. Do not state a disease as certain; use Thai wording equivalent to preliminary risk or trend.\n"
        "2. Do not give medication names, doses, or instructions.\n"
        "3. Avoid alarming Thai wording equivalent to dangerous, critical, or severe.\n\n"
        "JSON format (no additional text):\n"
        "If all rules pass: "
        '{"action": "PASSED", "revised_content": null}\n'
        "If a revision is needed: "
        '{"action": "MODIFIED", "revised_content": "complete revised Thai response"}\n\n'
        f"Answer to review:\n{last_ai_message}"
    )

    result = timed_llm_invoke(
        chat_model,
        guard_prompt,
        "output_guardrail_review",
    ).content.strip()

    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        action = parsed.get("action", "PASSED").upper()
        revised = parsed.get("revised_content")
    except (json.JSONDecodeError, AttributeError):
        action = "PASSED"
        revised = None

    if action == "MODIFIED" and revised:
        print("    ⚠️ MODIFIED: Response was adjusted by guardrail")
        followup_question = _structured_followup_question_for_slot(state.get("pending_slot"))
        if followup_question:
            revised = _append_structured_followup(str(revised), followup_question)
        new_message = AIMessage(content=revised, id=state["messages"][-1].id)

        return {
            "messages": [new_message],
            "steps": state.get("steps", []) + ["guardrail_modified"],
        }

    print("    ✅ PASSED: Response is safe")
    return {
        "steps": state.get("steps", []) + ["guardrail_passed"],
    }
