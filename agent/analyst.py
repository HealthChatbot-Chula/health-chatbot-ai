import json
from typing import Any

from langchain_core.messages import SystemMessage

from .followups import _append_structured_followup, _next_structured_followup
from .llm_timing import timed_llm_invoke
from .models import chat_model
from .prompts import lab_prompt, no_context_prompt
from .rag_utils import retrieve_context
from .state import AgentState


def _analysis_slot_context(state: AgentState) -> str:
    labs = state.get("extracted_lab_values") or {}
    lab_text = ", ".join(f"{name}: {value}" for name, value in labs.items()) or "-"
    profile_metrics = state.get("profile_metrics")
    health_state = state.get("health_state")
    return (
        "\n\nข้อมูล structured ที่สกัดได้ก่อนวิเคราะห์:"
        f"\n- intent: {state.get('intent')}"
        f"\n- age: {state.get('age')}"
        f"\n- gender: {state.get('gender')}"
        f"\n- fasting_status: {state.get('fasting_status')}"
        f"\n- underlying_disease: {state.get('underlying_disease')}"
        f"\n- current_medications: {state.get('current_medications')}"
        f"\n- current_symptoms: {state.get('current_symptoms')}"
        f"\n- extracted_lab_values: {lab_text}"
        f"\n- profile_metrics: {_compact_json(profile_metrics) if profile_metrics else '-'}"
        f"\n- health_state: {_compact_json(health_state) if health_state else '-'}"
    )


def _compact_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _has_structured_health_data(state: AgentState) -> bool:
    return bool(
        state.get("extracted_lab_values")
        or state.get("profile_metrics")
        or state.get("health_state")
    )


def _analysis_retrieval_query(state: AgentState, fallback_query: str) -> str:
    labs = state.get("extracted_lab_values") or {}
    if not labs:
        return fallback_query

    query_parts = [
        "health checkup lab interpretation",
        "diabetes dyslipidemia kidney liver blood pressure",
        f"age {state.get('age')}",
        f"gender {state.get('gender')}",
        f"fasting {state.get('fasting_status')}",
        f"intent {state.get('intent')}",
    ]
    query_parts.extend(f"{name} {value}" for name, value in labs.items())
    return " ".join(str(part) for part in query_parts if part)


def call_model(state: AgentState):
    """
    Node สำหรับตอบคำถาม: ดึง Context มาใส่ใน Prompt จริงๆ
    """
    messages = state["messages"]
    summary = state.get("summary", "")
    last_user_message = messages[-1].content

    retrieval_query = _analysis_retrieval_query(state, last_user_message)
    context = retrieve_context(retrieval_query)

    summary_context = f"\n\nสรุปบริบทการสนทนาก่อนหน้านี้: {summary}" if summary else ""
    summary_context += _analysis_slot_context(state)

    print("=== RETRIEVAL QUERY ===", retrieval_query)
    print("=== CONTEXT ===", context)

    if context or _has_structured_health_data(state):
        if not context:
            context = (
                "ไม่มีข้อมูลจาก RAG ที่ตรงพอในรอบนี้ ให้ใช้ข้อมูล structured "
                "จากผู้ใช้ด้านบนเป็นหลัก และห้ามตอบว่าผู้ใช้ยังไม่ได้ให้ค่าผลตรวจ"
            )
        system_prompt = lab_prompt(context, summary_context)
    else:
        system_prompt = no_context_prompt()

    print("[2] >>> AGENT NODE: Generating response...")
    response = timed_llm_invoke(
        chat_model,
        [SystemMessage(content=system_prompt)] + messages,
        "analyst_response",
    )
    pending_slot, followup_question = _next_structured_followup(state)
    if pending_slot and followup_question:
        updated_content = _append_structured_followup(str(response.content), followup_question)
        if hasattr(response, "model_copy"):
            response = response.model_copy(update={"content": updated_content})
        else:
            response = response.copy(update={"content": updated_content})

    result = {
        "messages": [response],
        "steps": state.get("steps", []) + ["retrieval", "generate"],
    }
    if pending_slot:
        result["pending_slot"] = pending_slot

    return result
