import json
from typing import Any

from langchain_core.messages import SystemMessage

from .followups import _append_structured_followup, _next_structured_followup
from .llm_timing import timed_llm_invoke
from .models import chat_model
from .prompts import lab_prompt, no_context_prompt
from .rag_utils import format_citations, retrieve_context
from .state import AgentState


# A lab value has a clear clinical home.  Constraining retrieval prevents a
# plausible-but-wrong citation from another guideline (e.g. diabetes) being
# attached to an LDL answer.
LAB_SOURCE_IDS = {
    "LDL": "dyslipidemia",
    "HDL": "dyslipidemia",
    "Triglycerides": "dyslipidemia",
    "Total Cholesterol": "dyslipidemia",
    "Cholesterol": "dyslipidemia",
    "FBS": "diabetes",
    "Glucose": "diabetes",
    "HbA1c": "diabetes",
    "Creatinine": "kidney",
    "eGFR": "kidney",
    "ACR": "kidney",
    "Urine Albumin": "kidney",
    "Systolic BP": "hypertension",
    "Diastolic BP": "hypertension",
    "Blood Pressure": "hypertension",
}

SOURCE_QUERY_TERMS = {
    "dyslipidemia": "ไขมันในเลือด LDL HDL ไตรกลีเซอไรด์ การแปลผล",
    "diabetes": "เบาหวาน น้ำตาลในเลือด HbA1c FBS การแปลผล",
    "kidney": "โรคไต creatinine eGFR การแปลผล",
    "hypertension": "ความดันโลหิต การแปลผล",
}


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


def _source_ids_for_retrieval(state: AgentState, fallback_query: str) -> list[str]:
    labs = state.get("extracted_lab_values") or {}
    source_ids = {LAB_SOURCE_IDS[name] for name in labs if name in LAB_SOURCE_IDS}
    if source_ids:
        return sorted(source_ids)

    # This covers text-only questions before slot extraction finds a numeric
    # lab value.  Do not filter ambiguous general-health questions.
    text = fallback_query.lower()
    keyword_sources = {
        "dyslipidemia": ("ldl", "hdl", "cholesterol", "ไขมัน", "ไตรกลีเซอไรด์"),
        "diabetes": ("hba1c", "fbs", "glucose", "เบาหวาน", "น้ำตาล"),
        "kidney": ("egfr", "creatinine", "ไต", "ครีอะตินิน"),
        "hypertension": ("ความดัน", "blood pressure", "bp"),
    }
    return sorted(
        source_id
        for source_id, keywords in keyword_sources.items()
        if any(keyword in text for keyword in keywords)
    )


def _analysis_retrieval_query(state: AgentState, fallback_query: str) -> str:
    labs = state.get("extracted_lab_values") or {}
    source_ids = _source_ids_for_retrieval(state, fallback_query)
    if not labs:
        return fallback_query

    # Keep the embedding query clinical and specific.  Previously, including
    # every disease in one query caused cross-guideline matches for LDL.
    query_parts = [SOURCE_QUERY_TERMS[source_id] for source_id in source_ids]
    query_parts.append("การแปลผลตรวจสุขภาพ")
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
    source_ids = _source_ids_for_retrieval(state, last_user_message)
    retrieved = retrieve_context(retrieval_query, source_ids=source_ids)
    context = retrieved.context

    summary_context = f"\n\nสรุปบริบทการสนทนาก่อนหน้านี้: {summary}" if summary else ""
    summary_context += _analysis_slot_context(state)

    print("=== RETRIEVAL QUERY ===", retrieval_query)
    print("=== RETRIEVAL SOURCE FILTER ===", source_ids or "all")
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
        "citations": retrieved.citations,
    }
    if pending_slot:
        result["pending_slot"] = pending_slot

    return result


def append_citations_node(state: AgentState):
    """Append only retrieval-backed textbook citations after safety review."""
    citation_text = format_citations(state.get("citations") or [])
    if not citation_text:
        return {}

    last_message = state["messages"][-1]
    content = f"{last_message.content}\n\n{citation_text}"
    if hasattr(last_message, "model_copy"):
        message = last_message.model_copy(update={"content": content})
    else:
        message = last_message.copy(update={"content": content})
    return {"messages": [message]}
