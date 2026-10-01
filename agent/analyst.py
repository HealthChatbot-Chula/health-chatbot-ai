import json
import re
from typing import Any

from langchain_core.messages import SystemMessage

from .followups import _append_structured_followup, _next_structured_followup
from .llm_timing import timed_llm_invoke
from .models import chat_model
from .prompts import (
    general_health_prompt,
    health_overview_prompt,
    lab_prompt,
    no_context_prompt,
)
from .rag_utils import RetrievedContext, format_citations, retrieve_context
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
    "HbA1c": "diabetes",
    "SBP": "hypertension",
    "DBP": "hypertension",
    "Creatinine": "kidney",
    "eGFR": "kidney",
    "ACR": "kidney",
    "Urine Albumin": "kidney",
    "Systolic BP": "hypertension",
    "Diastolic BP": "hypertension",
    "Blood Pressure": "hypertension",
}

SOURCE_KEYWORDS = {
    "dyslipidemia": (
        "ldl", "hdl", "cholesterol", "ไขมัน", "ไตรกลีเซอไรด์", "tg",
        "triglyceride", "statin", "lipid profile", "non-hdl",
    ),
    "diabetes": (
        "hba1c", "fbs", "glucose", "เบาหวาน", "น้ำตาล", "metformin",
        "ogtt", "น้ำตาลต่ำ", "น้ำตาลสูง",
    ),
    "kidney": (
        "egfr", "creatinine", "ไต", "ครีอะตินิน", "ckd", "acr",
        "โปรตีนรั่ว", "อัลบูมิน", "ฟอกไต", "dialysis", "โพแทสเซียม",
        "nsaid",
    ),
    "hypertension": (
        "ความดัน", "blood pressure", "bp", "amlodipine", "ace inhibitor",
        "arb", "ความดันตัวบน", "ความดันตัวล่าง", "โซเดียม",
    ),
}

# These terms identify the clinical question's primary guideline.  They take
# precedence over a related disease named in the same question, e.g. CKD + LDL
# should cite the CKD guideline's lipid recommendation rather than a generic
# dyslipidemia page.
PRIMARY_SOURCE_KEYWORDS = {
    "kidney": (
        "egfr", "creatinine", "ครีอะตินิน", "ckd", "acr", "โปรตีนรั่ว",
        "ฟอกไต", "dialysis", "โพแทสเซียม", "โรคไต", "ไตเรื้อรัง",
    ),
}

EVIDENCE_BACKED_LIFESTYLE_KEYWORDS = (
    "อาหาร", "กิน", "รับประทาน", "เมนู", "โภชนาการ",
    "ออกกำลังกาย", "กิจกรรมทางกาย", "เดินเร็ว", "วิ่ง", "ปั่นจักรยาน",
    "น้ำหนัก", "ลดเค็ม", "โซเดียม", "ของหวาน", "ไขมันอิ่มตัว",
    "สูบบุหรี่", "ดูแลสุขภาพ", "ปรับพฤติกรรม", "ควบคุมสุขภาพ",
    "diet", "exercise", "nutrition", "lifestyle", "weight",
)


def _analysis_slot_context(state: AgentState) -> str:
    labs = state.get("extracted_lab_values") or {}
    lab_text = ", ".join(f"{name}: {value}" for name, value in labs.items()) or "-"
    rejected = state.get("rejected_lab_values") or {}
    rejected_text = (
        ", ".join(
            f"{name}: {info['value']} (ต้องอยู่ระหว่าง {info['min']}-{info['max']})"
            for name, info in rejected.items()
        )
        or "-"
    )
    profile_metrics = state.get("profile_metrics")
    health_state = state.get("health_state")
    return (
        "\n\nStructured data extracted before analysis:"
        f"\n- intent: {state.get('intent')}"
        f"\n- age: {state.get('age')}"
        f"\n- gender: {state.get('gender')}"
        f"\n- fasting_status: {state.get('fasting_status')}"
        f"\n- underlying_disease: {state.get('underlying_disease')}"
        f"\n- current_medications: {state.get('current_medications')}"
        f"\n- current_symptoms: {state.get('current_symptoms')}"
        f"\n- extracted_lab_values: {lab_text}"
        f"\n- rejected_lab_values (NOT saved, outside the allowed range): {rejected_text}"
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


def _current_turn_source_ids(query: str) -> list[str]:
    """Return only guideline domains explicitly named in the latest turn."""

    text = query.lower()
    for source_id, keywords in PRIMARY_SOURCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return [source_id]

    source_ids = {
        source_id
        for source_id, keywords in SOURCE_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }
    if re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", text):
        source_ids.add("hypertension")
    return sorted(source_ids)


def _source_ids_for_retrieval(state: AgentState, fallback_query: str) -> list[str]:
    explicit_source_ids = _current_turn_source_ids(fallback_query)
    if explicit_source_ids:
        return explicit_source_ids

    # A short follow-up such as "ควรเริ่มปรับอาหารอย่างไร" may omit the lab
    # names. In that case the saved results determine which guideline domains
    # are relevant, but they must not broaden an explicitly scoped LDL/CKD turn.
    labs = state.get("extracted_lab_values") or {}
    return sorted({LAB_SOURCE_IDS[name] for name in labs if name in LAB_SOURCE_IDS})


def _should_retrieve_for_turn(state: AgentState, latest_user_message: str) -> bool:
    """Keep general wellness turns from inheriting stale clinical citations."""

    intent = state.get("intent")
    if intent in {"lab_interpretation", "medication_safety", "urgent_red_flag"}:
        return True
    if _current_turn_source_ids(latest_user_message):
        return True
    normalized = latest_user_message.strip().lower()
    return bool(
        _has_structured_health_data(state)
        and any(term in normalized for term in EVIDENCE_BACKED_LIFESTYLE_KEYWORDS)
    )


def _analysis_retrieval_query(state: AgentState, fallback_query: str) -> str:
    # Guideline filters carry the disease context. The embedding query should
    # represent only what the user asks now; appending every saved value or the
    # phrase "lab interpretation" overwhelms nearby food/exercise semantics.
    return fallback_query.strip()


def call_model(state: AgentState):
    """
    Node สำหรับตอบคำถาม: ดึง Context มาใส่ใน Prompt จริงๆ
    """
    messages = state["messages"]
    summary = state.get("summary", "")
    last_user_message = messages[-1].content

    is_overview = bool(state.get("health_overview_request"))
    should_retrieve = not is_overview and _should_retrieve_for_turn(
        state,
        last_user_message,
    )
    retrieval_query = _analysis_retrieval_query(state, last_user_message)
    source_ids = _source_ids_for_retrieval(state, last_user_message)
    # A dashboard-style overview is based on the user's saved measurements.
    # Do not retrieve textbook excerpts or expose citations on this route.
    retrieved = (
        RetrievedContext(context="", citations=[])
        if not should_retrieve
        else retrieve_context(retrieval_query, source_ids=source_ids)
    )
    context = retrieved.context

    summary_context = f"\n\nPrevious conversation summary: {summary}" if summary else ""
    summary_context += _analysis_slot_context(state)

    print("=== RETRIEVAL QUERY ===", retrieval_query if should_retrieve else "skipped for this turn")
    print(
        "=== RETRIEVAL SOURCE FILTER ===",
        (source_ids or "all") if should_retrieve else "none",
    )
    print("=== CONTEXT ===", context)

    if is_overview:
        system_prompt = health_overview_prompt(summary_context)
    elif should_retrieve and (context or _has_structured_health_data(state)):
        if not context:
            context = (
                "No sufficiently relevant RAG material was found. Use the structured user data above "
                "as the primary source, and do not state that the user has not provided laboratory values."
            )
        system_prompt = lab_prompt(context, summary_context, last_user_message)
    elif state.get("intent") == "general_health":
        system_prompt = general_health_prompt(summary_context, last_user_message)
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
        "steps": state.get("steps", []) + (["retrieval"] if should_retrieve else []) + ["generate"],
        "citations": retrieved.citations,
    }
    if pending_slot:
        result["pending_slot"] = pending_slot

    return result


def _drop_incomplete_trailing_markdown(content: str) -> str:
    """Remove an unfinished Markdown fragment if generation stopped mid-line.

    This is a presentation safeguard only.  The prompt keeps the substantive
    safety action first; this prevents a dangling token such as ``**การ`` from
    being shown immediately before the application-owned citations.
    """
    lines = content.rstrip().splitlines()
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith("**") and last.count("**") % 2:
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip()


def append_citations_node(state: AgentState):
    """Append only retrieval-backed textbook citations after safety review."""
    if "retrieval" not in state.get("steps", []):
        return {}
    if state.get("health_overview_request"):
        return {"citations": []}

    citation_text = format_citations(state.get("citations") or [])
    if not citation_text:
        return {}

    last_message = state["messages"][-1]
    # The model is instructed not to cite sources itself, but strip a citation
    # section defensively if it still does. The application owns this section
    # so references appear exactly once and contain retrieval-backed pages.
    content_without_model_citations = re.split(
        # Accept Thai headings generated in natural variants, including
        # "การอ้างอิงจากตำรา:" seen in the UI, and optional Markdown styling.
        r"\n\s*(?:#+\s*)?(?:\*\*)?\s*(?:(?:การ)?อ้างอิง(?:จากตำรา)?|references?)(?:\*\*)?\s*:\s*",
        str(last_message.content),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].rstrip()
    content_without_model_citations = _drop_incomplete_trailing_markdown(
        content_without_model_citations
    )
    content = f"{content_without_model_citations}\n\n{citation_text}"
    if hasattr(last_message, "model_copy"):
        message = last_message.model_copy(update={"content": content})
    else:
        message = last_message.copy(update={"content": content})
    return {"messages": [message]}
