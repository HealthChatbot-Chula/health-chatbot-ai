"""
Slot-filling / information-extraction LangGraph for a medical checkup chatbot.

This graph collects structured patient information before handing off to the
main medical analyst node. It is intentionally self-contained so it can be used
directly, or composed into a larger application graph later.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .intake import (
    classify_interaction_intent,
    conversation_user_text,
    extract_lab_values,
    latest_user_text,
    looks_like_greeting,
    looks_like_scope_question,
    mentioned_lab_topics,
    needs_fasting_status,
    next_required_slot,
    parse_intake_updates,
)
from .metric_extraction import extract_metrics_with_llm
from .state import AgentState


def _latest_user_text(messages: List[BaseMessage]) -> str:
    """Return the latest human message content, or an empty string."""
    return latest_user_text(messages)


def _latest_assistant_text_before_latest_user(messages: List[BaseMessage]) -> str:
    """Return the assistant message immediately before the latest human message."""

    latest_human_index: Optional[int] = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            latest_human_index = index
            break

    if latest_human_index is None:
        return ""

    for message in reversed(messages[:latest_human_index]):
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)

    return ""


def _looks_like_greeting(text: str) -> bool:
    return looks_like_greeting(text)


def _looks_like_scope_question(text: str) -> bool:
    return looks_like_scope_question(text)


def _classify_interaction_intent(text: str, state: AgentState) -> str:
    return classify_interaction_intent(text, state)


def _intake_question_text() -> str:
    return (
        "ส่งค่าผลตรวจที่อยากให้ช่วยดูได้เลยครับ เช่น LDL 178, HbA1c 6.1 "
        "หรือความดัน 145/90 mmHg ถ้ามีอายุ เพศ โรคประจำตัว หรือยาที่ใช้อยู่ "
        "ส่งเพิ่มได้ จะช่วยตีความให้เหมาะกับบริบทมากขึ้นครับ"
    )


def _conversation_user_text(messages: List[BaseMessage]) -> str:
    return conversation_user_text(messages)


def _mentioned_lab_topics(text: str) -> List[str]:
    return mentioned_lab_topics(text)


def _extract_lab_values_from_text(text: str, context: str = "") -> Optional[Dict[str, float]]:
    return extract_lab_values(text, context)


def _known_profile_phrase(state: AgentState) -> str:
    parts: List[str] = []
    age = state.get("age")
    gender = state.get("gender")
    if age is not None:
        parts.append(f"อายุ {age} ปี")
    if gender == "male":
        parts.append("เพศชาย")
    elif gender == "female":
        parts.append("เพศหญิง")
    if not parts:
        return ""
    return f"โอเคครับ ทราบว่าเจ้าของผลตรวจ{' '.join(parts)}แล้ว "


def _topic_specific_lab_question(state: AgentState, messages: List[BaseMessage]) -> Optional[str]:
    latest_user_message = _latest_user_text(messages)
    previous_assistant_message = _latest_assistant_text_before_latest_user(messages)
    all_user_text = _conversation_user_text(messages)
    topics = _mentioned_lab_topics(f"{all_user_text}\n{latest_user_message}")
    prefix = _known_profile_phrase(state)

    if "blood_pressure" in topics:
        if "ความดันที่สูงมีผลได้" in previous_assistant_message:
            return (
                f"{prefix}เหลือขอค่าความดันที่วัดได้เป็นตัวเลขครับ เช่น 145/90 mmHg "
                "หรือบอกตัวบน/ตัวล่างก็ได้ครับ"
            )
        return (
            f"{prefix}ความดันที่สูงมีผลได้ครับ โดยเฉพาะถ้าสูงซ้ำ ๆ เพราะเพิ่มความเสี่ยงต่อหัวใจ "
            "หลอดเลือด และไต แต่ต้องดูจากตัวเลขที่วัดได้ก่อน "
            "ขอค่าความดันเป็นตัวเลขหน่อยครับ เช่น 145/90 mmHg "
            "หรือบอกตัวบน/ตัวล่างก็ได้ครับ ถ้ามีอาการเจ็บหน้าอก หอบ เหนื่อยมาก "
            "ปวดศีรษะรุนแรง แขนขาอ่อนแรง หรือพูดไม่ชัด ให้รีบพบแพทย์ทันทีครับ"
        )

    if "fbs" in topics or "hba1c" in topics:
        return (
            f"{prefix}ขอค่าน้ำตาลที่ขึ้นในใบตรวจหน่อยครับ เช่น FBS 112, Glucose 130 "
            "หรือ HbA1c 6.1 ถ้าเป็น FBS/Glucose บอกได้ด้วยว่าตรวจหลังงดอาหารไหมครับ"
        )

    if {"ldl", "hdl", "triglycerides", "cholesterol"} & set(topics):
        return (
            f"{prefix}ขอค่าชุดไขมันที่เห็นในใบตรวจหน่อยครับ เช่น LDL, HDL, "
            "Triglycerides หรือ Total cholesterol พร้อมตัวเลขที่ขึ้นสูง/ต่ำครับ"
        )

    if "egfr" in topics or "creatinine" in topics:
        return (
            f"{prefix}ขอค่าไตที่อยู่ในใบตรวจหน่อยครับ เช่น eGFR หรือ Creatinine "
            "พร้อมตัวเลข ถ้ามีค่าเดิมครั้งก่อนส่งมาด้วยจะช่วยดูแนวโน้มได้ครับ"
        )

    if "alt_ast" in topics:
        return (
            f"{prefix}ขอค่าตับที่ขึ้นในใบตรวจหน่อยครับ เช่น ALT หรือ AST พร้อมตัวเลข "
            "ถ้ามีประวัติดื่มแอลกอฮอล์ ยาที่ใช้ หรือไวรัสตับอักเสบ บอกเพิ่มได้ครับ"
        )

    return None


def _needs_fasting_status(lab_values: Optional[Dict[str, float]]) -> bool:
    return needs_fasting_status(lab_values)


def _updates_from_pending_slot(state: AgentState, latest_user_message: str) -> Dict[str, Any]:
    """Interpret short answers using the slot that the graph asked for last."""

    pending_slot = state.get("pending_slot")
    if not pending_slot:
        return {}

    return parse_intake_updates(
        state,
        latest_user_message,
        _conversation_user_text(state.get("messages", [])),
    )


def extract_info_node(state: AgentState) -> Dict[str, Any]:
    """
    Extract patient slots from the latest user message and merge them into state.

    Deterministic rules run first and always win. The model pass only fills
    catalog fields the patterns cannot reach, and is skipped entirely when the
    message has no digits, so digit-free intake turns stay LLM-free and fast.
    """

    messages = state.get("messages", [])
    latest_user_message = _latest_user_text(messages)
    if not latest_user_message:
        return {}

    conversation_context = _conversation_user_text(messages)
    extracted_updates = parse_intake_updates(
        state,
        latest_user_message,
        conversation_context,
    )

    model_values = extract_metrics_with_llm(latest_user_message, conversation_context)
    if model_values is not None:
        # The model read the whole sentence, so it decides for this turn. Patterns
        # cannot tell "LDL 154" from "LDL ควรต่ำกว่า 100 ไหม" and would save the
        # threshold out of a question. An empty result means "nothing to save here",
        # which is why it still overrides the patterns.
        known_values = state.get("extracted_lab_values") or {}
        merged_values = {**known_values, **model_values}
        if merged_values != known_values:
            extracted_updates["extracted_lab_values"] = merged_values
            extracted_updates["pending_slot"] = None
        else:
            extracted_updates.pop("extracted_lab_values", None)
            if not merged_values:
                # Nothing was captured, so the turn must not look like the lab
                # slot got answered; leave the pending question standing.
                extracted_updates.pop("pending_slot", None)

    merged_state: AgentState = {**state, **extracted_updates}
    intent = _classify_interaction_intent(latest_user_message, merged_state)

    print(
        "[slot_filling] intake: "
        f"updates={{{', '.join(sorted(extracted_updates.keys()))}}}, "
        f"model_fields={sorted(model_values) if model_values is not None else 'skipped'}, "
        f"intent={intent}"
    )

    return {
        **extracted_updates,
        "intent": intent,
    }


def route_after_extraction(state: AgentState) -> str:
    """Route after extraction without blocking answers on optional profile slots."""

    latest_user_message = _latest_user_text(state.get("messages", []))
    labs = state.get("extracted_lab_values") or {}

    # When the preceding turn asked for a laboratory value, a reply containing
    # only profile information (for example, "I am 26") must stay on the
    # deterministic intake path.  Previously it could fall through to the
    # analyst, trigger RAG, and append irrelevant textbook citations.
    if state.get("pending_slot") == "extracted_lab_values" and not labs:
        return "ask_lab_node"

    if state.get("intent") == "general_info" and (
        _looks_like_greeting(latest_user_message)
        or _looks_like_scope_question(latest_user_message)
    ):
        return "ask_lab_node"

    next_slot = next_required_slot(state)
    if next_slot == "extracted_lab_values":
        if state.get("intent") not in {"lab_interpretation", "general_info"}:
            return "our_agent"
        return "ask_lab_node"

    # Once lab/profile data is available, answer the primary question first.
    # Age, gender, and fasting status are collected by the analyst follow-up.
    return "our_agent"


def ask_lab_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    latest_user_message = _latest_user_text(messages)
    topic_specific_question = _topic_specific_lab_question(state, messages)
    labs = state.get("extracted_lab_values") or {}

    if topic_specific_question:
        content = topic_specific_question
    elif _looks_like_scope_question(latest_user_message):
        content = (
            "ผมช่วยให้ข้อมูลเบื้องต้นและช่วยแปลผลตรวจในกลุ่มหลัก ๆ เหล่านี้ครับ:\n"
            "- เบาหวาน\n"
            "- ความดันโลหิตสูง\n"
            "- ไขมันในเลือดสูง\n"
            "- โรคไตเรื้อรัง (CKD)\n"
            "- ภาวะที่เกี่ยวข้องกับการทำงานของตับ\n\n"
            f"{_intake_question_text()}"
        )
    elif _looks_like_greeting(latest_user_message):
        if labs:
            content = (
                "สวัสดีครับ ผมยังใช้ข้อมูลผลตรวจที่คุยกันไว้เป็นบริบทได้ครับ "
                "ถ้าอยากให้ช่วยดูค่าล่าสุดหรือถามต่อจากผลตรวจเดิม ส่งคำถามมาได้เลยครับ"
            )
        else:
            content = (
                "สวัสดีครับ ผมช่วยดูผลตรวจสุขภาพเบื้องต้นได้ครับ\n\n"
                f"{_intake_question_text()}"
            )
    else:
        content = _intake_question_text()

    result: Dict[str, Any] = {
        "messages": [
            AIMessage(content=content)
        ],
        # Explicitly clear any transient retrieval data on an intake-only turn.
        "citations": [],
    }
    if not labs:
        result["pending_slot"] = "extracted_lab_values"
    return result


def ask_fasting_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "fasting_status",
        "citations": [],
        "messages": [
            AIMessage(
                content=(
                    "ผลเลือดชุดนี้ตรวจหลังงดอาหารหรือไม่ครับ? กรุณาตอบว่าใช่หรือไม่ใช่"
                )
            )
        ]
    }


def ask_age_node(state: AgentState) -> Dict[str, Any]:
    if state.get("gender") is None:
        question = "ผู้ที่เป็นเจ้าของผลตรวจอายุเท่าไร และเพศชายหรือหญิงครับ?"
    else:
        question = "ผู้ที่เป็นเจ้าของผลตรวจอายุเท่าไรครับ?"

    return {
        "pending_slot": "age",
        "citations": [],
        "messages": [
            AIMessage(content=question)
        ]
    }


def ask_gender_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "gender",
        "citations": [],
        "messages": [
            AIMessage(
                content=(
                    "เพศของผู้ที่เป็นเจ้าของผลตรวจคือชายหรือหญิงครับ? "
                    "ข้อมูลนี้ช่วยให้เทียบช่วงอ้างอิงได้เหมาะสมขึ้น"
                )
            )
        ]
    }


def our_agent(state: AgentState) -> Dict[str, Any]:
    """
    Placeholder for the main medical analyst agent.

    Replace this node with the real RAG/analysis agent once slot collection is
    complete.
    """

    labs = state.get("extracted_lab_values") or {}
    lab_summary = ", ".join(f"{name}: {value:g}" for name, value in labs.items())

    content = (
        "ขอบคุณครับ ตอนนี้มีข้อมูลสำคัญพอสำหรับส่งต่อให้ agent วิเคราะห์หลักแล้ว: "
        f"อายุ {state.get('age')}, เพศ {state.get('gender')}, "
        f"สถานะการงดอาหาร {state.get('fasting_status')}, ค่าผลตรวจ [{lab_summary}] "
        "ขั้นถัดไปจะเป็นการทำงานของ medical analyst agent หลัก โดย placeholder นี้ยังไม่วินิจฉัยโรคหรือให้แผนการรักษาครับ"
    )

    return {"messages": [AIMessage(content=content)], "pending_slot": None}


def build_slot_filling_graph():
    """Build and compile the slot-filling LangGraph."""

    graph = StateGraph(AgentState)

    graph.add_node("extract_info_node", extract_info_node)
    graph.add_node("ask_lab_node", ask_lab_node)
    graph.add_node("ask_fasting_node", ask_fasting_node)
    graph.add_node("ask_age_node", ask_age_node)
    graph.add_node("ask_gender_node", ask_gender_node)
    graph.add_node("our_agent", our_agent)

    graph.set_entry_point("extract_info_node")

    graph.add_conditional_edges(
        "extract_info_node",
        route_after_extraction,
        {
            "ask_lab_node": "ask_lab_node",
            "ask_fasting_node": "ask_fasting_node",
            "ask_age_node": "ask_age_node",
            "ask_gender_node": "ask_gender_node",
            "our_agent": "our_agent",
        },
    )

    # These nodes intentionally end the turn. On the next user response, invoke
    # the graph again with the accumulated state and the new HumanMessage.
    graph.add_edge("ask_lab_node", END)
    graph.add_edge("ask_fasting_node", END)
    graph.add_edge("ask_age_node", END)
    graph.add_edge("ask_gender_node", END)
    graph.add_edge("our_agent", END)

    return graph.compile()


app = build_slot_filling_graph()


if __name__ == "__main__":
    example_state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "I am a 45 year old male. My FBS is 126 and LDL is 154. "
                    "I fasted before the test."
                )
            )
        ],
        "age": None,
        "gender": None,
        "underlying_disease": None,
        "current_medications": None,
        "current_symptoms": None,
        "fasting_status": None,
        "extracted_lab_values": None,
    }

    result = app.invoke(example_state)
    print(result)
