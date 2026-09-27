from typing import Optional, Tuple

from .constants import FASTING_FOLLOWUP_LABS
from .state import AgentState


def _structured_followup_question_for_slot(pending_slot: Optional[str]) -> Optional[str]:
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


def _next_structured_followup(state: AgentState) -> Tuple[Optional[str], Optional[str]]:
    labs = state.get("extracted_lab_values") or {}
    if not labs or state.get("pending_slot"):
        return None, None

    if state.get("gender") is None:
        return "gender", _structured_followup_question_for_slot("gender")

    if state.get("age") is None:
        return "age", _structured_followup_question_for_slot("age")

    if state.get("fasting_status") is None and any(name in FASTING_FOLLOWUP_LABS for name in labs):
        return "fasting_status", _structured_followup_question_for_slot("fasting_status")

    return None, None


def _append_structured_followup(content: str, followup_question: str) -> str:
    trimmed_content = content.rstrip()
    if followup_question in trimmed_content:
        return trimmed_content
    return f"{trimmed_content}\n\n{followup_question}"
