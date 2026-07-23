import json
import re
from typing import Any, Dict, Iterable, Optional

from langchain_core.messages import BaseMessage

from .state import AgentState


FASTING_REQUIRED_LABS = {"FBS", "Glucose", "Triglycerides"}
SAFETY_FIRST_INTENTS = {"medication_safety", "urgent_red_flag"}

LAB_ALIASES = {
    "fbs": "FBS",
    "fasting blood sugar": "FBS",
    "glucose": "Glucose",
    "hba1c": "HbA1c",
    "a1c": "HbA1c",
    "total cholesterol": "Total Cholesterol",
    "cholesterol": "Total Cholesterol",
    "คอเลสเตอรอล": "Total Cholesterol",
    "hdl": "HDL",
    "ldl": "LDL",
    "tg": "Triglycerides",
    "triglyceride": "Triglycerides",
    "triglycerides": "Triglycerides",
    "ไตรกลีเซอไรด์": "Triglycerides",
    "creatinine": "Creatinine",
    "ครีเอตินิน": "Creatinine",
    "egfr": "eGFR",
    "bun": "BUN",
    "ast": "AST",
    "alt": "ALT",
    "sbp": "SBP",
    "systolic": "SBP",
    "ความดันตัวบน": "SBP",
    "ตัวบน": "SBP",
    "dbp": "DBP",
    "diastolic": "DBP",
    "ความดันตัวล่าง": "DBP",
    "ตัวล่าง": "DBP",
}

LAB_VALUE_PATTERNS = (
    ("FBS", r"\b(?:fbs|fasting blood sugar)\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Glucose", r"\bglucose\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Glucose", r"น้ำตาล(?:ในเลือด)?\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("HbA1c", r"\b(?:hba1c|a1c)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
    ("HbA1c", r"น้ำตาลสะสม\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
    ("LDL", r"\bldl\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("HDL", r"\bhdl\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Triglycerides", r"\b(?:tg|triglycerides?|ไตรกลีเซอไรด์)\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Total Cholesterol", r"\b(?:total cholesterol|cholesterol)\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Total Cholesterol", r"คอเลสเตอรอล\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("eGFR", r"\begfr\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Creatinine", r"\bcreatinine\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("Creatinine", r"ครีเอตินิน\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("BUN", r"\bbun\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("ALT", r"\balt\s*[:=]?\s*(\d+(?:\.\d+)?)"),
    ("AST", r"\bast\s*[:=]?\s*(\d+(?:\.\d+)?)"),
)

LAB_TOPIC_PATTERNS = (
    ("blood_pressure", ("ความดัน", "ค่าความดัน", "bp", "blood pressure")),
    ("ldl", ("ldl",)),
    ("hdl", ("hdl",)),
    ("triglycerides", ("triglyceride", "triglycerides", "ไตรกลีเซอไรด์", "ไขมัน")),
    ("hba1c", ("hba1c", "a1c", "น้ำตาลสะสม")),
    ("fbs", ("fbs", "น้ำตาล", "glucose")),
    ("egfr", ("egfr", "ค่าไต")),
    ("creatinine", ("creatinine", "ครีเอตินิน")),
    ("alt_ast", ("alt", "ast", "ค่าตับ")),
    ("cholesterol", ("cholesterol", "คอเลสเตอรอล")),
)


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def latest_user_text(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if getattr(message, "type", None) == "human":
            return message_text(message)
    return ""


def conversation_user_text(messages: Iterable[BaseMessage]) -> str:
    return "\n".join(
        message_text(message)
        for message in messages
        if getattr(message, "type", None) == "human"
    )


def mentioned_lab_topics(text: str) -> list[str]:
    normalized = text.strip().lower()
    topics: list[str] = []
    for topic, patterns in LAB_TOPIC_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            topics.append(topic)
    return topics


def extract_lab_values(text: str, context: str = "") -> Optional[Dict[str, float]]:
    normalized_context = f"{context}\n{text}".lower()
    values: Dict[str, float] = {}

    bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if bp_match and (
        "ความดัน" in normalized_context
        or "bp" in normalized_context
        or "blood pressure" in normalized_context
    ):
        sbp = float(bp_match.group(1))
        dbp = float(bp_match.group(2))
        if 50 <= sbp <= 260 and 30 <= dbp <= 180:
            values["SBP"] = sbp
            values["DBP"] = dbp

    thai_bp_match = re.search(
        r"ตัวบน\s*(\d{2,3}).{0,20}ตัวล่าง\s*(\d{2,3})",
        text,
        flags=re.IGNORECASE,
    )
    if thai_bp_match:
        sbp = float(thai_bp_match.group(1))
        dbp = float(thai_bp_match.group(2))
        if 50 <= sbp <= 260 and 30 <= dbp <= 180:
            values["SBP"] = sbp
            values["DBP"] = dbp

    for canonical_name, pattern in LAB_VALUE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values[canonical_name] = float(match.group(1))

    return values or None


def normalize_age(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None

    raw_text = str(value).strip().lower()
    polite_stripped = re.sub(r"\s*(?:ครับ|ค่ะ|คะ)$", "", raw_text).strip()
    age_match = (
        re.search(r"(?:อายุ|age)\s*[:：-]?\s*(\d{1,3})", polite_stripped)
        or re.search(r"\b(\d{1,3})\s*(?:ปี|years?|yrs?|yo)", polite_stripped)
        or re.search(r"^\s*(\d{1,3})\s*(?:ชาย|หญิง|male|female|m|f)?\s*$", polite_stripped)
        or re.search(r"^\s*(?:ชาย|หญิง|male|female|m|f)\s*(\d{1,3})\s*$", polite_stripped)
    )
    if not age_match:
        return None

    age = int(age_match.group(1))
    if 0 < age < 130:
        return age
    return None


def normalize_gender(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    normalized = re.sub(r"\s*(?:ครับ|ค่ะ|คะ)$", "", normalized).strip()
    if re.search(r"\b(?:male|man|m)\b", normalized) or "เพศชาย" in normalized or "ผู้ชาย" in normalized:
        return "male"
    if re.search(r"\b(?:female|woman|f)\b", normalized) or "เพศหญิง" in normalized or "ผู้หญิง" in normalized:
        return "female"
    if re.search(r"(^|\s)ชาย($|\s)", normalized):
        return "male"
    if re.search(r"(^|\s)หญิง($|\s)", normalized):
        return "female"
    return None


def normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    no_terms = (
        "ไม่ได้งดอาหาร",
        "ไม่ได้งด",
        "ไม่งดอาหาร",
        "ไม่งด",
        "ไม่ใช่",
        "ไม่ใช่ครับ",
        "ไม่ใช่ค่ะ",
        "non-fasting",
        "non fasting",
        "not fasting",
    )
    if any(term in normalized for term in no_terms) or normalized in {"no", "n", "false", "ไม่"}:
        return "no"

    yes_terms = (
        "งดอาหาร",
        "งด",
        "fasting",
        "fasted",
        "ใช่",
        "ใช่ครับ",
        "ใช่ค่ะ",
    )
    if any(term in normalized for term in yes_terms) or normalized in {"yes", "y", "true"}:
        return "yes"

    return None


def _normalize_string_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if value == []:
        return []
    if isinstance(value, str):
        if not value.strip():
            return None
        raw_items = re.split(r",|;|\n", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def merge_lists(current: Optional[list[str]], extracted: Optional[list[str]]) -> Optional[list[str]]:
    if extracted is None:
        return current
    if extracted == []:
        return []
    if not current:
        return extracted

    merged = list(current)
    seen = {item.casefold() for item in current}
    for item in extracted:
        key = item.casefold()
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def merge_lab_values(
    current: Optional[Dict[str, float]],
    extracted: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if extracted is None:
        return current
    if not current:
        return extracted
    return {**current, **extracted}


def normalize_lab_values(value: Any) -> Optional[Dict[str, float]]:
    if value is None or not isinstance(value, dict):
        return None

    normalized_values: Dict[str, float] = {}
    for raw_name, raw_value in value.items():
        if raw_value is None or raw_value == "":
            continue
        canonical_key = LAB_ALIASES.get(str(raw_name).strip().casefold(), str(raw_name).strip())
        try:
            normalized_values[canonical_key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return normalized_values or None


def _list_update_from_no_phrase(text: str, slot_name: str) -> Optional[list[str]]:
    normalized = text.strip().lower()
    no_markers = {
        "underlying_disease": ("ไม่มีโรค", "ไม่มีโรคประจำตัว"),
        "current_medications": ("ไม่มียา", "ไม่ได้กินยา", "ไม่กินยา"),
        "current_symptoms": ("ไม่มีอาการ",),
    }
    if any(marker in normalized for marker in no_markers.get(slot_name, ())):
        return []
    return None


def parse_intake_updates(state: AgentState, latest_user_message: str, context: str = "") -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    pending_slot = state.get("pending_slot")

    age = normalize_age(latest_user_message)
    if age is not None:
        updates["age"] = age

    gender = normalize_gender(latest_user_message)
    if gender is not None:
        updates["gender"] = gender

    fasting_status = normalize_yes_no(latest_user_message)
    if fasting_status is not None:
        updates["fasting_status"] = fasting_status

    lab_values = extract_lab_values(latest_user_message, context)
    merged_labs = merge_lab_values(state.get("extracted_lab_values"), lab_values)
    if merged_labs is not state.get("extracted_lab_values"):
        updates["extracted_lab_values"] = merged_labs

    for slot_name in ("underlying_disease", "current_medications", "current_symptoms"):
        extracted_list = _list_update_from_no_phrase(latest_user_message, slot_name)
        merged_list = merge_lists(state.get(slot_name), extracted_list)
        if merged_list is not state.get(slot_name):
            updates[slot_name] = merged_list

    if pending_slot == "age" and "age" in updates:
        updates["pending_slot"] = None
    elif pending_slot == "gender" and "gender" in updates:
        updates["pending_slot"] = None
    elif pending_slot == "fasting_status" and "fasting_status" in updates:
        updates["pending_slot"] = None
    elif pending_slot == "extracted_lab_values" and "extracted_lab_values" in updates:
        updates["pending_slot"] = None
    elif not pending_slot and updates:
        updates["pending_slot"] = None

    return updates


def looks_like_greeting(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "สวัสดี",
        "สวัสดีครับ",
        "สวัสดีค่ะ",
        "หวัดดี",
        "หวัดดีครับ",
        "หวัดดีค่ะ",
        "hello",
        "hi",
    }


def looks_like_scope_question(text: str) -> bool:
    normalized = text.strip().lower()
    scope_phrases = (
        "โรคอะไร",
        "โรคอะไรได้บ้าง",
        "ช่วยอะไรได้บ้าง",
        "ทำอะไรได้บ้าง",
        "ให้ข้อมูลเกี่ยวกับ",
        "ขอบเขต",
        "scope",
        "capability",
    )
    return any(phrase in normalized for phrase in scope_phrases)


def classify_interaction_intent(text: str, state: AgentState) -> str:
    normalized = text.strip().lower()
    lab_values = state.get("extracted_lab_values") or {}

    medication_terms = (
        "หยุดยา",
        "หยุดยาเอง",
        "ลดยา",
        "เพิ่มยา",
        "ปรับยา",
        "กินยา",
        "ยาความดัน",
        "ยาเบาหวาน",
        "ยาไต",
        "ยามันทำให้",
        "ต้องหยุดไหม",
        "หยุดไหม",
        "หยุดได้ไหม",
        "ไม่กินยา",
    )
    if any(term in normalized for term in medication_terms):
        return "medication_safety"

    urgent_terms = (
        "เจ็บหน้าอก",
        "หอบ",
        "หายใจไม่ออก",
        "หมดสติ",
        "ซึม",
        "ชัก",
        "เวียนหัวมาก",
        "มึนหัวมาก",
        "อ่อนเพลียมาก",
        "ปากแห้ง",
        "ปัสสาวะบ่อย",
        "รอดู",
        "ไปพรุ่งนี้",
        "ฉุกเฉิน",
    )
    if any(term in normalized for term in urgent_terms):
        return "urgent_red_flag"

    if "Potassium" in lab_values:
        return "urgent_red_flag"

    glucose_value = lab_values.get("Glucose") or lab_values.get("FBS")
    if glucose_value is not None and glucose_value >= 300:
        return "urgent_red_flag"

    if lab_values or mentioned_lab_topics(normalized):
        return "lab_interpretation"

    if looks_like_greeting(normalized) or looks_like_scope_question(normalized):
        return "general_info"

    return state.get("intent") or "general_health"


def needs_fasting_status(lab_values: Optional[Dict[str, float]]) -> bool:
    if not lab_values:
        return False
    return any(name in FASTING_REQUIRED_LABS for name in lab_values)


def next_required_slot(state: AgentState) -> Optional[str]:
    intent = state.get("intent")
    if intent in SAFETY_FIRST_INTENTS:
        return None

    labs = state.get("extracted_lab_values") or {}
    if not labs:
        return "extracted_lab_values"

    if state.get("age") is None:
        return "age"

    if state.get("gender") is None:
        return "gender"

    if state.get("fasting_status") is None and needs_fasting_status(labs):
        return "fasting_status"

    return None
