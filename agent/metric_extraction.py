"""
LLM-judged extraction of health metrics from free-text chat.

The deterministic patterns in `intake.py` only reach 14 fields and need a number
sitting next to a keyword. This module covers the rest of the profile catalog and
the phrasings regex cannot follow, by asking a small model to decide which
numbers in a message are the user's own current measurements.

Canonical ids here MUST match `metric-catalog.ts` in health-chatbot-web; they are
the keys the web app writes onto the profile page.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_timing import timed_llm_invoke


# id -> (Thai label, unit, plausible min, plausible max)
METRIC_CATALOG: Dict[str, tuple[str, str, float, float]] = {
    "SBP": ("ความดันตัวบน (SBP)", "มม.ปรอท", 50, 260),
    "DBP": ("ความดันตัวล่าง (DBP)", "มม.ปรอท", 30, 180),
    "HeartRate": ("ชีพจร (Heart Rate)", "ครั้ง/นาที", 25, 220),
    "Weight": ("น้ำหนัก (Weight)", "กก.", 10, 400),
    "Height": ("ส่วนสูง (Height)", "ซม.", 50, 250),
    "BMI": ("ดัชนีมวลกาย (BMI)", "กก./ม.²", 8, 100),
    "WaistCircumference": ("รอบเอว (Waist)", "ซม.", 30, 250),
    "FBS": ("น้ำตาลในเลือดขณะอดอาหาร (FPG)", "มก./ดล.", 20, 900),
    "Glucose": ("น้ำตาลในเลือด (Glucose)", "มก./ดล.", 20, 900),
    "2-hr PG": ("น้ำตาลในเลือดที่ 2 ชั่วโมง (2-hr PG)", "มก./ดล.", 20, 900),
    "HbA1c": ("น้ำตาลสะสม (HbA1c)", "%", 3, 20),
    "Ketone": ("คีโตนในเลือด (Ketone)", "มิลลิโมล/ล.", 0, 30),
    "Total Cholesterol": ("คอเลสเตอรอลรวม (Total Cholesterol)", "มก./ดล.", 50, 600),
    "LDL": ("แอล ดี แอล คอเลสเตอรอล (LDL-C)", "มก./ดล.", 10, 500),
    "HDL": ("เอช ดี แอล คอเลสเตอรอล (HDL-C)", "มก./ดล.", 5, 150),
    "non-HDL": ("นอน-เอช ดี แอล คอเลสเตอรอล (non-HDL-C)", "มก./ดล.", 10, 550),
    "Triglycerides": ("ไตรกลีเซอไรด์ (Triglycerides)", "มก./ดล.", 20, 2000),
    "eGFR": ("อัตราการกรองของไต (eGFR)", "มล./นาที/1.73 ม.²", 1, 200),
    "Creatinine": ("ครีแอตินีน (Creatinine)", "มก./ดล.", 0.1, 25),
    "BUN": ("ยูเรียไนโตรเจนในเลือด (BUN)", "มก./ดล.", 1, 200),
    "UACR": ("อัลบูมินต่อครีแอตินีน (UACR)", "มก./ก.", 0, 10000),
    "Potassium": ("โพแทสเซียม (Potassium)", "มิลลิโมล/ล.", 1, 10),
    "Calcium": ("แคลเซียม (Calcium)", "มก./ดล.", 3, 20),
    "Phosphate": ("ฟอสเฟต (Phosphate)", "มก./ดล.", 0.5, 15),
    "Hemoglobin": ("ฮีโมโกลบิน (Hemoglobin)", "ก./ดล.", 3, 25),
    "AST": ("เอนไซม์ตับ (AST)", "ยูนิต/ล.", 1, 5000),
    "ALT": ("เอนไซม์ตับ (ALT)", "ยูนิต/ล.", 1, 5000),
}

_FIELD_LIST = "\n".join(
    f"- {metric_id} = {label} ({unit})"
    for metric_id, (label, unit, _, _) in METRIC_CATALOG.items()
)

_SYSTEM_PROMPT = (
    "You extract health measurements from Thai or English chat messages.\n"
    "Return ONLY a JSON object mapping field id to a number. No prose, no code fence.\n"
    "Return {} when the message contains no measurement.\n\n"
    "Allowed field ids:\n"
    f"{_FIELD_LIST}\n\n"
    "Record a value when the user reports it as their own — their latest result, what a\n"
    "doctor told them, or what they measured at home. Phrasing does not matter and the\n"
    "test name may be missing, implied, or written in Thai. Extract it whenever you can\n"
    "tell which field is meant.\n"
    "  'หมอบอกว่าไตผมกรองได้ 45'      -> {\"eGFR\": 45}\n"
    "  'เมื่อเช้าชั่งได้ 78 รอบเอว 92' -> {\"Weight\": 78, \"WaistCircumference\": 92}\n"
    "  'ค่าน้ำตาลสะสมผมล่าสุด 6.1'     -> {\"HbA1c\": 6.1}\n\n"
    "Return {} instead when the number is not the user's own present measurement:\n"
    "  'LDL ควรต่ำกว่า 100 ไหม'   -> {}  (a question about a target)\n"
    "  'แม่ผมน้ำตาล 200'          -> {}  (another person)\n"
    "  'เมื่อก่อน LDL เคย 200'     -> {}  (a past value they are not reporting now)\n"
    "  'กินยา 500 mg วันละ 2 ครั้ง' -> {}  (a medicine dose, not a lab value)\n\n"
    "Other rules:\n"
    "- Blood pressure written as 145/90 means SBP 145 and DBP 90.\n"
    "- Convert to the unit shown above; drop any value you cannot convert.\n"
    "- Never guess or infer a value that was not stated. Do not compute BMI yourself.\n"
)


def _coerce_catalog_value(metric_id: Any, raw_value: Any) -> Optional[float]:
    """Accept only known ids carrying a physiologically plausible number."""

    definition = METRIC_CATALOG.get(str(metric_id).strip())
    if not definition:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    _, _, minimum, maximum = definition
    if value < minimum or value > maximum:
        return None

    return value


def _parse_response(content: Any) -> Dict[str, float]:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    values: Dict[str, float] = {}
    for metric_id, raw_value in payload.items():
        value = _coerce_catalog_value(metric_id, raw_value)
        if value is not None:
            values[str(metric_id).strip()] = value

    return values


def extract_metrics_with_llm(text: str, context: str = "") -> Optional[Dict[str, float]]:
    """
    Ask the small model which numbers in `text` are the user's own measurements.

    Returns a dict when the model ran, including an empty one, which is a real
    decision that nothing here should be saved. Returns None when the model did
    not run at all, so the caller can fall back to the deterministic patterns
    instead of mistaking a failure for "nothing found".
    """

    if not text or not re.search(r"\d", text):
        return None

    from .models import intent_model

    prompt = f"ข้อความล่าสุดของผู้ใช้:\n{text}"
    if context:
        prompt = f"บริบทก่อนหน้า (ใช้ประกอบการตีความเท่านั้น):\n{context[-1500:]}\n\n{prompt}"

    try:
        response = timed_llm_invoke(
            intent_model,
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            "metric_extraction",
        )
    except Exception as exc:
        print(f"[metric_extraction] extraction failed, falling back to patterns: {exc}")
        return None

    return _parse_response(response.content)
