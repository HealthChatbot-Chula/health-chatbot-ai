"""Small routing hints; conversational follow-ups use the semantic classifier."""


def is_appointment_request(text: str) -> bool:
    normalized = text.lower()
    return any(term in normalized for term in (
        "นัดหมาย", "จองนัด", "เลื่อนนัด", "ยกเลิกนัด", "ดูนัด", "เช็กนัด",
        "เช็คนัด", "เวลาว่าง", "สล็อต", "appointment", "จองเวลา",
        "ขอนัด", "นัดของ", "มีนัด", "วันนัด", "นัดว่าง", "นัดได้",
    ))


def route_after_turn_intent(state):
    return "appointment_agent" if state.get("intent") == "appointment" else "extract_info_node"
