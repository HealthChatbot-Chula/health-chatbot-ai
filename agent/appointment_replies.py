"""Fixed Thai wording for appointment facts. Never changes booking data."""

import json
import re
from datetime import date as Date, datetime
from zoneinfo import ZoneInfo


BANGKOK = ZoneInfo("Asia/Bangkok")
MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
          "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม")
WEEKDAYS = ("วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์")

_FOREIGN_SCRIPTS = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯぀-ヿ㐀-鿿]")
UNKNOWN_OUTCOME = "ยังยืนยันผลรายการนี้ไม่ได้ครับ กรุณาพิมพ์ \"ดูนัดหมาย\" เพื่อตรวจสอบสถานะล่าสุดก่อนทำรายการซ้ำ"


def reply_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def strip_foreign(text: str) -> str:
    """The small model occasionally emits stray CJK/Korean characters."""
    return _FOREIGN_SCRIPTS.sub("", text)


def tool_data(message) -> dict:
    try:
        value = json.loads(reply_text(message.content))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def local_slot(instant: str):
    """Calendar instant -> ('YYYY-MM-DD', 'HH:MM') in Bangkok, or None."""
    try:
        start = datetime.fromisoformat(instant.replace("Z", "+00:00"))
        if start.tzinfo is None:
            return None
        start = start.astimezone(BANGKOK)
        return start.date().isoformat(), start.strftime("%H:%M")
    except (ValueError, TypeError, AttributeError):
        return None


def weekday_name(date: str) -> str:
    try:
        return WEEKDAYS[Date.fromisoformat(date).weekday()]
    except (ValueError, TypeError):
        return ""


def date_label(date: str) -> str:
    try:
        day = Date.fromisoformat(date)
    except (ValueError, TypeError):
        return str(date)
    return f"{WEEKDAYS[day.weekday()]}ที่ {day.day} {MONTHS[day.month - 1]} {day.year}"


def slot_label(slot: dict) -> str:
    return f"{date_label(slot.get('date'))} เวลา {slot.get('time')} น."


# --- confirmation -----------------------------------------------------------

CONFIRM_CHOICES = [{"label": "ยืนยัน", "value": "ยืนยัน"}, {"label": "ไม่ยืนยัน", "value": "ไม่ยืนยัน"}]
_YES = ("ยืนยัน", "ใช่", "ตกลง", "โอเค", "ok", "yes", "confirm", "คอนเฟิร์ม", "ได้", "เอาตามนี้", "ตามนี้", "ถูกต้อง", "เอาเลย")
_NO = ("ไม่ยืนยัน", "ไม่ยกเลิก", "ไม่ใช่", "ไม่เอา", "ไม่ต้อง", "ยังไม่", "ยังก่อน", "เปลี่ยนใจ", "no", "ยกเลิก")
_NOT_AN_ANSWER = re.compile(r"\d|แต่|ไหม|มั้ย|หรือ|เป็น|ขอ")


def pending_action(pending) -> str:
    """'cancel' only when every stored operation is a cancellation."""
    operations = [pending] if isinstance(pending, dict) else list(pending or [])
    actions = {str(item.get("action", "")) for item in operations if isinstance(item, dict)}
    return actions.pop() if len(actions) == 1 else ""


def confirmation_decision(text: str, action: str = ""):
    """'yes' / 'no' for a short answer to the confirmation question.

    The buttons send exactly "ยืนยัน" / "ไม่ยืนยัน". Typed answers may carry
    endings ("ยืนยันจร้า", "ใช่ค่ะ"), so a short message that starts with a
    yes/no word counts. Anything carrying new details ("ขอเป็น 15:00 แทน",
    "ใช่ แต่...") returns None and goes to the model as a new request.
    """
    answer = re.sub(r"[\s.!?,\"'“”]+", "", text or "").lower()
    if not answer or len(answer) > 16 or _NOT_AN_ANSWER.search(answer):
        return None
    if action == "cancel" and answer.startswith("ยกเลิก"):
        return "yes"
    if answer == "ไม่" or answer.startswith(_NO):
        return "no"
    if answer.startswith(_YES):
        return "yes"
    return None


def _operation_text(operation: dict) -> str:
    action = operation.get("action")
    if action == "cancel":
        return f"ยกเลิกนัดหมาย{slot_label(operation['from'])}"
    if action == "reschedule":
        return f"เลื่อนนัดหมายจาก{slot_label(operation['from'])} เป็น{slot_label(operation['to'])}"
    return f"จองนัดหมาย{slot_label(operation['to'])}"


def confirmation_text(pending: list) -> str:
    ending = "กด \"ยืนยัน\" เพื่อทำรายการ หรือแจ้งวันและเวลาอื่นได้เลยครับ"
    if len(pending) == 1:
        return f"ต้องการ{_operation_text(pending[0])} ใช่ไหมครับ {ending}"
    lines = "\n".join(f"- {_operation_text(item)}" for item in pending)
    return f"ต้องการทำ {len(pending)} รายการต่อไปนี้ใช่ไหมครับ\n{lines}\n{ending}"


# --- results ----------------------------------------------------------------

def _nearest(times, near: str = "", count: int = 6) -> list:
    """The free slots closest to what the user asked for, in clock order."""
    times = [value for value in times or [] if isinstance(value, str)]
    if len(times) <= count:
        return times

    def minutes(value):
        return int(value[:2]) * 60 + int(value[3:])
    try:
        target = minutes(near) if near else minutes(times[0])
        return sorted(sorted(times, key=lambda value: abs(minutes(value) - target))[:count])
    except ValueError:
        return times[:count]


def _suggestion_text(suggestion, near: str = "") -> str:
    if not isinstance(suggestion, dict) or not suggestion.get("times"):
        return ""
    return f"{date_label(suggestion.get('date'))} มีเวลาว่าง เช่น {', '.join(_nearest(suggestion['times'], near))} น."


def _short_date(date: str) -> str:
    day = Date.fromisoformat(date)
    return f"{day.day}/{day.month}/{day.year}"


def suggestion_choices(result: dict):
    """Buttons for the suggested free slots. Each sends a complete request, so
    choosing one never depends on how the previous turn was understood."""
    suggestion, retry = result.get("suggestion"), result.get("retry") or {}
    if not isinstance(suggestion, dict) or retry.get("action") not in {"create", "reschedule"}:
        return None
    requested = result.get("requested") or {}
    try:
        target = f"วันที่ {_short_date(suggestion['date'])} เวลา "
        if retry["action"] == "reschedule":
            lead = f"ขอเลื่อนนัดวันที่ {_short_date(retry['from']['date'])} เวลา {retry['from']['time']} เป็น{target}"
        else:
            lead = f"ขอจองนัด{target}"
        day = Date.fromisoformat(suggestion["date"])
    except (KeyError, ValueError, TypeError):
        return None
    other_day = suggestion["date"] != requested.get("date")
    return [{"label": f"{day.day}/{day.month} {time}" if other_day else time, "value": lead + time}
            for time in _nearest(suggestion.get("times"), requested.get("time") or "", 4)] or None


def error_reply(result: dict, wrote: bool = False) -> str:
    """Explain a refused/failed operation using only what the backend said.

    `wrote` marks a real write attempt: if its outcome is unknown the calendar
    may already have changed, so the user must check before retrying.
    """
    code, status = result.get("code"), result.get("status")
    message = result.get("error") if isinstance(result.get("error"), str) else ""
    if code in {"ambiguous", "not_found"} and isinstance(result.get("appointments"), list):
        lines = "\n".join(f"- {slot_label(item)}" for item in result["appointments"])
        lead = "พบนัดหมายมากกว่าหนึ่งรายการครับ ต้องการทำรายการกับนัดใดครับ" if code == "ambiguous" \
            else "ไม่พบนัดหมายตามวันเวลาที่ระบุครับ นัดหมายของคุณมีดังนี้"
        return f"{lead}\n{lines}"
    if status == 401:
        return "สิทธิ์ทำรายการหมดอายุครับ กรุณาส่งคำขออีกครั้ง"
    refused = message and code != "unknown_outcome" and (status in {400, 404, 409} or (code and not status))
    if not refused:
        return UNKNOWN_OUTCOME if wrote else "ขณะนี้ติดต่อระบบนัดหมายไม่ได้ครับ กรุณาลองใหม่อีกครั้งในอีกสักครู่"
    text = message if message.endswith(("ครับ", "ค่ะ")) else message + "ครับ"
    requested = result.get("requested") or {}
    suggestion = _suggestion_text(result.get("suggestion"), requested.get("time") or "")
    if suggestion:
        text += f" {suggestion} สะดวกเวลาไหนครับ"
    return text


def write_failure(operation: dict, result: dict) -> dict:
    """A failed write plus what was being attempted, for wording and buttons."""
    return {**result, "requested": operation.get("to"),
            "retry": {"action": operation.get("action"), "from": operation.get("from")}}


def write_reply(operation: dict, result: dict) -> str:
    if result.get("error"):
        return error_reply(write_failure(operation, result), wrote=True)
    if operation.get("action") == "cancel" and result.get("status") == "cancelled":
        return f"ยกเลิกนัดหมาย{slot_label(operation['from'])} เรียบร้อยแล้วครับ"
    saved = local_slot((result.get("appointment") or {}).get("start", ""))
    if result.get("status") == "confirmed" and saved:
        label = slot_label({"date": saved[0], "time": saved[1]})
        if operation.get("action") == "reschedule":
            return f"เลื่อนนัดหมายเรียบร้อยแล้วครับ นัดใหม่คือ{label}"
        return f"บันทึกนัดหมายเรียบร้อยแล้วครับ {label}"
    return UNKNOWN_OUTCOME


def read_reply(result: dict) -> str:
    """Fallback wording when the model returns no text after a read tool."""
    if result.get("error"):
        return error_reply(result)
    if isinstance(result.get("appointments"), list):
        if not result["appointments"]:
            return "ขณะนี้คุณไม่มีนัดหมายที่กำลังจะถึงครับ"
        return "นัดหมายของคุณมีดังนี้ครับ\n" + "\n".join(f"- {slot_label(item)}" for item in result["appointments"])
    if isinstance(result.get("free_times"), list):
        if result["free_times"]:
            return f"{date_label(result.get('date'))} มีเวลาว่าง {', '.join(result['free_times'])} น. ครับ ต้องการเวลาไหนครับ"
        reason = {"closed_day": "เป็นวันหยุด (เปิดจันทร์–ศุกร์)", "past": "ผ่านมาแล้ว"}.get(result.get("reason"), "ไม่มีเวลาว่างแล้ว")
        text = f"{date_label(result.get('date'))} {reason}ครับ"
        nearby = result.get("next_available") or {}
        suggestion = _suggestion_text({"date": nearby.get("date"), "times": nearby.get("free_times")})
        return f"{text} {suggestion} สะดวกเวลาไหนครับ" if suggestion else text
    return ""
