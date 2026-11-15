"""Appointment tools call the web backend; Google credentials never enter AI.

The web backend owns every booking rule (opening days, 30-minute grid, no past
or double bookings). Tools here only translate between the model and that API.
Write tools do NOT write: they validate with the backend and return a proposal
that the user must confirm (see appointments.py).
"""

import json
import os
import time as clock
from typing import Annotated, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .appointment_inputs import normalize_appointment_time
from .appointment_replies import local_slot, weekday_name


def _web_request(state: dict, action: str, **arguments) -> dict:
    if "time" in arguments:
        normalized_time = normalize_appointment_time(arguments["time"])
        if normalized_time is None:
            return {"error": "รูปแบบเวลาไม่ถูกต้อง กรุณาระบุเวลาแบบ 14:30", "status": 400, "code": "invalid_time"}
        arguments["time"] = normalized_time
    context = state.get("appointment_context") or {}
    token = context.get("token")
    endpoint = os.getenv("WEB_APPOINTMENTS_URL", "")
    if not token or not endpoint:
        return {"error": "ระบบนัดหมายยังไม่เปิดใช้งาน กรุณาติดต่อผู้ดูแล"}
    # The endpoint is server configuration, never a model/user-provided URL.
    request = Request(
        endpoint,
        data=json.dumps({"action": action, **arguments}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    started = clock.monotonic()
    try:
        with urlopen(request, timeout=70) as response:
            result = json.load(response)
        print(f"[Appointment API] action={action} status=200 elapsed_ms={int((clock.monotonic() - started) * 1000)}")
        return result
    except HTTPError as exc:
        try:
            body = json.loads(exc.read()).get("error", {})
        except (ValueError, AttributeError):
            body = {}
        message = body.get("message") or "ระบบนัดหมายไม่พร้อมใช้งาน"
        print(f"[Appointment API] action={action} status={exc.code} code={body.get('code')} reason={str(message)[:160]}")
        if message == "Invalid request body":
            message = "ข้อมูลวันหรือเวลาไม่ถูกต้อง กรุณาระบุวันและเวลาให้ชัดเจน"
        result = {"error": message, "status": exc.code}
        if body.get("code"):
            result["code"] = body["code"]
        if isinstance(body.get("suggestion"), dict):
            result["suggestion"] = body["suggestion"]
        return result
    except (URLError, TimeoutError, OSError, ValueError):
        print(f"[Appointment API] action={action} outcome=unknown")
        return {"error": "ติดต่อระบบนัดหมายไม่สำเร็จ", "code": "unknown_outcome"}


def _slot(date: str, time: str) -> dict:
    return {"date": date, "weekday": weekday_name(date), "time": time}


def _my_appointments(state: dict):
    """(appointments with ids, error). IDs stay in code; the model never sees them."""
    raw = _web_request(state, "list")
    if raw.get("error"):
        return None, raw
    items = []
    for item in raw.get("appointments") or []:
        local = local_slot(item.get("start", "")) if isinstance(item, dict) else None
        if local:
            items.append({"id": item.get("appointmentId"), **_slot(*local)})
    return items, None


def _public(items: list) -> list:
    return [{key: item[key] for key in ("date", "weekday", "time")} for item in items]


def _matching(state: dict, current_date: Optional[str], current_time: Optional[str]):
    """(existing appointments matching the given date/time, error)."""
    items, error = _my_appointments(state)
    if error:
        return None, error
    if not items:
        return None, {"error": "ขณะนี้คุณไม่มีนัดหมายที่กำลังจะถึง", "code": "no_appointments"}
    wanted_time = normalize_appointment_time(current_time) if current_time else None
    matches = [item for item in items
               if (not current_date or item["date"] == current_date)
               and (not wanted_time or item["time"] == wanted_time)]
    if not matches:
        return None, {"error": "เลือกนัดหมายไม่ได้", "code": "not_found", "appointments": _public(items)}
    return matches, None


def _select(state: dict, current_date: Optional[str], current_time: Optional[str]):
    """Pick the one existing appointment the user means, or explain why not."""
    matches, error = _matching(state, current_date, current_time)
    if error:
        return None, error
    if len(matches) == 1:
        return matches[0], None
    return None, {"error": "เลือกนัดหมายไม่ได้", "code": "ambiguous", "appointments": _public(matches)}


def _operation(action: str, args: dict, **slots) -> dict:
    return {"action": action, "args": args, **slots}


def _proposal(*operations: dict) -> dict:
    return {"status": "needs_confirmation", "pending": list(operations)}


@tool
def get_available_slots(date: str, state: Annotated[dict, InjectedState]) -> dict:
    """Check which 30-minute appointment times are still free on one date. Use it for any question about availability. date is YYYY-MM-DD (Gregorian)."""
    raw = _web_request(state, "available", date=date)
    if raw.get("error"):
        return raw
    times = [local[1] for local in (local_slot(item.get("start", "")) for item in raw.get("slots") or []) if local]
    result = {"date": date, "weekday": weekday_name(date), "free_times": times}
    if not times:
        result["reason"] = raw.get("reason") or "full"
        nearby = raw.get("suggestion")
        if isinstance(nearby, dict) and nearby.get("times"):
            result["next_available"] = {"date": nearby["date"], "weekday": weekday_name(nearby["date"]),
                                        "free_times": nearby["times"]}
    return result


@tool
def get_my_appointments(state: Annotated[dict, InjectedState]) -> dict:
    """List the user's own upcoming appointments (date, weekday, time). Use it when the user asks to see or check their appointments."""
    items, error = _my_appointments(state)
    return error or {"appointments": _public(items)}


@tool
def create_appointment(date: str, time: str, state: Annotated[dict, InjectedState]) -> dict:
    """Book a new appointment. Call it once the user has given both a date and a time. date is YYYY-MM-DD (Gregorian); time is 24-hour HH:mm exactly as the user said it (never round). The system checks the slot and asks the user to confirm."""
    checked = _web_request(state, "check", date=date, time=time)
    if checked.get("error"):
        return {**checked, "requested": {"date": date, "time": time}, "retry": {"action": "create"}}
    time = normalize_appointment_time(time)
    return _proposal(_operation("create", {"date": date, "time": time}, to=_slot(date, time)))


@tool
def reschedule_appointment(
    state: Annotated[dict, InjectedState],
    new_date: Optional[str] = None,
    new_time: Optional[str] = None,
    current_date: Optional[str] = None,
    current_time: Optional[str] = None,
) -> dict:
    """Move one of the user's existing appointments to a new date and/or time. new_date is YYYY-MM-DD; new_time is 24-hour HH:mm exactly as the user said it. Leave new_time empty to keep the same time ("เวลาเดิม"); leave new_date empty to keep the same day. current_date/current_time say which existing appointment to move; leave them empty if the user did not say. The system checks the slot and asks the user to confirm."""
    chosen, error = _select(state, current_date, current_time)
    if error:
        return error
    date = new_date or chosen["date"]
    time = normalize_appointment_time(new_time) if new_time else chosen["time"]
    if new_time and time is None:
        return {"error": "รูปแบบเวลาไม่ถูกต้อง กรุณาระบุเวลาแบบ 14:30", "status": 400, "code": "invalid_time"}
    if (date, time) == (chosen["date"], chosen["time"]):
        return {"error": "วันและเวลาใหม่ตรงกับนัดเดิมอยู่แล้ว ต้องการเลื่อนเป็นวันและเวลาใด", "status": 400, "code": "same_slot"}
    checked = _web_request(state, "check", date=date, time=time, appointmentId=chosen["id"])
    if checked.get("error"):
        return {**checked, "requested": {"date": date, "time": time},
                "retry": {"action": "reschedule", "from": _slot(chosen["date"], chosen["time"])}}
    return _proposal(_operation("reschedule", {"appointmentId": chosen["id"], "date": date, "time": time},
                                to=_slot(date, time), **{"from": _slot(chosen["date"], chosen["time"])}))


@tool
def cancel_appointment(
    state: Annotated[dict, InjectedState],
    current_date: Optional[str] = None,
    current_time: Optional[str] = None,
    all_matching: bool = False,
) -> dict:
    """Cancel the user's existing appointment. current_date (YYYY-MM-DD) / current_time (HH:mm) say which one; leave them empty if the user did not say. Set all_matching=true only when the user asks to cancel every appointment that matches (e.g. "ทั้งหมด", "ทั้งสองนัด", "ทั้ง 2 เวลา"). The system asks the user to confirm."""
    if all_matching:
        chosen, error = _matching(state, current_date, current_time)
    else:
        one, error = _select(state, current_date, current_time)
        chosen = [one]
    if error:
        return error
    return _proposal(*[_operation("cancel", {"appointmentId": item["id"]}, **{"from": _slot(item["date"], item["time"])})
                       for item in chosen])


WRITE_TOOLS = {"create_appointment", "reschedule_appointment", "cancel_appointment"}
appointment_tools = [get_available_slots, get_my_appointments, create_appointment, reschedule_appointment, cancel_appointment]
