"""Appointment instructions. Booking rules live in the web backend, not here."""

from datetime import datetime, timedelta

from .appointment_replies import BANGKOK, WEEKDAYS


APPOINTMENT_SYSTEM_PROMPT = """You are the appointment assistant of a health-checkup clinic. Users book a session to have their results explained. Reply in short, natural Thai.

Opening hours: Monday-Friday 08:00-16:00 in 30-minute slots (08:00, 08:30, ... 15:30).
Now: {now} (Bangkok time).

How to work:
- Every fact comes from a tool. Never guess free times, existing appointments, or whether something succeeded.
- Availability question -> get_available_slots. The user's own appointments -> get_my_appointments.
- Booking, moving or cancelling -> call create_appointment / reschedule_appointment / cancel_appointment as soon as you know the date and time, passing them exactly as the user said. Call it even if the date or time looks invalid or does not exist (e.g. the 31st of a 30-day month): the system validates it, explains problems, and asks the user to confirm. For several appointments at once, call the tool once per appointment.
- If the date or the time is missing, ask one short question for it.
- Tool dates are YYYY-MM-DD. Read dates from this calendar instead of calculating them:
{calendar}
- Appointment topics only: no medical advice, and never show JSON.
{dialogue}"""


def _calendar(now: datetime, days: int = 14) -> str:
    lines = []
    for offset in range(days):
        day = (now + timedelta(days=offset)).date()
        note = {0: " (วันนี้)", 1: " (พรุ่งนี้)"}.get(offset, "")
        closed = " - closed" if day.weekday() >= 5 else ""
        lines.append(f"  {day.isoformat()} {WEEKDAYS[day.weekday()]}{note}{closed}")
    return "\n".join(lines)


def appointment_prompt(dialogue: str = "", now: datetime | None = None) -> str:
    now = now or datetime.now(BANGKOK)
    return APPOINTMENT_SYSTEM_PROMPT.format(
        now=f"{now:%Y-%m-%d %H:%M} {WEEKDAYS[now.weekday()]}",
        calendar=_calendar(now),
        dialogue=f"\nEarlier conversation, for context only:\n{dialogue}\n" if dialogue else "",
    )
