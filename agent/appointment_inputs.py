"""Normalize representation only; never round a requested booking time."""

import re


def normalize_appointment_time(value: str) -> str | None:
    match = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})(?::00)?\s*", value)
    if not match:
        return None
    hour, minute = map(int, match.groups())
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"
