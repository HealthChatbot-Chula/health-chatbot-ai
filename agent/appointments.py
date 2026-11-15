"""Appointment tool-use loop, separate from clinical intake and RAG.

The model's only jobs are to pick a tool and fill in its arguments. Booking
rules are enforced by the web backend, and anything that changes the calendar
is first stored as `appointment_pending` and executed only after the user's
next message confirms it - so a bare "ใช่" never has to be re-interpreted.
"""

import json

from langchain_core.messages import AIMessage, SystemMessage

from . import appointment_tools as api
from .appointment_prompts import appointment_prompt
from .appointment_replies import (
    CONFIRM_CHOICES, confirmation_decision, confirmation_text, error_reply,
    pending_action, read_reply, reply_text, strip_foreign, suggestion_choices,
    tool_data, write_failure, write_reply,
)
from .llm_timing import timed_llm_invoke


MAX_MODEL_ROUNDS = 4
_ASK_AGAIN = "ขออภัยครับ ช่วยระบุวันและเวลาที่ต้องการอีกครั้งได้ไหมครับ"


def _turn_tools(messages):
    """Tool results produced since the latest user message."""
    result = []
    for message in reversed(messages):
        if message.type == "human":
            break
        if message.type == "tool":
            result.append(message)
    return list(reversed(result))


def _dialogue(messages, limit: int = 6) -> str:
    latest = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].type == "human"), 0)
    lines = []
    for message in messages[:latest][-limit:]:
        if message.type in {"human", "ai"} and reply_text(message.content).strip():
            speaker = "User" if message.type == "human" else "Assistant"
            lines.append(f"{speaker}: {reply_text(message.content).strip()[:400]}")
    return "\n".join(lines)


def appointment_agent_node(state):
    messages = state["messages"]

    def finish(text, pending=None, choices=None):
        # `choices` become quick-reply buttons in the web chat.
        return {"messages": [AIMessage(content=text)], "appointment_pending": pending,
                "appointment_choices": choices, "pending_slot": None, "citations": [],
                "steps": state.get("steps", []) + ["appointment"]}

    if not (state.get("appointment_context") or {}).get("token"):
        return finish("ระบบนัดหมายยังไม่เปิดใช้งานในช่องทางนี้ กรุณาติดต่อผู้ดูแลครับ")

    latest = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].type == "human"), 0)
    latest_user = reply_text(messages[latest].content)
    done = _turn_tools(messages)

    # 1) The previous reply asked the user to confirm stored operation(s).
    pending = state.get("appointment_pending")
    pending = [pending] if isinstance(pending, dict) else pending
    if pending and not done:
        decision = confirmation_decision(latest_user, pending_action(pending))
        print(f"[Appointment Flow] pending={[item.get('action') for item in pending]} decision={decision}")
        if decision == "yes":
            # The backend re-validates everything; these are the only writes.
            outcomes = [(item, api._web_request(state, item["action"], **item["args"])) for item in pending]
            failed = [write_failure(item, result) for item, result in outcomes if result.get("error")]
            return finish("\n".join(write_reply(item, result) for item, result in outcomes),
                          choices=suggestion_choices(failed[0]) if len(outcomes) == 1 and failed else None)
        if decision == "no":
            return finish("ยังไม่ได้ทำรายการครับ หากต้องการวันหรือเวลาอื่นแจ้งได้เลยครับ")
        # Anything else is a new request; the old proposal is dropped below.

    # 2) Write tools just validated a request: ask to confirm, or explain why not.
    results = [tool_data(message) for message in done if getattr(message, "name", "") in api.WRITE_TOOLS]
    if results:
        proposed, seen = [], set()
        for data in results:
            for item in (data.get("pending") or []) if data.get("status") == "needs_confirmation" else []:
                key = json.dumps([item["action"], item["args"]], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    proposed.append(item)
        refused = list(dict.fromkeys(error_reply(data) for data in results if data.get("status") != "needs_confirmation"))
        if proposed:
            text = confirmation_text(proposed)
            if refused:
                text += "\n\nรายการที่ทำไม่ได้: " + " ".join(refused)
            return finish(text, proposed, CONFIRM_CHOICES)
        return finish("\n".join(refused), choices=suggestion_choices(results[0]) if len(results) == 1 else None)

    # 3) Let the model choose the next tool, or answer / ask for what is missing.
    rounds = state.get("appointment_tool_calls", 0)
    fallback = (read_reply(tool_data(done[-1])) if done else "") or _ASK_AGAIN
    if rounds >= MAX_MODEL_ROUNDS:
        return finish(fallback)
    from .models import appointment_model

    response = timed_llm_invoke(
        appointment_model.bind_tools(api.appointment_tools),
        [SystemMessage(content=appointment_prompt(_dialogue(messages))),
         *[m for m in messages[latest:] if m.type != "system"]],
        "appointment_response",
    )
    calls = getattr(response, "tool_calls", None) or []
    print(f"[Appointment Model] round={rounds + 1} tools={[call['name'] for call in calls]}")
    if calls:
        # Several look-ups are fine, and so are several changes of one kind
        # ("cancel both"). Mixed kinds are not combined: only the first kind runs.
        writes = [call for call in calls if call["name"] in api.WRITE_TOOLS]
        keep = [call for call in writes if call["name"] == writes[0]["name"]][:4] if writes else calls[:5]
        response = response.model_copy(update={"content": "", "tool_calls": keep})
        return {"messages": [response], "appointment_pending": None, "appointment_choices": None, "pending_slot": None,
                "citations": [], "appointment_tool_calls": rounds + 1}
    return finish(strip_foreign(reply_text(response.content)).strip() or fallback)


def route_appointment_tools(state):
    return "appointment_tools" if getattr(state["messages"][-1], "tool_calls", None) else "__end__"
