"""End-to-end appointment turns with a scripted model and a fake web backend."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.appointment_tools import appointment_tools
from agent.appointments import appointment_agent_node, route_appointment_tools
from agent.intent_classifier import classify_turn_intent_node
from agent.state import AgentState


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}])


class FakeBackend:
    """Mimics the web API: one appointment on Wed 18 Nov 2026 10:00 Bangkok."""

    def __init__(self):
        self.appointments = {"evt01": "2026-11-18T03:00:00Z"}
        self.calls = []

    def __call__(self, state, action, **args):
        self.calls.append((action, args))
        if action == "list":
            return {"appointments": [{"appointmentId": key, "start": value} for key, value in self.appointments.items()]}
        if action in {"check", "reschedule", "create"}:
            minute = args["time"][-2:]
            if args["date"] == "2026-11-21":
                return {"error": "วันดังกล่าวเป็นวันหยุด เปิดให้นัดเฉพาะวันจันทร์–ศุกร์", "status": 400, "code": "closed_day",
                        "suggestion": {"date": "2026-11-23", "times": ["09:00", "09:30"]}}
            if minute not in {"00", "30"}:
                return {"error": "นัดหมายเป็นช่วงละ 30 นาที เริ่มได้เฉพาะนาทีที่ :00 หรือ :30", "status": 400, "code": "off_grid",
                        "suggestion": {"date": args["date"], "times": ["14:00", "14:30"]}}
        if action == "available":
            return {"slots": [{"start": f"{args['date']}T0{h}:{m}:00Z"} for h in (1, 2) for m in ("00", "30")]}
        if action == "check":
            return {"ok": True}
        if action == "reschedule":
            hour = int(args["time"][:2]) - 7
            self.appointments[args["appointmentId"]] = f"{args['date']}T{hour:02d}:{args['time'][-2:]}:00Z"
            return {"status": "confirmed", "appointment": {"appointmentId": args["appointmentId"], "start": self.appointments[args["appointmentId"]]}}
        if action == "create":
            return {"status": "confirmed", "appointment": {"appointmentId": "new01", "start": f"{args['date']}T02:00:00Z"}}
        if action == "cancel":
            self.appointments.pop(args["appointmentId"])
            return {"appointmentId": args["appointmentId"], "status": "cancelled"}
        raise AssertionError(action)

    def writes(self):
        return [item for item in self.calls if item[0] in {"create", "reschedule", "cancel"}]


class AppointmentFlowTests(unittest.TestCase):
    def setUp(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", appointment_agent_node)
        graph.add_node("appointment_tools", ToolNode(appointment_tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route_appointment_tools, {"appointment_tools": "appointment_tools", "__end__": END})
        graph.add_edge("appointment_tools", "agent")
        self.graph = graph.compile()
        self.backend = FakeBackend()
        self.history = []
        self.pending = None
        self.choices = None
        self.model_calls = 0
        self.model_inputs = []

    def say(self, text, *model_replies):
        """One chat turn. Like the real service, only user/assistant text and
        the stored pending operation survive between turns."""
        script = list(model_replies)
        fake_model = Mock()
        fake_model.bind_tools.return_value = fake_model
        models = ModuleType("agent.models")
        models.appointment_model = fake_model

        def invoke(model, messages, label):
            self.model_calls += 1
            self.model_inputs.append(messages)
            return script.pop(0)

        self.history.append(HumanMessage(content=text))
        with patch.dict(sys.modules, {"agent.models": models}), patch(
            "agent.appointments.timed_llm_invoke", side_effect=invoke
        ), patch("agent.appointment_tools._web_request", side_effect=self.backend):
            result = self.graph.invoke({
                "messages": self.history[-6:], "appointment_context": {"token": "hidden"},
                "appointment_tool_calls": 0, "appointment_pending": self.pending, "steps": [],
            })
        self.assertEqual(script, [], "model was expected to be called again")
        self.pending = result.get("appointment_pending")
        self.choices = result.get("appointment_choices")
        reply = result["messages"][-1].content
        self.history.append(AIMessage(content=reply))
        return reply

    def test_conversation_from_the_screenshot(self):
        move = dict(new_date="2026-11-20", current_date="2026-11-18")
        reply = self.say("ขอเลื่อนนัดวันที่18/11 เป็นวันที่20/11 เวลา 14.14", call("reschedule_appointment", new_time="14.14", **move))
        self.assertIn("30 นาที", reply)
        self.assertIn("14:00, 14:30", reply)
        self.assertIsNone(self.pending)

        reply = self.say("เป็นเวลา 14.30แทน", call("reschedule_appointment", new_time="14:30", **move))
        self.assertIn("วันพุธที่ 18 พฤศจิกายน 2026 เวลา 10:00", reply)
        self.assertIn("วันศุกร์ที่ 20 พฤศจิกายน 2026 เวลา 14:30", reply)
        self.assertEqual(self.backend.writes(), [], "nothing is written before the user confirms")

        models_before = self.model_calls
        reply = self.say("ใช่")
        self.assertEqual(self.model_calls, models_before, "confirmation must not go through the model")
        self.assertIn("เลื่อนนัดหมายเรียบร้อย", reply)
        self.assertNotIn("31", reply)
        self.assertEqual(self.backend.writes(), [("reschedule", {"appointmentId": "evt01", "date": "2026-11-20", "time": "14:30"})])
        self.assertIsNone(self.pending)

        # A second "ใช่" has nothing to confirm and must not repeat the write.
        self.say("ใช่", AIMessage(content="ต้องการให้ช่วยเรื่องนัดหมายอะไรครับ"))
        self.assertEqual(len(self.backend.writes()), 1)

    def test_empty_or_failed_appointment_list_is_reported_exactly(self):
        self.backend.appointments = {}
        # The model only picks the tool; it is not asked to describe the result.
        reply = self.say("ดูนัดหมาย", call("get_my_appointments"))
        self.assertEqual(reply, "ขณะนี้คุณไม่มีนัดหมายที่กำลังจะถึงครับ")

        self.backend.appointments = {"evt20": "2026-11-20T08:00:00Z"}
        reply = self.say("ดูนัดหมายทั้งหมด", call("get_my_appointments"))
        self.assertIn("วันศุกร์ที่ 20 พฤศจิกายน 2026 เวลา 15:00", reply)

        failing = lambda state, action, **args: {"error": "ยังไม่ได้ตั้งค่าการเชื่อมต่อ Google Calendar", "status": 503}
        with patch.object(self, "backend", failing):
            reply = self.say("ดูนัดหมาย", call("get_my_appointments"))
        self.assertIn("ยังไม่ได้ตั้งค่าการเชื่อมต่อ Google Calendar", reply)
        self.assertNotIn("เรียบร้อย", reply)

    def test_time_only_follow_up_sees_the_day_just_offered(self):
        reply = self.say("พรุ่งนี้มีเวลาว่างไหม", call("get_available_slots", date="2026-11-18"),
                         AIMessage(content="วันพุธที่ 18 พฤศจิกายน 2026 มีเวลาว่าง 08:00, 08:30, 09:00, 09:30 น. ครับ ต้องการเวลาไหนครับ"))
        self.assertEqual([item["label"] for item in self.choices], ["08:00", "08:30", "09:00", "09:30"])
        self.assertEqual(self.choices[3]["value"], "ขอจองนัดวันที่ 18/11/2026 เวลา 09:30")

        self.model_inputs.clear()
        self.say("ต้องการนัดเวลา 9.30", call("create_appointment", date="2026-11-18", time="9.30"))
        sent = self.model_inputs[0]
        # The earlier question and the bot's offer reach the model as real turns.
        self.assertEqual([m.type for m in sent[1:]], ["human", "ai", "human"])
        self.assertIn("วันพุธที่ 18", sent[2].content)
        self.assertIn("09:30", self.history[-1].content)
        self.assertEqual(self.pending[0]["args"], {"date": "2026-11-18", "time": "09:30"})

    def test_saturday_request_is_refused_with_a_real_alternative(self):
        reply = self.say("ขอจองวันเสาร์ 9 โมง", call("create_appointment", date="2026-11-21", time="09:00"))
        self.assertIn("วันหยุด", reply)
        self.assertIn("วันจันทร์ที่ 23 พฤศจิกายน 2026", reply)
        self.assertEqual(self.backend.writes(), [])

    def test_declining_or_changing_the_request_discards_the_proposal(self):
        self.say("ขอจองวันที่ 20/11 9 โมง", call("create_appointment", date="2026-11-20", time="09:00"))
        self.assertEqual(self.pending[0]["action"], "create")
        self.assertIn("ยังไม่ได้ทำรายการ", self.say("ไม่เอาแล้วครับ"))
        self.assertIsNone(self.pending)
        self.say("ใช่", AIMessage(content="ต้องการนัดวันไหนครับ"))
        self.assertEqual(self.backend.writes(), [])

        self.say("ขอจองวันที่ 20/11 9 โมง", call("create_appointment", date="2026-11-20", time="09:00"))
        reply = self.say("ขอเป็น 10 โมงแทน", call("create_appointment", date="2026-11-20", time="10:00"))
        self.assertIn("10:00", reply)
        self.assertEqual(self.pending[0]["args"], {"date": "2026-11-20", "time": "10:00"})
        self.assertEqual(self.backend.writes(), [])

    def test_cancel_and_same_time_reschedule_resolve_the_single_appointment(self):
        reply = self.say("เลื่อนนัดไปวันศุกร์ เวลาเดิม", call("reschedule_appointment", new_date="2026-11-20"))
        self.assertIn("เวลา 10:00 น. เป็นวันศุกร์ที่ 20 พฤศจิกายน 2026 เวลา 10:00", reply)
        self.say("ยืนยันครับ")
        reply = self.say("ขอยกเลิกนัด", call("cancel_appointment"))
        self.assertIn("ยกเลิกนัดหมายวันศุกร์ที่ 20", reply)
        self.assertIn("เรียบร้อย", self.say("ยกเลิกเลย"))
        self.assertEqual([item[0] for item in self.backend.writes()], ["reschedule", "cancel"])
        reply = self.say("ขอยกเลิกนัด", call("cancel_appointment"))
        self.assertIn("ไม่มีนัดหมาย", reply)

    def test_model_cannot_write_twice_or_leak_ids(self):
        both = AIMessage(content="", tool_calls=[
            {"name": "cancel_appointment", "args": {}, "id": "c1", "type": "tool_call"},
            {"name": "create_appointment", "args": {"date": "2026-11-20", "time": "09:00"}, "id": "c2", "type": "tool_call"},
        ])
        self.say("ยกเลิกนัดแล้วจองใหม่ 20/11 9 โมง", both)
        self.assertEqual([item["action"] for item in self.pending], ["cancel"])
        self.assertEqual(self.backend.writes(), [])
        reply = self.say("ดูนัดหมาย", call("get_my_appointments"))
        self.assertIn("วันพุธที่ 18 พฤศจิกายน 2026 เวลา 10:00", reply)
        self.assertIsNone(self.pending)
        self.assertNotIn("evt01", repr(self.history))

    def test_bare_confirmation_routes_to_appointments_only_while_pending(self):
        pending = [{"action": "create", "args": {}}]
        state = {"messages": [HumanMessage(content="ยืนยันจร้า")], "appointment_pending": pending}
        self.assertEqual(classify_turn_intent_node(state)["intent"], "appointment")

    def test_casual_confirmation_from_the_screenshot_does_not_loop(self):
        self.backend.appointments = {"evt30": "2026-11-30T01:00:00Z"}
        reply = self.say("เลื่อนนัดวันที่30 เวลา8.00 เป็นเวลา8.30",
                         call("reschedule_appointment", new_time="8.30", current_date="2026-11-30", current_time="8.00"))
        self.assertIn("08:30", reply)
        self.assertEqual([item["value"] for item in self.choices], ["ยืนยัน", "ไม่ยืนยัน"])
        models_before = self.model_calls
        reply = self.say("ยืนยันจร้า")
        self.assertIn("เลื่อนนัดหมายเรียบร้อย", reply)
        self.assertEqual(self.model_calls, models_before)
        self.assertEqual(len(self.backend.writes()), 1)
        self.assertIsNone(self.choices)

    def test_cancelling_both_appointments_of_a_day(self):
        both_days = {"evt09": "2026-11-18T02:00:00Z", "evt12": "2026-11-18T05:00:00Z", "evt20": "2026-11-20T01:00:00Z"}
        two_calls = AIMessage(content="", tool_calls=[
            {"name": "cancel_appointment", "args": {"current_date": "2026-11-18", "current_time": "09:00"}, "id": "c1", "type": "tool_call"},
            {"name": "cancel_appointment", "args": {"current_date": "2026-11-18", "current_time": "12:00"}, "id": "c2", "type": "tool_call"},
        ])
        one_call = call("cancel_appointment", current_date="2026-11-18", all_matching=True)
        for model_reply in (two_calls, one_call):
            with self.subTest(model_reply=model_reply.tool_calls):
                self.backend.appointments = dict(both_days)
                self.backend.calls.clear()
                reply = self.say("ยกเลิกนัดวันที่18", call("cancel_appointment", current_date="2026-11-18"))
                self.assertIn("มากกว่าหนึ่งรายการ", reply)
                self.assertNotIn("20 พฤศจิกายน", reply)
                reply = self.say("ทั้ง 2 เวลา", model_reply)
                self.assertIn("2 รายการ", reply)
                self.assertIn("09:00", reply)
                self.assertIn("12:00", reply)
                self.assertEqual(self.backend.writes(), [])
                reply = self.say("ยืนยัน")
                self.assertEqual(reply.count("เรียบร้อย"), 2)
                self.assertEqual(list(self.backend.appointments), ["evt20"])

    def test_refused_time_offers_free_slots_as_buttons(self):
        self.say("ขอเลื่อนนัดวันที่18/11 เป็นวันที่20/11 เวลา 14.14",
                 call("reschedule_appointment", new_date="2026-11-20", new_time="14.14", current_date="2026-11-18"))
        self.assertEqual([item["label"] for item in self.choices], ["14:00", "14:30"])
        self.assertIn("ขอเลื่อนนัดวันที่ 18/11/2026 เวลา 10:00 เป็นวันที่ 20/11/2026 เวลา 14:30", self.choices[1]["value"])

    def test_unclear_follow_up_stays_with_appointments_but_health_topics_leave(self):
        def classify(text, model_intent, active=True):
            reply = AIMessage(content='{"intent": "%s"}' % model_intent)
            models = ModuleType("agent.models")
            models.intent_model = Mock()
            with patch.dict(sys.modules, {"agent.models": models}), patch(
                "agent.intent_classifier.timed_llm_invoke", return_value=reply
            ):
                return classify_turn_intent_node({"messages": [
                    AIMessage(content="ต้องการนัดวันไหนครับ"), HumanMessage(content=text),
                ], "appointment_active": active})["intent"]

        self.assertEqual(classify("ไม่รู้", "unknown"), "appointment")
        self.assertEqual(classify("ไม่รู้", "unknown", active=False), "unknown")
        self.assertEqual(classify("ผลเลือดผมเป็นยังไงบ้าง", "lab_interpretation"), "lab_interpretation")
        self.assertEqual(classify("ขอบคุณครับ", "greeting"), "greeting")
        self.assertEqual(classify("เจ็บหน้าอกมาก หายใจไม่ออก", "unknown"), "urgent_red_flag")


if __name__ == "__main__":
    unittest.main()
