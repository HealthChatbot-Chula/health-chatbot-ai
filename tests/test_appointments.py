import io
import json
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.appointment_routing import is_appointment_request, route_after_turn_intent
from agent.appointment_tools import appointment_tools, _web_request
from agent.appointments import appointment_agent_node, route_appointment_tools
from agent.intent_classifier import classify_turn_intent_node
from agent.state import AgentState
from backend.schemas import ChatRequest
from services.health_chat_service import pending_slot_fast_response, slot_memory_store


class AppointmentTests(unittest.TestCase):
    def test_route_does_not_send_appointments_through_medical_intake(self):
        self.assertEqual(route_after_turn_intent({"intent": "appointment"}), "appointment_agent")
        self.assertEqual(route_after_turn_intent({"intent": "lab_interpretation"}), "extract_info_node")
        self.assertTrue(is_appointment_request("ขอเลื่อนนัดเป็นวันจันทร์"))
        result = classify_turn_intent_node({
            "messages": [HumanMessage(content="ขอจองนัดอธิบายผลตรวจวันที่ 19 ตุลาคม")],
            "pending_slot": "age",
        })
        self.assertEqual(result["intent"], "appointment")

    def test_available_appointment_wording_from_web_chat(self):
        text = "วันที่ 16 พฤศจิกายน 2026 มีเวลานัดว่างช่วงไหนบ้าง"
        self.assertTrue(is_appointment_request(text))
        result = classify_turn_intent_node({"messages": [HumanMessage(content=text)]})
        self.assertEqual(result["intent"], "appointment")

    def test_clinical_urgency_still_takes_priority_over_booking(self):
        result = classify_turn_intent_node({"messages": [HumanMessage(content="Glucose 350 ขอจองนัดหมาย")]})
        self.assertEqual(result["intent"], "urgent_red_flag")

    def test_appointment_interrupts_medical_fast_path(self):
        slot_memory_store["appointment-test"] = {"pending_slot": "age"}
        try:
            self.assertIsNone(pending_slot_fast_response("appointment-test", "ขอจองนัดหมาย"))
        finally:
            slot_memory_store.pop("appointment-test", None)

    def test_token_and_user_identity_are_not_in_model_tool_schema(self):
        for tool in appointment_tools:
            schema = json.dumps(tool.tool_call_schema.model_json_schema())
            for hidden in ('"state"', '"token"', '"userId"', '"appointment_context"'):
                self.assertNotIn(hidden, schema)

    def test_disabled_tools_do_not_make_network_requests(self):
        with patch("agent.appointment_tools.urlopen") as send:
            result = _web_request({}, "list")
        self.assertIn("error", result)
        send.assert_not_called()

    def test_api_uses_signed_context_and_configured_endpoint(self):
        response = io.BytesIO(b'{"appointments": []}')
        with patch.dict(os.environ, {"WEB_APPOINTMENTS_URL": "http://localhost:3000/api/internal/appointments"}), patch(
            "agent.appointment_tools.urlopen", return_value=response
        ) as send:
            result = _web_request({"appointment_context": {"token": "signed-secret"}}, "list")
        self.assertEqual(result, {"appointments": []})
        request = send.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer signed-secret")
        self.assertEqual(json.loads(request.data), {"action": "list"})

    def test_errors_do_not_report_booking_success(self):
        for error in [
            HTTPError("http://local", 409, "busy", {}, io.BytesIO('{"error":{"message":"ไม่ว่าง"}}'.encode())),
            URLError("offline"),
        ]:
            with patch.dict(os.environ, {"WEB_APPOINTMENTS_URL": "http://localhost/api/internal/appointments"}), patch(
                "agent.appointment_tools.urlopen", side_effect=error
            ):
                self.assertIn("error", _web_request({"appointment_context": {"token": "signed"}, "messages": [HumanMessage(content="ขอจองนัด 19/10 เวลา 08:00")]}, "create", date="2026-10-19", time="08:00"))

    def test_missing_context_returns_clear_unavailable_reply(self):
        result = appointment_agent_node({"messages": [HumanMessage(content="ขอจองนัด")]})
        self.assertIsNone(result["pending_slot"])
        self.assertIn("ยังไม่เปิดใช้งาน", result["messages"][0].content)

    def test_empty_or_broken_model_output_never_writes(self):
        models = ModuleType("agent.models")
        models.appointment_model = Mock()
        variants = [
            AIMessage(content=""),
            AIMessage(content="", invalid_tool_calls=[{"name": "create_appointment", "args": "{", "id": "bad-call", "error": "invalid JSON"}]),
        ]
        for response in variants:
            with self.subTest(response=response), patch.dict(sys.modules, {"agent.models": models}), patch(
                "agent.appointments.timed_llm_invoke", return_value=response
            ), patch("agent.appointment_tools._web_request") as api:
                result = appointment_agent_node({"messages": [HumanMessage(content="เพิ่มนัด 14.30")], "appointment_context": {"token": "hidden"}})
            self.assertTrue(result["messages"][0].content.strip())
            self.assertNotIn("เรียบร้อย", result["messages"][0].content)
            api.assert_not_called()

    def test_dot_time_is_normalized_before_calling_web_api(self):
        with patch.dict(os.environ, {"WEB_APPOINTMENTS_URL": "http://localhost:3000/api/internal/appointments"}), patch(
            "agent.appointment_tools.urlopen", return_value=io.BytesIO(b'{"status":"confirmed"}')
        ) as api:
            _web_request({"appointment_context": {"token": "hidden"}, "messages": [HumanMessage(content="ขอจองนัด 30/11 เวลา 14.30")]}, "create", date="2026-11-30", time="14.30")
        self.assertEqual(json.loads(api.call_args.args[0].data)["time"], "14:30")

    def test_real_langgraph_tool_loop_injects_context_without_leaking_token(self):
        fake_model = Mock()
        fake_model.bind_tools.return_value = fake_model
        responses = [
            AIMessage(content="", tool_calls=[{"name": "get_available_slots", "args": {"date": "2026-10-19"}, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="วันที่ 19 ตุลาคม ว่างเวลา 09:00 ครับ"),
        ]
        captured = []
        def invoke(model, messages, label):
            captured.append(messages)
            return responses.pop(0)
        models = ModuleType("agent.models")
        models.chat_model = fake_model
        models.appointment_model = fake_model
        graph = StateGraph(AgentState)
        graph.add_node("agent", appointment_agent_node)
        graph.add_node("appointment_tools", ToolNode(appointment_tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route_appointment_tools, {"appointment_tools": "appointment_tools", "__end__": END})
        graph.add_edge("appointment_tools", "agent")
        with patch.dict(sys.modules, {"agent.models": models}), patch("agent.appointments.timed_llm_invoke", side_effect=invoke), patch(
            "agent.appointment_tools._web_request", return_value={"slots": [{"start": "2026-10-19T02:00:00Z", "end": "2026-10-19T02:30:00Z"}]}
        ) as call:
            result = graph.compile().invoke({
                "messages": [HumanMessage(content="วันที่ 19 ตุลาคม มีเวลาไหนว่าง")],
                "appointment_context": {"token": "do-not-expose"}, "appointment_tool_calls": 0,
            })
        self.assertEqual(call.call_args.args[0]["appointment_context"]["token"], "do-not-expose")
        self.assertNotIn("do-not-expose", repr(captured))
        self.assertEqual(len(captured), 2)
        tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertIn('"free_times": ["09:00"]', tool_message.content)
        self.assertIn("09:00", result["messages"][-1].content)

    def test_context_is_a_known_request_field_not_logged_as_extra(self):
        from services.health_chat_service import request_extra_fields
        request = ChatRequest(model="health-agent", messages=[], appointment_context={"token": "hidden"})
        self.assertNotIn("appointment_context", request_extra_fields(request))

    def test_main_graph_bypasses_medical_nodes_for_appointments(self):
        models = ModuleType("agent.models")
        models.chat_model = Mock()
        models.appointment_model = Mock()
        models.intent_model = Mock()
        rag = ModuleType("agent.rag_utils")
        rag.RetrievedContext = Mock()
        rag.format_citations = Mock()
        rag.retrieve_context = Mock()
        with patch.dict(sys.modules, {"agent.models": models, "agent.rag_utils": rag}):
            from agent.graph import build_graph
            with patch("agent.appointments.timed_llm_invoke", return_value=AIMessage(content="สะดวกนัดวันไหนครับ")):
                result = build_graph().invoke({
                    "messages": [HumanMessage(content="ขอจองนัดหมาย")],
                    "steps": [], "pending_slot": "age",
                    "appointment_context": {"token": "hidden"}, "appointment_tool_calls": 0,
                })
        self.assertEqual(result["intent"], "appointment")
        self.assertIsNone(result["pending_slot"])
        self.assertNotIn("retrieval", result["steps"])
        self.assertEqual(result["messages"][-1].content, "สะดวกนัดวันไหนครับ")
        rag.retrieve_context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
