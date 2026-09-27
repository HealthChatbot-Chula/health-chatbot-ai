import unittest

from langchain_core.messages import HumanMessage

from agent.intent_classifier import _parse_intent_response, classify_turn_intent_node
from agent.slot_filling_graph import ask_lab_node, route_after_extraction


class IntentRoutingTests(unittest.TestCase):
    def test_parser_accepts_known_intent(self):
        self.assertEqual(
            _parse_intent_response('{"intent":"greeting"}'),
            "greeting",
        )

    def test_parser_rejects_unknown_intent(self):
        self.assertEqual(
            _parse_intent_response('{"intent":"stored_lab_memory"}'),
            "unknown",
        )

    def test_greeting_with_saved_labs_skips_analyst(self):
        state = {
            "messages": [HumanMessage(content="มอนิ่งงงง")],
            "intent": "greeting",
            "extracted_lab_values": {"HbA1c": 6.2, "LDL": 80.0},
        }

        self.assertEqual(route_after_extraction(state), "ask_lab_node")
        result = ask_lab_node(state)
        self.assertEqual(result["citations"], [])
        self.assertIn("สวัสดี", result["messages"][0].content)

    def test_out_of_scope_with_saved_labs_skips_analyst(self):
        state = {
            "messages": [HumanMessage(content="เล่าเรื่องหนังให้ฟังหน่อย")],
            "intent": "out_of_scope",
            "extracted_lab_values": {"HbA1c": 6.2},
        }

        self.assertEqual(route_after_extraction(state), "ask_lab_node")
        result = ask_lab_node(state)
        self.assertEqual(result["citations"], [])
        self.assertIn("เฉพาะเรื่องสุขภาพ", result["messages"][0].content)

    def test_medication_safety_uses_current_turn_override(self):
        result = classify_turn_intent_node(
            {"messages": [HumanMessage(content="ควรหยุดยาไหม")]}
        )

        self.assertEqual(result["intent"], "medication_safety")

    def test_critical_current_lab_uses_urgent_override(self):
        result = classify_turn_intent_node(
            {"messages": [HumanMessage(content="Glucose 350 mg/dL")]}
        )

        self.assertEqual(result["intent"], "urgent_red_flag")


if __name__ == "__main__":
    unittest.main()
