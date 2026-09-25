import os
import sys
import types
import unittest
from unittest.mock import patch

from agent.llm_timing import timed_llm_invoke
from agent.observability import observe_chat_request, sanitize_langfuse_content


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **details):
        self.updates.append(details)


class FakeObservationContext:
    def __init__(self, observation, on_enter=None):
        self.observation = observation
        self.on_enter = on_enter
        self.exit_exception = None

    def __enter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self.observation

    def __exit__(self, exc_type, exc, traceback):
        self.exit_exception = exc
        return False


class FakeClient:
    def __init__(self, events=None):
        self.started = []
        self.events = events

    def start_as_current_observation(self, **details):
        if self.events is not None:
            self.events.append("observation-started")
        observation = FakeObservation()
        context = FakeObservationContext(observation)
        self.started.append((details, observation, context))
        return context


class FakeModel:
    model_name = "gemini-test"

    def invoke(self, prompt):
        return types.SimpleNamespace(
            content="answer",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
            },
            response_metadata={},
        )


class ObservabilityTests(unittest.TestCase):
    def test_chat_trace_groups_generations_without_capturing_health_content(self):
        events = []
        client = FakeClient(events)
        propagated = []

        def propagate_attributes(**details):
            propagated.append(details)
            return FakeObservationContext(
                FakeObservation(),
                on_enter=lambda: events.append("attributes-entered"),
            )

        fake_langfuse = types.ModuleType("langfuse")
        fake_langfuse.get_client = lambda: client
        fake_langfuse.propagate_attributes = propagate_attributes

        with (
            patch.dict(os.environ, {"LANGFUSE_ENABLED": "true"}),
            patch.dict(sys.modules, {"langfuse": fake_langfuse}),
            observe_chat_request(
                user_id="user-1",
                session_id="session-1",
                conversation_id="conversation-1",
                model="health-agent",
                message_count=3,
                latest_user_chars=14,
                latest_user_message="sensitive health prompt",
            ),
        ):
            response = timed_llm_invoke(
                FakeModel(),
                "sensitive health prompt",
                "health-response",
            )

        self.assertEqual(response.content, "answer")
        self.assertEqual(len(client.started), 2)

        root_details, _, _ = client.started[0]
        generation_details, generation, _ = client.started[1]
        self.assertEqual(root_details["name"], "health-chat-turn")
        self.assertNotIn("sensitive health prompt", str(root_details))
        self.assertEqual(generation_details["name"], "health-response")
        self.assertNotIn("sensitive health prompt", str(generation_details))
        self.assertEqual(
            generation.updates[-1]["usage_details"],
            {"input": 12, "output": 4, "total": 16},
        )
        self.assertEqual(propagated[0]["session_id"], "session-1")
        self.assertEqual(propagated[0]["user_id"], "user-1")
        self.assertEqual(events[:2], ["attributes-entered", "observation-started"])

    def test_content_capture_masks_common_direct_identifiers(self):
        client = FakeClient()
        fake_langfuse = types.ModuleType("langfuse")
        fake_langfuse.get_client = lambda: client
        fake_langfuse.propagate_attributes = (
            lambda **_: FakeObservationContext(FakeObservation())
        )
        message = "Call me at 081-234-5678 or a@b.com on 2026-09-25. ID 1-2345-67890-12-3"

        with patch.dict(
            os.environ,
            {"LANGFUSE_ENABLED": "true", "LANGFUSE_CAPTURE_CONTENT": "true"},
        ), patch.dict(sys.modules, {"langfuse": fake_langfuse}), observe_chat_request(
            user_id=None,
            session_id="session-1",
            conversation_id="conversation-1",
            model="health-agent",
            message_count=1,
            latest_user_chars=len(message),
            latest_user_message=message,
        ) as trace:
            trace.update(
                output=[
                    {
                        "role": "assistant",
                        "content": sanitize_langfuse_content("Email b@c.com"),
                    }
                ]
            )

        root_details, root_observation, _ = client.started[0]
        captured_input = root_details["input"][0]["content"]
        captured_output = root_observation.updates[-1]["output"][0]["content"]
        self.assertIn("[REDACTED_PHONE]", captured_input)
        self.assertIn("[REDACTED_EMAIL]", captured_input)
        self.assertIn("[REDACTED_DATE]", captured_input)
        self.assertIn("[REDACTED_NATIONAL_ID]", captured_input)
        self.assertNotIn("081-234-5678", captured_input)
        self.assertEqual(captured_output, "Email [REDACTED_EMAIL]")

    def test_content_capture_includes_masked_generation_messages(self):
        client = FakeClient()
        fake_langfuse = types.ModuleType("langfuse")
        fake_langfuse.get_client = lambda: client
        fake_langfuse.propagate_attributes = (
            lambda **_: FakeObservationContext(FakeObservation())
        )
        prompt = [
            types.SimpleNamespace(type="system", content="Classify this message."),
            types.SimpleNamespace(type="human", content="My email is a@b.com"),
        ]

        with patch.dict(
            os.environ,
            {"LANGFUSE_ENABLED": "true", "LANGFUSE_CAPTURE_CONTENT": "true"},
        ), patch.dict(sys.modules, {"langfuse": fake_langfuse}), observe_chat_request(
            user_id=None,
            session_id="session-1",
            conversation_id="conversation-1",
            model="health-agent",
            message_count=1,
            latest_user_chars=4,
            latest_user_message="test",
        ):
            timed_llm_invoke(FakeModel(), prompt, "turn_intent_classification")

        generation_details, generation, _ = client.started[1]
        self.assertEqual(
            generation_details["input"],
            [
                {"role": "system", "content": "Classify this message."},
                {"role": "user", "content": "My email is [REDACTED_EMAIL]"},
            ],
        )
        self.assertEqual(
            generation.updates[-1]["output"],
            [{"role": "assistant", "content": "answer"}],
        )


if __name__ == "__main__":
    unittest.main()
