import os
import sys
import types
import unittest
from unittest.mock import patch

from agent.llm_timing import timed_llm_invoke
from agent.observability import observe_chat_request


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **details):
        self.updates.append(details)


class FakeObservationContext:
    def __init__(self, observation):
        self.observation = observation
        self.exit_exception = None

    def __enter__(self):
        return self.observation

    def __exit__(self, exc_type, exc, traceback):
        self.exit_exception = exc
        return False


class FakeClient:
    def __init__(self):
        self.started = []

    def start_as_current_observation(self, **details):
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
        client = FakeClient()
        propagated = []

        def propagate_attributes(**details):
            propagated.append(details)
            return FakeObservationContext(FakeObservation())

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


if __name__ == "__main__":
    unittest.main()
