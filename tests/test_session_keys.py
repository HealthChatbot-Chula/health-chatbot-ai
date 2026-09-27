import os
import unittest
from unittest.mock import patch

from backend.schemas import ChatRequest, Message
from services.health_chat_service import (
    memory_key,
    monitoring_session_key,
    run_chat_completion,
    session_usage_store,
)


def chat_request(**overrides):
    values = {
        "model": "health-agent",
        "messages": [Message(role="user", content="LDL 178 สูงไหม")],
        "user": "user-1",
    }
    values.update(overrides)
    return ChatRequest(**values)


class SessionKeyTests(unittest.TestCase):
    def setUp(self):
        session_usage_store.clear()

    def tearDown(self):
        session_usage_store.clear()

    def test_memory_stays_with_conversation_while_monitoring_uses_session(self):
        first = chat_request(conversation_id="conversation-1", session_id="session-1")
        second = chat_request(conversation_id="conversation-1", session_id="session-2")

        self.assertEqual(memory_key(first), "conversation-1")
        self.assertEqual(memory_key(second), "conversation-1")
        self.assertEqual(monitoring_session_key(first), "session-1")
        self.assertEqual(monitoring_session_key(second), "session-2")

    def test_legacy_client_falls_back_to_conversation_for_monitoring(self):
        request = chat_request(conversation_id="conversation-1")

        self.assertEqual(memory_key(request), "conversation-1")
        self.assertEqual(monitoring_session_key(request), "conversation-1")

    def test_metadata_ids_follow_the_same_separation(self):
        request = chat_request(
            metadata={
                "conversation_id": "conversation-from-metadata",
                "session_id": "session-from-metadata",
            }
        )

        self.assertEqual(memory_key(request), "conversation-from-metadata")
        self.assertEqual(monitoring_session_key(request), "session-from-metadata")

    def test_fingerprint_fallback_is_stable_for_legacy_anonymous_requests(self):
        first = chat_request(user=None)
        second = chat_request(user=None)

        self.assertEqual(memory_key(first), memory_key(second))
        self.assertEqual(memory_key(first), monitoring_session_key(first))

    def test_runtime_token_totals_are_isolated_by_monitoring_session(self):
        first_session = chat_request(
            conversation_id="conversation-1",
            session_id="session-1",
        )
        second_session = chat_request(
            conversation_id="conversation-1",
            session_id="session-2",
        )
        request_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }

        with (
            patch.dict(os.environ, {"LANGFUSE_ENABLED": "false"}),
            patch(
                "services.health_chat_service._run_chat_completion_with_timing",
                return_value=("answer", request_usage, None),
            ),
            patch(
                "services.health_chat_service.collected_token_usage",
                return_value=request_usage,
            ),
        ):
            run_chat_completion(first_session, [], "first", is_webui_task=False)
            run_chat_completion(first_session, [], "second", is_webui_task=False)
            _, _, metadata = run_chat_completion(
                second_session,
                [],
                "third",
                is_webui_task=False,
            )

        self.assertEqual(session_usage_store["session-1"]["total_tokens"], 24)
        self.assertEqual(session_usage_store["session-2"]["total_tokens"], 12)
        self.assertEqual(metadata["token_usage"]["session"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
