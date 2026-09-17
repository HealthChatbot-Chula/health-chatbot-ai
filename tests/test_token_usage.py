import unittest
from unittest.mock import Mock

from agent.llm_timing import collected_token_usage, request_timing, timed_llm_invoke


class TokenUsageTests(unittest.TestCase):
    def test_all_llm_calls_are_aggregated_for_request(self):
        first_response = Mock(
            content="first",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
            },
        )
        second_response = Mock(
            content="second",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
            },
        )
        model = Mock()
        model.invoke.side_effect = [first_response, second_response]

        with request_timing(route="test") as timing:
            timed_llm_invoke(model, "one", "first")
            timed_llm_invoke(model, "two", "second")
            usage = collected_token_usage(timing)

        self.assertEqual(
            usage,
            {
                "prompt_tokens": 30,
                "completion_tokens": 7,
                "total_tokens": 37,
            },
        )


if __name__ == "__main__":
    unittest.main()
