import json
import uuid
from typing import Any, Optional


def empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def response_payload(
    content: str,
    usage_data: Optional[dict[str, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if metadata:
        message["metadata"] = metadata

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage_data or empty_usage(),
    }


def stream_chunk(content: str, chunk_id: str) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_done(chunk_id: str) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
