from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.openai_compat import empty_usage, response_payload
from backend.schemas import ChatRequest
from backend.settings import EVAL_SIMULATOR_MODEL, HEALTH_MODEL
from services.eval_service import (
    is_simulation_command,
    run_openwebui_simulation,
    stream_openwebui_simulation,
)
from services.health_chat_service import (
    is_openwebui_task,
    langchain_messages,
    log_openwebui_request,
    run_chat_completion,
)


router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    converted_messages = langchain_messages(req.messages)
    last_message_content = req.messages[-1].content if req.messages else ""
    is_webui_task = is_openwebui_task(last_message_content)

    log_openwebui_request(req, is_webui_task=is_webui_task)

    if not is_webui_task and (
        req.model == EVAL_SIMULATOR_MODEL or is_simulation_command(last_message_content)
    ):
        print("\n[Eval Simulator] Running user simulation through Open WebUI.")
        if req.stream:
            return StreamingResponse(
                stream_openwebui_simulation(last_message_content),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        assistant_content = run_openwebui_simulation(last_message_content)
        return response_payload(assistant_content, empty_usage())

    assistant_content, usage_data, metadata = run_chat_completion(
        req,
        converted_messages,
        last_message_content,
        is_webui_task=is_webui_task,
    )
    return response_payload(assistant_content, usage_data, metadata)


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": HEALTH_MODEL,
                "object": "model",
                "created": 1667925528,
                "owned_by": "health-chatbot",
            },
            {
                "id": "health-model",
                "object": "model",
                "created": 1667925528,
                "owned_by": "health-chatbot",
            },
            {
                "id": EVAL_SIMULATOR_MODEL,
                "object": "model",
                "created": 1667925528,
                "owned_by": "health-chatbot",
            },
        ],
    }
