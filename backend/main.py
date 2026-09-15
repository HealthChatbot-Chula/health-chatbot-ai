import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat
from backend.routes import eval as eval_routes
from backend.dependencies import load_agent_resources
from agent.rag_utils import warm_retrieval_resources


app = FastAPI()


@app.on_event("startup")
async def warm_health_agent() -> None:
    """Move one-time model/vector initialization out of the first chat request."""

    if os.getenv("WARM_RAG_ON_STARTUP", "true").lower() not in {"1", "true", "yes"}:
        print("[Startup Warmup] Disabled by WARM_RAG_ON_STARTUP.")
        return
    try:
        await asyncio.to_thread(load_agent_resources)
        await asyncio.to_thread(warm_retrieval_resources)
        print("[Startup Warmup] Health agent and RAG are ready.")
    except Exception as exc:
        # A temporary model/database issue must not prevent FastAPI from
        # starting; the normal lazy path remains available and logs its error.
        print(f"[Startup Warmup] Skipped: {exc}")

# Allow OpenWebUI or local clients to call the OpenAI-compatible endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(eval_routes.router)
