from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat
from backend.routes import eval as eval_routes


app = FastAPI()

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
