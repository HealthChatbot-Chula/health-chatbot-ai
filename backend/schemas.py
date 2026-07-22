from typing import Any, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    user: Optional[str] = None
    conversation_id: Optional[str] = None
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    stream: Optional[bool] = False

    class Config:
        extra = "allow"
