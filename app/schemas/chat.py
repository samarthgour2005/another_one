from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatMessage(BaseModel):
    id: int
    role: str
    content: str
    metrics: dict | None = None
