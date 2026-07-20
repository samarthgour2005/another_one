import uuid

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.sql import func

from app.core.database import Base


class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metrics = Column(JSON, default=lambda: {"total_tokens": 0, "total_messages": 0})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
