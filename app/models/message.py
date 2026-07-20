import enum

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.sql import func

from app.core.database import Base


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(String, nullable=False)
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
