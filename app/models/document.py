import enum

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.sql import func

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    filename = Column(String, index=True)
    file_path = Column(String, nullable=True)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    extracted_data_path = Column(String, nullable=True)
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
