from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, ForeignKey, Integer, String

from app.core.config import settings
from app.core.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    text = Column(String, nullable=False)
    payload = Column(JSON, default=dict)  # filename, page, headings, image_paths, etc.
    embedding = Column(Vector(settings.EMBEDDING_DIM))
