from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str
    status: str
    metrics: dict | None = None
