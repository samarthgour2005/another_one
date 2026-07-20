from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import ChatSession
from app.schemas.chat import ChatRequest
from app.services.chat_service import stream_chat_response

router = APIRouter()


@router.post("/chat/")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    session = await db.get(ChatSession, request.session_id)
    if not session:
        raise HTTPException(404, f"Session '{request.session_id}' not found. Upload a document first.")

    return StreamingResponse(
        stream_chat_response(request.session_id, request.message),
        media_type="text/event-stream",
    )
