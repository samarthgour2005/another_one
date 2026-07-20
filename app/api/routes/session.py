from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import ChatSession, Document, Message

router = APIRouter()


@router.post("/sessions/")
async def create_session(db: AsyncSession = Depends(get_db)):
    """Lets the frontend get a session_id before any file is uploaded."""
    session = ChatSession()
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return {"session_id": session.id}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"id": session.id, "created_at": session.created_at, "metrics": session.metrics}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    await db.delete(session)
    await db.commit()
    return {"status": "deleted"}


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()

    if not messages and not await db.get(ChatSession, session_id):
        raise HTTPException(404, "Session not found")

    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role.value,
                "content": m.content,
                "metrics": m.metrics,
                "created_at": m.created_at,
            }
            for m in messages
        ]
    }


@router.get("/sessions/{session_id}/documents")
async def get_documents(session_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Document).where(Document.session_id == session_id)
    result = await db.execute(stmt)
    docs = result.scalars().all()
    return {
        "documents": [
            {"id": d.id, "filename": d.filename, "status": d.status.value, "metrics": d.metrics}
            for d in docs
        ]
    }
