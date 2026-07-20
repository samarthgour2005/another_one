import os
import shutil
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.logger import log
from app.models import ChatSession, Document
from app.services.ingestion_service import run_background_ingestion_task

router = APIRouter()


@router.post("/upload/")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Get or create the session
    if session_id:
        session = await db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found.")
    else:
        session = ChatSession()
        db.add(session)
        await db.flush()
        session_id = str(session.id)

    if len(files) > settings.MAX_PDF_UPLOADS:
        raise HTTPException(400, f"Max {settings.MAX_PDF_UPLOADS} files per upload.")

    doc_tasks = []
    response_docs = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            log.warning(f"Skipping non-PDF upload: {file.filename}")
            continue

        new_doc = Document(session_id=session_id, filename=file.filename, status="pending")
        db.add(new_doc)
        await db.flush()

        doc_dir = os.path.join(settings.UPLOAD_DIR, session_id, str(new_doc.id))
        os.makedirs(doc_dir, exist_ok=True)
        file_path = os.path.join(doc_dir, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()

        new_doc.file_path = file_path
        doc_tasks.append((new_doc.id, file_path, session_id))
        response_docs.append({"filename": file.filename, "document_id": new_doc.id})

    if not doc_tasks:
        raise HTTPException(400, "No valid PDF files were uploaded.")

    await db.commit()
    background_tasks.add_task(run_background_ingestion_task, doc_tasks)

    return {"status": "accepted", "session_id": session_id, "documents": response_docs}
