from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, session, upload
from app.core.config import settings
from app.core.database import Base, engine
from app.core.logger import log
from app.models import *  # noqa: F401,F403  (ensures all models register with Base)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up: creating tables if they don't exist...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    log.info("Shutting down.")


app = FastAPI(title="Research RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_CORS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(session.router, prefix="/api/v1", tags=["Sessions"])


@app.get("/health")
async def health():
    return {"status": "ok"}
