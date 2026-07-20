# Research RAG 

Upload research PDFs, ask questions, get streamed answers with citations and
relevant figures — powered by FastAPI, pgvector, Docling, FlashRank, and OpenRouter.

## How it works

1. **Upload** a PDF → Docling extracts text + images, splits into chunks (tracks page/heading).
2. **Embed** each chunk into a vector, stored in Postgres via `pgvector`.
3. **Ask a question** →
   - question is expanded into a few alternate phrasings
   - vector search finds candidate chunks
   - FlashRank re-ranks them for precision
   - top chunks + related images go to a vision LLM
   - answer streams back token-by-token with `[Source: file, Page: N]` citations

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI (async) |
| Database | PostgreSQL + pgvector |
| PDF parsing | Docling |
| Reranking | FlashRank |
| LLM & embeddings | OpenRouter |
| Frontend (original repo) | Next.js |

## Setup

**Prereqs:** Python 3.11+, Docker, an [OpenRouter](https://openrouter.ai) API key, `uv` (or pip)

```bash
# 1. Start Postgres + pgvector
docker compose up -d
docker compose exec db psql -U raguser -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Install deps
uv sync   # or: pip install -e .

# 3. Configure
cp .env.example .env   # add your OPENROUTER_API_KEY

# 4. Run
uv run uvicorn app.main:app --reload --port 8000
```

Tables auto-create on startup. Docs at `http://localhost:8000/docs`.

> `docling` and `flashrank` are heavy — first run downloads model weights, so expect a delay on first PDF/first query.

## Try it

```bash
# Upload (creates a session)
curl -X POST http://localhost:8000/api/v1/upload/ -F "files=@paper.pdf"

# Poll status
curl http://localhost:8000/api/v1/sessions/<session_id>/documents

# Chat (SSE stream)
curl -N -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "Summarize the key findings"}'
```

Or: `python scripts/smoke_test.py paper.pdf "your question"`

## Project layout

```
app/
├── main.py                 FastAPI app, CORS, routers, table creation
├── core/                   settings, DB engine/session, logging
├── models/                 sessions, documents, chunks, messages
├── schemas/                 Pydantic request/response models
├── api/routes/              upload.py, chat.py, session.py
└── services/
    ├── ingestion_service.py  Docling parsing (process pool) + embedding
    ├── retrieval_service.py  query expansion + vector search + rerank
    └── chat_service.py        SSE streaming, multimodal prompt, persistence
scripts/smoke_test.py        manual end-to-end test
docker-compose.yml            local Postgres + pgvector
```