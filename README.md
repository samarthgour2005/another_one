# Research RAG — Backend

A multimodal RAG backend: FastAPI + PostgreSQL/pgvector + Docling (PDF parsing) +
FlashRank (reranking) + OpenRouter (embeddings/chat), streamed over SSE.

## 1. Prerequisites

- Python 3.11+
- Docker (for Postgres/pgvector) — or a local Postgres with the `vector` extension
- An [OpenRouter](https://openrouter.ai) API key
- `uv` (recommended) — `pip install uv` — or plain `pip` works too

## 2. Start Postgres (pgvector)

```bash
docker compose up -d
```

This starts Postgres on `localhost:5432` with user `raguser` / password `ragpass` / db `ragdb`,
using the `pgvector/pgvector:pg16` image (pgvector extension pre-installed).

Enable the extension once:
```bash
docker compose exec db psql -U raguser -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 3. Install dependencies

```bash
uv sync
```
(or `pip install -e .` if not using uv)

> **Heads up:** `docling` and `flashrank` are heavy — Docling in particular pulls in
> layout/OCR models and can take a while on first run (and downloads model weights on
> first PDF you parse). Reranking (`flashrank`) also downloads its cross-encoder model
> the first time it's used.

## 4. Configure environment

```bash
cp .env.example .env
```
Edit `.env` and set your `OPENROUTER_API_KEY`. If you used `docker-compose.yml` as-is,
`DATABASE_URL` in `.env.example` already matches it.

## 5. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```
(or `uvicorn app.main:app --reload --port 8000` if not using uv)

On startup it creates all tables automatically (no migrations yet — see the guide's
"things to add" section for why you'd want Alembic eventually).

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## 6. Try it end-to-end

**Option A — smoke test script:**
```bash
python scripts/smoke_test.py /path/to/paper.pdf "What is this paper about?"
```

**Option B — curl:**
```bash
# 1. Upload a PDF (creates a session)
curl -X POST http://localhost:8000/api/v1/upload/ \
  -F "files=@/path/to/paper.pdf"
# -> {"status": "accepted", "session_id": "...", "documents": [...]}

# 2. Poll until ingestion completes
curl http://localhost:8000/api/v1/sessions/<session_id>/documents

# 3. Chat (streams Server-Sent Events)
curl -N -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "Summarize the key findings"}'
```

## 7. Notes on running without a GPU

`ACCELERATOR_DEVICE` in `.env`/`config.py` defaults to `"cpu"` in this version so it
runs out of the box on any machine. Set it to `"cuda"` if you have an NVIDIA GPU set up
for faster PDF parsing — Docling is the slowest part of the pipeline on CPU.

## 8. Project layout

```
app/
├── main.py                 FastAPI app, CORS, router mounting, table creation
├── core/                   settings, DB engine/session, logging
├── models/                 SQLAlchemy ORM tables (sessions, documents, chunks, messages)
├── schemas/                 Pydantic request/response models
├── api/routes/              upload.py, chat.py, session.py
└── services/
    ├── ingestion_service.py  Docling parsing (process pool) + embedding
    ├── retrieval_service.py  query expansion + vector search + FlashRank rerank
    └── chat_service.py        SSE streaming, multimodal prompt building, persistence
scripts/smoke_test.py        manual end-to-end test against a running server
docker-compose.yml            local Postgres + pgvector
```

## 9. Common issues

- **`ConnectionRefusedError` on startup** → Postgres isn't running or `DATABASE_URL` is wrong.
- **`relation "vector" does not exist` / vector type errors** → you forgot `CREATE EXTENSION vector;`.
- **Ingestion stuck on `parsing` forever** → check server logs; Docling errors (corrupt PDF,
  out-of-memory) show up in `Document.metrics.error` once the background task finishes.
- **401/402 from OpenRouter** → check your API key and that the free-tier models in
  `config.py` haven't been deprecated/rate-limited (swap `GENERATION_MODEL` /
  `EMBEDDING_MODEL` for any OpenRouter-supported model as needed).
