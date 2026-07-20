"""
The retrieval pipeline:
  1. Expand the user's query into several phrasings (LLM call)
  2. Embed all query variants
  3. Vector-search each variant against this session's chunks, merge & de-dupe
  4. Re-rank the merged candidates with a cross-encoder (FlashRank)
"""

import asyncio
import json
import time

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import log
from app.models import Document, DocumentChunk

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Reranker is initialized once, lazily, so the app can still start if it fails to load.
_ranker = None
_ranker_init_attempted = False


def _get_ranker():
    global _ranker, _ranker_init_attempted
    if not settings.USE_RERANKER:
        return None
    if _ranker_init_attempted:
        return _ranker
    _ranker_init_attempted = True
    try:
        from flashrank import Ranker

        _ranker = Ranker(model_name=settings.RANKER_MODEL, cache_dir="uploads/models")
    except Exception as e:
        log.error(f"Reranker failed to initialize, continuing without it: {e}")
        _ranker = None
    return _ranker


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.EMBEDDING_MODEL, "input": texts}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=30.0
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


async def _expand_query(user_query: str, client: httpx.AsyncClient) -> list[str]:
    """Ask an LLM for alternative phrasings of the query to widen vector recall."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    system_prompt = (
        f"You are a search query optimizer. Generate exactly {settings.IMPROVED_QUERIES_COUNT} "
        "alternative phrasings of the user's query to improve recall in a vector database. "
        "Include synonyms and both broader and narrower phrasings. "
        'Respond ONLY with JSON: {"queries": ["...", "..."]}'
    )
    payload = {
        "model": settings.QUERY_IMPROVER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        response = await client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=15.0)
        response.raise_for_status()
        content_str = response.json()["choices"][0]["message"]["content"]
        return json.loads(content_str).get("queries", [])
    except Exception as e:
        log.warning(f"Query expansion failed, falling back to original query only: {e}")
        return []


def _rerank_sync(user_query: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    ranker = _get_ranker()
    if not ranker or not chunks:
        return chunks

    from flashrank import RerankRequest

    passages = [{"id": c.id, "text": c.text, "meta": c.payload} for c in chunks]
    results = ranker.rerank(RerankRequest(query=user_query, passages=passages))
    chunk_map = {c.id: c for c in chunks}
    return [chunk_map[r["id"]] for r in results if r["id"] in chunk_map]


async def retrieve_and_rerank(
    session_id: str, user_query: str, db: AsyncSession
) -> tuple[list[DocumentChunk], dict]:
    start = time.time()

    async with httpx.AsyncClient() as client:
        expanded_queries = await _expand_query(user_query, client)

    all_queries = [user_query] + expanded_queries
    query_vectors = await _get_embeddings(all_queries)

    unique_chunks: dict[int, DocumentChunk] = {}
    for vector in query_vectors:
        stmt = (
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.session_id == session_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(settings.TOP_K_INITIAL)
        )
        result = await db.execute(stmt)
        for chunk in result.scalars().all():
            unique_chunks[chunk.id] = chunk

    initial_chunks = list(unique_chunks.values())
    if not initial_chunks:
        return [], {
            "retrieval_time": round(time.time() - start, 2),
            "queries_generated": len(expanded_queries),
            "initial_chunks_found": 0,
            "final_chunks_used": 0,
        }

    reranked = await asyncio.to_thread(_rerank_sync, user_query, initial_chunks)
    final_chunks = reranked[: settings.TOP_K_RERANK]

    metrics = {
        "retrieval_time": round(time.time() - start, 2),
        "queries_generated": len(expanded_queries),
        "initial_chunks_found": len(initial_chunks),
        "final_chunks_used": len(final_chunks),
        "reranker_used": _get_ranker() is not None,
    }
    return final_chunks, metrics
