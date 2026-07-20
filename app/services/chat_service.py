import base64
import json
import os
import time

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logger import log
from app.models import ChatSession, Message, MessageRole
from app.services.retrieval_service import retrieve_and_rerank

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def _get_history(db: AsyncSession, session_id: str) -> list[dict]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(desc(Message.created_at))
        .limit(settings.MAX_CHAT_HISTORY)
    )
    result = await db.execute(stmt)
    rows = list(reversed(result.scalars().all()))
    return [{"role": m.role.value, "content": m.content} for m in rows]


async def stream_chat_response(session_id: str, user_query: str):
    """
    Generator that yields SSE-formatted strings:
      - `context` : sources found for this query (sent early, before generation)
      - `token`   : one streamed token of the answer
      - `error`   : if the upstream call fails
      - `end`     : final metrics once generation is complete
    """
    async with AsyncSessionLocal() as db:
        db.add(Message(session_id=session_id, role=MessageRole.USER, content=user_query))
        await db.commit()

        final_chunks, retrieval_metrics = await retrieve_and_rerank(session_id, user_query, db)

        context_string = ""
        sources: list[dict] = []
        images: list[str] = []

        for chunk in final_chunks:
            meta = chunk.payload or {}
            filename = meta.get("filename", "Unknown")
            page = meta.get("page_no") or meta.get("pages", "N/A")

            source = {"filename": filename, "page": page}
            if source not in sources:
                sources.append(source)

            context_string += f"\n[Source: {filename}, Page: {page}]\n{chunk.text}\n"

            if meta.get("has_image"):
                for path in meta.get("image_paths", []):
                    if os.path.exists(path) and path not in images:
                        images.append(path)

        yield _sse("context", {"sources": sources})

        history = await _get_history(db, session_id)

        # Attach up to 3 images to the latest user turn (multimodal payload)
        content_payload: list[dict] = [{"type": "text", "text": user_query}]
        for img_path in images[:3]:
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_encode_image(img_path)}"},
                }
            )
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = content_payload

        system_prompt = (
            "You are an advanced AI research assistant with vision capabilities. "
            "Answer the user's question using ONLY the provided context. "
            "For every claim drawn from the context, cite it as [Source: filename, Page: N]. "
            "If the context doesn't contain the answer, say so plainly.\n\n"
            f"CONTEXT:{context_string or 'No relevant documents were found for this query.'}"
        )
        messages_payload = [{"role": "system", "content": system_prompt}] + history

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.GENERATION_MODEL,
            "messages": messages_payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        gen_start = time.time()
        assistant_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=60.0
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if choices:
                            token = choices[0].get("delta", {}).get("content")
                            if token:
                                assistant_text += token
                                yield _sse("token", {"token": token})

                        if usage := data.get("usage"):
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
        except Exception as e:
            log.error(f"Streaming generation failed: {e}")
            yield _sse("error", {"detail": str(e)})

        generation_time = round(time.time() - gen_start, 2)
        final_metrics = {
            "generation_time_seconds": generation_time,
            "response_model_used": settings.GENERATION_MODEL,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "retrieved_chunks": [{"id": c.id, "text": c.text} for c in final_chunks],
            **retrieval_metrics,
        }

        db.add(
            Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=assistant_text,
                metrics=final_metrics,
            )
        )

        session_obj = await db.get(ChatSession, session_id)
        if session_obj:
            m = dict(session_obj.metrics or {})
            m["total_tokens"] = m.get("total_tokens", 0) + prompt_tokens + completion_tokens
            m["total_messages"] = m.get("total_messages", 0) + 2
            session_obj.metrics = m

        await db.commit()
        yield _sse("end", {"metrics": final_metrics})
