"""
Handles turning an uploaded PDF into searchable, embedded chunks.

Two phases, run separately because they have very different performance profiles:
  1. PARSE  (CPU-bound)   -> runs in a ProcessPoolExecutor, one process per PDF
  2. EMBED  (network-bound) -> runs on the asyncio event loop, batched HTTP calls
"""

import asyncio
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import httpx
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logger import log
from app.models import Document, DocumentChunk, DocumentStatus

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


# --------------------------------------------------------------------------
# Phase 1: Parsing (runs in a worker process, must be picklable / self-contained)
# --------------------------------------------------------------------------


def _flatten_metadata(raw_meta: dict, document_id: int, session_id: str) -> dict:
    """Pull out the bits of Docling's chunk metadata that we actually need."""
    pages: set[int] = set()
    for item in raw_meta.get("doc_items", []):
        for prov in item.get("prov", []):
            if "page_no" in prov:
                pages.add(prov["page_no"])

    filename = raw_meta.get("origin", {}).get("filename", f"doc_{document_id}")

    return {
        "session_id": session_id,
        "document_id": document_id,
        "filename": filename,
        "headings": " | ".join(raw_meta.get("headings", [])),
        "pages": ", ".join(map(str, sorted(pages))) if pages else "unknown",
        "page_no": next(iter(sorted(pages)), "unknown"),
    }


def _parse_and_chunk_document(
    file_path: str, document_id: int, session_id: str
) -> tuple[str, dict]:
    """
    Synchronous, CPU-heavy work: parse the PDF with Docling, pull out images,
    chunk the document, and write the result to a JSON sidecar file.

    Runs inside a separate OS process so it never blocks the API's event loop.
    """
    from docling.chunking import HierarchicalChunker
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    start = time.perf_counter()

    pipeline_options = PdfPipelineOptions(
        generate_picture_images=settings.EXTRACT_CHUNK_IMAGES,
        do_ocr=settings.ENABLE_OCR,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice(settings.ACCELERATOR_DEVICE),
            num_threads=settings.DOCLING_THREADS,
        ),
    )
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    docling_doc = converter.convert(file_path).document

    # Extract images to disk, remembering which internal ref maps to which file
    img_dir = os.path.join(settings.UPLOAD_DIR, session_id, str(document_id), "images")
    os.makedirs(img_dir, exist_ok=True)

    image_map: dict[str, str] = {}
    image_records = []
    for i, picture in enumerate(docling_doc.pictures):
        pil_image = picture.get_image(docling_doc)
        if pil_image is None:
            continue
        img_path = os.path.join(img_dir, f"pic_{i}.png")
        pil_image.save(img_path, "PNG")
        image_map[picture.self_ref] = img_path
        image_records.append({"path": img_path, "ref": picture.self_ref})

    # Chunk, and attach any images that belong to the same parent element
    chunks = list(HierarchicalChunker().chunk(docling_doc))
    serialized_chunks = []
    for chunk in chunks:
        chunk_image_paths = []
        for item in chunk.meta.doc_items:
            parent_ref = getattr(item.parent, "cref", None)
            if parent_ref and (img_path := image_map.get(parent_ref)):
                if img_path not in chunk_image_paths:
                    chunk_image_paths.append(img_path)

        meta = _flatten_metadata(chunk.meta.export_json_dict(), document_id, session_id)
        meta["has_image"] = bool(chunk_image_paths)
        meta["image_paths"] = chunk_image_paths
        serialized_chunks.append({"text": chunk.text, "metadata": meta})

    data_path = file_path.replace(".pdf", "_parsed_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": serialized_chunks, "images": image_records}, f, indent=2)

    metrics = {
        "parsing_time_seconds": round(time.perf_counter() - start, 2),
        "total_chunks_yielded": len(serialized_chunks),
        "total_images_extracted": len(image_records),
    }
    return data_path, metrics


# --------------------------------------------------------------------------
# Phase 2: Embedding (async, network-bound)
# --------------------------------------------------------------------------


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.EMBEDDING_MODEL, "input": texts}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=60.0
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


async def _embed_and_store_chunks(document_id: int, data_path: str, db: AsyncSession) -> dict:
    start = time.time()
    with open(data_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    chunks = parsed.get("chunks", [])
    if not chunks:
        return {"embedding_time_seconds": 0, "total_vectors_stored": 0}

    batch_size = 20
    total_vectors = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        embeddings = await _get_embeddings(texts)
        db_chunks = [
            DocumentChunk(
                document_id=document_id,
                text=chunk_data["text"],
                payload=chunk_data.get("metadata", {}),
                embedding=embeddings[j],
            )
            for j, chunk_data in enumerate(batch)
        ]
        db.add_all(db_chunks)
        await db.commit()
        total_vectors += len(db_chunks)

    return {
        "embedding_time_seconds": round(time.time() - start, 2),
        "total_vectors_stored": total_vectors,
        "embedding_model_used": settings.EMBEDDING_MODEL,
    }


# --------------------------------------------------------------------------
# Orchestration — this is what FastAPI's BackgroundTasks calls
# --------------------------------------------------------------------------


async def run_background_ingestion_task(doc_tasks: list[tuple[int, str, str]]) -> None:
    """
    doc_tasks: list of (document_id, file_path, session_id)
    Runs parsing for all docs in parallel processes, then embeds each sequentially.
    """
    log.info(f"Starting ingestion for {len(doc_tasks)} document(s)")

    async with AsyncSessionLocal() as db:
        for doc_id, _, _ in doc_tasks:
            await db.execute(
                update(Document).where(Document.id == doc_id).values(status=DocumentStatus.PARSING)
            )
        await db.commit()

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=settings.MAX_CPU_WORKERS) as pool:
        futures = [
            loop.run_in_executor(pool, _parse_and_chunk_document, file_path, doc_id, session_id)
            for doc_id, file_path, session_id in doc_tasks
        ]
        parse_results = await asyncio.gather(*futures, return_exceptions=True)

    async with AsyncSessionLocal() as db:
        for (doc_id, file_path, _), result in zip(doc_tasks, parse_results):
            doc = await db.get(Document, doc_id)
            if not doc:
                continue

            if isinstance(result, Exception):
                log.error(f"Parsing failed for document {doc_id}: {result}")
                doc.status = DocumentStatus.FAILED
                doc.metrics = {"error": str(result)}
                await db.commit()
                continue

            data_path, parse_metrics = result
            try:
                doc.status = DocumentStatus.EMBEDDING
                await db.commit()

                embed_metrics = await _embed_and_store_chunks(doc_id, data_path, db)

                doc.status = DocumentStatus.COMPLETED
                doc.extracted_data_path = data_path
                doc.metrics = {**parse_metrics, **embed_metrics}

                if os.path.exists(file_path):
                    os.remove(file_path)

                await db.commit()
                log.info(f"Document {doc_id} ready for chat")
            except Exception as e:
                log.error(f"Embedding failed for document {doc_id}: {e}")
                doc.status = DocumentStatus.FAILED
                doc.metrics = {**parse_metrics, "error": f"Embedding error: {e}"}
                await db.commit()
