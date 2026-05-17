"""Resume-collection wiring: the only place that combines PDF extraction,
embeddings, and Qdrant. Retrieval here is hard-scoped to `resume_kb` and never
falls back to any other collection (grounding non-negotiable in CLAUDE.md).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import PointStruct

from app.embedding import embed_model
from app.qdrant_store import RESUME_COLLECTION, client, ensure_resume_collection
from app.cv.extract import chunk_text, extract_contact_info, extract_cv_text


def index_cv(pdf_path: str | Path) -> tuple[int, str, dict]:
    """Extract → chunk → embed → upsert into resume_kb.
    Returns (chunk_count, source_file, contact_info)."""
    ensure_resume_collection()
    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    text = extract_cv_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    contact = extract_contact_info(text)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    uploaded_at = datetime.now(timezone.utc).isoformat()

    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=emb,
            payload={
                "source_file": source_file,
                "uploaded_at": uploaded_at,
                "chunk_idx": i,
                "text": chunk,
                "contact": contact,  # duplicated per chunk so retrieval returns it
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=RESUME_COLLECTION, points=points)
    print(
        f"Indexed {len(points)} chunks into {RESUME_COLLECTION} from {source_file} "
        f"(contact: name={contact['name'] or '∅'} email={contact['email'] or '∅'})"
    )
    return len(points), source_file, contact


def retrieve_resume(query: str, top_k: int = 8) -> list[dict]:
    """Retrieve resume chunks ONLY from resume_kb — never falls back to ISRO data."""
    ensure_resume_collection()
    qv = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name=RESUME_COLLECTION,
        query=qv,
        limit=top_k,
    )
    return [
        {
            "source_file": p.payload.get("source_file", "?"),
            "chunk_idx": p.payload.get("chunk_idx", -1),
            "text": p.payload["text"],
            "contact": p.payload.get("contact", {}),
            "score": p.score,
        }
        for p in results.points
    ]


def fetch_full_cv(source_file: str) -> list[dict]:
    """Pull EVERY chunk for a given source_file, ordered by chunk_idx.

    Used at CV-generation time so the LLM sees the entire resume — preventing
    it from inventing plausible-but-missing facts to fill gaps from top-k
    semantic retrieval.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
    )
    points, _ = client.scroll(
        collection_name=RESUME_COLLECTION,
        scroll_filter=flt,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    chunks = [
        {
            "source_file": source_file,
            "chunk_idx": p.payload.get("chunk_idx", 0),
            "text": p.payload["text"],
            "contact": p.payload.get("contact", {}),
        }
        for p in points
    ]
    chunks.sort(key=lambda c: c["chunk_idx"])
    return chunks
