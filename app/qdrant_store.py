"""Qdrant infrastructure: the shared client, collection names, and the
resume-collection bootstrap. Pure infra — no CV- or ISRO-specific logic lives
here, so higher-level modules can import this without circular dependencies.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.embedding import VECTOR_SIZE

ISRO_COLLECTION = "isro_knowledge_base"
RESUME_COLLECTION = "resume_kb"
QDRANT_URL = "http://localhost:6333"

try:
    client = QdrantClient(url=QDRANT_URL, timeout=5)
    client.get_collections()  # ping
except Exception as e:
    raise SystemExit(
        f"\nCannot reach Qdrant at {QDRANT_URL}.\n"
        "Start it with:\n"
        "  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \\\n"
        "      -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant\n"
        f"\nUnderlying error: {e}"
    )


def ensure_resume_collection() -> None:
    """Create resume_kb if missing. Never wipes — accumulation mode."""
    if client.collection_exists(RESUME_COLLECTION):
        return
    client.create_collection(
        collection_name=RESUME_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: {RESUME_COLLECTION}")
