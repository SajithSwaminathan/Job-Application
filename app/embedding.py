"""Sentence-transformer embedding model — shared by every retrieval path."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast & light
VECTOR_SIZE = 384
