"""BGE embeddings via fastembed (ONNX runtime).

Same BAAI/bge weights as sentence-transformers, but without PyTorch, so the
API container fits in a free-tier 512 MB instance. Query embeddings are cached
in Postgres (embedding_cache) because the same policy questions recur.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from sqlalchemy import text

from app.core.config import get_settings

# BGE v1.5 recommends this instruction on queries only (not on passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embed_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    return np.array(list(_model().embed(texts)), dtype=np.float32)


def embed_query(query: str, use_cache: bool = True) -> np.ndarray:
    s = get_settings()
    key = hashlib.sha256(f"{s.embed_model}|{query}".encode()).hexdigest()
    if use_cache:
        try:
            from app.db.session import rw_engine

            with rw_engine().connect() as conn:
                row = conn.execute(
                    text("SELECT embedding FROM embedding_cache WHERE text_hash = :h"), {"h": key}
                ).scalar()
            if row is not None:
                return np.array(row, dtype=np.float32)
        except Exception:  # cache is best-effort
            pass
    vec = np.array(next(iter(_model().embed([QUERY_INSTRUCTION + query]))), dtype=np.float32)
    if use_cache:
        try:
            from app.db.session import rw_engine

            with rw_engine().begin() as conn:
                conn.execute(
                    text("INSERT INTO embedding_cache (text_hash, model, embedding) VALUES (:h, :m, :e) "
                         "ON CONFLICT (text_hash) DO NOTHING"),
                    {"h": key, "m": s.embed_model, "e": str(vec.tolist())},
                )
        except Exception:
            pass
    return vec
