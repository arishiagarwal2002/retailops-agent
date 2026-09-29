"""Cross-encoder re-ranking (ms-marco MiniLM, ONNX via fastembed)."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=get_settings().rerank_model)


def rerank(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    return [float(s) for s in _model().rerank(query, documents)]
