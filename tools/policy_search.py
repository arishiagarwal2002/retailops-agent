"""search_policy: dense top-20 -> cross-encoder -> top-k, with citation labels."""
from __future__ import annotations

from retrieval.store import search

MAX_K = 8


def search_policy(query: str, k: int = 5) -> dict:
    k = max(1, min(int(k), MAX_K))
    hits = search(query, k=k)
    return {
        "query": query,
        "results": [
            {"citation": h.citation, "doc_name": h.doc_name, "section": h.section, "text": h.chunk_text,
             "score": round(h.rerank_score if h.rerank_score is not None else h.dense_score, 3)}
            for h in hits
        ],
        "instruction": "Cite every policy claim with its citation label, e.g. [Returns Policy §3.2].",
    }
