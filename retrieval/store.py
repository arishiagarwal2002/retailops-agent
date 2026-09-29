"""pgvector read/write for policy chunks, plus the ingest entry point.

    python -m retrieval.store      # (re)ingest data/policy/*.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import rw_engine
from retrieval.chunk import chunk_directory
from retrieval.embed import embed_passages, embed_query
from retrieval.rerank import rerank

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policy"


@dataclass
class Hit:
    id: int
    doc_name: str
    section: str
    chunk_text: str
    dense_score: float
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return f"[{self.doc_name} {self.section.split(' ', 1)[0]}]"


def ingest(policy_dir: Path = POLICY_DIR) -> int:
    chunks = chunk_directory(policy_dir)
    vectors = embed_passages([c.embed_text for c in chunks])
    engine = rw_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE policy_chunks RESTART IDENTITY"))
        conn.execute(text("DROP INDEX IF EXISTS ix_policy_chunks_embedding"))
        for c, v in zip(chunks, vectors):
            conn.execute(
                text("INSERT INTO policy_chunks (doc_name, section, chunk_text, embedding) "
                     "VALUES (:d, :s, :t, :e)"),
                {"d": c.doc_name, "s": c.section, "t": c.text, "e": str(v.tolist())},
            )
        # IVFFlat must be built after the data exists (it clusters existing rows).
        # pgvector guidance is lists ~ rows/1000 below 1M rows; we use a small value
        # and set probes = lists at query time, which makes the search exact.
        lists = max(1, int(math.sqrt(len(chunks)) // 2))
        conn.execute(text(
            f"CREATE INDEX ix_policy_chunks_embedding ON policy_chunks "
            f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"))
    return len(chunks)


def dense_search(query: str, k: int) -> list[Hit]:
    vec = embed_query(query)
    with rw_engine().begin() as conn:
        conn.execute(text("SET LOCAL ivfflat.probes = 10"))
        rows = conn.execute(
            text("SELECT id, doc_name, section, chunk_text, 1 - (embedding <=> CAST(:q AS vector)) AS score "
                 "FROM policy_chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"),
            {"q": str(vec.tolist()), "k": k},
        ).all()
    return [Hit(r.id, r.doc_name, r.section, r.chunk_text, float(r.score)) for r in rows]


def search(query: str, k: int | None = None, use_rerank: bool = True) -> list[Hit]:
    s = get_settings()
    k = k or s.rerank_k
    hits = dense_search(query, s.retrieve_k)
    if not use_rerank:
        return hits[:k]
    scores = rerank(query, [f"{h.doc_name} - {h.section}\n{h.chunk_text}" for h in hits])
    for h, sc in zip(hits, scores):
        h.rerank_score = sc
    return sorted(hits, key=lambda h: h.rerank_score or 0, reverse=True)[:k]


if __name__ == "__main__":
    n = ingest()
    print(f"ingested {n} policy chunks")
    for h in search("customer wants to return an order from 20 days ago"):
        print(f"{h.rerank_score:6.2f}  {h.citation}  {h.section}")
