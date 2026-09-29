"""Heading-aware chunking for the policy corpus.

Each `### N.M Title` subsection becomes one chunk, carrying its document
name and section label as metadata, which is exactly what the citation shows
(e.g. [Returns Policy §3.2]). Subsections over ~400 tokens are split into
windows with ~50 tokens of overlap. Token counts are approximated as
words x 1.3, which is close enough for sizing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_TOKENS = 400
OVERLAP_TOKENS = 50
WORDS_PER_TOKEN = 1 / 1.3


@dataclass(frozen=True)
class Chunk:
    doc_name: str
    section: str  # e.g. "§3.2 Refund timing"
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.doc_name} {self.section.split(' ', 1)[0]}]"

    @property
    def embed_text(self) -> str:
        return f"{self.doc_name} - {self.section}\n{self.text}"


def _split_long(words: list[str]) -> list[str]:
    size = int(MAX_TOKENS * WORDS_PER_TOKEN)
    step = size - int(OVERLAP_TOKENS * WORDS_PER_TOKEN)
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - OVERLAP_TOKENS, 1), step)]


def chunk_markdown(text: str) -> list[Chunk]:
    doc_name = ""
    section = ""
    buf: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        body = " ".join(" ".join(buf).split())
        if section and body:
            for piece in _split_long(body.split()):
                chunks.append(Chunk(doc_name, section, piece))
        buf.clear()

    for line in text.splitlines():
        if m := re.match(r"^# (.+)$", line):
            doc_name = m.group(1).strip()
        elif re.match(r"^## ", line):
            flush()
            section = ""
        elif m := re.match(r"^### (\d+(?:\.\d+)*)\s+(.+)$", line):
            flush()
            section = f"§{m.group(1)} {m.group(2).strip()}"
        else:
            buf.append(line)
    flush()
    return chunks


def chunk_directory(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for f in sorted(path.glob("*.md")):
        out.extend(chunk_markdown(f.read_text(encoding="utf-8")))
    return out
