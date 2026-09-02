"""Lexical retrieval over the local knowledge base.

Deliberately BM25 (``rank-bm25``, pure Python, no model download) rather than
an embedding + vector-store pipeline. The corpus here is a handful of
curated markdown documents chunked by section — at that scale, an embedding
index adds a dependency and a model download for no retrieval-quality
benefit; BM25's term-overlap scoring is enough to find the right section for
the kind of short, keyword-heavy queries an analyst agent issues ("emerging
market equities volatility", "risk band 3 allocation"). This stops being the
right call once the corpus grows past a few dozen documents or queries need
to match on meaning rather than shared vocabulary — see DECISIONS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    heading: str
    text: str


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_chunks(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        sections = re.split(r"(?m)^## ", path.read_text())
        for section in sections:
            section = section.strip()
            if not section or section.startswith("# "):
                continue
            heading, _, body = section.partition("\n")
            chunks.append(Chunk(doc_id=path.stem, heading=heading.strip(), text=body.strip()))
    return chunks


class KnowledgeStore:
    """A small BM25-indexed corpus, loaded once and queried many times."""

    def __init__(self, directory: Path = KNOWLEDGE_BASE_DIR) -> None:
        self._chunks = _load_chunks(directory)
        corpus = [_tokenize(f"{c.heading} {c.text}") for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self._chunks, strict=True), key=lambda pair: pair[0], reverse=True)
        return [chunk for score, chunk in ranked[:top_k] if score > 0]

    def search_as_text(self, query: str, top_k: int = 3) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "No relevant results in the knowledge base."
        return "\n\n".join(f"[{c.doc_id}#{c.heading}]\n{c.text}" for c in hits)
