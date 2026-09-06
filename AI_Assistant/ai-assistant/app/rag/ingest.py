"""
Document ingestion: load raw text/PDF files and split them into
overlapping chunks suitable for embedding.

Chunking strategy: recursive character splitting that tries to break on
paragraph -> sentence -> word boundaries before falling back to a hard
character cut, so chunks stay semantically coherent.
"""
import hashlib
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader


def load_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Split `text` into pieces no larger than chunk_size, trying separators
    from coarsest to finest and recursing into any piece that is still too
    big after splitting on the current separator."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # No separator left to try (or none matched) -> hard character split.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    raw_pieces = text.split(sep) if sep else list(text)

    # If this separator didn't actually split anything, fall through to the next one.
    if len(raw_pieces) == 1:
        return _split_recursive(text, chunk_size, rest)

    result: List[str] = []
    for piece in raw_pieces:
        if not piece:
            continue
        if len(piece) > chunk_size:
            result.extend(_split_recursive(piece, chunk_size, rest))
        else:
            result.append(piece)

    # Greedily re-merge small adjacent pieces back up to chunk_size so we
    # don't end up with lots of tiny fragments (e.g. one per sentence).
    merged: List[str] = []
    current = ""
    for piece in result:
        candidate = current + (sep if current else "") + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def recursive_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Recursively split `text` on decreasing-granularity separators
    (paragraph -> line -> sentence -> word -> hard cut), then stitch a
    small overlap between consecutive chunks for retrieval continuity."""
    text = re.sub(r"\s+\n", "\n", text).strip()
    if not text:
        return []

    pieces = _split_recursive(text, chunk_size, _SEPARATORS)

    overlapped = []
    for i, piece in enumerate(pieces):
        if i == 0 or overlap <= 0:
            overlapped.append(piece)
        else:
            tail = pieces[i - 1][-overlap:]
            overlapped.append(tail + piece)
    return [c.strip() for c in overlapped if c.strip()]


def make_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text[:50]}".encode()).hexdigest()[:12]
    return f"{Path(source).stem}-{index}-{digest}"


def ingest_file(path: str, chunk_size: int = 800, overlap: int = 120):
    """Returns list of (chunk_id, text, metadata) tuples ready for embedding."""
    text = load_text(path)
    chunks = recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)
    records = []
    for i, chunk in enumerate(chunks):
        cid = make_chunk_id(path, i, chunk)
        records.append((cid, chunk, {"source": Path(path).name, "chunk_index": i}))
    return records
