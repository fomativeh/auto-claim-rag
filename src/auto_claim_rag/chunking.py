from __future__ import annotations

from auto_claim_rag.types import Chunk, Document


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def chunk_document(
    doc: Document,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    text = _normalize(doc.text)
    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for b in blocks:
        b_len = len(b) + 2
        if current_len + b_len <= chunk_chars:
            current.append(b)
            current_len += b_len
            continue

        if current:
            chunks.append("\n\n".join(current).strip())

        if len(b) > chunk_chars:
            start = 0
            while start < len(b):
                end = min(len(b), start + chunk_chars)
                chunks.append(b[start:end].strip())
                if end == len(b):
                    break
                start = max(0, end - overlap_chars)
            current = []
            current_len = 0
            continue

        current = [b]
        current_len = b_len

    if current:
        chunks.append("\n\n".join(current).strip())

    out: list[Chunk] = []
    for i, chunk_text in enumerate(chunks):
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{i}",
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                ordinal=i,
                text=chunk_text,
            )
        )
    return out


def chunk_documents(
    docs: list[Document],
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(
            chunk_document(
                doc, chunk_chars=chunk_chars, overlap_chars=overlap_chars
            )
        )
    return out
