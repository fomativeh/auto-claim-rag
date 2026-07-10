from __future__ import annotations

from pathlib import Path

from auto_claim_rag.chunking import chunk_documents
from auto_claim_rag.embeddings import Embedder
from auto_claim_rag.types import Document, RetrievedChunk
from auto_claim_rag.vectorstore import VectorIndex


def build_index(
    docs: list[Document],
    *,
    embedder: Embedder,
    chunk_chars: int,
    chunk_overlap_chars: int,
) -> VectorIndex:
    chunks = chunk_documents(
        docs, chunk_chars=chunk_chars, overlap_chars=chunk_overlap_chars
    )
    vectors = embedder.embed_texts([c.text for c in chunks])
    return VectorIndex(chunks=chunks, embeddings=vectors)


def build_or_load_index(
    *,
    kb_dir: str,
    embedder: Embedder,
    chunk_chars: int,
    chunk_overlap_chars: int,
    index_dir: str,
) -> VectorIndex:
    root = Path(index_dir)
    chunks_path = root / "chunks.jsonl"
    emb_path = root / "embeddings.npy"

    if chunks_path.exists() and emb_path.exists():
        return VectorIndex.load(str(root))

    from auto_claim_rag.ingest.loaders import load_documents_from_dir

    docs = load_documents_from_dir(kb_dir).documents
    index = build_index(
        docs,
        embedder=embedder,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    index.save(str(root))
    return index



def retrieve(
    *,
    index: VectorIndex,
    embedder: Embedder,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    q = embedder.embed_query(query)
    return index.search(q, top_k=top_k)
