from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from auto_claim_rag.types import Chunk, RetrievedChunk


@dataclass
class VectorIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray

    def search(self, query_vec: np.ndarray, *, top_k: int) -> list[RetrievedChunk]:
        if self.embeddings.size == 0:
            return []

        scores = self.embeddings @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        out: list[RetrievedChunk] = []
        for idx in top_idx:
            out.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(scores[idx])))
        return out

    def save(self, dir_path: str) -> None:
        root = Path(dir_path)
        root.mkdir(parents=True, exist_ok=True)

        chunks_path = root / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "source_path": c.source_path,
                            "ordinal": c.ordinal,
                            "text": c.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        np.save(str(root / "embeddings.npy"), self.embeddings)

    @staticmethod
    def load(dir_path: str) -> "VectorIndex":
        root = Path(dir_path)
        chunks_path = root / "chunks.jsonl"
        emb_path = root / "embeddings.npy"

        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=item["chunk_id"],
                        doc_id=item["doc_id"],
                        source_path=item["source_path"],
                        ordinal=int(item["ordinal"]),
                        text=item["text"],
                    )
                )

        embeddings = np.load(str(emb_path)).astype(np.float32)
        return VectorIndex(chunks=chunks, embeddings=embeddings)
