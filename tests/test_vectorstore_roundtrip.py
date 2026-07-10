import numpy as np

from auto_claim_rag.types import Chunk
from auto_claim_rag.vectorstore import VectorIndex


def test_vectorstore_roundtrip(tmp_path) -> None:
    chunks = [
        Chunk(
            chunk_id="d1::c0",
            doc_id="d1",
            source_path="x",
            ordinal=0,
            text="hello world",
        ),
        Chunk(
            chunk_id="d1::c1",
            doc_id="d1",
            source_path="x",
            ordinal=1,
            text="second chunk",
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    index = VectorIndex(chunks=chunks, embeddings=embeddings)
    index.save(str(tmp_path))

    loaded = VectorIndex.load(str(tmp_path))
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]
    assert [c.text for c in loaded.chunks] == [c.text for c in chunks]
    assert loaded.embeddings.shape == embeddings.shape
    assert np.allclose(loaded.embeddings, embeddings)
