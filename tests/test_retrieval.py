"""
Tests for semantic retrieval logic and thresholds.
"""

import numpy as np
from backend.embeddings.embedder import embed_documents
from backend.retrieval.retriever import retrieve
from backend.retrieval.vector_store import VectorStore


def test_retrieval_empty_store(isolated_vector_store: VectorStore):
    res = retrieve("What is DocuMind?")
    assert res["relevant"] is False
    assert res["answerable"] is False
    assert len(res["results"]) == 0
    assert len(res["deduplicated_sources"]) == 0


def test_retrieval_with_matching_content(isolated_vector_store: VectorStore):
    texts = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "DocuMind supports PDF, Word, and PowerPoint file formats for documentation.",
    ]
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_index": 0,
            "text": texts[0],
            "source": "fastapi_manual.txt",
            "location": "Page 1",
            "page": 1,
            "type": "txt",
        },
        {
            "chunk_id": "c2",
            "chunk_index": 1,
            "text": texts[1],
            "source": "documind_guide.txt",
            "location": "Page 2",
            "page": 2,
            "type": "txt",
        },
    ]

    embeddings = embed_documents(texts)
    isolated_vector_store.add(embeddings, chunks)

    # Search with low threshold to ensure match
    res = retrieve("FastAPI is a modern web framework", similarity_threshold=0.1)
    assert res["relevant"] is True
    assert res["answerable"] is True
    assert len(res["results"]) >= 1
    assert len(res["deduplicated_sources"]) >= 1


def test_retrieval_rejection_below_threshold(isolated_vector_store: VectorStore):
    texts = ["Quantum entanglement in superconducting circuits."]
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_index": 0,
            "text": texts[0],
            "source": "physics.txt",
            "location": "Page 1",
            "page": 1,
            "type": "txt",
        }
    ]
    embeddings = embed_documents(texts)
    isolated_vector_store.add(embeddings, chunks)

    # Force threshold higher than maximum possible to verify rejection
    res = retrieve("How do I make chocolate chip cookies?", similarity_threshold=0.9999)
    assert res["relevant"] is False
    assert res["answerable"] is False
    assert len(res["results"]) == 0
