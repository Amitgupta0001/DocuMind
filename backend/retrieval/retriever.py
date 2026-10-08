"""
DocuMind Retriever Module.
Performs semantic similarity search, score thresholding, deduplication,
and context assembly for the RAG generation pipeline.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from backend.config import (
    MAX_CONTEXT_CHUNKS,
    MIN_SIMILARITY_GAP,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from backend.embeddings.embedder import embed_query
from backend.retrieval.vector_store import get_store

logger = logging.getLogger(__name__)


def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate chunks based on deterministic chunk_id or identical text."""
    seen_ids: Set[str] = set()
    seen_texts: Set[str] = set()
    unique: List[Dict[str, Any]] = []

    for item in results:
        chunk_id = item.get("chunk_id")
        text_snippet = item.get("text", "").strip()[:100]

        if chunk_id and chunk_id in seen_ids:
            continue
        if text_snippet in seen_texts:
            continue

        if chunk_id:
            seen_ids.add(chunk_id)
        if text_snippet:
            seen_texts.add(text_snippet)

        unique.append(item)

    return unique


def _deduplicate_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produce clean, grouped source citations with highest matching score and page numbers.
    """
    grouped: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}

    for res in results:
        source = res.get("source", "unknown")
        location = res.get("location", "Document")
        page = res.get("page")
        score = res.get("score", 0.0)
        doc_type = res.get("type", "unknown")

        key = (source, location, page)
        if key not in grouped or score > grouped[key]["score"]:
            grouped[key] = {
                "source": source,
                "location": location,
                "page": page,
                "type": doc_type,
                "score": score,
            }

    # Sort sources by descending similarity score
    sorted_sources = sorted(grouped.values(), key=lambda s: s["score"], reverse=True)
    return sorted_sources


def retrieve(
    question: str,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_context_chunks: int = MAX_CONTEXT_CHUNKS,
) -> Dict[str, Any]:
    """
    Core RAG retrieval step:
    1. Embed user query
    2. Perform FAISS top-K similarity search
    3. Filter out weak matches below similarity_threshold
    4. Deduplicate chunks
    5. Verify similarity gap
    6. Select top context chunks up to max_context_chunks
    """
    clean_q = question.strip() if question else ""
    if not clean_q:
        return {
            "relevant": False,
            "answerable": False,
            "results": [],
            "deduplicated_sources": [],
            "query": clean_q,
        }

    store = get_store()
    if store.count() == 0:
        logger.info("Retrieval requested but vector store is empty.")
        return {
            "relevant": False,
            "answerable": False,
            "results": [],
            "deduplicated_sources": [],
            "query": clean_q,
        }

    try:
        query_embedding = embed_query(clean_q)
    except Exception as exc:
        logger.error("Failed to generate query embedding: %s", exc)
        return {
            "relevant": False,
            "answerable": False,
            "results": [],
            "deduplicated_sources": [],
            "query": clean_q,
        }

    raw_results = store.search(query_embedding, top_k=top_k)
    if not raw_results:
        return {
            "relevant": False,
            "answerable": False,
            "results": [],
            "deduplicated_sources": [],
            "query": clean_q,
        }

    # Similarity threshold gating
    filtered = [r for r in raw_results if r.get("score", 0.0) >= similarity_threshold]

    if not filtered:
        max_seen = max((r.get("score", 0.0) for r in raw_results), default=0.0)
        logger.info(
            "Query '%s' rejected: highest score %.3f below threshold %.3f",
            clean_q[:50],
            max_seen,
            similarity_threshold,
        )
        return {
            "relevant": False,
            "answerable": False,
            "results": [],
            "deduplicated_sources": [],
            "query": clean_q,
        }

    unique_chunks = _deduplicate_results(filtered)

    # Optional check: similarity gap between top-1 and top-2
    if len(unique_chunks) >= 2:
        gap = unique_chunks[0]["score"] - unique_chunks[1]["score"]
        if MIN_SIMILARITY_GAP > 0 and gap < MIN_SIMILARITY_GAP:
            logger.debug("Tight similarity gap between top-2 results: %.4f", gap)

    selected = unique_chunks[:max_context_chunks]
    sources = _deduplicate_sources(selected)

    logger.info(
        "Retrieved %d relevant chunks (top score: %.3f) for query: '%s'",
        len(selected),
        selected[0]["score"],
        clean_q[:50],
    )

    return {
        "relevant": True,
        "answerable": len(selected) > 0,
        "results": selected,
        "deduplicated_sources": sources,
        "query": clean_q,
    }