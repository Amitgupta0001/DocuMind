"""
DocuMind Embedding Engine.
Generates normalized dense vector embeddings using local sentence-transformers models.
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None

# Prefix recommended for BAAI/bge retrieval query models
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the embedding model instance.
    """
    global _model

    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def set_model(model: Optional[SentenceTransformer]) -> None:
    """Set or mock the global model (useful for unit testing)."""
    global _model
    _model = model


def embed_documents(texts: List[str]) -> np.ndarray:
    """
    Generate normalized embeddings for document chunks.
    Maintains strict 1:1 length correspondence with input texts.
    """
    if not texts:
        model = get_model()
        dimension = model.get_sentence_embedding_dimension()
        return np.zeros((0, dimension), dtype=np.float32)

    # Sanitize texts while preserving strict array index alignment
    safe_texts = [t.strip() if (t and t.strip()) else " " for t in texts]

    model = get_model()
    embeddings = model.encode(
        safe_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    arr = np.asarray(embeddings, dtype=np.float32)
    logger.debug("Generated %d document embeddings with dimension %d", len(arr), arr.shape[1] if arr.ndim > 1 else 0)
    return arr


def embed_query(text: str) -> np.ndarray:
    """
    Generate a normalized embedding for a user query.
    Applies BGE query prefix if applicable.
    """
    if not text or not text.strip():
        raise ValueError("Query cannot be empty.")

    model = get_model()
    query_str = BGE_QUERY_PREFIX + text.strip()

    embedding = model.encode(
        [query_str],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embedding, dtype=np.float32)