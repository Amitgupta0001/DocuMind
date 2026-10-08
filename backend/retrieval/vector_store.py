"""
DocuMind Vector Store.
Manages persistent FAISS vector indexing and chunk metadata mapping.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from backend.config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thread-safe persistent FAISS vector store.
    Uses IndexFlatIP with normalized vectors for exact cosine similarity search.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or VECTORSTORE_DIR
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 0
        self._lock = threading.Lock()

    def build(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Initialize or replace the index with a set of embeddings and chunks."""
        with self._lock:
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a 2D NumPy array.")

            if embeddings.shape[0] != len(chunks):
                raise ValueError(
                    f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
                )

            if embeddings.shape[0] == 0:
                self._clear_internal()
                return

            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

            normalized = np.asarray(embeddings, dtype=np.float32)
            self.index.add(normalized)
            self.metadata = list(chunks)

            logger.info(
                "Built FAISS index: total_vectors=%d, dimension=%d",
                self.index.ntotal,
                self.dimension,
            )

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Incrementally add new embeddings and chunks to the existing index."""
        with self._lock:
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a 2D NumPy array.")

            if embeddings.shape[0] != len(chunks):
                raise ValueError(
                    f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
                )

            if embeddings.shape[0] == 0:
                return

            if self.index is None or self.dimension == 0:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []

            if embeddings.shape[1] != self.dimension:
                raise ValueError(
                    f"Dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}."
                )

            normalized = np.asarray(embeddings, dtype=np.float32)
            self.index.add(normalized)
            self.metadata.extend(chunks)

            logger.info(
                "Added %d vectors to FAISS index. Total: %d",
                embeddings.shape[0],
                self.index.ntotal,
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute similarity search against indexed document vectors.
        Returns top-K matching chunks with similarity scores.
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0 or top_k <= 0:
                return []

            query = np.asarray(query_embedding, dtype=np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)

            if query.shape[1] != self.dimension:
                raise ValueError(
                    f"Query dimension mismatch. Expected {self.dimension}, got {query.shape[1]}."
                )

            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query, k)

            results: List[Dict[str, Any]] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0 or index >= len(self.metadata):
                    continue

                meta = self.metadata[index]
                # Cosine similarity score (clamped [-1.0, 1.0])
                sim_score = max(-1.0, min(1.0, float(score)))

                results.append(
                    {
                        "chunk_id": meta.get("chunk_id", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "text": meta.get("text", ""),
                        "source": meta.get("source", "unknown"),
                        "location": meta.get("location", "Document"),
                        "page": meta.get("page"),
                        "type": meta.get("type", "unknown"),
                        "score": round(sim_score, 4),
                    }
                )

            return results

    def save(self, directory: Optional[Path] = None) -> None:
        """Persist FAISS index and metadata JSON to disk."""
        with self._lock:
            target = directory or self.storage_dir
            target.mkdir(parents=True, exist_ok=True)

            index_path = target / "index.faiss"
            meta_path = target / "metadata.json"

            if self.index is not None and self.index.ntotal > 0:
                faiss.write_index(self.index, str(index_path))
            elif index_path.exists():
                index_path.unlink()

            meta_path.write_text(
                json.dumps(self.metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Vector store persisted to %s", target)

    def load(self, directory: Optional[Path] = None) -> bool:
        """Load persisted FAISS index and metadata from disk."""
        with self._lock:
            target = directory or self.storage_dir
            index_path = target / "index.faiss"
            meta_path = target / "metadata.json"

            if not index_path.exists() or not meta_path.exists():
                return False

            try:
                index = faiss.read_index(str(index_path))
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))

                if not isinstance(metadata, list):
                    raise ValueError("metadata.json must contain a JSON list.")

                if index.ntotal != len(metadata):
                    raise ValueError(
                        f"Index vector count ({index.ntotal}) differs from metadata entries ({len(metadata)})."
                    )

                self.index = index
                self.metadata = metadata
                self.dimension = index.d
                logger.info(
                    "Loaded FAISS vector store: %d vectors, dim=%d from %s",
                    index.ntotal,
                    index.d,
                    target,
                )
                return True
            except Exception as exc:
                logger.error("Failed to load vector store from %s: %s", target, exc)
                self._clear_internal()
                return False

    def clear(self) -> None:
        """Reset in-memory state."""
        with self._lock:
            self._clear_internal()

    def reset_storage(self, directory: Optional[Path] = None) -> None:
        """Clear in-memory state and delete disk persistence files."""
        with self._lock:
            self._clear_internal()
            target = directory or self.storage_dir
            for fname in ("index.faiss", "metadata.json"):
                f = target / fname
                if f.exists():
                    f.unlink()
            logger.info("Vector store storage reset at %s", target)

    def _clear_internal(self) -> None:
        self.index = None
        self.metadata = []
        self.dimension = 0

    def count(self) -> int:
        """Return the number of indexed vectors."""
        with self._lock:
            return int(self.index.ntotal) if self.index is not None else 0


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Singleton getter for the global vector store."""
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store


def set_store(store: Optional[VectorStore]) -> None:
    """Set or mock the global vector store (useful for testing)."""
    global _store
    _store = store