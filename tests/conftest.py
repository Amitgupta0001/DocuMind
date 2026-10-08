"""
Pytest configuration and shared test fixtures for DocuMind.
Provides isolated vector stores, mocked embedding models, and mocked LLM generators.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.chatbot.generator import ChatbotGenerator, set_generator
from backend.config import FALLBACK_ANSWER
from backend.embeddings.embedder import set_model
from backend.main import app
from backend.retrieval.vector_store import VectorStore, set_store


class DummyEmbeddingModel:
    """
    Fast, deterministic bag-of-words feature hashing embedding model for testing.
    Shared words yield high cosine similarity; disjoint words yield near-zero similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = []
        for s in sentences:
            vec = np.zeros(self.dimension, dtype=np.float32)
            # Word-level hashing
            words = [w.strip(".,!?:;\"'()[]{}") for w in s.lower().split()]
            for word in words:
                if word:
                    # Deterministic hash bucket
                    bucket = (hash(word) ^ 0x12345678) % self.dimension
                    vec[bucket] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


class DummyLLMClient:
    """Mock OpenAI client responding with predictable answers."""

    def __init__(self, canned_answer: str = "This is a verified test answer based on documentation."):
        self.canned_answer = canned_answer
        self.chat = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = self.canned_answer
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.chat.completions.create.return_value = mock_response


@pytest.fixture(scope="session", autouse=True)
def setup_mock_embedder():
    """Ensure tests run lightning-fast without waiting on PyTorch weights."""
    mock_model = DummyEmbeddingModel(dimension=384)
    set_model(mock_model)
    yield
    set_model(None)


@pytest.fixture
def mock_llm_generator():
    """Fixture providing a mock generator with controllable responses."""
    dummy_client = DummyLLMClient()
    gen = ChatbotGenerator(client=dummy_client, model="mock-model")
    set_generator(gen)
    yield gen
    set_generator(None)


@pytest.fixture
def isolated_vector_store():
    """Fixture providing a temporary, completely isolated vector store."""
    temp_dir = Path(tempfile.mkdtemp())
    store = VectorStore(storage_dir=temp_dir)
    set_store(store)
    yield store
    set_store(None)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client(isolated_vector_store, mock_llm_generator):
    """FastAPI TestClient with isolated vector store and mocked LLM."""
    return TestClient(app)
