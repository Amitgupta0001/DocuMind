"""
Tests for API root and health endpoints.
"""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "DocuMind"
    assert data["status"] == "running"
    assert "endpoints" in data


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["vector_count"], int)
    assert "embedding_model" in data
    assert "supported_extensions" in data
    assert ".pdf" in data["supported_extensions"]
