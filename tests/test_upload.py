"""
Tests for document upload and ingestion endpoint.
"""

import io
from fastapi.testclient import TestClient


def test_upload_valid_text_file(client: TestClient):
    file_content = b"DocuMind is an open-source documentation chatbot.\nIt uses FAISS for vector search."
    files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "test_doc.txt"
    assert data["chunks"] >= 1
    assert data["total_vectors"] >= 1


def test_upload_unsupported_extension(client: TestClient):
    files = {"file": ("malicious.exe", io.BytesIO(b"executable content"), "application/octet-stream")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file(client: TestClient):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_path_traversal_sanitization(client: TestClient):
    file_content = b"Safe content inside path traversal test."
    files = {"file": ("../../etc/passwd.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    # Ensure directory path components were stripped
    assert "/" not in data["filename"]
    assert ".." not in data["filename"]
    assert "passwd.txt" in data["filename"]
