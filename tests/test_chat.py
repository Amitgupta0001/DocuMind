"""
End-to-end tests for the /chat API endpoint.
"""

import io
from fastapi.testclient import TestClient
from backend.config import FALLBACK_ANSWER


def test_chat_empty_question(client: TestClient):
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_chat_fallback_when_no_documents_indexed(client: TestClient):
    response = client.post("/chat", json={"question": "How do I deploy with Docker?"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == FALLBACK_ANSWER
    assert data["relevant"] is False
    assert len(data["sources"]) == 0


def test_chat_grounded_answer_after_upload(client: TestClient):
    # 1. Upload documentation
    doc_text = b"DocuMind features a FastAPI backend and a Streamlit UI.\nThe port for FastAPI is 8000."
    files = {"file": ("deployment_guide.txt", io.BytesIO(doc_text), "text/plain")}
    up_res = client.post("/upload", files=files)
    assert up_res.status_code == 200

    # 2. Query chat
    response = client.post("/chat", json={"question": "FastAPI backend and Streamlit UI port"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["relevant"] is True
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["source"] == "deployment_guide.txt"
    assert "verified test answer" in data["answer"].lower()
