"""
Tests for text cleaner and chunker modules.
"""

import pytest
from backend.ingestion.chunker import build_chunks, chunk_text, clean_text


def test_clean_text():
    dirty = "Hello\xa0World!\r\n\r\n\r\nThis has \t extra    spaces.\x00"
    cleaned = clean_text(dirty)
    assert "\xa0" not in cleaned
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "extra spaces" in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_normal():
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 200


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError, match="positive"):
        chunk_text("test", chunk_size=-10)

    with pytest.raises(ValueError, match="less than chunk_size"):
        chunk_text("test", chunk_size=100, overlap=100)


def test_build_chunks_metadata():
    units = [
        {
            "text": "This is sample paragraph one.\n\nThis is sample paragraph two.",
            "source": "manual.pdf",
            "location": "Page 5",
            "page": 5,
            "type": "pdf",
        }
    ]
    chunks = build_chunks(units)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c["source"] == "manual.pdf"
    assert c["page"] == 5
    assert c["location"] == "Page 5"
    assert c["type"] == "pdf"
    assert "chunk_id" in c
    assert len(c["chunk_id"]) == 16
