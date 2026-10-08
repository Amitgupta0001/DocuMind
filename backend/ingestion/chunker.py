"""
DocuMind Chunker Module.
Cleans raw text and generates overlapping text chunks with rich deterministic metadata.
"""

import hashlib
import re
from typing import Any, Dict, List

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    """
    Sanitize and normalize extracted text:
    - Removes control characters and null bytes
    - Normalizes unicode whitespace and non-breaking spaces
    - Collapses excessive blank lines
    """
    if not text:
        return ""

    # Remove null bytes and non-printable control characters (preserve \n, \r, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize unicode whitespace (non-breaking spaces, zero-width spaces)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # Normalize line breaks to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace runs of horizontal whitespace with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3 or more newlines to double newlines (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Deterministic character-based chunking with sliding window overlap.
    Guarantees non-empty chunks and monotonic window advancement.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        # Advance window
        step = chunk_size - overlap
        start += step

    return chunks


def build_chunks(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw extraction units into chunk dicts with full metadata.
    Each chunk gets a unique deterministic SHA-256 ID.
    """
    chunks: List[Dict[str, Any]] = []

    for unit in units:
        raw_text = unit.get("text", "")
        raw_chunks = chunk_text(raw_text)

        source = unit.get("source", "unknown")
        location = unit.get("location", "Document")
        page = unit.get("page")
        doc_type = unit.get("type", "unknown")

        for idx, chunk_content in enumerate(raw_chunks):
            # Deterministic chunk hash
            hash_input = f"{source}|{location}|{idx}|{chunk_content[:64]}"
            chunk_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "text": chunk_content,
                    "source": source,
                    "location": location,
                    "page": page,
                    "type": doc_type,
                }
            )

    return chunks