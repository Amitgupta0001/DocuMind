"""
DocuMind Configuration Module.
Loads environment variables and validates system-wide RAG parameters.
"""

import os
from pathlib import Path
from typing import Set

from dotenv import load_dotenv


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LLM CONFIGURATION (OpenRouter & OpenAI)
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "liquid/lfm-2.5-2.6b:free",
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
).strip()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
if not LLM_PROVIDER:
    if OPENROUTER_API_KEY:
        LLM_PROVIDER = "openrouter"
    elif OPENAI_API_KEY:
        LLM_PROVIDER = "openai"
    else:
        LLM_PROVIDER = "openrouter"

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
).strip()


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
MIN_SIMILARITY_GAP = float(os.getenv("MIN_SIMILARITY_GAP", "0.0"))


# ============================================================
# FALLBACK ANSWER
# ============================================================

FALLBACK_ANSWER = os.getenv(
    "FALLBACK_ANSWER",
    "I couldn't find the answer in the provided documentation.",
).strip()


# ============================================================
# SUPPORTED DOCUMENT TYPES
# ============================================================

SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
}


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than 0.")

if CHUNK_OVERLAP < 0:
    raise ValueError("CHUNK_OVERLAP cannot be negative.")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than 0.")

if MAX_CONTEXT_CHUNKS <= 0:
    raise ValueError("MAX_CONTEXT_CHUNKS must be greater than 0.")

if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
    raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0.")

if not 0.0 <= MIN_SIMILARITY_GAP <= 1.0:
    raise ValueError("MIN_SIMILARITY_GAP must be between 0.0 and 1.0.")

if MAX_FILE_SIZE_MB <= 0:
    raise ValueError("MAX_FILE_SIZE_MB must be greater than 0.")

if REQUEST_TIMEOUT_SECONDS <= 0:
    raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than 0.")