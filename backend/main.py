"""
DocuMind FastAPI Application.
Production-ready REST API for documentation-grounded AI question answering,
document ingestion, vector store management, and semantic retrieval.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.chatbot.generator import generate_answer
from backend.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    FALLBACK_ANSWER,
    LLM_PROVIDER,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
)
from backend.embeddings.embedder import embed_documents
from backend.ingestion.chunker import build_chunks
from backend.ingestion.document_loader import load_document
from backend.retrieval.retriever import retrieve
from backend.retrieval.vector_store import get_store

# =========================================================
# LOGGING SETUP
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("documind")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="DocuMind API",
    description="Documentation-Grounded AI Assistant & Semantic RAG Engine",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query to evaluate against uploaded documentation.",
        json_schema_extra={"example": "How do I configure the authentication middleware?"},
    )


class SourceItem(BaseModel):
    source: str
    location: Optional[str] = "Document"
    page: Optional[int] = None
    type: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool = True
    answer: str
    sources: List[SourceItem] = []
    relevant: bool = True


class UploadResponse(BaseModel):
    success: bool = True
    filename: str
    extracted_units: int
    chunks: int
    total_vectors: int
    status: str


class RetrieveResponse(BaseModel):
    success: bool = True
    relevant: bool
    answerable: bool
    results: List[Dict[str, Any]]
    sources: List[SourceItem]


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    embedding_model: str
    llm_provider: str
    supported_extensions: List[str]


class ResetResponse(BaseModel):
    success: bool
    message: str
    total_vectors: int


# =========================================================
# UTILITIES
# =========================================================


def sanitize_filename(filename: str) -> str:
    """
    Prevent directory traversal and invalid characters in uploaded filenames.
    """
    clean_name = Path(filename).name
    # Strip null bytes and non-printable chars
    clean_name = re.sub(r"[^\w\s\.-]", "_", clean_name).strip()
    if not clean_name:
        clean_name = "uploaded_document"
    return clean_name


# =========================================================
# ENDPOINTS
# =========================================================


@app.get(
    "/",
    tags=["General"],
    summary="Root Information",
)
def root():
    return {
        "app": "DocuMind",
        "description": "Documentation-grounded AI chatbot",
        "version": "1.1.0",
        "status": "running",
        "llm_provider": LLM_PROVIDER,
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "retrieve": "/retrieve",
            "chat": "/chat",
            "clear": "/clear",
            "docs": "/docs",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health and Vector Count",
)
def health():
    try:
        store = get_store()
        return HealthResponse(
            status="ok",
            vector_count=store.count(),
            embedding_model=EMBEDDING_MODEL_NAME,
            llm_provider=LLM_PROVIDER,
            supported_extensions=sorted(list(SUPPORTED_EXTENSIONS)),
        )
    except Exception as exc:
        logger.exception("Health check failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed.",
        )


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Ingestion"],
    summary="Upload and Index Documentation",
)
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload payload.",
        )

    safe_name = sanitize_filename(file.filename)
    extension = Path(safe_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{extension or 'unknown'}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Validate file size
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    content = bytearray()
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum upload size limit of {MAX_FILE_SIZE_MB}MB.",
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error while reading file upload.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # Save to disk
    save_path = DOCUMENTS_DIR / safe_name
    try:
        save_path.write_bytes(content)
        logger.info("Saved upload to %s (%d bytes)", save_path, len(content))
    except Exception as exc:
        logger.exception("Failed to write document to disk.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded document.",
        )

    # Ingestion Pipeline
    try:
        logger.info("Extracting document units from: %s", safe_name)
        units = load_document(save_path)
        if not units:
            raise ValueError("No readable text found in document.")

        chunks = build_chunks(units)
        if not chunks:
            raise ValueError("Could not extract any chunks from document.")

        texts = [c["text"] for c in chunks]
        embeddings = embed_documents(texts)

        store = get_store()
        store.add(embeddings, chunks)
        store.save()

        logger.info("Successfully indexed %s (%d chunks, %d total vectors)", safe_name, len(chunks), store.count())

        return UploadResponse(
            success=True,
            filename=safe_name,
            extracted_units=len(units),
            chunks=len(chunks),
            total_vectors=store.count(),
            status="indexed",
        )

    except ValueError as val_err:
        logger.warning("Ingestion validation error on %s: %s", safe_name, val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Document ingestion failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed due to internal error.",
        )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Retrieval"],
    summary="Semantic Retrieval & Similarity Search",
)
def retrieve_endpoint(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        result = retrieve(question)
        sources = [
            SourceItem(
                source=s.get("source", "unknown"),
                location=s.get("location"),
                page=s.get("page"),
                type=s.get("type"),
                score=s.get("score"),
            )
            for s in result.get("deduplicated_sources", [])
        ]

        return RetrieveResponse(
            success=True,
            relevant=result["relevant"],
            answerable=result["answerable"],
            results=result["results"],
            sources=sources,
        )
    except Exception as exc:
        logger.exception("Retrieval endpoint error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval query failed.",
        )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Generate Grounded Answer with Citations",
)
def chat(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    logger.info("Chat query received: '%s'", question[:60])

    # 1. Retrieval
    try:
        result = retrieve(question)
    except Exception as exc:
        logger.exception("Retrieval failure during chat.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documentation.",
        )

    sources = [
        SourceItem(
            source=s.get("source", "unknown"),
            location=s.get("location"),
            page=s.get("page"),
            type=s.get("type"),
            score=s.get("score"),
        )
        for s in result.get("deduplicated_sources", [])
    ]

    # 2. Grounding Gate
    if not result.get("relevant") or not result.get("answerable"):
        logger.info("Query '%s' rejected by grounding gate. Returning fallback.", question[:50])
        return ChatResponse(
            success=True,
            answer=FALLBACK_ANSWER,
            sources=[],
            relevant=False,
        )

    # 3. LLM Generation
    try:
        answer = generate_answer(question, result["results"])
    except Exception as exc:
        logger.exception("LLM generation error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from documentation.",
        )

    return ChatResponse(
        success=True,
        answer=answer,
        sources=sources,
        relevant=True,
    )


@app.post(
    "/clear",
    response_model=ResetResponse,
    tags=["System"],
    summary="Reset Vector Store and Remove Persisted Embeddings",
)
def clear_vectorstore():
    try:
        store = get_store()
        store.reset_storage()
        logger.info("Vector store index and metadata successfully cleared.")
        return ResetResponse(
            success=True,
            message="Vector store successfully cleared.",
            total_vectors=store.count(),
        )
    except Exception as exc:
        logger.exception("Failed to reset vector store.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset vector store.",
        )