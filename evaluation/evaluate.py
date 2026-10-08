"""
DocuMind RAG Evaluation Script.
Measures empirical retrieval success, grounding fallback correctness,
precision, and query latency on a standardized evaluation dataset.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings.embedder import embed_documents
from backend.ingestion.chunker import build_chunks
from backend.retrieval.retriever import retrieve
from backend.retrieval.vector_store import VectorStore, set_store

logging.basicConfig(level=logging.WARNING)

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"

SAMPLE_DOC_TEXT = """
DocuMind Architecture Overview:
DocuMind is an open-source, documentation-grounded AI assistant.
It uses FAISS (Facebook AI Similarity Search) with IndexFlatIP for dense vector retrieval.
Embeddings are computed using local sentence-transformers models such as BAAI/bge-small-en-v1.5.

Supported File Formats:
The system accepts PDF, DOC, DOCX, PPT, PPTX, and TXT files for automated text ingestion.
Files are cleaned, split into chunks of 800 characters with 120 character overlap, and indexed.

Hallucination Control & Grounding:
The default similarity threshold is 0.35. Any query whose top matching chunk scores below 0.35
is immediately rejected by the grounding gate without triggering the LLM.
When rejected, the system returns: "I couldn't find the answer in the provided documentation."
"""


def setup_benchmark_index() -> VectorStore:
    """Prepare an isolated in-memory vector store with sample documentation."""
    store = VectorStore()
    units = [
        {
            "text": SAMPLE_DOC_TEXT,
            "source": "documind_architecture.txt",
            "location": "Page 1",
            "page": 1,
            "type": "txt",
        }
    ]
    chunks = build_chunks(units)
    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)
    store.add(embeddings, chunks)
    set_store(store)
    return store


def run_evaluation() -> Dict[str, Any]:
    """Execute the evaluation benchmark suite."""
    setup_benchmark_index()

    questions: List[Dict[str, Any]] = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    total = len(questions)
    retrieval_correct = 0
    fallback_correct = 0
    total_latency_ms = 0.0

    print("=" * 70)
    print("      🚀 DOCUMIND RAG EVALUATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"Total benchmark questions: {total}\n")

    results_table = []

    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        exp_relevant = item["expected_relevant"]
        exp_fallback = item["expected_fallback"]

        t_start = time.perf_counter()
        ret_result = retrieve(q_text)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_latency_ms += t_elapsed_ms

        actual_relevant = ret_result["relevant"]
        actual_fallback = not actual_relevant or not ret_result["answerable"]

        is_ret_match = actual_relevant == exp_relevant
        is_fall_match = actual_fallback == exp_fallback

        if is_ret_match:
            retrieval_correct += 1
        if is_fall_match:
            fallback_correct += 1

        top_score = ret_result["results"][0]["score"] if ret_result["results"] else 0.0

        status_mark = "✅ PASS" if (is_ret_match and is_fall_match) else "❌ FAIL"
        print(f"[{q_id}] {status_mark} | Latency: {t_elapsed_ms:5.1f}ms | Top Score: {top_score:.3f}")
        print(f"     Query: '{q_text[:55]}...'")

        results_table.append(
            {
                "id": q_id,
                "latency_ms": round(t_elapsed_ms, 2),
                "top_score": round(top_score, 4),
                "retrieval_passed": is_ret_match,
                "fallback_passed": is_fall_match,
            }
        )

    retrieval_acc = (retrieval_correct / total) * 100.0
    fallback_acc = (fallback_correct / total) * 100.0
    avg_latency = total_latency_ms / total

    print("\n" + "=" * 70)
    print("               📊 EMPIRICAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  • Total Evaluated Queries:       {total}")
    print(f"  • Retrieval Classification Acc:  {retrieval_acc:.1f}% ({retrieval_correct}/{total})")
    print(f"  • Grounding Fallback Accuracy:   {fallback_acc:.1f}% ({fallback_correct}/{total})")
    print(f"  • Average Retrieval Latency:     {avg_latency:.2f} ms")
    print("=" * 70)

    return {
        "total_queries": total,
        "retrieval_accuracy_percent": round(retrieval_acc, 2),
        "fallback_accuracy_percent": round(fallback_acc, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "details": results_table,
    }


if __name__ == "__main__":
    run_evaluation()
