# 📊 DocuMind RAG Evaluation Suite

This evaluation suite provides a deterministic, repeatable benchmark for measuring DocuMind's retrieval accuracy, similarity filtering gates, grounding fallback precision, and query latencies.

---

## 📌 Benchmark Metrics

1. **Retrieval Classification Accuracy**: Measures whether relevant in-domain questions successfully retrieve matching documentation excerpts above the similarity threshold ($0.35$).
2. **Grounding Fallback Accuracy**: Measures whether out-of-domain queries and prompt injection attempts correctly trigger the grounding gate fallback ($100\%$ rejection of irrelevant context).
3. **Query Latency (ms)**: Measures dense vector encoding and FAISS inner-product retrieval latency per query.

---

## 📂 Evaluation Dataset Structure (`questions.json`)

Each entry in `questions.json` specifies:

```json
{
  "id": "q1",
  "question": "What is DocuMind and what vector database does it use?",
  "expected_relevant": true,
  "expected_fallback": false,
  "target_document": "documind_architecture.txt",
  "expected_keywords": ["DocuMind", "FAISS", "documentation"]
}
```

---

## 🚀 Running the Evaluation Benchmark

Run the automated evaluation runner from the project root:

```bash
python evaluation/evaluate.py
```

The script runs without external API dependencies, measuring real retrieval performance against the local embedding model and FAISS vector index.
