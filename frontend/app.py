"""
DocuMind Streamlit Frontend.
Production-ready interactive web application for document ingestion,
semantic search inspection, and documentation-grounded AI chat.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="DocuMind - Grounded AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4f46e5;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .score-badge {
        background-color: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR: DOCUMENT MANAGEMENT & SYSTEM METRICS
# =========================================================

with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Production RAG Documentation Engine")
    st.divider()

    # Backend Connection & Health Monitor
    st.subheader("System Status")
    try:
        health_res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_res.ok:
            health_data = health_res.json()
            st.success("🟢 Backend Connected")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Indexed Vectors", health_data.get("vector_count", 0))
            with col2:
                provider = health_data.get("llm_provider", "openrouter").upper()
                st.metric("LLM Provider", provider)
            st.caption(f"Embedding: `{health_data.get('embedding_model', 'bge-small')}`")
        else:
            st.warning("🟡 Backend Degraded")
    except Exception:
        st.error("🔴 Backend Offline")
        st.info(f"Connecting to: `{BACKEND_URL}`\nPlease ensure the FastAPI server is running.")

    st.divider()

    # Document Upload Section
    st.subheader("📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload documentation file",
        type=["pdf", "doc", "docx", "ppt", "pptx", "txt"],
        help="Supported formats: PDF, DOC, DOCX, PPT, PPTX, TXT (Max 25MB)",
    )

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

        if file_size_mb > 25.0:
            st.error("File exceeds 25MB upload limit.")
        else:
            if st.button("🚀 Upload & Index", use_container_width=True, type="primary"):
                with st.spinner("Parsing text, generating chunks, and computing embeddings..."):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        }
                        response = requests.post(
                            f"{BACKEND_URL}/upload",
                            files=files,
                            timeout=180,
                        )

                        if response.ok:
                            data = response.json()
                            st.success("✅ Ingestion & Indexing Complete!")
                            st.write(f"- **Filename:** `{data.get('filename')}`")
                            st.write(f"- **Chunks Generated:** `{data.get('chunks')}`")
                            st.write(f"- **Total Database Vectors:** `{data.get('total_vectors')}`")
                            st.rerun()
                        else:
                            try:
                                err_detail = response.json().get("detail", response.text)
                            except Exception:
                                err_detail = response.text
                            st.error(f"Upload failed: {err_detail}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to FastAPI backend.")
                    except requests.exceptions.Timeout:
                        st.error("Upload request timed out.")
                    except Exception as exc:
                        st.error(f"Unexpected upload error: {exc}")

    st.divider()

    # Maintenance Controls
    st.subheader("🛠️ Index Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🗑️ Reset DB", use_container_width=True, help="Deletes all indexed vectors"):
            try:
                clear_res = requests.post(f"{BACKEND_URL}/clear", timeout=5)
                if clear_res.ok:
                    st.success("Database cleared!")
                    st.rerun()
                else:
                    st.error("Failed to reset database.")
            except Exception as exc:
                st.error(f"Error resetting database: {exc}")


# =========================================================
# MAIN CHAT AREA
# =========================================================

st.markdown('<div class="main-header">📚 DocuMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Documentation-Grounded AI Assistant • Zero Assumptions • Verified Citations</div>',
    unsafe_allow_html=True,
)

# Onboarding Empty State
if not st.session_state.messages:
    with st.container():
        st.info(
            "👋 **Welcome to DocuMind!**\n\n"
            "This chatbot answers questions **strictly** based on documentation you upload into the system.\n\n"
            "**Getting Started:**\n"
            "1. Upload your product specs, whitepapers, manuals, or notes using the left sidebar.\n"
            "2. Click **'Upload & Index'** to process text and generate FAISS embeddings.\n"
            "3. Ask any technical question below to receive grounded answers with exact citations and page references."
        )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📎 Verified Citations ({len(sources)} sources)"):
                for src in sources:
                    doc_name = src.get("source", "Document")
                    loc = src.get("location", "Document")
                    page = src.get("page")
                    score = src.get("score")

                    page_str = f" • Page/Slide {page}" if page is not None else ""
                    score_str = f" • Match: {score*100:.1f}%" if score is not None else ""

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>📄 {doc_name}</strong><br>
                            <small>{loc}{page_str}{score_str}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat Input & Response Flow
user_query = st.chat_input("Ask a question about the uploaded documentation...")

if user_query:
    clean_query = user_query.strip()
    if clean_query:
        # 1. Append & render user message
        st.session_state.messages.append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(clean_query)

        # 2. Query backend
        with st.chat_message("assistant"):
            with st.spinner("Scanning vectorized documentation..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={"question": clean_query},
                        timeout=90,
                    )

                    if resp.ok:
                        data = resp.json()
                        answer_text = data.get("answer", "No response returned.")
                        citations = data.get("sources", [])

                        st.markdown(answer_text)

                        if citations:
                            with st.expander(f"📎 Verified Citations ({len(citations)} sources)"):
                                for src in citations:
                                    doc_name = src.get("source", "Document")
                                    loc = src.get("location", "Document")
                                    page = src.get("page")
                                    score = src.get("score")

                                    page_str = f" • Page/Slide {page}" if page is not None else ""
                                    score_str = f" • Match: {score*100:.1f}%" if score is not None else ""

                                    st.markdown(
                                        f"""
                                        <div class="source-card">
                                            <strong>📄 {doc_name}</strong><br>
                                            <small>{loc}{page_str}{score_str}</small>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "sources": citations,
                            }
                        )

                    else:
                        try:
                            err_msg = resp.json().get("detail", resp.text)
                        except Exception:
                            err_msg = resp.text
                        st.error(f"Backend error: {err_msg}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"⚠️ An error occurred while retrieving answers: {err_msg}",
                                "sources": [],
                            }
                        )

                except requests.exceptions.ConnectionError:
                    err_text = "❌ Cannot connect to the FastAPI backend. Please ensure the backend server is running at " + BACKEND_URL
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text, "sources": []})
                except requests.exceptions.Timeout:
                    err_text = "⏱️ The request timed out. Please try asking again."
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text, "sources": []})
                except Exception as exc:
                    err_text = f"⚠️ Unexpected error: {exc}"
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text, "sources": []})