import streamlit as st
from rag_chain import load_pdf, split_document, create_vectore_store, build_qa_chain
import tempfile
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="🏦",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("🏦 Financial Document Q&A")
st.markdown("*Powered by RAG · LangChain · Ollama · ChromaDB*")
st.divider()

# ── Session state ─────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── PDF Upload ───────────────────────────────────────────────
st.subheader("📄 Upload Financial Document")
uploaded_file = st.file_uploader(
    "Upload a PDF (annual report, prospectus, financial statement)",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        with st.spinner("🔄 Processing document... this takes 1-2 mins"):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Build RAG pipeline
            pages = load_pdf(tmp_path)
            chunks = split_document(pages)
            vector_store = create_vectore_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store)

            # Cleanup temp file
            os.unlink(tmp_path)

        st.success(f"✅ Document ready — {len(pages)} pages processed")

# ── Q&A Section ───────────────────────────────────────────────
if st.session_state.qa_chain is not None:
    st.divider()
    st.subheader("❓ Ask a Question")

    question = st.text_input(
        "Type your question about the document",
        placeholder="What were the total net sales? What risks are mentioned?"
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

        # Store in chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

# ── Chat History ──────────────────────────────────────────────
if st.session_state.chat_history:
    st.divider()
    st.subheader("💬 Results")

    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**❓ {item['question']}**")
        st.success(item["answer"])

        with st.expander("📚 View Sources"):
            for i, doc in enumerate(item["sources"]):
                st.markdown(f"**Source {i + 1}** — Page {doc.metadata.get('page', 'N/A')}")
                st.caption(doc.page_content[:300] + "...")

        st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **RAG (Retrieval Augmented Generation)** to answer 
    questions about financial documents.

    **How it works:**
    1. Upload a PDF document
    2. Document is chunked and embedded
    3. Your question finds relevant chunks
    4. Mistral LLM answers from those chunks

    **Tech Stack**
    - 🐍 Python 3.11
    - 🦜 LangChain
    - 🗄️ ChromaDB
    - 🤖 Ollama · Mistral
    - 🎨 Streamlit
    """)

    if st.session_state.qa_chain is not None:
        st.divider()
        if st.button("🗑️ Clear & Upload New Document"):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.rerun()