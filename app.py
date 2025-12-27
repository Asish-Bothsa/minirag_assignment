import streamlit as st
from rag_pipeline import (
    build_vectorstore,
    retrieve_context,
    generate_answer
)

st.set_page_config(
    page_title="Mini RAG Assignment",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

/* Header card */
.header-box {
    background: linear-gradient(135deg, #374151, #1f2937);
    padding: 1.6rem;
    border-radius: 14px;
    color: #f9fafb;
    margin-bottom: 1.8rem;
    border-left: 6px solid #9ca3af;
}

.header-box h2 {
    margin-bottom: 0.4rem;
    color: #f9fafb;
}

.header-box p {
    margin: 0;
    color: #e5e7eb;
    font-size: 1rem;
}

/* Context chunks */
.context-box {
    background-color: #f9fafb;
    color: #111827;
    padding: 0.9rem;
    border-radius: 10px;
    font-size: 0.95rem;
    line-height: 1.55;
    border: 1px solid #e5e7eb;
}

/* Final answer */
.answer-box {
    background-color: #ffffff;
    color: #111827;
    padding: 1.1rem;
    border-radius: 12px;
    border-left: 5px solid #6b7280;
    font-size: 1.05rem;
    line-height: 1.6;
    border: 1px solid #e5e7eb;
}

/* Sidebar note */
.sidebar-note {
    font-size: 0.85rem;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="header-box">
    <h2>📄 Mini RAG Chatbot</h2>
    <p>
    Upload <b>PDF</b> or <b>Markdown</b> documents and ask questions.
    Answers are generated <b>strictly from retrieved document content</b>.
    </p>
</div>
""", unsafe_allow_html=True)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


st.sidebar.header("📁 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / Markdown files",
    type=["pdf", "md"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")

top_k = st.sidebar.slider(
    "🔍 Top-K Retrieved Chunks",
    min_value=1,
    max_value=8,
    value=4
)

st.sidebar.markdown(
    "<p class='sidebar-note'>Higher K improves recall but may introduce noise.</p>",
    unsafe_allow_html=True
)


if uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = build_vectorstore(uploaded_files)
    st.sidebar.success("Documents indexed successfully.")

vectorstore = st.session_state.vectorstore


# QUERY INPUT

query = st.text_input(
    "Ask a question based on the uploaded documents",
    placeholder="e.g. What factors cause project delays?"
)


# RESPONSE PIPELINE
if query and vectorstore:
    retrieved_docs = retrieve_context(query, vectorstore, k=top_k)

    # Retrieved Context
    st.subheader("🔍 Retrieved Context")

    for i, doc in enumerate(retrieved_docs):
        with st.expander(f"📌 Chunk {i+1}", expanded=False):
            st.markdown(
                f"<div class='context-box'>{doc.page_content}</div>",
                unsafe_allow_html=True
            )
            st.caption(
                f"📄 Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Page: {doc.metadata.get('page', 'N/A')}"
            )

    # Generate Answer
    with st.spinner("Generating grounded answer..."):
        answer = generate_answer(query, retrieved_docs)

    # Final Answer
    st.subheader("✅ Final Answer")

    if "don't have enough information" in answer.lower():
        st.warning("The answer could not be found in the uploaded documents.")
    else:
        st.markdown(
            f"<div class='answer-box'>{answer}</div>",
            unsafe_allow_html=True
        )

elif query and not vectorstore:
    st.info("Please upload documents to begin.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "🔐 Documents are processed only within the runtime environment. "
    "No documents are stored permanently.your privacy is ensured."
)
