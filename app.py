import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with your Documents", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document." """,
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def build_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    """Load uploaded PDFs, split them, and build an in-memory Chroma store."""
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(file_path=tmp_path)
            all_docs.extend(loader.load())
        finally:
            os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=MistralAIEmbeddings(),
        # No persist_directory: each uploaded session gets a fresh in-memory
        # store. Swap this for persist_directory="ChromaDB" if you want the
        # index to survive a restart.
    )
    return vectorstore, len(chunks)


def answer_query(query, retriever):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": query})
    llm = ChatMistralAI(model="mistral-small-2506")
    response = llm.invoke(final_prompt)
    return response.content


# ---------------------------------------------------------------------------
# Sidebar — Step 1: upload + process documents
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload your documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    chunk_size = 1000
    chunk_overlap = 200

    process_clicked = st.button(
        "Process documents", type="primary", disabled=not uploaded_files
    )

    if process_clicked and uploaded_files:
        with st.spinner("Reading and indexing your documents..."):
            try:
                vectorstore, n_chunks = build_vectorstore(
                    uploaded_files, chunk_size, chunk_overlap
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = n_chunks
                st.session_state.docs_processed = True
                st.session_state.messages = []  # fresh chat for new docs
            except Exception as e:
                st.error(f"Failed to process documents: {e}")

    if st.session_state.docs_processed:
        st.success(
            f"✅ {len(uploaded_files) if uploaded_files else ''} document(s) ready "
            f"({st.session_state.num_chunks} chunks indexed)."
        )
        if st.button("Reset / upload new documents"):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.docs_processed = False
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — Step 2: chat (locked until documents are processed)
# ---------------------------------------------------------------------------
st.title("📄 Chat with your Documents")

if not st.session_state.docs_processed:
    st.info("👈 Upload one or more PDFs and click **Process documents** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5},
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_query(query, retriever)
                except Exception as e:
                    answer = f"Something went wrong while answering: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
