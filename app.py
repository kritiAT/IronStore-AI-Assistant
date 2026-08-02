"""
IronStore AI Assistant - Streamlit chat UI
------------------------------------------------
Run with:  streamlit run app.py

Requirements:
    pip install streamlit openai pinecone-client python-dotenv

.env file needed (same folder):
    OPENAI_API_KEY=sk-...
    PINECONE_API_KEY=pcsk-...

Assumes the Pinecone index "ironstore-enterprise-knowledge-base" has already been populated by
running 01_ingest_and_index.ipynb. This file duplicates the retrieval + generation logic
from 02_retrieve_and_answer.ipynb so it can run standalone as a Streamlit script.
"""

import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---------------- Configuration ----------------
PINECONE_INDEX_NAME = "ironstore-enterprise-knowledge-base"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

TOP_K = 6
SCORE_THRESHOLD = 0.72  # below this top-match score, we don't trust retrieval enough to answer

SYSTEM_PROMPT = """You are an internal enterprise assistant. Answer employee questions \
using ONLY the provided context excerpts from internal company documents.

Rules:
1. Base your answer strictly on the given context. Do not use outside knowledge or make assumptions.
2. Always cite which excerpt(s) you used, like this: (Source [1], [3]).
3. If the context does not contain enough information to answer confidently, say clearly: \
"I don't have enough information in the internal documents to answer this confidently," \
and suggest which department or document the employee might check with instead.
4. Be concise, clear, and professional. Explain procedures step by step when relevant.
5. Never fabricate policies, numbers, or procedures that are not present in the context.
"""

st.set_page_config(page_title="IronStore AI Assistant", page_icon="🧭", layout="wide")


# ---------------- Clients (cached) ----------------
@st.cache_resource
def init_clients():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX_NAME)
    return openai_client, index


openai_client, index = init_clients()


@st.cache_data(ttl=300)
def get_departments():
    stats = index.describe_index_stats()
    return sorted(stats.get("namespaces", {}).keys())


# ---------------- RAG pipeline ----------------
def embed_query(query: str):
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(query: str, top_k: int = TOP_K, department: str | None = None):
    query_vector = embed_query(query)
    namespaces = [department] if department else get_departments()

    all_matches = []
    for ns in namespaces:
        result = index.query(vector=query_vector, top_k=top_k, namespace=ns, include_metadata=True)
        all_matches.extend(result["matches"])

    all_matches.sort(key=lambda m: m["score"], reverse=True)
    return all_matches[:top_k]


def format_context(matches):
    context_blocks = []
    sources = []
    for i, m in enumerate(matches, start=1):
        meta = m["metadata"]
        context_blocks.append(
            f"[{i}] Department: {meta['department']} | Document: {meta['document_name']} "
            f"| Section: {meta['section_title']} | Pages: {meta['start_page']}-{meta['end_page']}\n"
            f"{meta['text']}"
        )
        sources.append({
            "ref": i,
            "department": meta["department"],
            "document_name": meta["document_name"],
            "section_title": meta["section_title"],
            "start_page": meta["start_page"],
            "end_page": meta["end_page"],
            "score": round(m["score"], 3),
        })
    return "\n\n---\n\n".join(context_blocks), sources


def generate_answer(query: str, department: str | None = None, top_k: int = TOP_K):
    matches = retrieve(query, top_k=top_k, department=department)

    # if not matches or matches[0]["score"] < SCORE_THRESHOLD:
    #     return {
    #         "answer": (
    #             "I couldn't find sufficiently relevant information in the internal documents "
    #             "to answer this confidently. You may want to check with the relevant department "
    #             "directly or try rephrasing your question."
    #         ),
    #         "sources": [],
    #         "confident": False,
    #     }

    context, sources = format_context(matches)
    user_prompt = f"Context excerpts:\n\n{context}\n\nEmployee question: {query}"

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "confident": True,
    }


def render_sources(sources):
    with st.expander("📎 Sources"):
        for s in sources:
            st.markdown(
                f"**[{s['ref']}] {s['department']} / {s['document_name']}** — {s['section_title']} · "
                f"{s['department']} · pages {s['start_page']}-{s['end_page']} "
                f"(relevance {s['score']})"
            )


# ---------------- UI ----------------
st.title("🧭 IronStore AI Assistant")
st.caption("Ask about internal policies, procedures, and department documentation.")

with st.sidebar:
    st.header("Filters")
    department_options = ["All departments"] + get_departments()
    selected_department = st.selectbox("Department", department_options)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "This assistant only answers from indexed internal documents and will tell you "
        "when it isn't confident enough to answer, rather than guessing."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

query = st.chat_input("Ask a question about company policies, procedures, or documentation...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching internal documents..."):
            dep_filter = None if selected_department == "All departments" else selected_department
            result = generate_answer(query, department=dep_filter)
        st.markdown(result["answer"])
        if result["sources"]:
            render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
