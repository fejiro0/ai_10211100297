import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="Academic City RAG Assistant", layout="wide")

NAME = "FAMOUS AKPOVOGBETA"
INDEX_NUMBER = "10211100297"
REPO_NAME = f"ai_{INDEX_NUMBER}"

ELECTION_DATA_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/"
    "Ghana_Election_Result.csv"
)
BUDGET_PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class ChunkRecord:
    source: str
    chunk_id: str
    text: str
    metadata: Dict


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    chunks = []
    start = 0
    cleaned = normalize_text(text)
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


@st.cache_resource(show_spinner=False)
def fetch_election_csv() -> pd.DataFrame:
    response = requests.get(ELECTION_DATA_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Cleaning: trim headers/strings, remove exact duplicates, fill numeric NaN with 0.
    # In handling the missing data set i filled the numeric columns with 0
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


@st.cache_resource(show_spinner=False)
def fetch_budget_text() -> List[Dict]:
    local_path = os.path.join(DATA_DIR, "2025_budget_statement.pdf")
    if not os.path.exists(local_path):
        response = requests.get(BUDGET_PDF_URL, timeout=60)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

    reader = PdfReader(local_path)
    pages = []
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = normalize_text(raw_text)
        if cleaned:
            pages.append({"page": i + 1, "text": cleaned})
    return pages


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=True)
def build_knowledge_base(chunk_size: int, overlap: int) -> Tuple[List[ChunkRecord], np.ndarray]:
    records: List[ChunkRecord] = []

    election_df = fetch_election_csv()
    election_rows_as_text = election_df.astype(str).agg(" | ".join, axis=1).tolist()

    for idx, row_text in enumerate(election_rows_as_text):
        for c_idx, piece in enumerate(chunk_text(row_text, chunk_size=chunk_size, overlap=overlap)):
            records.append(
                ChunkRecord(
                    source="Ghana_Election_Result.csv",
                    chunk_id=f"csv-{idx}-{c_idx}",
                    text=piece,
                    metadata={"row": idx},
                )
            )

    for page in fetch_budget_text():
        for c_idx, piece in enumerate(chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)):
            records.append(
                ChunkRecord(
                    source="2025 Budget Statement PDF",
                    chunk_id=f"pdf-{page['page']}-{c_idx}",
                    text=piece,
                    metadata={"page": page["page"]},
                )
            )

    texts = [r.text for r in records]
    vectors = get_embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return records, np.array(vectors)


def keyword_overlap_score(query: str, doc_text: str) -> float:
    q_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    d_terms = set(re.findall(r"[a-zA-Z]{3,}", doc_text.lower()))
    if not q_terms:
        return 0.0
    return len(q_terms.intersection(d_terms)) / len(q_terms)


def expand_query(query: str) -> str:
    synonyms = {
        "budget": ["appropriation", "expenditure", "fiscal"],
        "election": ["votes", "constituency", "results"],
        "inflation": ["cpi", "price levels"],
        "gdp": ["growth", "economic output"],
    }
    extra = []
    lower = query.lower()
    for key, vals in synonyms.items():
        if key in lower:
            extra.extend(vals)
    return query if not extra else f"{query} {' '.join(extra)}"


def retrieve(
    query: str,
    records: List[ChunkRecord],
    vectors: np.ndarray,
    top_k: int = 5,
    alpha: float = 0.8,
    use_expansion: bool = True,
) -> List[Dict]:
    retrieval_query = expand_query(query) if use_expansion else query
    q_vec = get_embedder().encode([retrieval_query], normalize_embeddings=True)[0]

    # Cosine similarity from normalized vectors.
    vec_scores = vectors @ q_vec

    combined = []
    for i, rec in enumerate(records):
        kw = keyword_overlap_score(query, rec.text)
        score = alpha * float(vec_scores[i]) + (1 - alpha) * kw
        combined.append(
            {
                "record": rec,
                "vector_score": float(vec_scores[i]),
                "keyword_score": kw,
                "score": score,
            }
        )

    ranked = sorted(combined, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def build_prompt(user_query: str, selected_chunks: List[Dict], max_chars: int = 3000) -> str:
    parts = []
    running = 0
    for item in selected_chunks:
        rec = item["record"]
        snippet = f"[{rec.source}::{rec.chunk_id}] {rec.text}"
        if running + len(snippet) > max_chars:
            break
        parts.append(snippet)
        running += len(snippet)

    context_block = "\n\n".join(parts)
    prompt = f"""
You are the Academic City RAG assistant.
Use ONLY the provided context. If the answer is unavailable, reply exactly:
"I do not have enough grounded context to answer this safely."

Context:
{context_block}

User Question:
{user_query}

Answer in 4-7 concise bullet points and cite chunk IDs used
""".strip()
    return prompt


def grounded_answer_stub(prompt: str, retrieved: List[Dict]) -> str:
    if not retrieved:
        return "I do not have enough grounded context to answer this safely."

    bullets = []
    for item in retrieved[:3]:
        rec = item["record"]
        preview = rec.text[:180].strip()
        bullets.append(f"- {preview}... (source: {rec.chunk_id})")
    return "\n".join(bullets)


st.title("Academic City Exam RAG Assistant")
st.caption(f"Name: {NAME} | Index Number: {INDEX_NUMBER} | Repository: {REPO_NAME}")

with st.sidebar:
    st.header("RAG Controls")
    chunk_size = st.slider("Chunk size", min_value=300, max_value=1200, value=800, step=100)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=120, step=20)
    top_k = st.slider("Top-k", min_value=2, max_value=8, value=5)
    alpha = st.slider("Hybrid alpha (vector weight)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    query_expansion = st.checkbox("Enable query expansion", value=True)

records, vectors = build_knowledge_base(chunk_size=chunk_size, overlap=overlap)

st.success(
    f"Knowledge base loaded with {len(records)} chunks from Ghana election CSV and 2025 budget PDF."
)

user_query = st.text_input(
    "Ask a question about the Ghana election result dataset or 2025 budget statement:"
)

if user_query:
    retrieved = retrieve(
        query=user_query,
        records=records,
        vectors=vectors,
        top_k=top_k,
        alpha=alpha,
        use_expansion=query_expansion,
    )

    prompt = build_prompt(user_query=user_query, selected_chunks=retrieved, max_chars=2800)
    answer = grounded_answer_stub(prompt, retrieved)

    st.subheader("Final Response")
    st.markdown(answer)

    st.subheader("Retrieved Chunks + Scores")
    for idx, item in enumerate(retrieved, start=1):
        rec = item["record"]
        with st.expander(f"{idx}. {rec.chunk_id} | score={item['score']:.4f}"):
            st.write(f"Source: {rec.source}")
            st.write(f"Metadata: {rec.metadata}")
            st.write(f"Vector score: {item['vector_score']:.4f}")
            st.write(f"Keyword score: {item['keyword_score']:.4f}")
            st.write(rec.text)

    st.subheader("Final Prompt Sent to LLM")
    st.code(prompt)

st.markdown("---")
st.markdown(
    "### Chunking Design Justification\n"
    "- Chunk size default is **800 chars** with **120 overlap** to preserve paragraph-level coherence while keeping embeddings focused.\n"
    "- Smaller chunks increase precision but can fragment context; larger chunks improve recall but may introduce noise.\n"
    "- Use the sliders to compare retrieval behavior and document this in your experiment logs."
)
