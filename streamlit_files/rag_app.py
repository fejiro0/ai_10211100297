import io
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


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


def timestamp_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def append_stage_log(stage: str, payload: Dict) -> None:
    if "stage_logs" not in st.session_state:
        st.session_state.stage_logs = []
    st.session_state.stage_logs.append(
        {
            "timestamp_utc": timestamp_utc(),
            "stage": stage,
            "payload": payload,
        }
    )


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
    source_bias = st.session_state.get("source_feedback_bias", {})
    for i, rec in enumerate(records):
        kw = keyword_overlap_score(query, rec.text)
        bias = float(source_bias.get(rec.source, 0.0))
        score = alpha * float(vec_scores[i]) + (1 - alpha) * kw + bias
        combined.append(
            {
                "record": rec,
                "vector_score": float(vec_scores[i]),
                "keyword_score": kw,
                "feedback_bias": bias,
                "score": score,
            }
        )

    ranked = sorted(combined, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def build_prompt(
    user_query: str,
    selected_chunks: List[Dict],
    max_chars: int = 3000,
    prompt_style: str = "strict",
) -> str:
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
    if prompt_style == "concise":
        prompt = f"""
You are the Academic City RAG assistant.
Use only the context below. Keep answer short and factual.
If unsupported by context, respond:
"I do not have enough grounded context to answer this safely."

Context:
{context_block}

Question: {user_query}
Return 3-5 bullets with chunk IDs.
""".strip()
    elif prompt_style == "analyst":
        prompt = f"""
You are an evidence-focused policy analyst assistant.
Ground every claim in provided context and cite chunk IDs.
If evidence is missing, state:
"I do not have enough grounded context to answer this safely."
Do not speculate.

Context:
{context_block}

User Question:
{user_query}

Output format:
1) Direct answer
2) Evidence bullets with chunk IDs
3) Confidence: High/Medium/Low
""".strip()
    else:
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


def pure_llm_baseline_stub(user_query: str) -> str:
    return (
        "- This is a baseline (no retrieval) placeholder response.\n"
        f"- It answers the question directly: {user_query[:120]}...\n"
        "- Warning: Without retrieval, this mode may hallucinate or miss source-grounded details."
    )


def evaluate_response_consistency(response_a: str, response_b: str) -> float:
    a_terms = set(re.findall(r"[a-zA-Z]{3,}", response_a.lower()))
    b_terms = set(re.findall(r"[a-zA-Z]{3,}", response_b.lower()))
    if not a_terms and not b_terms:
        return 1.0
    union = len(a_terms.union(b_terms))
    return (len(a_terms.intersection(b_terms)) / union) if union else 0.0


def persist_logs_to_disk() -> Optional[str]:
    logs = st.session_state.get("stage_logs", [])
    if not logs:
        return None
    file_name = f"rag_stage_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path = os.path.join(LOGS_DIR, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


st.title("Academic City Exam RAG Assistant")
st.caption(f"Name: {NAME} | Index Number: {INDEX_NUMBER} | Repository: {REPO_NAME}")

with st.sidebar:
    st.header("RAG Controls")
    chunk_size = st.slider("Chunk size", min_value=300, max_value=1200, value=800, step=100)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=120, step=20)
    top_k = st.slider("Top-k", min_value=2, max_value=8, value=5)
    alpha = st.slider("Hybrid alpha (vector weight)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    query_expansion = st.checkbox("Enable query expansion", value=True)
    prompt_style = st.selectbox("Prompt style", options=["strict", "concise", "analyst"], index=0)

    st.header("Experiment & Evaluation")
    show_baseline = st.checkbox("Compare with no-retrieval baseline", value=True)
    run_adversarial = st.checkbox("Show adversarial query panel", value=True)
    if st.button("Clear session logs"):
        st.session_state.stage_logs = []
        st.success("Session logs cleared.")

records, vectors = build_knowledge_base(chunk_size=chunk_size, overlap=overlap)

st.success(
    f"Knowledge base loaded with {len(records)} chunks from Ghana election CSV and 2025 budget PDF."
)
append_stage_log(
    "knowledge_base_loaded",
    {
        "records_count": len(records),
        "chunk_size": chunk_size,
        "overlap": overlap,
    },
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
    append_stage_log(
        "retrieval",
        {
            "query": user_query,
            "top_k": top_k,
            "alpha": alpha,
            "query_expansion": query_expansion,
            "retrieved_ids": [x["record"].chunk_id for x in retrieved],
            "scores": [round(float(x["score"]), 6) for x in retrieved],
        },
    )

    prompt = build_prompt(
        user_query=user_query,
        selected_chunks=retrieved,
        max_chars=2800,
        prompt_style=prompt_style,
    )
    answer = grounded_answer_stub(prompt, retrieved)
    append_stage_log(
        "prompt_and_generation",
        {
            "prompt_style": prompt_style,
            "prompt_length_chars": len(prompt),
            "response_length_chars": len(answer),
        },
    )

    st.subheader("Final Response")
    st.markdown(answer)

    if show_baseline:
        baseline = pure_llm_baseline_stub(user_query)
        consistency = evaluate_response_consistency(answer, baseline)
        st.subheader("RAG vs Pure-LLM (No Retrieval) Baseline")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RAG Response**")
            st.markdown(answer)
        with col2:
            st.markdown("**No-Retrieval Baseline**")
            st.markdown(baseline)
        st.info(f"Response consistency score (Jaccard terms): {consistency:.2f}")
        append_stage_log(
            "baseline_comparison",
            {
                "consistency_score": round(float(consistency), 4),
                "baseline_enabled": True,
            },
        )

    st.subheader("Retrieved Chunks + Scores")
    for idx, item in enumerate(retrieved, start=1):
        rec = item["record"]
        with st.expander(f"{idx}. {rec.chunk_id} | score={item['score']:.4f}"):
            st.write(f"Source: {rec.source}")
            st.write(f"Metadata: {rec.metadata}")
            st.write(f"Vector score: {item['vector_score']:.4f}")
            st.write(f"Keyword score: {item['keyword_score']:.4f}")
            st.write(f"Feedback bias: {item['feedback_bias']:.4f}")
            st.write(rec.text)
            helpful_key = f"helpful_{rec.chunk_id}"
            not_helpful_key = f"not_helpful_{rec.chunk_id}"
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Helpful 👍", key=helpful_key):
                    if "source_feedback_bias" not in st.session_state:
                        st.session_state.source_feedback_bias = {}
                    st.session_state.source_feedback_bias[rec.source] = (
                        st.session_state.source_feedback_bias.get(rec.source, 0.0) + 0.02
                    )
                    st.success(f"Increased retrieval bias for source: {rec.source}")
            with c2:
                if st.button("Not helpful 👎", key=not_helpful_key):
                    if "source_feedback_bias" not in st.session_state:
                        st.session_state.source_feedback_bias = {}
                    st.session_state.source_feedback_bias[rec.source] = (
                        st.session_state.source_feedback_bias.get(rec.source, 0.0) - 0.02
                    )
                    st.warning(f"Decreased retrieval bias for source: {rec.source}")

    st.subheader("Final Prompt Sent to LLM")
    st.code(prompt)
    st.subheader("Stage Logs (Current Session)")
    st.json(st.session_state.get("stage_logs", []))
    if st.button("Persist logs to logs/ as JSONL"):
        saved_path = persist_logs_to_disk()
        if saved_path:
            st.success(f"Saved logs to: {saved_path}")
        else:
            st.warning("No logs available yet.")

if run_adversarial:
    st.markdown("### Adversarial Testing Panel")
    adversarial_queries = [
        "What happened there in that year?",
        "The budget removed all taxes, explain.",
    ]
    selected_adv = st.selectbox("Adversarial query", adversarial_queries)
    if st.button("Run adversarial retrieval test"):
        adv_retrieved = retrieve(
            query=selected_adv,
            records=records,
            vectors=vectors,
            top_k=top_k,
            alpha=alpha,
            use_expansion=query_expansion,
        )
        st.write("Top retrieved chunk IDs:", [x["record"].chunk_id for x in adv_retrieved])
        st.write("Top scores:", [round(float(x["score"]), 4) for x in adv_retrieved])
        append_stage_log(
            "adversarial_test",
            {
                "query": selected_adv,
                "retrieved_ids": [x["record"].chunk_id for x in adv_retrieved],
                "scores": [round(float(x["score"]), 6) for x in adv_retrieved],
            },
        )

st.markdown("---")
st.markdown(
    "### Chunking Design Justification\n"
    "- Chunk size default is **800 chars** with **120 overlap** to preserve paragraph-level coherence while keeping embeddings focused.\n"
    "- Smaller chunks increase precision but can fragment context; larger chunks improve recall but may introduce noise.\n"
)
