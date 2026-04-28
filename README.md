## CS4241 - Introduction to Artificial Intelligence (2026)
## Name:FAMOUS AKPOVOGBETA
## Index Number: 10211100297
## Repository Name: ai_10211100297

This repository contains my **project-based examination submission** for CS4241.  
It implements a **custom Retrieval-Augmented Generation (RAG) chatbot** for Academic City, using the required datasets and avoiding end-to-end frameworks such as LangChain or LlamaIndex.

## 1) Project Scope and Exam Alignment

The implementation follows the required pipeline:

**User Query -> Retrieval -> Context Selection -> Prompt -> LLM/Generator -> Response**

Implemented with:
- Manual chunking
- Manual embedding + vector matrix handling
- Manual similarity scoring (cosine)
- Hybrid retrieval extension (vector + keyword)
- Query expansion extension
- Prompt construction with hallucination guardrail
- Stage-by-stage logs in UI
- Prompt-style A/B/C testing in UI
- RAG vs no-retrieval baseline comparison panel
- Adversarial testing panel
- Exportable JSONL experiment/stage logs
- Innovation: feedback-driven source bias for retrieval ranking

## 2) Required Datasets

- Ghana election result CSV:  
  `https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv`
- 2025 Budget Statement PDF:  
  `https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf`

The app downloads and processes these at runtime.

## 3) Run Instructions

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4) PART-BY-PART Mapping

### Part A: Data Engineering & Preparation
- Data cleaning for CSV:
  - trim column names
  - trim string values
  - drop duplicates
  - fill numeric nulls
- Chunking strategy implemented with configurable:
  - `chunk_size`
  - `overlap`
- Comparative chunking analysis enabled through Streamlit sliders and logs.

### Part B: Custom Retrieval System
- Embedding pipeline: `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage: in-memory NumPy embedding matrix
- Top-k retrieval: configurable in UI
- Similarity scoring: cosine similarity
- Extension implemented:
  - Hybrid search (vector + keyword overlap)
  - Query expansion (simple domain synonym map)
- Failure case + fix documented in `docs/EXPERIMENT_LOG.md`.

### Part C: Prompt Engineering & Generation
- Prompt template injects retrieved chunk context.
- Hallucination control enforced with fallback rule:
  - "I do not have enough grounded context..."
- Context window management through max character budget and ranked truncation.
- Prompt comparison experiments documented in `docs/EXPERIMENT_LOG.md`.

### Part D: Full RAG Pipeline
- Complete pipeline implemented in `streamlit_files/rag_app.py`.
- UI displays:
  - Retrieved chunks
  - Similarity/keyword/final scores
  - Final prompt sent to generator
  - Session stage logs and JSONL export

### Part E: Critical Evaluation & Adversarial Testing
- Includes adversarial and ambiguous queries in experiment log.
- Compares RAG response behavior against no-retrieval baseline (manual comparison log).
- Provides in-app adversarial panel + baseline comparison panel for repeatable tests.

### Part F: Architecture & System Design
- Architecture documented in `docs/ARCHITECTURE.md`.
- Includes component interactions and design justification.

### Part G: Innovation Component
- Novel feature included: **feedback-driven source bias** (`Helpful/Not helpful`) that adjusts future retrieval scoring.

## 5) Final Deliverables Checklist

- [x] GitHub codebase
- [x] Streamlit UI with:
  - query input
  - retrieved chunk display
  - final response
- [x] Detailed documentation in this README + `/docs`
- [ ] Video walkthrough (<=2 minutes): add your link in `docs/DELIVERY.md`
- [x] Manual experiment logs in `docs/EXPERIMENT_LOG.md`
- [x] Architecture explanation in `docs/ARCHITECTURE.md`
- [x] Answer strategy in `docs/ANSWER_STRATEGY.md`
- [ ] Deployment URL: add your cloud URL in `docs/DELIVERY.md`

## 6) Submission Compliance Notes

- Add/invite collaborator: `godwin.danso@acity.edu.gh` or `GodwinDansoAcity`.
- Send GitHub + deployment links to: `godwin.danso@acity.edu.gh`.
- Email subject format:
  `CS4241-Introduction to Artificial Intelligence-2026:10211100297 Famous Akpovogbeta`

## 7) Academic Integrity

This submission is structured as unique work. All experiments and logs are documented manually.
