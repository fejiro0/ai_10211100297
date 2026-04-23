## Best Way to Answer the Exam (Step-by-Step)
## Name:FAMOUS AKPOVOGBETA
## Index Number: 10211100297

Use this exact workflow when preparing your final submission and viva/video.

### 1) Demonstrate Part A (Data + Chunking)
1. Show cleaning code in `streamlit_files/rag_app.py`.
2. Run one query with three chunk settings (e.g., 400/80, 800/120, 1100/200).
3. Record retrieval differences manually in `docs/EXPERIMENT_LOG.md`.

### 2) Demonstrate Part B (Custom Retrieval + Failure/Fix)
1. Explain embedding model and in-memory vector matrix.
2. Show top-k retrieval scores in the UI.
3. Use a known failure query and capture irrelevant top chunks.
4. Re-run with hybrid scoring/query expansion and show improved ranking.

### 3) Demonstrate Part C (Prompt Engineering)
1. Use same query with prompt styles:
   - `strict`
   - `concise`
   - `analyst`
2. Compare grounding quality and hallucination behavior.
3. Keep evidence in manual logs (not AI summaries).

### 4) Demonstrate Part D (Full Pipeline + Logging)
For one sample query, screenshot and explain:
- Retrieved chunks and scores
- Final prompt sent to model
- Final response
- Stage logs panel
- Exported `logs/*.jsonl` file

### 5) Demonstrate Part E (Adversarial + Baseline)
1. Run at least 2 adversarial queries from the app panel.
2. Compare RAG response vs no-retrieval baseline.
3. Report:
   - Accuracy (manual judgement)
   - Hallucination rate (unsupported claims count)
   - Consistency score (from app)

### 6) Demonstrate Part F (Architecture)
1. Use `docs/ARCHITECTURE.md` diagram.
2. Explain each component and why mixed CSV/PDF domain needs hybrid retrieval.

### 7) Demonstrate Part G (Innovation)
1. Show feedback buttons (`Helpful/Not helpful`) on retrieved chunks.
2. Explain how this updates source bias and affects future ranking.

### 8) Final Packaging Checklist
- Update `docs/DELIVERY.md` with deployment + video links.
- Keep `docs/EXPERIMENT_LOG.md` human-written with dates, settings, and observations.
- Ensure README links all evidence docs.
