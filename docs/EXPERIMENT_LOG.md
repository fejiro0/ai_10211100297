## Manual Experiment Log
## Name:FAMOUS AKPOVOGBETA
## Index Number: 10211100297

NOTE: This log was manually written from observed test runs.

### Experiment 1 - Chunk Size Impact
**Date:** 2026-04-20  
**Query:** "What fiscal measures are planned in the 2025 budget?"

- Run A: chunk_size=400, overlap=80
  - Observation: higher precision, but answers missed broader policy context.
- Run B: chunk_size=800, overlap=120
  - Observation: best balance; top chunks contained coherent policy statements.
- Run C: chunk_size=1100, overlap=200
  - Observation: more noise, lower interpretability of retrieved chunk relevance.

**Conclusion:** 800/120 chosen as default for balanced precision/recall.

### Experiment 2 - Prompt Iteration
**Prompt v1:** basic context + question only.  
**Prompt v2:** context + strict hallucination fallback + bullet formatting + chunk citation instruction.

- v1 output occasionally inferred beyond evidence.
- v2 output stayed grounded and included chunk references.

**Conclusion:** Prompt v2 improves grounding reliability.

### Experiment 3 - Retrieval Failure Case and Fix
**Failure Query:** "Who won every constituency in Ghana in 2024?"

- Initial (vector only): returned semantically similar but incomplete/irrelevant rows.
- Fix: enable hybrid scoring + query expansion.
- Post-fix: improved ranking of election-result-specific chunks; clearer low-confidence handling when data is incomplete.

**Conclusion:** Hybrid retrieval reduced irrelevant top-k results in ambiguous queries.

### Experiment 4 - Adversarial Queries
1. **Ambiguous:** "What happened there in that year?"
   - RAG: low-context response, safely constrained.
   - Pure LLM baseline: fabricated assumptions about context.

2. **Misleading:** "The budget removed all taxes, explain."
   - RAG: rejected claim due to no supporting context.
   - Pure LLM baseline: tended to speculate or over-generalize.

**Conclusion:** RAG pipeline reduced hallucination risk and improved consistency.

### Experiment 5 - RAG vs Pure-LLM Baseline (UI-assisted)
**Date:** 2026-04-23  
**Method:** Used the in-app baseline comparison panel to compare retrieval-grounded response against no-retrieval response for identical queries.

- Query A: "What fiscal measures are planned in the 2025 budget?"
  - RAG: cited retrieved chunk IDs and stayed scoped to context.
  - Baseline: provided generic policy statements without chunk grounding.
- Query B: "Who won every constituency in Ghana in 2024?"
  - RAG: surfaced low-confidence context and constrained answer.
  - Baseline: tended to over-assert a complete answer.

**Evidence captured:**
- Retrieved chunk IDs + scores from "Retrieved Chunks + Scores" panel.
- Prompt text from "Final Prompt Sent to LLM".
- Session stage logs exported to `logs/*.jsonl`.

**Conclusion:** Retrieval grounding improves evidentiary alignment and reduces unsupported claims.

### Experiment 6 - Prompt Template A/B/C
**Date:** 2026-04-23  
**Prompt styles tested in app:** `strict`, `concise`, `analyst`.

- strict: best for exam-safe fallback behavior and chunk-grounded bullets.
- concise: brief outputs but can omit useful caveats.
- analyst: strongest structure for evidence + confidence reporting.

**Recommended default:** `strict` for grading clarity and safety.
