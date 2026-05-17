# Project conventions — Job-Application

RAG over a CV → tailor to a JD → render PDF → email via MCP. Local stack:
Qdrant (Docker), Ollama running `phi4-mini`, Gradio UI, MCP email server.

## Run / dev commands

- Gradio UI (default): `python rag_with_qdrant_ollama.py`
- ISRO grounding demo (smoke test): `python rag_with_qdrant_ollama.py --demo-isro`
- Index a CV: `python rag_with_qdrant_ollama.py --index-cv /path/to/resume.pdf`
- Tailor from a JD: `python rag_with_qdrant_ollama.py --jd "..."` (add `--send` to email)

Qdrant and Ollama must be up first — see the README for the `docker run …` and
`ollama run phi4-mini` invocations.

## Module map

The pipeline lives in the `app/` package; `rag_with_qdrant_ollama.py` is a thin
shim re-exporting public names (CLI/imports unchanged):

- `app/embedding.py` — SentenceTransformer + `VECTOR_SIZE`
- `app/qdrant_store.py` — Qdrant client, collection names, `ensure_resume_collection`
- `app/ollama_client.py` — `generate()` (pinned grounded sampling), `warm_up()`
- `app/isro_demo.py` — ISRO demo: `DOCUMENTS`, `rag()`, `_demo_isro()` (NOT in the graph)
- `app/cv/` — `extract` (PDF/regex), `store` (index/retrieve/fetch_full),
  `verifier`, `prompts`, `render`, `email_client` (MCP transport)
- `app/graph/` — LangGraph: `state` (TypedDict), `nodes`, `build`
  (`run_cv_pipeline` is the legacy `build_application` replacement)
- `app/ui/gradio_app.py`, `app/cli.py` — presentation/entry

The CV flow is now a `StateGraph` with one verifier-driven retry edge:
`retrieve_top → fetch_full → generate_cv → verify_cv ⇄ (one stricter retry) →
assemble → render → (send if flagged)`. Warnings are still surfaced after the
retry, never silently stripped.

## Active skills

Both apply when editing this project:

- `rag-grounding` — `~/.claude/skills/rag-grounding/SKILL.md` (three-layer grounding standard)
- `cv-layout` — `FDE/.claude/skills/cv-layout/SKILL.md` (CV-specific layout / anti-fabrication rules built on top of rag-grounding)

## Non-negotiables

- **Three-layer grounding** for any new retrieve-then-generate path:
  isolation (dedicated Qdrant collection), grounded prompt with concrete
  `✗ / ✓` examples + sampling at `temperature=0.1, top_p=0.85, repeat_penalty=1.1`,
  and a post-generation pass through `find_unsupported_claims`.
- **Never share a Qdrant collection across data domains.** `isro_knowledge_base`
  and `resume_kb` exist for exactly this reason.
- For "tailor / summarise *this* document" tasks, feed the **full** document
  via `fetch_full_cv` — not top-k chunks. Top-k invites the model to fabricate
  gap-fillers.
- Verifier warnings are **surfaced to the user**, never silently stripped.

## Test before claiming done

Minimum regression check after any pipeline change:

```bash
python rag_with_qdrant_ollama.py --demo-isro
```

The demo includes "Who led India's mission to Pluto?" — the answer must mention
**Sajith** (a planted document) and the verifier must pass. This is the canary
for grounding fidelity: a real ISRO chairman name appearing instead means the
model leaked training knowledge through the prompt.

## What not to do

- Don't add the `tailored.pdf`, `.env`, `qdrant_storage/`, or `.venv/` to git —
  `.gitignore` already excludes them.
- Don't pin exact dependency versions in `requirements.txt` without asking —
  the convention here is loose minimums.
- Don't widen the verifier to LLM-as-judge unless asked; the heuristic version
  is the documented contract.
