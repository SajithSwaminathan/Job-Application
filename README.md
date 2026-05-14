# Job-Application — RAG-tailored CV with email automation

A small RAG pipeline that takes your indexed CV and a job description, generates a
job-specific tailored CV grounded **only** in facts from your resume, renders it to
PDF, and (optionally) emails it via an MCP-backed Gmail server. Built on Qdrant
(vector store), Ollama running `phi4-mini` (local LLM), Gradio (UI), and the
Model Context Protocol Python SDK.

## Prerequisites

- Python 3.10+
- Docker (to run Qdrant locally)
- [Ollama](https://ollama.com) with the `phi4-mini` model pulled
- A Gmail account with 2-Step Verification enabled and an App Password

## Setup

```bash
git clone https://github.com/SajithSwaminathan/Job-Application.git
cd Job-Application

# Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Qdrant (persistent storage in ./qdrant_storage)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Ollama model
ollama run phi4-mini

# Credentials
cp .env.example .env
# then edit .env and paste your 16-char Gmail App Password
# (full steps are in .env.example)
```

## Quick start

Launch the Gradio UI (ISRO chat + Job-Application tabs):

```bash
python rag_with_qdrant_ollama.py
```

Or use the CLI:

```bash
# Index a CV into the resume_kb collection
python rag_with_qdrant_ollama.py --index-cv /path/to/resume.pdf

# Generate a tailored CV from an inline JD (writes tailored.pdf)
python rag_with_qdrant_ollama.py --jd "Senior Python role at Acme..."

# Generate AND email it via the MCP server
python rag_with_qdrant_ollama.py --jd-file jd.txt --send

# Run the ISRO sample-questions demo
python rag_with_qdrant_ollama.py --demo-isro
```

## Architecture

Two Qdrant collections, never mixed:

| Collection | Purpose | Seeded by |
|---|---|---|
| `isro_knowledge_base` | Demo corpus on the Indian space programme | Hard-coded `DOCUMENTS` list |
| `resume_kb` | Chunked resume(s) for CV tailoring | `index_cv(pdf_path)` |

The CV pipeline is a deterministic five-step flow (no agent loop):

```
retrieve → generate → verify → render → email
```

Key functions to read first:

- [`build_application`](rag_with_qdrant_ollama.py#L925) — the end-to-end orchestration
- [`fetch_full_cv`](rag_with_qdrant_ollama.py#L520) — pulls **every** chunk of the chosen resume so the LLM sees the whole document, not just top-k semantic hits
- [`generate_tailored_cv`](rag_with_qdrant_ollama.py#L702) + [`CV_TEMPLATE_INSTRUCTIONS`](rag_with_qdrant_ollama.py#L633) — strict-grounding prompt with concrete negative examples
- [`find_unsupported_claims`](rag_with_qdrant_ollama.py#L572) — heuristic verifier that flags proper-noun phrases not present in the source CV

The email sender is a **separate MCP server** ([`mcp_email_server.py`](mcp_email_server.py)). The main script spawns it over stdio and calls `send_application_email` through the MCP client SDK. Because it's a real MCP server, it can also be registered with Claude Code (see `.claude/settings.json`) and called directly by the assistant — not only by the Python pipeline.

## Grounding rationale

Every claim the LLM writes must trace to a chunk that was actually retrieved. We
enforce this in **three layers** — any single layer is known to be insufficient:

1. **Physical isolation.** One Qdrant collection per data domain. Retrieval
   functions hard-code their collection name and never fall back.
2. **Generation rules.** Explicit prompt instructions ("use ONLY context", plus
   concrete `✗ / ✓` examples), and Ollama sampling pinned at
   `temperature=0.1, top_p=0.85, repeat_penalty=1.1` instead of the ~0.8 default.
3. **Post-generation verifier.** A programmatic scan for title-case proper-noun
   phrases that don't appear in the source corpus — flagged to the user, never
   silently stripped.

One detail people find counter-intuitive: at CV-generation time we feed the
**entire** resume, not just top-k chunks. With top-k, relevant facts that fall
below the cutoff create gaps the model wants to fill — and it fills them by
inventing plausible alternatives. Feeding the whole document removes the
incentive.

The canonical statement of this standard lives in [`~/.claude/skills/rag-grounding/SKILL.md`](../.claude/skills/rag-grounding/SKILL.md), with a CV-specific layer at [`FDE/.claude/skills/cv-layout/SKILL.md`](../.claude/skills/cv-layout/SKILL.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Cannot reach Qdrant at http://localhost:6333` | Qdrant container not running. See the error block at [`rag_with_qdrant_ollama.py:222`](rag_with_qdrant_ollama.py#L222) for the exact `docker run` command. |
| `ERROR: Cannot connect to Ollama` | Ollama not started. Run `ollama serve` (or `ollama run phi4-mini`) and retry. |
| `SMTPAuthenticationError` / app-password rejected | Most likely a typo, or 2-Step Verification not enabled. Workspace admins sometimes disable App Passwords org-wide — see the diagnostic hints at [`mcp_email_server.py:79`](mcp_email_server.py#L79). |
| `resume_kb is empty` when generating a CV | Index a resume first via `--index-cv /path/to/resume.pdf` or the Gradio upload box. |

## Files

- `rag_with_qdrant_ollama.py` — main pipeline + Gradio UI + CLI
- `mcp_email_server.py` — MCP server exposing `send_application_email`
- `.env.example` — Gmail credential template (copy to `.env`)
- `requirements.txt` — Python dependencies (loose minimums)
- `qdrant_storage/` — Docker volume mount for Qdrant (gitignored)
- `tailored.pdf` — default output path for generated CVs (gitignored)
