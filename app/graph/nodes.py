"""LangGraph node functions. Each returns a partial-state dict. Nodes contain
no control flow — routing lives in build.py's conditional edges.
"""

from __future__ import annotations

from app.cv.email_client import send_application
from app.cv.prompts import generate_tailored_cv
from app.cv.render import _pick_contact_info, assemble_full_cv, render_cv_to_pdf
from app.cv.store import fetch_full_cv, retrieve_resume
from app.cv.verifier import find_unsupported_claims
from app.graph.state import CVApplicationState

_EMPTY_CONTACT = {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}


def retrieve_top(state: CVApplicationState) -> dict:
    """Semantic top-k against resume_kb to pick which indexed CV is relevant.

    On an empty collection, short-circuit the rest of the pipeline: produce the
    canonical empty-collection error body and route straight to render (no
    verify, no header) — matching the legacy build_application behaviour.
    """
    top_chunks = retrieve_resume(state["jd"], top_k=state.get("top_k", 8))
    if not top_chunks:
        err = generate_tailored_cv(state["jd"], [])  # canonical error string
        return {
            "top_chunks": [],
            "source_file": None,
            "cv_body": err,
            "cv_text": err,
            "contact": dict(_EMPTY_CONTACT),
            "warnings": [],
        }
    return {"top_chunks": top_chunks, "source_file": top_chunks[0]["source_file"]}


def fetch_full(state: CVApplicationState) -> dict:
    """Pull the ENTIRE resume for the chosen source_file (not just top-k)."""
    full_chunks = fetch_full_cv(state["source_file"])
    contact = _pick_contact_info(full_chunks) or _pick_contact_info(state["top_chunks"])
    return {"full_chunks": full_chunks, "contact": contact}


def generate_cv(state: CVApplicationState) -> dict:
    """Strict-grounded generation. retry_count>0 → stricter retry prompt naming
    the phrases the verifier flagged on the previous pass."""
    retry_count = state.get("retry_count", 0)
    body = generate_tailored_cv(
        state["jd"],
        state["full_chunks"],
        strict=retry_count > 0,
        flagged=state.get("warnings"),
    )
    return {"cv_body": body, "retry_count": retry_count + 1}


def verify_cv(state: CVApplicationState) -> dict:
    """Heuristic verifier over the full source text. Flags surface, never strip."""
    source_text = "\n\n".join(c["text"] for c in state["full_chunks"])
    return {"warnings": find_unsupported_claims(state["cv_body"], source_text)}


def assemble(state: CVApplicationState) -> dict:
    """Prepend the deterministic name/contact header."""
    return {"cv_text": assemble_full_cv(state["cv_body"], state["contact"])}


def render(state: CVApplicationState) -> dict:
    """Render the assembled CV text to PDF."""
    pdf_path = render_cv_to_pdf(state["cv_text"], state.get("out_pdf", "tailored.pdf"))
    return {"pdf_path": str(pdf_path)}


def send(state: CVApplicationState) -> dict:
    """Email the PDF via the MCP server. Errors propagate (isError contract)."""
    result = send_application(state["pdf_path"], state["jd"], state["cv_text"])
    return {"send_result": result}
