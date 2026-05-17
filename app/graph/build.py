"""StateGraph wiring + the `run_cv_pipeline` façade. The façade returns the
same dict shape the legacy `build_application` did, so call sites change in one
line.

Flow:
    START → retrieve_top
    retrieve_top ─(empty)→ render
                 └(hit)──→ fetch_full → generate_cv → verify_cv
    verify_cv ─(warnings & retry_count<=1)→ generate_cv   (one stricter retry)
              └(else)────────────────────→ assemble → render
    render ─(should_send & source_file & pdf_path)→ send → END
           └(else)──────────────────────────────→ END
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import CVApplicationState


def _after_retrieve(state: CVApplicationState) -> str:
    return "render" if not state.get("top_chunks") else "fetch_full"


def _after_verify(state: CVApplicationState) -> str:
    # retry_count is post-increment: 1 after first pass, 2 after the retry pass.
    if state.get("warnings") and state.get("retry_count", 0) <= 1:
        return "generate_cv"
    return "assemble"


def _after_render(state: CVApplicationState) -> str:
    if (
        state.get("should_send")
        and state.get("source_file")
        and state.get("pdf_path")
    ):
        return "send"
    return END


@lru_cache(maxsize=1)
def build_cv_graph():
    """Compile (and cache) the CV-application graph."""
    g = StateGraph(CVApplicationState)
    g.add_node("retrieve_top", nodes.retrieve_top)
    g.add_node("fetch_full", nodes.fetch_full)
    g.add_node("generate_cv", nodes.generate_cv)
    g.add_node("verify_cv", nodes.verify_cv)
    g.add_node("assemble", nodes.assemble)
    g.add_node("render", nodes.render)
    g.add_node("send", nodes.send)

    g.add_edge(START, "retrieve_top")
    g.add_conditional_edges(
        "retrieve_top", _after_retrieve, {"render": "render", "fetch_full": "fetch_full"}
    )
    g.add_edge("fetch_full", "generate_cv")
    g.add_edge("generate_cv", "verify_cv")
    g.add_conditional_edges(
        "verify_cv", _after_verify, {"generate_cv": "generate_cv", "assemble": "assemble"}
    )
    g.add_edge("assemble", "render")
    g.add_conditional_edges("render", _after_render, {"send": "send", END: END})
    g.add_edge("send", END)
    return g.compile()


def run_cv_pipeline(
    jd: str,
    out_pdf: str = "tailored.pdf",
    top_k: int = 8,
    should_send: bool = False,
) -> dict:
    """Run the full CV-application graph and return the legacy result shape:
    cv_text, pdf_path, source_file, top_chunks, full_chunks, contact, warnings
    (+ send_result when emailed)."""
    final = build_cv_graph().invoke(
        {
            "jd": jd,
            "out_pdf": out_pdf,
            "top_k": top_k,
            "should_send": should_send,
            "retry_count": 0,
            "errors": [],
        }
    )
    return {
        "cv_text": final.get("cv_text", ""),
        "pdf_path": final.get("pdf_path"),
        "source_file": final.get("source_file"),
        "top_chunks": final.get("top_chunks", []),
        "full_chunks": final.get("full_chunks", []),
        "contact": final.get("contact", {}),
        "warnings": final.get("warnings", []),
        "send_result": final.get("send_result"),
    }
