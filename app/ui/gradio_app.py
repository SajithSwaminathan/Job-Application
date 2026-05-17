"""Gradio UI: ISRO chat tab + Job-Application tab. The Job-Application tab
drives the LangGraph via `run_cv_pipeline`. The separate "Send" button calls
the MCP transport directly so it does not re-run (and re-generate) the graph.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from app.cv.email_client import RECIPIENT_EMAIL, send_application
from app.cv.store import index_cv
from app.isro_demo import _verify_against_context, ensure_isro_collection, generate_answer, retrieve
from app.ollama_client import warm_up
from app.qdrant_store import ISRO_COLLECTION, RESUME_COLLECTION, ensure_resume_collection
from app.graph.build import run_cv_pipeline


def isro_chat_fn(message: str, _history: list) -> str:
    docs = retrieve(message, top_k=3)
    answer = generate_answer(message, docs)
    sources = "\n".join(f"  • {d['title']}  (score: {d['score']:.3f})" for d in docs)
    warnings = _verify_against_context(answer, docs)
    if warnings:
        verifier_md = (
            "⚠️ **Verifier flagged these phrases (not in retrieved context):**\n"
            + "\n".join(f"  • `{w}`" for w in warnings)
        )
    else:
        verifier_md = "✅ **Verifier passed** — no out-of-context proper-noun phrases."
    return (
        f"{answer}\n\n---\n**Retrieved context:**\n{sources}\n\n{verifier_md}"
    )


def _ui_index_cv(file_obj) -> str:
    if file_obj is None:
        return "No file uploaded."
    try:
        n, src, contact = index_cv(file_obj.name)
        bits = [f"Indexed **{n}** chunks from **{src}** into `{RESUME_COLLECTION}`.", ""]
        bits.append("**Extracted contact info:**")
        for k in ("name", "email", "phone", "linkedin", "github"):
            v = contact.get(k) or "_(not found)_"
            bits.append(f"  • {k}: {v}")
        return "\n".join(bits)
    except Exception as e:
        return f"ERROR: {e}"


def _ui_generate(jd: str) -> tuple[str, str, str]:
    if not jd.strip():
        return "Please paste a job description.", "", ""
    try:
        result = run_cv_pipeline(jd, out_pdf="tailored.pdf")
        cv_text = result["cv_text"]
        pdf_path = result["pdf_path"]
        warnings = result["warnings"]
        top_chunks = result["top_chunks"]

        sources = "\n".join(
            f"  • {c['source_file']} (chunk {c['chunk_idx']}, score {c['score']:.3f})"
            for c in top_chunks
        )
        meta = [f"**Source CV:** `{result['source_file']}`"]
        meta.append(
            f"**Context fed to LLM:** {len(result['full_chunks'])} chunks "
            f"(entire resume, not just top-k)."
        )
        meta.append(f"**Top-k semantic hits (for transparency):**\n{sources}")

        if warnings:
            meta.append("")
            meta.append("⚠️  **Possible fabrications flagged by verifier:**")
            for w in warnings:
                meta.append(f"  • `{w}` — not found in source CV")
            meta.append("Review these in the preview; the LLM may have invented them.")
        else:
            meta.append("")
            meta.append("✅ Verifier found no unsupported proper-noun phrases.")

        return cv_text, str(pdf_path), "\n".join(meta)
    except Exception as e:
        return f"ERROR: {e}", "", ""


def _ui_send(jd: str, cv_text: str, pdf_path: str) -> str:
    if not pdf_path or not Path(pdf_path).is_file():
        return "Generate the tailored CV first."
    try:
        return send_application(pdf_path, jd, cv_text)
    except Exception as e:
        return f"ERROR: {e}"


def launch_chat_ui() -> None:
    ensure_isro_collection()
    ensure_resume_collection()
    warm_up()

    with gr.Blocks(title="RAG: ISRO + Job Application Assistant") as ui:
        gr.Markdown("# RAG Assistant — phi4-mini grounded on Qdrant")
        with gr.Tabs():
            with gr.Tab("ISRO Chat"):
                gr.ChatInterface(
                    fn=isro_chat_fn,
                    description=(
                        "Ask questions about ISRO. Answers are grounded in the "
                        f"`{ISRO_COLLECTION}` collection."
                    ),
                    examples=[
                        "When did Chandrayaan-3 land on the Moon?",
                        "Who led India's mission to Pluto?",
                        "What is the cost of India's Mars mission?",
                    ],
                )

            with gr.Tab("Job Application"):
                gr.Markdown(
                    f"Upload a CV (PDF) → it is chunked and indexed into "
                    f"`{RESUME_COLLECTION}`. Then paste a job description: a "
                    f"tailored CV is generated **using only your indexed resume "
                    f"data** (no other sources), rendered to PDF, and (optionally) "
                    f"emailed to **{RECIPIENT_EMAIL}** via the MCP server."
                )
                cv_file = gr.File(label="Upload CV (PDF)", file_types=[".pdf"])
                index_status = gr.Markdown()
                cv_file.change(_ui_index_cv, inputs=cv_file, outputs=index_status)

                jd_box = gr.Textbox(label="Job Description", lines=10)
                gen_btn = gr.Button("Generate Tailored CV", variant="primary")

                cv_preview = gr.Textbox(label="Tailored CV (preview)", lines=20)
                pdf_out = gr.File(label="Tailored CV PDF")
                sources_md = gr.Markdown()

                gen_btn.click(
                    _ui_generate,
                    inputs=jd_box,
                    outputs=[cv_preview, pdf_out, sources_md],
                )

                send_btn = gr.Button(f"Send to {RECIPIENT_EMAIL}")
                send_status = gr.Markdown()
                send_btn.click(
                    _ui_send,
                    inputs=[jd_box, cv_preview, pdf_out],
                    outputs=send_status,
                )

    print("\n" + "=" * 60)
    print("UI starting — Gradio will print the URL below once it's bound.")
    print("=" * 60 + "\n")
    # server_port=None → Gradio auto-picks a free port starting at 7860.
    ui.launch(server_name="127.0.0.1", server_port=None, inbrowser=False)
