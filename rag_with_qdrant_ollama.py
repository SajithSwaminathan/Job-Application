"""Thin shim — preserves the CLI and import surface from before the `app/`
refactor. The implementation now lives in the `app/` package:

  app/embedding.py      embedding model
  app/qdrant_store.py   Qdrant client + collections
  app/ollama_client.py  phi4-mini HTTP client
  app/isro_demo.py      ISRO grounding demo (rag(), --demo-isro)
  app/cv/               CV pipeline (extract, store, verifier, prompts, render, email)
  app/graph/            LangGraph orchestration (state, nodes, build)
  app/ui/gradio_app.py  Gradio UI
  app/cli.py            argparse entry point

Run it exactly as before:
  python rag_with_qdrant_ollama.py                 # Gradio UI
  python rag_with_qdrant_ollama.py --demo-isro
  python rag_with_qdrant_ollama.py --index-cv resume.pdf
  python rag_with_qdrant_ollama.py --jd "..." [--send]
"""

from __future__ import annotations

from app.embedding import VECTOR_SIZE as VECTOR_SIZE
from app.embedding import embed_model as embed_model
from app.qdrant_store import QDRANT_URL as QDRANT_URL
from app.qdrant_store import ISRO_COLLECTION as ISRO_COLLECTION
from app.qdrant_store import RESUME_COLLECTION as RESUME_COLLECTION
from app.qdrant_store import client as client
from app.qdrant_store import ensure_resume_collection as ensure_resume_collection
from app.ollama_client import OLLAMA_URL as OLLAMA_URL
from app.ollama_client import MODEL_NAME as MODEL_NAME
from app.ollama_client import generate as generate
from app.isro_demo import DOCUMENTS as DOCUMENTS
from app.isro_demo import ensure_isro_collection as ensure_isro_collection
from app.isro_demo import retrieve as retrieve
from app.isro_demo import generate_answer as generate_answer
from app.isro_demo import _verify_against_context as _verify_against_context
from app.isro_demo import rag as rag
from app.isro_demo import _demo_isro as _demo_isro
from app.cv.extract import extract_cv_text as extract_cv_text
from app.cv.extract import extract_contact_info as extract_contact_info
from app.cv.extract import chunk_text as chunk_text
from app.cv.store import index_cv as index_cv
from app.cv.store import retrieve_resume as retrieve_resume
from app.cv.store import fetch_full_cv as fetch_full_cv
from app.cv.verifier import find_unsupported_claims as find_unsupported_claims
from app.cv.verifier import SECTION_HEADINGS as SECTION_HEADINGS
from app.cv.verifier import _VERIFIER_IGNORE as _VERIFIER_IGNORE
from app.cv.prompts import CV_TEMPLATE_INSTRUCTIONS as CV_TEMPLATE_INSTRUCTIONS
from app.cv.prompts import generate_tailored_cv as generate_tailored_cv
from app.cv.prompts import compose_cover_letter as compose_cover_letter
from app.cv.render import render_cv_to_pdf as render_cv_to_pdf
from app.cv.render import assemble_full_cv as assemble_full_cv
from app.cv.email_client import RECIPIENT_EMAIL as RECIPIENT_EMAIL
from app.cv.email_client import send_application as send_application
from app.graph.build import run_cv_pipeline as run_cv_pipeline
from app.graph.build import run_cv_pipeline as build_application  # back-compat alias
from app.ui.gradio_app import launch_chat_ui as launch_chat_ui
from app.cli import main as main

if __name__ == "__main__":
    main()
