"""CLI entry point. Same flags/behaviour as before the refactor."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.cv.store import index_cv
from app.isro_demo import _demo_isro
from app.graph.build import run_cv_pipeline
from app.ui.gradio_app import launch_chat_ui


def main() -> None:
    p = argparse.ArgumentParser(description="Qdrant + phi4-mini RAG + job-app automation")
    p.add_argument("--index-cv", metavar="PDF", help="Index a CV PDF into resume_kb")
    p.add_argument("--jd", metavar="TEXT", help="Job description text (inline)")
    p.add_argument("--jd-file", metavar="PATH", help="Job description from a file")
    p.add_argument("--out", default="tailored.pdf", help="Output PDF path (default: tailored.pdf)")
    p.add_argument("--top-k", type=int, default=8, help="# of resume chunks to retrieve")
    p.add_argument("--send", action="store_true", help="Email the generated PDF via MCP")
    p.add_argument("--demo-isro", action="store_true", help="Run the original ISRO sample-question demo")
    args = p.parse_args()

    did_something = False

    if args.demo_isro:
        _demo_isro()
        did_something = True

    if args.index_cv:
        _, _, contact = index_cv(args.index_cv)
        print("Extracted contact:")
        for k, v in contact.items():
            print(f"  {k:>10}: {v or '(not found)'}")
        did_something = True

    jd = args.jd
    if args.jd_file:
        jd = Path(args.jd_file).read_text()

    if jd:
        result = run_cv_pipeline(
            jd, out_pdf=args.out, top_k=args.top_k, should_send=args.send
        )
        cv_text = result["cv_text"]
        pdf_path = result["pdf_path"]
        print("\n" + "=" * 60)
        print("TAILORED CV")
        print("=" * 60)
        print(cv_text)
        print("=" * 60)
        print(f"PDF written to: {pdf_path}")
        print(f"Source CV: {result['source_file']}")
        print(
            f"LLM context: full resume "
            f"({len(result['full_chunks'])} chunks, "
            f"~{sum(len(c['text']) for c in result['full_chunks'])} chars)"
        )
        print("Top-k semantic hits:")
        for c in result["top_chunks"]:
            print(f"  - {c['source_file']} #{c['chunk_idx']}  score={c['score']:.3f}")

        if result["warnings"]:
            print("\n⚠️  Verifier flagged these as possibly fabricated:")
            for w in result["warnings"]:
                print(f"  - {w!r}  (not in source CV)")
            print("Review the CV; the LLM may have invented these.")
        else:
            print("\n✅ Verifier: no unsupported proper-noun phrases detected.")

        if args.send:
            print(f"\nMCP send result: {result['send_result']}")
        did_something = True

    if not did_something:
        launch_chat_ui()


if __name__ == "__main__":
    main()
