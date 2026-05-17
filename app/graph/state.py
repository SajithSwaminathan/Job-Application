"""Shared state for the CV-application LangGraph.

Reducer notes:
- `warnings` uses the default (replace) reducer on purpose — the retry pass's
  warnings supersede the first pass's; we surface the latest, not the union.
- `errors` accumulates across nodes via operator.add so a node can append a
  diagnostic without clobbering an earlier one.
- `retry_count` is an int; nodes return the incremented value explicitly (no
  reducer) so the conditional edge can cap retries deterministically.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CVApplicationState(TypedDict, total=False):
    # Inputs
    jd: str
    out_pdf: str
    top_k: int
    should_send: bool
    # Retrieval
    top_chunks: list[dict]
    source_file: str | None
    full_chunks: list[dict]
    contact: dict
    # Generation
    cv_body: str
    cv_text: str
    retry_count: int
    # Verification
    warnings: list[str]
    # Output
    pdf_path: str | None
    send_result: str | None
    # Diagnostics
    errors: Annotated[list[str], operator.add]
