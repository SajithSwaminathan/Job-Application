"""CV text extraction — pure Python (PDF + regex). No Qdrant, no embeddings,
so this module is freely importable from anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_cv_text(pdf_path: str | Path) -> str:
    """Extract text from a PDF resume."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


# Pre-compiled regexes for contact-info extraction.
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_LINKEDIN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w%-]+/?", re.IGNORECASE)
_RE_GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.IGNORECASE)
_RE_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.()-]*)?(?:\(?\d{2,4}\)?[\s.()-]*)?\d{3,4}[\s.()-]?\d{3,4}"
)


def extract_contact_info(text: str) -> dict:
    """Pull name/email/phone/linkedin/github from raw CV text using regex heuristics."""
    info = {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}

    if m := _RE_EMAIL.search(text):
        info["email"] = m.group(0)
    if m := _RE_LINKEDIN.search(text):
        info["linkedin"] = m.group(0).rstrip("/")
    if m := _RE_GITHUB.search(text):
        info["github"] = m.group(0).rstrip("/")

    # Phone: keep first candidate whose digit-count is plausible (8-15) and which
    # is not the year of an email/postcode hit.
    for m in _RE_PHONE.finditer(text):
        candidate = m.group(0).strip(" .-()")
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15:
            info["phone"] = candidate
            break

    # Name: scan the first ~15 non-empty lines for a 2-5 token mostly-Title-Case
    # line containing no digits/@/url. Resumes almost always lead with the name.
    for raw in text.splitlines()[:30]:
        s = raw.strip()
        if not s or len(s) > 80:
            continue
        low = s.lower()
        if "@" in s or "http" in low or "www." in low or any(c.isdigit() for c in s):
            continue
        tokens = s.split()
        if 1 < len(tokens) <= 5 and sum(t[:1].isupper() for t in tokens) >= len(tokens) - 1:
            info["name"] = s
            break
    return info


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Paragraph-aware chunker: merges paragraphs until ~size chars, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= size:
            buf += "\n\n" + para
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
    if buf:
        chunks.append(buf)
    # Fallback: hard-split anything still much larger than `size`.
    final: list[str] = []
    for c in chunks:
        if len(c) <= size * 2:
            final.append(c)
        else:
            step = max(1, size - overlap)
            for i in range(0, len(c), step):
                final.append(c[i : i + size])
    return final
