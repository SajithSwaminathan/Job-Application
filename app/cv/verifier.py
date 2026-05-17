"""Post-generation anti-fabrication verifier — layer 3 of the grounding
contract. Heuristic only (documented contract in CLAUDE.md): never widen to
LLM-as-judge without an explicit ask. Flagged phrases are surfaced to the user,
never silently stripped.
"""

from __future__ import annotations

import re

# Section headers used by the CV template. Lives here (not render.py) because
# the verifier must skip our own headers, and render.py imports it back.
SECTION_HEADINGS = {
    "PROFESSIONAL SUMMARY",
    "CORE SKILLS",
    "SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "EXPERIENCE",
    "EDUCATION",
    "CERTIFICATIONS",
    "PROJECTS",
    "SUMMARY",
    "TECHNICAL SKILLS",
}

# Words we ignore when scanning for "proper-noun phrases" because they only get
# capitalised because of sentence-start or section-header rules.
_VERIFIER_IGNORE = {
    "professional", "summary", "core", "skills", "experience", "education",
    "certifications", "projects", "technical", "candidate", "name", "contact",
    "led", "built", "designed", "shipped", "delivered", "developed", "managed",
    "implemented", "created", "improved", "reduced", "increased", "the", "and",
    "for", "with", "from", "of", "to", "in", "on", "at", "by", "as", "a", "an",
    "or", "but", "if", "then", "than", "this", "that", "those", "these",
}


def _normalize(s: str) -> str:
    """Lowercase + collapse non-word characters into single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s)).strip().lower()


def find_unsupported_claims(body: str, source: str) -> list[str]:
    """Heuristic verifier: returns proper-noun-style phrases in `body` that do
    not appear in `source`. False positives are possible (it's a heuristic),
    but for the categories users actually complain about — invented company
    names, certifications, product names, project titles — it catches them
    reliably.

    Detection: title-case 2-5 token phrases. Match: substring match against a
    whitespace-normalised lowercased source corpus.
    """
    src_norm = _normalize(source)

    phrase_re = re.compile(
        r"\b("
        r"(?:[A-Z][A-Za-z0-9+&./-]{1,})"           # first token, title-case
        r"(?:[ -]+(?:[A-Z][A-Za-z0-9+&./-]+|of|and|the|for|in|on|to|with)){1,4}"
        r")\b"
    )

    flagged: list[str] = []
    seen: set[str] = set()
    for line in body.splitlines():
        # Skip our own section headers (all-caps lines)
        if line.strip().rstrip(":").upper() in SECTION_HEADINGS:
            continue
        for m in phrase_re.finditer(line):
            phrase = m.group(1).strip(" -")
            tokens = re.split(r"[ -]+", phrase)
            if len(tokens) < 2:
                continue
            # Skip phrases that are entirely common verbs/stopwords capitalised at line start.
            if all(t.lower() in _VERIFIER_IGNORE for t in tokens):
                continue
            norm = _normalize(phrase)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if norm not in src_norm:
                flagged.append(phrase)
    return flagged
