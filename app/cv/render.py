"""Deterministic CV assembly + PDF rendering. The name/contact header is built
here (never by the LLM) so it can't be fabricated.
"""

from __future__ import annotations

from pathlib import Path

from app.cv.verifier import SECTION_HEADINGS


def _pick_contact_info(chunks: list[dict]) -> dict:
    """Pick contact info from the highest-scored chunk that has any contact data.
    If no chunk has contact info, return empty defaults."""
    for c in chunks:  # already ordered by score desc
        contact = c.get("contact") or {}
        if any(contact.get(k) for k in ("name", "email", "phone", "linkedin", "github")):
            return contact
    return {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}


def _format_contact_line(contact: dict) -> str:
    """Build the contact line that sits under the name."""
    parts = [contact.get("email"), contact.get("phone"), contact.get("linkedin"), contact.get("github")]
    return "  •  ".join(p for p in parts if p)


def assemble_full_cv(body: str, contact: dict) -> str:
    """Prepend a deterministic header (name + contact line) to the LLM-generated body."""
    name = (contact.get("name") or "").strip() or "Candidate"
    contact_line = _format_contact_line(contact)
    parts = [name]
    if contact_line:
        parts.append(contact_line)
    parts.append("")  # blank line separating header from body
    parts.append(body)
    return "\n".join(parts)


def render_cv_to_pdf(cv_text: str, out_path: str | Path) -> Path:
    """Render the tailored CV text to a one-column PDF.

    Convention: first non-empty line is the candidate's name (large title);
    second non-empty line (if present and not a recognised section header) is
    the contact line. Everything else is rendered as body + section headings.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    out_path = Path(out_path)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "Name", parent=styles["Title"], fontSize=22, leading=26, spaceAfter=2,
        textColor="#0b2a52",
    )
    contact_style = ParagraphStyle(
        "Contact", parent=styles["Normal"], fontSize=10, leading=12,
        textColor="#444444", spaceAfter=10, alignment=1,  # center
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, leading=14,
        spaceBefore=10, spaceAfter=4, textColor="#0b2a52",
    )
    body_style = styles["BodyText"]
    body_style.spaceAfter = 3
    bullet_style = ParagraphStyle(
        "Bullet", parent=body_style, leftIndent=14, bulletIndent=2, spaceAfter=2,
    )

    def escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def is_section_heading(line: str) -> bool:
        s = line.strip().rstrip(":")
        return s.upper() in SECTION_HEADINGS

    lines = cv_text.splitlines()
    # Find first/second non-empty lines for name/contact treatment.
    nonblank_indices = [i for i, ln in enumerate(lines) if ln.strip()]
    name_idx = nonblank_indices[0] if nonblank_indices else None
    contact_idx = None
    if len(nonblank_indices) >= 2 and not is_section_heading(lines[nonblank_indices[1]]):
        contact_idx = nonblank_indices[1]

    story = []
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 4))
            continue

        safe = escape(line)
        if i == name_idx:
            story.append(Paragraph(safe, name_style))
        elif i == contact_idx:
            story.append(Paragraph(safe, contact_style))
        elif is_section_heading(line):
            story.append(Paragraph(safe.rstrip(":").upper(), section_style))
        elif line.lstrip().startswith(("-", "•", "*")):
            text = line.lstrip()[1:].strip()
            story.append(Paragraph(escape(text), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(safe, body_style))

    doc.build(story)
    return out_path
