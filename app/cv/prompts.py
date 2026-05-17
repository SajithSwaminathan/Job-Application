"""CV-generation prompts + the cover-letter prompt. Strict-grounding wording
(✗/✓ examples, "use ONLY context") is the documented contract — keep new rules
in the same style.
"""

from __future__ import annotations

import re

from app.ollama_client import generate

CV_TEMPLATE_INSTRUCTIONS = """You are a CV-tailoring assistant. You will rewrite the candidate's
resume to match a specific job description, using ONLY facts from the resume.

HARD RULES — non-negotiable. Violating any of these is failure:
- Use ONLY facts present in the RESUME CONTEXT below.
- Do NOT invent or add ANY new content:
    * no new employers, no new job titles, no new dates
    * no new skills, technologies, tools, frameworks
    * no new certifications, degrees, awards
    * no new projects, no new project responsibilities, no new project tech
    * no new metrics, numbers, percentages, team sizes, budgets
- If the job description asks for something not in the resume, OMIT it.
  Better to ship a shorter CV than a CV with one fabricated detail.
- You MAY re-order, re-phrase, summarise, and emphasise existing resume content
  so it speaks to the job description.
- Every concrete noun (company name, certification name, technology, project)
  in your output must appear verbatim (or as a clear morphological variant) in
  the resume context above. If you cannot find it in the resume, do not write it.

EXAMPLES of forbidden fabrication:
  ✗ Adding "AWS Solutions Architect" to certifications if the resume only lists
    "Google Cloud Associate".
  ✗ Adding "Migrated Lambda applications across regions" to a project bullet if
    the project section in the resume only mentions a diary app.
  ✗ Adding "Led a team of 8" if the resume never mentions team size.
  ✓ Re-phrasing "Built a REST API" as "Designed and shipped a REST API" — same
    fact, different wording.

OUTPUT FORMAT — follow this template EXACTLY. Use the exact section headers shown
in CAPS. Omit a whole section (header included) if the resume has no material for
it. Do NOT output the name or contact line — those will be prepended automatically.

PROFESSIONAL SUMMARY
A 3-4 sentence paragraph positioning the candidate for the target role. Lead
with years of experience and core domain. Reference 2-3 strengths from the
resume that map to the JD.

CORE SKILLS
- Skill or technology (one per line, dash-prefixed)
- Group related items where possible (e.g. "Python, FastAPI, Pytest")
- 8-15 bullets max, ordered by relevance to the JD

PROFESSIONAL EXPERIENCE
Job Title — Company — Dates
- Achievement bullet starting with a strong verb (Led, Built, Designed, Shipped…)
- Quantify with numbers ONLY when the resume already contains them
- 3-5 bullets per role; most recent role first
(repeat block per role)

EDUCATION
Degree — Institution — Year
(repeat per credential)

CERTIFICATIONS
- Certification name — Issuer — Year
(omit section if none in resume)

PROJECTS
Project Name — one-line description
- Optional bullet on impact or stack
(omit section if not relevant)

STYLE:
- Plain text only. No markdown fences, no asterisks for bold, no backticks.
- Keep the whole thing to ~450 words / one page.
- Be specific, concise, and active-voice.
"""

# Layered on top of CV_TEMPLATE_INSTRUCTIONS for the retry pass only, after the
# heuristic verifier flagged phrases in the first draft. Same ✗/✓ style — no new
# grounding rules, just a sharper reminder naming the offending phrases.
STRICT_RETRY_INSTRUCTIONS = """RETRY — your previous draft FAILED verification.

The following phrases appeared in your draft but DO NOT appear anywhere in the
RESUME CONTEXT. They are fabrications and must not appear again in ANY form:

{flagged}

Re-read the RESUME CONTEXT. Regenerate the CV body. For each flagged phrase
above, either (a) drop it entirely, or (b) replace it with the closest fact
that IS present verbatim in the resume context. Do not substitute one invented
detail for another.
  ✗ Replacing fabricated "AWS Solutions Architect" with fabricated "Azure Architect".
  ✓ Dropping a certification line the resume never mentions.
"""


def generate_tailored_cv(
    jd: str,
    resume_chunks: list[dict],
    *,
    strict: bool = False,
    flagged: list[str] | None = None,
) -> str:
    """Call phi4-mini with the strict-grounding prompt to produce the CV BODY
    (sections only — header is prepended by the caller).

    `strict=True` prepends STRICT_RETRY_INSTRUCTIONS naming the `flagged`
    phrases the verifier rejected on the previous pass.
    """
    if not resume_chunks:
        return (
            "ERROR: resume_kb is empty. Index a CV first with "
            "`--index-cv /path/to/resume.pdf` or upload one in the Gradio UI."
        )

    context = "\n\n".join(
        f"[chunk {c['chunk_idx']} from {c['source_file']}]\n{c['text']}"
        for c in resume_chunks
    )

    retry_preamble = ""
    if strict:
        flagged_block = "\n".join(f"  - {p}" for p in (flagged or [])) or "  (none recorded)"
        retry_preamble = STRICT_RETRY_INSTRUCTIONS.format(flagged=flagged_block) + "\n\n"

    prompt = f"""{retry_preamble}{CV_TEMPLATE_INSTRUCTIONS}

RESUME CONTEXT (the only source of truth — do not go beyond this):
{context}

JOB DESCRIPTION:
{jd}

Now produce the tailored CV body, starting at PROFESSIONAL SUMMARY.
Do NOT include the candidate's name or contact line — those are added separately.
"""
    body = generate(prompt).strip()
    if body.startswith("ERROR:"):
        return body
    # Defensive cleanup: strip any markdown code fences the model leaks.
    if body.startswith("```"):
        body = re.sub(r"^```[\w]*\n?|\n?```$", "", body).strip()
    return body


def compose_cover_letter(jd: str, cv_text: str) -> str:
    """Short plain-text email body referencing the JD. Signs with the name
    parsed from the first line of the (already-assembled) CV text."""
    jd_snippet = jd.strip().splitlines()[0][:120] if jd.strip() else "the advertised role"
    # First non-empty line of the assembled CV is the candidate's name.
    sender = next(
        (ln.strip() for ln in cv_text.splitlines() if ln.strip()),
        "the candidate",
    )
    return (
        f"Hello,\n\n"
        f"Please find attached my CV tailored for the following role:\n"
        f"  {jd_snippet}\n\n"
        f"The attached CV highlights the experience and skills from my "
        f"background most relevant to this opportunity. I would welcome the "
        f"chance to discuss how I can contribute to your team.\n\n"
        f"Best regards,\n"
        f"{sender}\n"
    )
