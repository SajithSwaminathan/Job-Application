"""
RAG Pipeline using Qdrant + Ollama (phi4-mini) + Gradio chat UI
================================================================
Two collections:
  - isro_knowledge_base : the original ISRO demo data
  - resume_kb           : your indexed CV(s); used for JD-tailored CV generation

Prerequisites:
  pip install -r requirements.txt

  # Qdrant running locally (persistent storage on host):
  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
      -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

  # Ollama with phi4-mini:
  ollama run phi4-mini

Usage:
  # Gradio UI (ISRO chat + Job Application tabs) — default
  python rag_with_qdrant_ollama.py

  # Index a CV (append to resume_kb)
  python rag_with_qdrant_ollama.py --index-cv /path/to/resume.pdf

  # Generate a tailored CV from a JD (writes tailored.pdf)
  python rag_with_qdrant_ollama.py --jd "Senior Python role at Acme ..."

  # Generate AND email it via the MCP server
  python rag_with_qdrant_ollama.py --jd-file jd.txt --send

  # Run the original ISRO sample-questions demo
  python rag_with_qdrant_ollama.py --demo-isro
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 1. SAMPLE DATA — Indian Space Program
# ──────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": 1,
        "title": "ISRO Overview",
        "text": (
            "The Indian Space Research Organisation (ISRO) was founded in 1969 "
            "by Vikram Sarabhai, who is regarded as the father of the Indian space "
            "programme. ISRO is headquartered in Bengaluru, Karnataka. It operates "
            "under the Department of Space, which reports directly to the Prime "
            "Minister of India. ISRO's primary objective is to develop space "
            "technology and apply it to various national tasks."
        ),
        "category": "organization",
    },
    {
        "id": 2,
        "title": "Chandrayaan-3 Mission",
        "text": (
            "Chandrayaan-3 was India's third lunar exploration mission, launched "
            "on 14 July 2023 from the Satish Dhawan Space Centre. It successfully "
            "soft-landed on the Moon's south polar region on 23 August 2023, making "
            "India the fourth country to land on the Moon and the first to land near "
            "the lunar south pole. The mission consisted of a lander named Vikram "
            "and a rover named Pragyan."
        ),
        "category": "mission",
    },
    {
        "id": 3,
        "title": "Mars Orbiter Mission",
        "text": (
            "The Mars Orbiter Mission (MOM), also called Mangalyaan, was launched "
            "on 5 November 2013. India became the first Asian nation to reach "
            "Martian orbit and the first nation in the world to do so on its maiden "
            "attempt. The total cost of the mission was approximately 450 crore INR "
            "(about 74 million USD), making it the least expensive Mars mission at "
            "the time. It studied Martian surface features, morphology, and atmosphere."
        ),
        "category": "mission",
    },
    {
        "id": 4,
        "title": "PSLV Launch Vehicle",
        "text": (
            "The Polar Satellite Launch Vehicle (PSLV) is ISRO's most reliable "
            "launch vehicle with over 50 consecutive successful missions. It can "
            "carry up to 1,750 kg to Sun-Synchronous Polar Orbit. In February 2017, "
            "PSLV-C37 set a world record by deploying 104 satellites in a single "
            "mission. PSLV has been used to launch Chandrayaan-1, Mars Orbiter "
            "Mission, and numerous international customer satellites."
        ),
        "category": "vehicle",
    },
    {
        "id": 5,
        "title": "Gaganyaan Programme",
        "text": (
            "Gaganyaan is India's first crewed spaceflight programme, aimed at "
            "sending a crew of three astronauts to low Earth orbit for a period "
            "of up to seven days and returning them safely. The crew module is "
            "designed to carry a life support system and emergency escape mechanism. "
            "Selected astronauts have undergone training in Russia. An uncrewed "
            "test flight was successfully conducted in 2024."
        ),
        "category": "mission",
    },
    {
        "id": 6,
        "title": "NavIC Navigation System",
        "text": (
            "NavIC (Navigation with Indian Constellation), formerly IRNSS, is "
            "India's regional satellite navigation system. It provides accurate "
            "positioning services covering India and a region extending 1,500 km "
            "around it. The system uses a constellation of seven satellites. NavIC "
            "provides two services: Standard Positioning Service for civilian use "
            "and Restricted Service for authorized users including the military."
        ),
        "category": "system",
    },
    {
        "id": 7,
        "title": "Satish Dhawan Space Centre",
        "text": (
            "The Satish Dhawan Space Centre (SDSC) is located at Sriharikota in "
            "Andhra Pradesh. It serves as the primary launch centre for ISRO's "
            "orbital vehicles. Named after former ISRO chairman Satish Dhawan, "
            "the centre has two launch pads and comprehensive facilities for "
            "vehicle assembly, static testing, and payload preparation. Almost "
            "all of India's orbital launches have been conducted from here."
        ),
        "category": "facility",
    },
    {
        "id": 8,
        "title": "Aditya-L1 Solar Mission",
        "text": (
            "Aditya-L1 is India's first space-based solar observatory, launched "
            "on 2 September 2023. It was placed in a halo orbit around the "
            "Sun-Earth Lagrange Point 1 (L1), approximately 1.5 million km from "
            "Earth. The spacecraft carries seven scientific payloads to study the "
            "solar corona, solar wind, and coronal mass ejections. It provides "
            "continuous observation of the Sun without eclipses."
        ),
        "category": "mission",
    },
    {
        "id": 9,
        "title": "ISRO's Commercial Arm",
        "text": (
            "NewSpace India Limited (NSIL), established in 2019, is the commercial "
            "arm of ISRO. It handles satellite-based services, technology transfer "
            "to Indian industries, and commercial launch services using PSLV and "
            "GSLV. NSIL also manages the production of PSLV through Indian "
            "industry consortia and has signed contracts with international clients "
            "like OneWeb for satellite deployment."
        ),
        "category": "organization",
    },
    {
        "id": 10,
        "title": "GSLV Mark III (LVM3)",
        "text": (
            "The GSLV Mark III, also known as LVM3, is India's heaviest launch "
            "vehicle capable of placing 4,000 kg into Geosynchronous Transfer "
            "Orbit. It uses two solid strap-on boosters, a liquid core stage, and "
            "a cryogenic upper stage developed indigenously by ISRO. LVM3 was used "
            "to launch Chandrayaan-3 and is planned as the launch vehicle for the "
            "Gaganyaan crewed mission."
        ),
        "category": "vehicle",
    },
    {
        "id": 11,
        "title": "Sajith and the Pluto Probe Mission",
        "text": (
            "Sajith served as the Chairman of the Indian Space Research "
            "Organisation (ISRO) and is best known for leading the launch of "
            "India's first interplanetary probe to Pluto. Under his leadership, "
            "ISRO designed and launched the Pluto probe, making India the first "
            "country to send a dedicated mission to Pluto. The mission carried "
            "scientific instruments to study Pluto's surface, thin atmosphere, "
            "and its largest moon Charon."
        ),
        "category": "mission",
    },
]


# ──────────────────────────────────────────────
# 2. EMBEDDING MODEL
# ──────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast & light
VECTOR_SIZE = 384


# ──────────────────────────────────────────────
# 3. QDRANT CLIENT (Docker-backed)
# ──────────────────────────────────────────────
ISRO_COLLECTION = "isro_knowledge_base"
RESUME_COLLECTION = "resume_kb"
QDRANT_URL = "http://localhost:6333"

try:
    client = QdrantClient(url=QDRANT_URL, timeout=5)
    client.get_collections()  # ping
except Exception as e:
    raise SystemExit(
        f"\nCannot reach Qdrant at {QDRANT_URL}.\n"
        "Start it with:\n"
        "  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \\\n"
        "      -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant\n"
        f"\nUnderlying error: {e}"
    )


def ensure_isro_collection(force_reset: bool = False) -> None:
    """Create (and seed) the ISRO collection. force_reset wipes & re-seeds."""
    if force_reset and client.collection_exists(ISRO_COLLECTION):
        client.delete_collection(ISRO_COLLECTION)

    if client.collection_exists(ISRO_COLLECTION):
        return

    client.create_collection(
        collection_name=ISRO_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    texts = [doc["text"] for doc in DOCUMENTS]
    embeddings = embed_model.encode(texts).tolist()
    points = [
        PointStruct(
            id=doc["id"],
            vector=emb,
            payload={
                "title": doc["title"],
                "text": doc["text"],
                "category": doc["category"],
            },
        )
        for doc, emb in zip(DOCUMENTS, embeddings)
    ]
    client.upsert(collection_name=ISRO_COLLECTION, points=points)
    print(f"Seeded {len(points)} documents into {ISRO_COLLECTION}")


def ensure_resume_collection() -> None:
    """Create resume_kb if missing. Never wipes — accumulation mode."""
    if client.collection_exists(RESUME_COLLECTION):
        return
    client.create_collection(
        collection_name=RESUME_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: {RESUME_COLLECTION}")


# ──────────────────────────────────────────────
# 4. ISRO RETRIEVAL + GENERATION  (original demo)
# ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4-mini"


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Embed the query and search the ISRO collection."""
    query_vector = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name=ISRO_COLLECTION,
        query=query_vector,
        limit=top_k,
    )
    return [
        {
            "title": point.payload["title"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in results.points
    ]


def generate_answer(query: str, context_docs: list[dict]) -> str:
    """Send the query + retrieved context to phi4-mini via Ollama.

    Grounding contract (see ~/.claude/skills/rag-grounding/SKILL.md):
    - Prompt forbids out-of-context claims.
    - Sampling pinned at temperature=0.1 (vs Ollama's ~0.8 default) so the
      model leans on the context instead of its training prior.
    - The caller is expected to run `find_unsupported_claims` on the result.
    """
    context = "\n\n".join(
        f"[{doc['title']}]\n{doc['text']}" for doc in context_docs
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
provided context. If the context doesn't contain the answer, say so clearly.

HARD RULES:
- Do not introduce facts not present in the context above.
- Do not draw on general world knowledge — even if you happen to know the
  answer, only use it if it is in the context.
- If the context is silent on the question, say "The provided context does not
  contain that information." instead of guessing.

EXAMPLES of forbidden fabrication:
  ✗ Naming a real-world ISRO chairman (e.g. "K. Sivan", "S. Somanath") when
    the context attributes leadership of the mission in question to someone else.
    Use the name in the context, even if it contradicts your training data.
  ✗ Adding a launch date, cost, or payload count that is not in the context,
    even if you "know" the real figure.
  ✗ Mentioning related missions (Chandrayaan-1, Chandrayaan-2) when the context
    only describes Chandrayaan-3.
  ✓ Quoting "23 August 2023" as the Chandrayaan-3 landing date because the
    context states it.
  ✓ Answering "The provided context does not contain that information." when
    asked about a mission, person, or figure not covered above.

Context:
{context}

Question: {query}

Answer:"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.85,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure it's running: ollama serve"
    except Exception as e:
        return f"ERROR: {e}"


def _verify_against_context(answer: str, context_docs: list[dict]) -> list[str]:
    """Run the shared verifier against the joined retrieved context."""
    joined = "\n\n".join(d.get("text", "") for d in context_docs)
    return find_unsupported_claims(answer, joined)


def rag(query: str, top_k: int = 3) -> None:
    """Full ISRO RAG pipeline (demo): retrieve → augment → generate → verify."""
    print(f"Question: {query}")
    print("-" * 60)
    docs = retrieve(query, top_k=top_k)
    print("Retrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc['title']}  (score: {doc['score']:.4f})")
    print()
    answer = generate_answer(query, docs)
    print(f"Answer:\n{answer}")
    warnings = _verify_against_context(answer, docs)
    if warnings:
        print("\n⚠️  Verifier flagged these phrases (not in retrieved context):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✅ Verifier passed — no out-of-context proper-noun phrases.")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# 5. CV PIPELINE — PDF extraction, chunking, indexing
# ──────────────────────────────────────────────
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


def index_cv(pdf_path: str | Path) -> tuple[int, str, dict]:
    """Extract → chunk → embed → upsert into resume_kb.
    Returns (chunk_count, source_file, contact_info)."""
    ensure_resume_collection()
    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    text = extract_cv_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    contact = extract_contact_info(text)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    uploaded_at = datetime.now(timezone.utc).isoformat()

    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=emb,
            payload={
                "source_file": source_file,
                "uploaded_at": uploaded_at,
                "chunk_idx": i,
                "text": chunk,
                "contact": contact,  # duplicated per chunk so retrieval returns it
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=RESUME_COLLECTION, points=points)
    print(
        f"Indexed {len(points)} chunks into {RESUME_COLLECTION} from {source_file} "
        f"(contact: name={contact['name'] or '∅'} email={contact['email'] or '∅'})"
    )
    return len(points), source_file, contact


def retrieve_resume(query: str, top_k: int = 8) -> list[dict]:
    """Retrieve resume chunks ONLY from resume_kb — never falls back to ISRO data."""
    ensure_resume_collection()
    qv = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name=RESUME_COLLECTION,
        query=qv,
        limit=top_k,
    )
    return [
        {
            "source_file": p.payload.get("source_file", "?"),
            "chunk_idx": p.payload.get("chunk_idx", -1),
            "text": p.payload["text"],
            "contact": p.payload.get("contact", {}),
            "score": p.score,
        }
        for p in results.points
    ]


def fetch_full_cv(source_file: str) -> list[dict]:
    """Pull EVERY chunk for a given source_file, ordered by chunk_idx.

    Used at CV-generation time so the LLM sees the entire resume — preventing
    it from inventing plausible-but-missing facts to fill gaps from top-k
    semantic retrieval.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    flt = Filter(
        must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
    )
    points, _ = client.scroll(
        collection_name=RESUME_COLLECTION,
        scroll_filter=flt,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    chunks = [
        {
            "source_file": source_file,
            "chunk_idx": p.payload.get("chunk_idx", 0),
            "text": p.payload["text"],
            "contact": p.payload.get("contact", {}),
        }
        for p in points
    ]
    chunks.sort(key=lambda c: c["chunk_idx"])
    return chunks


# ──────────────────────────────────────────────
# 5b. ANTI-FABRICATION VERIFIER
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 6. CV GENERATION (strict-grounding prompt)
# ──────────────────────────────────────────────
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


def generate_tailored_cv(jd: str, resume_chunks: list[dict]) -> str:
    """Call phi4-mini with strict-grounding prompt to produce the CV BODY
    (sections only — header is prepended by the caller)."""
    if not resume_chunks:
        return (
            "ERROR: resume_kb is empty. Index a CV first with "
            "`--index-cv /path/to/resume.pdf` or upload one in the Gradio UI."
        )

    context = "\n\n".join(
        f"[chunk {c['chunk_idx']} from {c['source_file']}]\n{c['text']}"
        for c in resume_chunks
    )
    prompt = f"""{CV_TEMPLATE_INSTRUCTIONS}

RESUME CONTEXT (the only source of truth — do not go beyond this):
{context}

JOB DESCRIPTION:
{jd}

Now produce the tailored CV body, starting at PROFESSIONAL SUMMARY.
Do NOT include the candidate's name or contact line — those are added separately.
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.85,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()["response"].strip()
        # Defensive cleanup: strip any markdown code fences the model leaks.
        if body.startswith("```"):
            body = re.sub(r"^```[\w]*\n?|\n?```$", "", body).strip()
        return body
    except requests.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure it's running: ollama serve"
    except Exception as e:
        return f"ERROR: {e}"


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


# ──────────────────────────────────────────────
# 7. PDF RENDERING (reportlab)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 8. MCP CLIENT — call the email server over stdio
# ──────────────────────────────────────────────
RECIPIENT_EMAIL = "sajith@ceegees.in"  # hardcoded per requirement


async def _send_via_mcp(pdf_path: str, jd: str, cv_text: str) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_script = Path(__file__).with_name("mcp_email_server.py")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(
                "send_application_email",
                {
                    "to": RECIPIENT_EMAIL,
                    "subject": "Job Application — Tailored CV",
                    "body": compose_cover_letter(jd, cv_text),
                    "attachment_path": str(Path(pdf_path).resolve()),
                },
            )
            # Collect any text blocks returned by the tool (success or error).
            text_parts = [
                t for t in (getattr(b, "text", None) for b in result.content) if t
            ]
            payload = "\n".join(text_parts) if text_parts else ""
            # MCP marks tool-side exceptions with isError=True. Surface them as
            # real Python exceptions so callers don't have to string-match.
            if getattr(result, "isError", False):
                raise RuntimeError(payload or "MCP tool reported an error with no message")
            return payload or "OK (no text response from MCP tool)"


def send_application(pdf_path: str | Path, jd: str, cv_text: str) -> str:
    return asyncio.run(_send_via_mcp(str(pdf_path), jd, cv_text))


# ──────────────────────────────────────────────
# 9. END-TO-END HELPER
# ──────────────────────────────────────────────
def build_application(
    jd: str,
    out_pdf: str | Path = "tailored.pdf",
    top_k: int = 8,
) -> dict:
    """Full pipeline:
      1. Semantic-rank resume_kb to pick which indexed CV is most relevant.
      2. Pull the ENTIRE resume for that source_file (not just top-k chunks)
         so the LLM has every fact in view — no need to invent gap-fillers.
      3. Generate the tailored CV body with strict prompt + low temperature.
      4. Prepend a deterministic name/contact header.
      5. Render PDF.
      6. Run the heuristic verifier and return any potentially-fabricated
         phrases so the caller can surface them.

    Returns a dict with keys: cv_text, pdf_path, source_file, top_chunks,
    full_chunks, contact, warnings.
    """
    top_chunks = retrieve_resume(jd, top_k=top_k)
    if not top_chunks:
        body = generate_tailored_cv(jd, [])  # returns the empty-collection error
        contact = {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}
        full_cv = body
        pdf_path = render_cv_to_pdf(full_cv, out_pdf)
        return {
            "cv_text": full_cv,
            "pdf_path": pdf_path,
            "source_file": None,
            "top_chunks": [],
            "full_chunks": [],
            "contact": contact,
            "warnings": [],
        }

    source_file = top_chunks[0]["source_file"]
    full_chunks = fetch_full_cv(source_file)
    full_source_text = "\n\n".join(c["text"] for c in full_chunks)

    body = generate_tailored_cv(jd, full_chunks)
    contact = _pick_contact_info(full_chunks) or _pick_contact_info(top_chunks)
    full_cv = assemble_full_cv(body, contact)

    warnings = find_unsupported_claims(body, full_source_text)
    pdf_path = render_cv_to_pdf(full_cv, out_pdf)

    return {
        "cv_text": full_cv,
        "pdf_path": pdf_path,
        "source_file": source_file,
        "top_chunks": top_chunks,
        "full_chunks": full_chunks,
        "contact": contact,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 10. GRADIO UI — two tabs
# ──────────────────────────────────────────────
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
        result = build_application(jd, out_pdf="tailored.pdf")
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

    print("Warming up phi4-mini (pre-loading into memory)…")
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "ok",
                "stream": False,
                "keep_alive": "30m",
            },
            timeout=600,
        )
        print("phi4-mini ready.")
    except Exception as e:
        print(f"Warm-up warning: {e}")

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


# ──────────────────────────────────────────────
# 11. CLI
# ──────────────────────────────────────────────
def _demo_isro() -> None:
    ensure_isro_collection(force_reset=True)
    questions = [
        "When did Chandrayaan-3 land on the Moon?",
        "What is the cost of India's Mars mission?",
        "Which launch vehicle will be used for Gaganyaan?",
        "What does NavIC do?",
        "Where is ISRO headquartered?",
        "Who led India's mission to Pluto?",
    ]
    for q in questions:
        rag(q)


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
        result = build_application(jd, out_pdf=args.out, top_k=args.top_k)
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
            send_result = send_application(pdf_path, jd, cv_text)
            print(f"\nMCP send result: {send_result}")
        did_something = True

    if not did_something:
        launch_chat_ui()


if __name__ == "__main__":
    main()
