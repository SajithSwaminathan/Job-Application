"""ISRO grounding demo — kept as the original simple retrieve→generate→verify
`rag()` function (NOT wrapped in the LangGraph). It is the CLAUDE.md canary:
`--demo-isro` must still mention the planted "Sajith" Pluto doc and pass the
verifier. Do not route stdout through a logger — the smoke test reads stdout.
"""

from __future__ import annotations

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embedding import VECTOR_SIZE, embed_model
from app.ollama_client import generate
from app.qdrant_store import ISRO_COLLECTION, client
from app.cv.verifier import find_unsupported_claims

# ──────────────────────────────────────────────
# SAMPLE DATA — Indian Space Program
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
    return generate(prompt)


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
