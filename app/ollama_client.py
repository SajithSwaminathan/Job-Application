"""Thin Ollama HTTP client. `generate()` posts a prompt with the project's
pinned low-temperature sampling defaults (see the grounding contract in
~/.claude/skills/rag-grounding/SKILL.md). `warm_up()` is a function — never
called at import time — so importing this module does not silently hit Ollama.
"""

from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4-mini"

# Pinned sampling — leans on the provided context instead of the training prior.
_GROUNDED_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.85,
    "repeat_penalty": 1.1,
}


def generate(prompt: str, *, options: dict | None = None) -> str:
    """Send a prompt to phi4-mini via Ollama and return the raw response text.

    Connection/other errors are returned as `ERROR: ...` strings so callers can
    surface them verbatim (the existing contract throughout the pipeline).
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": options if options is not None else _GROUNDED_OPTIONS,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure it's running: ollama serve"
    except Exception as e:
        return f"ERROR: {e}"


def warm_up() -> None:
    """Pre-load phi4-mini into memory so the first real request is fast."""
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
