"""MCP transport only: spawn mcp_email_server.py over stdio and call its
`send_application_email` tool. Cover-letter wording lives in cv/prompts.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.cv.prompts import compose_cover_letter

RECIPIENT_EMAIL = "sajith@ceegees.in"  # hardcoded per requirement


async def _send_via_mcp(pdf_path: str, jd: str, cv_text: str) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # mcp_email_server.py sits next to the project root, beside this package.
    server_script = Path(__file__).resolve().parents[2] / "mcp_email_server.py"
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
