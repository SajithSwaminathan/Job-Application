"""
MCP server: sends a job-application email with a PDF attachment via Gmail SMTP.

Run standalone for a quick check:
    python mcp_email_server.py

In practice, the main RAG script spawns this over stdio and calls
`send_application_email` through the MCP client SDK.

Credentials are loaded from a .env file (or the process environment) — see
`.env.example`. A Gmail "App Password" is required (Google account → Security →
2-Step Verification → App passwords). Regular account passwords will NOT work.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import certifi
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env sitting next to this file (and also fall back to CWD).
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()  # also read .env in CWD if any

GMAIL_USER = os.environ.get("GMAIL_USER", "").strip()
# Gmail accepts the 16-char app password with or without spaces, but spaces
# can confuse some shells/copy-paste flows. Normalise to the no-space form.
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").strip().replace(" ", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

mcp = FastMCP("job-application-email")


@mcp.tool()
def send_application_email(
    to: str,
    subject: str,
    body: str,
    attachment_path: str,
) -> str:
    """Send a job-application email with a PDF CV attached via Gmail SMTP."""
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        return (
            "ERROR: Gmail credentials not configured. "
            "Create a .env file next to mcp_email_server.py with "
            "GMAIL_USER and GMAIL_APP_PASSWORD (see .env.example)."
        )

    attachment = Path(attachment_path)
    if not attachment.is_file():
        return f"ERROR: attachment not found: {attachment_path}"

    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment.read_bytes(),
        maintype="application",
        subtype="pdf",
        filename=attachment.name,
    )

    # certifi gives Python a CA bundle on macOS where the system one is often
    # not wired up — fixes "CERTIFICATE_VERIFY_FAILED: unable to get local issuer".
    tls_ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls(context=tls_ctx)
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        is_workspace = not GMAIL_USER.endswith("@gmail.com")
        hints = [
            f"ERROR: SMTP auth failed for {GMAIL_USER!r}.",
            f"Gmail raw response: {e}",
            "",
            "Common causes (in order of likelihood):",
            "  1. The app password has a typo (look-alikes: l/1, 0/O, I/l).",
            "  2. 2-Step Verification is not enabled on this account — "
            "without 2SV, app passwords are silently invalid.",
            "  3. The app password has been revoked or expired.",
        ]
        if is_workspace:
            hints.append(
                "  4. (Workspace) Your admin has disabled App Passwords for the "
                "organisation. Many Workspace orgs force OAuth-only auth — in "
                "that case, smtp.gmail.com basic auth will never work and you "
                "need a different transport (OAuth2, SMTP relay, or a "
                "transactional provider like Brevo/Resend)."
            )
        hints.append("")
        hints.append(
            f"Loaded credentials: GMAIL_USER set ({len(GMAIL_USER)} chars), "
            f"GMAIL_APP_PASSWORD set ({len(GMAIL_APP_PASSWORD)} chars after "
            f"stripping spaces; Gmail expects 16)."
        )
        return "\n".join(hints)
    except Exception as e:
        return f"ERROR: failed to send mail: {e}"

    return f"Sent to {to} with attachment {attachment.name}"


if __name__ == "__main__":
    mcp.run()
