"""FastAPI webhook server for the Montferrand booking agent.

Endpoints:

    POST /sms            — Twilio incoming SMS webhook
    POST /admin/tenants  — Upsert a tenant configuration (bearer token auth)
    GET  /health         — Health check

Architecture note — async SMS replies:

    The SMS webhook does NOT return the agent's reply inline via TwiML.
    Instead it returns an empty TwiML response immediately (so Twilio
    never times out), and launches a background task that:

    1. Acquires a per-conversation lock (prevents race conditions)
    2. Calls ``process_message()`` (which can take arbitrarily long)
    3. Sends the reply via the Twilio REST API (``_send_sms``)

    If Twilio retries a webhook (e.g., because of a previous timeout or
    5xx), the ``MessageSid`` dedup logic detects the duplicate and skips
    processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from montferrand_agent.conversation import (
    ConversationError,
    conversation_key_for_sms,
    process_message,
    reset_tenant,
)
from montferrand_agent.tenant import (
    TenantConfig,
    TenantNotFoundError,
    load_tenant_config,
    save_tenant_config,
    save_tenant_profile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MessageSid deduplication
# ---------------------------------------------------------------------------

_DEDUP_TTL = 60.0  # seconds to remember a MessageSid

_seen_sids: dict[str, float] = {}  # MessageSid -> monotonic timestamp


def _is_duplicate(message_sid: str) -> bool:
    """Return True if this MessageSid was already seen within the TTL.

    Also prunes expired entries on each call to keep memory bounded.
    """
    now = time.monotonic()

    # Prune expired entries
    expired = [sid for sid, ts in _seen_sids.items() if now - ts > _DEDUP_TTL]
    for sid in expired:
        del _seen_sids[sid]

    if message_sid in _seen_sids:
        return True

    _seen_sids[message_sid] = now
    return False


# ---------------------------------------------------------------------------
# In-flight task tracking for graceful shutdown
# ---------------------------------------------------------------------------

_inflight_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _track_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """Add a background task to the tracked set; auto-remove on completion."""
    _inflight_tasks.add(task)
    task.add_done_callback(_inflight_tasks.discard)


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Application lifespan — drain in-flight tasks on shutdown."""
    yield
    # Graceful shutdown: wait for all background tasks to complete
    if _inflight_tasks:
        logger.info(
            "Waiting for %d in-flight task(s) to complete...",
            len(_inflight_tasks),
        )
        await asyncio.gather(*_inflight_tasks, return_exceptions=True)
        logger.info("All in-flight tasks completed.")


app = FastAPI(
    title="Montferrand Agent",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Twilio REST API — outbound SMS via httpx
# ---------------------------------------------------------------------------


async def _send_sms(to: str, from_: str, body: str) -> None:
    """Send an SMS via the Twilio REST API.

    Uses httpx (async) to POST to the Twilio Messages endpoint.

    Raises:
        RuntimeError: If TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is not set.
        httpx.HTTPStatusError: If the Twilio API returns an error.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()

    if not account_sid or not auth_token:
        raise RuntimeError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must both be set "
            "to send outbound SMS."
        )

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            auth=(account_sid, auth_token),
            data={"To": to, "From": from_, "Body": body},
            timeout=30.0,
        )
        response.raise_for_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_twiml() -> Response:
    """Return an empty TwiML response (no message)."""
    twiml = MessagingResponse()
    return Response(content=str(twiml), media_type="application/xml")


def _twiml_response(text: str) -> Response:
    """Build a TwiML XML response containing a single SMS message."""
    twiml = MessagingResponse()
    twiml.message(text)
    return Response(content=str(twiml), media_type="application/xml")


# ---------------------------------------------------------------------------
# Twilio signature validation
# ---------------------------------------------------------------------------


def _get_twilio_validator() -> RequestValidator | None:
    """Build a Twilio RequestValidator from the auth token, or None."""
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if not token:
        return None
    return RequestValidator(token)


async def _validate_twilio_signature(request: Request) -> None:
    """FastAPI dependency that validates the Twilio request signature.

    Skipped (with a warning) if ``TWILIO_AUTH_TOKEN`` is not set, which
    is useful during local development.
    """
    validator = _get_twilio_validator()
    if validator is None:
        logger.warning("TWILIO_AUTH_TOKEN not set — skipping signature validation")
        return

    signature = request.headers.get("X-Twilio-Signature", "")
    # Behind a reverse proxy (e.g., Fly.io), request.url uses http://
    # but Twilio signed against the public https:// URL.
    url = str(request.url.replace(scheme="https"))
    form = dict(await request.form())

    if not validator.validate(url, form, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


# ---------------------------------------------------------------------------
# Admin bearer token auth
# ---------------------------------------------------------------------------


async def _validate_admin_token(request: Request) -> None:
    """FastAPI dependency that validates the admin bearer token."""
    expected = os.environ.get("MONTFERRAND_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="MONTFERRAND_ADMIN_TOKEN is not configured on the server",
        )

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = auth[len("Bearer ") :]
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


# ---------------------------------------------------------------------------
# POST /sms — Twilio incoming SMS webhook
# ---------------------------------------------------------------------------


async def _handle_sms(
    twilio_number: str,
    from_number: str,
    body: str,
    tenant_profile: str,
    *,
    is_boss: bool = False,
) -> None:
    """Background task: process the message and send the reply via Twilio API.

    This runs after the webhook has already returned an empty TwiML to
    Twilio.  Errors are logged and an error SMS is sent to the sender.
    """
    key = conversation_key_for_sms(twilio_number, from_number)

    try:
        result = await process_message(
            key,
            body,
            tenant_profile=tenant_profile,
            twilio_number=twilio_number,
            is_boss=is_boss,
        )
        reply = result.message
    except ConversationError as exc:
        logger.error("Agent error for %s: %s", key, exc)
        reply = "Une erreur est survenue. Veuillez réessayer dans quelques instants."
    except Exception as exc:
        logger.exception("Unexpected error for %s: %s", key, exc)
        reply = "Une erreur est survenue. Veuillez réessayer dans quelques instants."

    try:
        await _send_sms(to=from_number, from_=twilio_number, body=reply)
    except Exception as exc:
        logger.exception("Failed to send SMS reply to %s: %s", from_number, exc)


@app.post("/sms", dependencies=[Depends(_validate_twilio_signature)])
async def sms_webhook(request: Request) -> Response:
    """Handle an incoming SMS from Twilio.

    Returns an empty TwiML response immediately (no message).  The actual
    agent reply is sent asynchronously via the Twilio REST API in a
    background task — so the LLM can take as long as it needs.
    """
    form = await request.form()

    twilio_number: str = form.get("To", "")  # type: ignore[assignment]
    from_number: str = form.get("From", "")  # type: ignore[assignment]
    body: str = form.get("Body", "")  # type: ignore[assignment]
    message_sid: str = form.get("MessageSid", "")  # type: ignore[assignment]

    if not twilio_number or not from_number:
        raise HTTPException(status_code=400, detail="Missing To or From")

    # Dedup: skip if we already saw this MessageSid
    if message_sid and _is_duplicate(message_sid):
        logger.info("Duplicate MessageSid %s — skipping", message_sid)
        return _empty_twiml()

    # Load tenant config (sync and fast — can respond inline on error)
    try:
        config = load_tenant_config(twilio_number)
    except TenantNotFoundError:
        logger.error("No tenant config for %s", twilio_number)
        return _twiml_response(
            "Desolé, ce service n'est pas configuré. "
            "Veuillez contacter l'entreprise directement."
        )

    boss = from_number in config.boss_numbers

    # Launch background task and return immediately
    task = asyncio.create_task(
        _handle_sms(
            twilio_number,
            from_number,
            body,
            config.profile,
            is_boss=boss,
        )
    )
    _track_task(task)

    return _empty_twiml()


# ---------------------------------------------------------------------------
# POST /admin/tenants — Upsert tenant config
# ---------------------------------------------------------------------------


class TenantUpsertRequest(BaseModel):
    """Request body for creating/updating a tenant."""

    twilio_number: str
    tenant_profile: str
    boss_numbers: list[str] = []


@app.post(
    "/admin/tenants",
    status_code=201,
    dependencies=[Depends(_validate_admin_token)],
)
async def upsert_tenant(payload: TenantUpsertRequest) -> dict[str, str]:
    """Create or update a tenant's configuration."""
    config = TenantConfig(
        phone=payload.twilio_number,
        profile=payload.tenant_profile,
        boss_numbers=payload.boss_numbers,
    )
    path = save_tenant_config(config)
    logger.info("Tenant upserted: %s -> %s", payload.twilio_number, path)
    return {"status": "ok", "path": str(path)}


# ---------------------------------------------------------------------------
# DELETE /admin/tenants/{twilio_number}/conversations
# ---------------------------------------------------------------------------


@app.delete(
    "/admin/tenants/{twilio_number}/conversations",
    dependencies=[Depends(_validate_admin_token)],
)
async def delete_tenant_conversations(twilio_number: str) -> dict[str, object]:
    """Delete all conversation data for a tenant."""
    count = reset_tenant(twilio_number)
    logger.info("Reset tenant %s: deleted %d conversation(s)", twilio_number, count)
    return {"status": "ok", "deleted": count}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
