"""FastAPI webhook server for the Montferrand booking agent.

Endpoints:

    POST /sms            — Twilio incoming SMS webhook
    POST /admin/tenants  — Upsert a tenant configuration (bearer token auth)
    GET  /health         — Health check
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from montferrand_agent.conversation import (
    ConversationError,
    conversation_key_for_sms,
    process_message,
)
from montferrand_agent.tenant import (
    TenantNotFoundError,
    load_tenant_profile,
    save_tenant_profile,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Montferrand Agent", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@app.post("/sms", dependencies=[Depends(_validate_twilio_signature)])
async def sms_webhook(request: Request) -> Response:
    """Handle an incoming SMS from Twilio and return a TwiML response."""
    form = await request.form()

    twilio_number: str = form.get("To", "")  # type: ignore[assignment]
    from_number: str = form.get("From", "")  # type: ignore[assignment]
    body: str = form.get("Body", "")  # type: ignore[assignment]

    if not twilio_number or not from_number:
        raise HTTPException(status_code=400, detail="Missing To or From")

    # TODO: handle MMS media (NumMedia, MediaUrl0, etc.)

    # Load tenant profile
    try:
        tenant_profile = load_tenant_profile(twilio_number)
    except TenantNotFoundError:
        logger.error("No tenant config for %s", twilio_number)
        return _twiml_response(
            "Desolé, ce service n'est pas configuré. "
            "Veuillez contacter l'entreprise directement."
        )

    # Derive conversation key and process
    key = conversation_key_for_sms(twilio_number, from_number)

    try:
        result = await process_message(key, body, tenant_profile=tenant_profile)
    except ConversationError as exc:
        logger.error("Agent error for %s: %s", key, exc)
        return _twiml_response(
            "Une erreur est survenue. Veuillez réessayer dans quelques instants."
        )

    return _twiml_response(result.message)


# ---------------------------------------------------------------------------
# POST /admin/tenants — Upsert tenant config
# ---------------------------------------------------------------------------


class TenantUpsertRequest(BaseModel):
    """Request body for creating/updating a tenant."""

    twilio_number: str
    tenant_profile: str


@app.post(
    "/admin/tenants",
    status_code=201,
    dependencies=[Depends(_validate_admin_token)],
)
async def upsert_tenant(payload: TenantUpsertRequest) -> dict[str, str]:
    """Create or update a tenant's profile."""
    path = save_tenant_profile(payload.twilio_number, payload.tenant_profile)
    logger.info("Tenant upserted: %s -> %s", payload.twilio_number, path)
    return {"status": "ok", "path": str(path)}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
